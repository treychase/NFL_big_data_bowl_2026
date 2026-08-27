"""End to end: raw files in, scouting artifacts out."""

import json

import numpy as np
import pandas as pd
import pytest

from nfl_scouting import data, metrics, model, pipeline


@pytest.fixture(scope="module")
def built(tmp_path_factory, synthetic_season):
    out = tmp_path_factory.mktemp("artifacts")
    return pipeline.run(out_dir=out, root=synthetic_season, weeks=[1, 2],
                        n_folds=3, refresh=True), out


class TestPlayTable:
    def test_one_row_per_pass_attempt(self, built, synthetic_season):
        artifacts, _ = built
        supp = data.analysis_plays(data.load_supplementary(synthetic_season))
        assert len(artifacts.plays) == len(supp)
        assert not artifacts.plays.duplicated(["game_id", "play_id"]).any()

    def test_tracking_and_context_are_both_present(self, built):
        artifacts, _ = built
        for column in ("separation_at_throw_yd", "rec_cod_total_deg",
                       "route_of_targeted_receiver", "is_complete", "week"):
            assert column in artifacts.plays

    def test_release_features_are_populated(self, built):
        artifacts, _ = built
        assert artifacts.plays["separation_at_throw_yd"].notna().all()
        assert artifacts.plays["air_yards_yd"].notna().all()

    def test_the_cache_is_reused_unless_refreshed(self, built, synthetic_season):
        _, out = built
        cache = out / pipeline.PLAY_TABLE_NAME
        assert cache.exists()
        stamp = cache.stat().st_mtime
        pipeline.load_or_build_play_table(out, weeks=[1, 2], root=synthetic_season)
        assert cache.stat().st_mtime == stamp


class TestScoring:
    def test_every_play_carries_an_out_of_fold_probability(self, built):
        artifacts, _ = built
        probability = artifacts.plays["catch_probability"]
        assert probability.notna().all()
        assert probability.between(0.0, 1.0).all()

    def test_catch_over_expected_is_the_residual(self, built):
        artifacts, _ = built
        plays = artifacts.plays
        assert np.allclose(
            plays["catch_over_expected"],
            plays["is_complete"].astype(float) - plays["catch_probability"])

    def test_the_model_beats_the_base_rate(self, built):
        artifacts, _ = built
        scores = artifacts.summary["model"]
        assert scores["scores"]["brier"] < scores["baselines"]["base_rate"]["brier"]


class TestTables:
    def test_receiver_and_defender_tables_are_gated_by_volume(self, built):
        artifacts, _ = built
        if not artifacts.receivers.empty:
            assert (artifacts.receivers["targets"]
                    >= artifacts.summary["min_targets_receiver"]).all()
        if not artifacts.defenders.empty:
            assert (artifacts.defenders["targets_covered"]
                    >= artifacts.summary["min_targets_defender"]).all()

    def test_route_by_coverage_splits_are_produced(self, built):
        artifacts, _ = built
        assert not artifacts.route_coverage.empty
        assert {"route_of_targeted_receiver", "team_coverage_man_zone",
                "plays", "catch_rate"} <= set(artifacts.route_coverage.columns)

    def test_defensive_backs_are_flagged(self, built):
        artifacts, _ = built
        if not artifacts.defenders.empty:
            assert artifacts.defenders["is_defensive_back"].dtype == bool

    def test_importance_covers_every_feature(self, built):
        artifacts, _ = built
        assert set(artifacts.importance["feature"]) == set(model.FEATURES)


class TestArtifactsOnDisk:
    EXPECTED = [
        "play_features.csv", "receiver_scouting.csv", "defender_coverage.csv",
        "route_by_coverage.csv", "route_by_coverage_type.csv",
        "receiver_route_splits.csv", "model_feature_importance.csv",
        "model_fold_scores.csv", "model_calibration.csv",
        "play_features.parquet", "summary.json",
    ]

    def test_every_artifact_is_written(self, built):
        _, out = built
        missing = [name for name in self.EXPECTED if not (out / name).exists()]
        assert not missing

    def test_the_summary_is_valid_json_with_the_headline_numbers(self, built):
        _, out = built
        summary = json.loads((out / "summary.json").read_text())
        assert summary["n_plays"] > 0
        assert 0.0 <= summary["completion_rate"] <= 1.0
        assert summary["model"]["n_folds"] == 3
        assert summary["model"]["grouped_by"] == "game_id"
        assert 0.0 <= summary["model"]["scores"]["roc_auc"] <= 1.0

    def test_the_parquet_round_trips(self, built):
        artifacts, out = built
        reloaded = pd.read_parquet(out / pipeline.PLAY_TABLE_NAME)
        assert len(reloaded) == len(artifacts.plays)


def test_the_shipped_model_never_sees_the_flight(built):
    """A last check at the level the pipeline actually runs at."""
    artifacts, _ = built
    model.assert_no_leakage()
    flight_columns = [c for c in artifacts.plays.columns
                      if c.startswith(("rec_", "cov_"))]
    assert flight_columns, "the play table should carry flight features"
    assert not set(flight_columns) & set(model.FEATURES)
