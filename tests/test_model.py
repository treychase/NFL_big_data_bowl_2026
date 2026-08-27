"""The catch probability model, and the guard that keeps it honest."""

import numpy as np
import pandas as pd
import pytest

from nfl_scouting import model


class TestLeakageGuard:
    """The single most important test in this file.

    A model that sees the flight will score beautifully and mean nothing, and
    nothing else in the suite would catch it.
    """

    def test_the_shipped_feature_list_is_clean(self):
        model.assert_no_leakage()

    @pytest.mark.parametrize("leaked", [
        "separation_at_arrival_yd",
        "separation_change_yd",
        "target_dist_to_spot_at_arrival_yd",
        "rec_speed_max_yps",
        "cov_pursuit_efficiency",
        "yards_gained",
        "is_touchdown",
        "pass_result",
    ])
    def test_a_post_throw_feature_is_rejected(self, leaked):
        with pytest.raises(model.LeakageError, match="after the ball was thrown"):
            model.assert_no_leakage(model.FEATURES + [leaked])

    def test_the_error_names_every_offender(self):
        with pytest.raises(model.LeakageError) as caught:
            model.assert_no_leakage(["air_yards_yd", "rec_cod_total_deg",
                                     "separation_at_arrival_yd"])
        message = str(caught.value)
        assert "rec_cod_total_deg" in message
        assert "separation_at_arrival_yd" in message
        assert "air_yards_yd" not in message

    def test_no_feature_is_listed_twice(self):
        assert len(model.FEATURES) == len(set(model.FEATURES))

    def test_numeric_and_categorical_lists_do_not_overlap(self):
        assert not set(model.NUMERIC_FEATURES) & set(model.CATEGORICAL_FEATURES)


def model_frame(n=1500, seed=26):
    """A synthetic season with a completion rule the model should recover.

    The intercept is set so the completion rate lands near the real league
    figure of 0.69. At an 84% base rate there is barely any variance left to
    explain, and the test stops measuring whether the model works.

    Every other feature is pure noise, on purpose: the model has to find two
    real signals among twenty-seven distractors, which is the situation it is
    actually in.
    """
    rng = np.random.default_rng(seed)
    separation = rng.gamma(3.0, 1.4, n)
    air_yards = rng.normal(9.0, 8.0, n)
    logit = 0.05 + 0.30 * separation - 0.06 * air_yards
    frame = pd.DataFrame({
        "game_id": rng.integers(1, 20, n),
        "is_complete": rng.random(n) < 1.0 / (1.0 + np.exp(-logit)),
        "separation_at_throw_yd": separation,
        "air_yards_yd": air_yards,
    })
    for column in model.NUMERIC_FEATURES:
        if column not in frame:
            frame[column] = rng.normal(0, 1, n)
    for column in model.CATEGORICAL_FEATURES:
        frame[column] = rng.choice(["A", "B", "C"], n)
    return frame


class TestScoring:
    def test_a_perfect_prediction_scores_perfectly(self):
        y = np.array([0, 1, 0, 1])
        scores = model.score_predictions(y, y.astype(float))
        assert scores["roc_auc"] == pytest.approx(1.0)
        assert scores["brier"] == pytest.approx(0.0)
        assert scores["accuracy"] == pytest.approx(1.0)

    def test_a_coin_flip_scores_like_one(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500)
        scores = model.score_predictions(y, np.full(500, 0.5))
        assert scores["roc_auc"] == pytest.approx(0.5, abs=0.02)
        assert scores["brier"] == pytest.approx(0.25, abs=0.01)


class TestCrossValidation:
    @pytest.fixture(scope="class")
    def result(self):
        return model.cross_validate(model_frame(), n_folds=4)

    def test_every_play_gets_an_out_of_fold_prediction(self, result):
        assert result.predictions.notna().all()
        assert len(result.predictions) == result.n_plays

    def test_predictions_are_probabilities(self, result):
        assert result.predictions.between(0.0, 1.0).all()

    def test_the_fixture_is_a_fair_test(self, result):
        """A lopsided base rate would make the other assertions meaningless."""
        assert 0.6 < result.base_rate < 0.78

    def test_it_beats_predicting_the_base_rate(self, result):
        """If it cannot beat the league average it has learned nothing."""
        assert result.scores["roc_auc"] > 0.62
        assert result.scores["brier"] < result.baseline_scores["base_rate"]["brier"]
        assert result.scores["log_loss"] < result.baseline_scores["base_rate"]["log_loss"]

    def test_folds_are_reported_individually(self, result):
        assert len(result.fold_scores) == 4
        assert result.fold_scores["n_test"].sum() == result.n_plays

    def test_calibration_is_reported(self, result):
        assert result.calibration is not None
        assert {"mean_predicted", "observed"} <= set(result.calibration.columns)

    def test_no_game_spans_the_train_and_test_side_of_a_fold(self):
        """Plays from one game share a quarterback and a secondary; splitting
        them across a fold boundary would flatter every score here."""
        from sklearn.model_selection import GroupKFold
        frame = model_frame()
        groups = frame["game_id"].to_numpy()
        for train_idx, test_idx in GroupKFold(n_splits=4).split(frame, frame["is_complete"], groups):
            assert not set(groups[train_idx]) & set(groups[test_idx])

    def test_it_is_reproducible(self):
        frame = model_frame()
        first = model.cross_validate(frame, n_folds=3, with_baselines=False)
        second = model.cross_validate(frame, n_folds=3, with_baselines=False)
        assert first.scores == second.scores


class TestFitAndAttach:
    def test_the_full_fit_predicts_probabilities(self):
        frame = model_frame(300)
        estimator = model.fit_full(frame)
        probability = estimator.predict_proba(frame[model.FEATURES])[:, 1]
        assert ((probability >= 0.0) & (probability <= 1.0)).all()

    def test_catch_over_expected_is_the_residual(self):
        frame = model_frame(200)
        predictions = pd.Series(np.full(len(frame), 0.6), index=frame.index)
        scored = model.add_catch_probability(frame, predictions)
        assert scored["catch_over_expected"].equals(
            scored["is_complete"].astype(float) - 0.6)

    def test_over_expected_sums_to_about_nothing_across_the_league(self):
        """A calibrated model gives up as much as it takes; a systematic
        offset means the expectation itself is biased."""
        frame = model_frame(800)
        result = model.cross_validate(frame, n_folds=4, with_baselines=False)
        scored = model.add_catch_probability(frame, result.predictions)
        assert scored["catch_over_expected"].mean() == pytest.approx(0.0, abs=0.03)

    def test_feature_importance_ranks_the_features_that_matter(self):
        """Separation and air yards are what the synthetic rule is built on."""
        frame = model_frame(600)
        importance = model.feature_importance(frame, model.fit_full(frame), n_repeats=3)
        assert len(importance) == len(model.FEATURES)
        top = set(importance.head(4)["feature"])
        assert {"separation_at_throw_yd", "air_yards_yd"} <= top
