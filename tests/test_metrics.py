"""The scouting tables, and the arithmetic behind the numbers on them."""

import numpy as np
import pandas as pd
import pytest

from nfl_scouting import metrics
from nfl_scouting.config import (
    MIN_TARGETS_FOR_DEFENDER,
    MIN_TARGETS_FOR_RECEIVER,
    OPEN_SEPARATION_YARDS,
)


class TestPasserRating:
    """Checked against lines whose published rating is a matter of record."""

    def test_a_perfect_rating_is_the_documented_ceiling(self):
        """Quoted as 158.3; the exact value from the formula is 158.33..."""
        assert metrics.passer_rating(10, 10, 200, 10, 0) == pytest.approx(158.3, abs=0.04)
        assert metrics.MAX_PASSER_RATING == pytest.approx(158.3333, abs=1e-3)

    def test_the_floor_is_zero_not_a_negative_number(self):
        assert metrics.passer_rating(10, 0, 0, 0, 10) == pytest.approx(0.0)

    def test_matches_a_real_season_line(self):
        """Brock Purdy, 2023: 444 attempts, 308 completions, 4280 yards,
        31 touchdowns, 11 interceptions - a published 113.0."""
        assert metrics.passer_rating(444, 308, 4280, 31, 11) == pytest.approx(113.0, abs=0.1)

    def test_matches_a_second_real_season_line(self):
        """Tua Tagovailoa, 2023: 560/388, 4624 yards, 29 TD, 14 INT - 101.1."""
        assert metrics.passer_rating(560, 388, 4624, 29, 14) == pytest.approx(101.1, abs=0.1)

    def test_no_attempts_is_unknown_rather_than_zero(self):
        """Never being thrown at is not the same as smothering coverage."""
        assert np.isnan(metrics.passer_rating(0, 0, 0, 0, 0))

    def test_components_are_capped_so_the_rating_stays_in_range(self):
        # An absurd line that would blow past 158.3 without the clamps.
        assert metrics.passer_rating(10, 10, 1000, 10, 0) == pytest.approx(
            metrics.MAX_PASSER_RATING, abs=1e-9)

    def test_works_elementwise_on_arrays(self):
        ratings = metrics.passer_rating(
            [10, 10, 0], [10, 0, 0], [200, 0, 0], [10, 0, 0], [0, 10, 0])
        assert ratings[0] == pytest.approx(158.3, abs=0.04)
        assert ratings[1] == pytest.approx(0.0)
        assert np.isnan(ratings[2])

    def test_every_random_line_lands_inside_the_scale(self):
        rng = np.random.default_rng(5)
        attempts = rng.integers(1, 60, 400)
        completions = rng.integers(0, attempts + 1)
        ratings = metrics.passer_rating(
            attempts, completions, rng.integers(0, 500, 400),
            rng.integers(0, 8, 400), rng.integers(0, 6, 400))
        assert np.all((ratings >= 0.0) & (ratings <= metrics.MAX_PASSER_RATING + 1e-9))


class TestOpenRate:
    def test_counts_the_share_at_or_above_the_threshold(self):
        series = pd.Series([1.0, OPEN_SEPARATION_YARDS, 5.0, 0.5])
        assert metrics._open_rate(series) == pytest.approx(0.5)

    def test_ignores_missing_values_rather_than_scoring_them_as_covered(self):
        """A defender who was not tracked in the air is unknown, not close."""
        assert metrics._open_rate(pd.Series([5.0, np.nan])) == pytest.approx(1.0)

    def test_all_missing_is_unknown(self):
        assert np.isnan(metrics._open_rate(pd.Series([np.nan, np.nan])))


class TestReceiverTable:
    def test_one_row_per_qualified_receiver(self, play_table):
        table = metrics.receiver_table(play_table, min_targets=1)
        assert len(table) == play_table["target_nfl_id"].nunique()
        assert table["targets"].sum() == len(play_table)

    def test_the_volume_gate_removes_small_samples(self, play_table):
        gated = metrics.receiver_table(play_table, min_targets=1000)
        assert gated.empty

    def test_rates_match_a_hand_computed_group(self, play_table):
        table = metrics.receiver_table(play_table, min_targets=1)
        first = table.iloc[0]
        group = play_table[play_table["target_nfl_id"] == first["nfl_id"]]
        assert first["targets"] == len(group)
        assert first["catch_rate"] == pytest.approx(group["is_complete"].mean())
        assert first["separation_at_throw_yd"] == pytest.approx(
            group["separation_at_throw_yd"].mean())
        assert first["open_rate_at_throw"] == pytest.approx(
            (group["separation_at_throw_yd"] >= OPEN_SEPARATION_YARDS).mean())

    def test_carries_the_flight_profile(self, play_table):
        table = metrics.receiver_table(play_table, min_targets=1)
        for column in ("air_cod_total_deg", "air_accel_burst_yps2"):
            assert column in table
            assert table[column].notna().all()

    def test_default_gate_is_the_configured_one(self, play_table):
        table = metrics.receiver_table(play_table)
        assert (table["targets"] >= MIN_TARGETS_FOR_RECEIVER).all()

    def test_an_empty_input_gives_an_empty_table(self, play_table):
        assert metrics.receiver_table(play_table.iloc[:0], min_targets=1).empty


class TestDefenderTable:
    def test_passer_rating_allowed_matches_the_formula_on_the_group(self, play_table):
        table = metrics.defender_table(play_table, min_targets=1)
        first = table.iloc[0]
        group = play_table[play_table["coverage_nfl_id"] == first["nfl_id"]]
        expected = metrics.passer_rating(
            len(group), group["is_complete"].sum(), group["yards_gained"].sum(),
            group["is_touchdown"].sum(), group["is_interception"].sum())
        assert first["passer_rating_allowed"] == pytest.approx(float(expected))

    def test_sorted_best_coverage_first(self, play_table):
        table = metrics.defender_table(play_table, min_targets=1)
        assert table["passer_rating_allowed"].is_monotonic_increasing

    def test_plays_with_no_charged_defender_are_excluded(self, play_table):
        frame = play_table.copy()
        frame.loc[frame.index[:40], "coverage_nfl_id"] = np.nan
        table = metrics.defender_table(frame, min_targets=1)
        assert table["targets_covered"].sum() == len(frame) - 40

    def test_default_gate_is_the_configured_one(self, play_table):
        table = metrics.defender_table(play_table)
        assert (table["targets_covered"] >= MIN_TARGETS_FOR_DEFENDER).all()


class TestSplitTable:
    def test_route_by_coverage_covers_every_qualifying_combination(self, play_table):
        table = metrics.split_table(
            play_table, ["route_of_targeted_receiver", "team_coverage_man_zone"],
            min_plays=1)
        assert len(table) == len(
            play_table.groupby(["route_of_targeted_receiver", "team_coverage_man_zone"]))
        assert table["plays"].sum() == len(play_table)

    def test_the_gate_drops_thin_cells(self, play_table):
        table = metrics.split_table(
            play_table, ["route_of_targeted_receiver", "team_coverage_man_zone"],
            min_plays=25)
        assert (table["plays"] >= 25).all()

    def test_a_single_key_split_works(self, play_table):
        table = metrics.split_table(play_table, ["route_of_targeted_receiver"],
                                    min_plays=1)
        assert set(table["route_of_targeted_receiver"]) == set(
            play_table["route_of_targeted_receiver"])

    def test_receiver_route_table_keys_on_player_route_and_coverage(self, play_table):
        table = metrics.receiver_route_table(play_table, min_plays=1)
        for column in ("target_nfl_id", "route_of_targeted_receiver",
                       "team_coverage_man_zone", "plays", "catch_rate"):
            assert column in table
        assert table["plays"].sum() == len(play_table)


def test_rates_stay_within_their_natural_bounds(play_table):
    receivers = metrics.receiver_table(play_table, min_targets=1)
    for column in ("catch_rate", "open_rate_at_throw", "open_rate_at_arrival"):
        valid = receivers[column].dropna()
        assert ((valid >= 0.0) & (valid <= 1.0)).all()
