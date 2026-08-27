"""Per-play extraction, against plays whose geometry we chose on purpose."""

import numpy as np
import pandas as pd
import pytest

from nfl_scouting import features
from nfl_scouting.config import DT, ROLE_COVERAGE, ROLE_TARGET
from tests.conftest import assemble_play, make_player, straight_path


class TestThrowState:
    def test_takes_the_last_pre_pass_frame(self, simple_play):
        tracking, _ = simple_play
        state = features.throw_state(tracking)
        assert (state["frame_id"] == tracking["frame_id"].max()).all()
        assert state.index.name == "nfl_id"
        assert len(state) == tracking["nfl_id"].nunique()

    def test_rejects_an_empty_play(self):
        with pytest.raises(ValueError, match="no tracking frames"):
            features.throw_state(pd.DataFrame(columns=["frame_id", "nfl_id"]))

    def test_rejects_a_duplicated_player_on_the_release_frame(self, simple_play):
        tracking, _ = simple_play
        last = tracking[tracking["frame_id"] == tracking["frame_id"].max()]
        with pytest.raises(ValueError, match="duplicate players"):
            features.throw_state(pd.concat([tracking, last.head(1)]))


class TestNearestPlayer:
    def test_finds_the_closest_candidate_and_its_distance(self, simple_play):
        tracking, _ = simple_play
        state = features.throw_state(tracking)
        coverage = state.index[state["player_role"] == ROLE_COVERAGE]
        target = state.loc[1001]
        nearest, dist = features.nearest_player(
            state, target["x"], target["y"], coverage)
        # The corner (2001) trails 2 yards behind; the safety is far upfield.
        assert nearest == 2001
        assert dist == pytest.approx(2.0, abs=1e-6)

    def test_returns_nothing_when_there_are_no_candidates(self, simple_play):
        tracking, _ = simple_play
        state = features.throw_state(tracking)
        nearest, dist = features.nearest_player(state, 0.0, 0.0, pd.Index([]))
        assert nearest is None and np.isnan(dist)

    def test_ignores_candidates_that_are_not_on_the_field(self, simple_play):
        tracking, _ = simple_play
        state = features.throw_state(tracking)
        nearest, _ = features.nearest_player(
            state, 40.0, 20.0, pd.Index([2002, 999999]))
        assert nearest == 2002


class TestAirTracks:
    def test_the_release_position_is_prepended_to_the_flight(self, simple_play):
        """Without it the first tenth of a second of the reaction is missing."""
        tracking, ball_air = simple_play
        state = features.throw_state(tracking)
        tracks = features.build_air_tracks(ball_air, state)
        n_out = (ball_air["nfl_id"] == 1001).sum()
        assert tracks[1001].n_frames == n_out + 1
        assert tracks[1001].x[0] == pytest.approx(state.at[1001, "x"])

    def test_only_predicted_players_get_a_track(self, simple_play):
        tracking, ball_air = simple_play
        tracks = features.build_air_tracks(ball_air, features.throw_state(tracking))
        assert set(tracks) == {1001, 2001}

    def test_no_flight_frames_gives_no_tracks(self, simple_play):
        tracking, ball_air = simple_play
        assert features.build_air_tracks(
            ball_air.iloc[:0], features.throw_state(tracking)) == {}


class TestExtractPlay:
    def test_release_geometry_matches_the_play_we_built(self, simple_play):
        row = features.extract_play(*simple_play)
        assert row["target_nfl_id"] == 1001
        assert row["coverage_nfl_id"] == 2001
        assert row["coverage_position"] == "CB"
        assert row["separation_at_throw_yd"] == pytest.approx(2.0, abs=1e-6)
        assert row["n_route_runners"] == 2
        assert row["n_coverage_defenders"] == 2

    def test_depth_and_air_yards_are_signed_toward_the_end_zone(self, simple_play):
        tracking, ball_air = simple_play
        row = features.extract_play(tracking, ball_air)
        # Receiver starts on the line at x=40 and runs to x=40+6*1.9.
        assert row["target_depth_yd"] == pytest.approx(6.0 * 19 * DT, abs=1e-6)
        assert row["air_yards_yd"] > row["target_depth_yd"] > 0

    def test_a_left_running_play_gives_the_same_signs(self, simple_play):
        """Direction must be normalised, or half the season reads negative."""
        tracking, ball_air = simple_play
        right = features.extract_play(tracking, ball_air)

        mirrored = tracking.copy()
        mirrored["x"] = 120.0 - mirrored["x"]
        mirrored["ball_land_x"] = 120.0 - mirrored["ball_land_x"]
        mirrored["absolute_yardline_number"] = 120.0 - mirrored["absolute_yardline_number"]
        mirrored["play_direction"] = "left"
        air = ball_air.copy()
        air["x"] = 120.0 - air["x"]
        left = features.extract_play(mirrored, air)

        assert left["target_depth_yd"] == pytest.approx(right["target_depth_yd"])
        assert left["air_yards_yd"] == pytest.approx(right["air_yards_yd"])
        assert left["separation_at_throw_yd"] == pytest.approx(
            right["separation_at_throw_yd"])

    def test_the_corner_losing_ground_shows_as_growing_separation(self, simple_play):
        """The fixture's corner runs 5 yd/s to the receiver's 6, so the gap opens."""
        row = features.extract_play(*simple_play)
        assert row["separation_at_arrival_yd"] > row["separation_at_throw_yd"]
        assert row["separation_change_yd"] > 0
        assert row["arrival_defender_nfl_id"] == 2001

    def test_the_receiver_arrives_at_the_ball(self, simple_play):
        row = features.extract_play(*simple_play)
        assert row["target_dist_to_spot_at_arrival_yd"] == pytest.approx(0.0, abs=1e-6)
        assert row["rec_pursuit_efficiency"] == pytest.approx(1.0, abs=1e-3)

    def test_flight_summaries_are_present_for_both_players(self, simple_play):
        row = features.extract_play(*simple_play)
        assert row["rec_speed_max_yps"] == pytest.approx(6.0, abs=0.05)
        assert row["cov_speed_max_yps"] == pytest.approx(5.0, abs=0.3)
        assert row["coverage_tracked_in_air"] is True
        assert row["rec_cod_total_deg"] == pytest.approx(0.0, abs=1.0)

    def test_a_play_with_no_targeted_receiver_is_skipped(self, simple_play):
        tracking, ball_air = simple_play
        tracking = tracking.copy()
        tracking.loc[tracking["player_role"] == ROLE_TARGET, "player_role"] = "Other Route Runner"
        assert features.extract_play(tracking, ball_air) is None

    def test_a_play_with_no_flight_frames_is_skipped(self, simple_play):
        tracking, ball_air = simple_play
        assert features.extract_play(tracking, ball_air.iloc[:0]) is None

    def test_an_untracked_defender_still_produces_a_full_row(self, simple_play):
        """The corner is in coverage but not in the predicted set on 11% of
        real plays; the row must keep its shape rather than go ragged."""
        tracking, ball_air = simple_play
        row = features.extract_play(tracking, ball_air[ball_air["nfl_id"] == 1001])
        full = features.extract_play(tracking, ball_air)
        assert set(row) == set(full)
        assert row["coverage_tracked_in_air"] is False
        assert np.isnan(row["cov_speed_max_yps"])
        # Release-time coverage is still known: it does not need the flight.
        assert row["separation_at_throw_yd"] == pytest.approx(2.0, abs=1e-6)


class TestSeparationChange:
    """The comparison has to be over the same players at both ends."""

    def test_change_is_measured_against_the_tracked_defenders_only(self, simple_play):
        """When the nearest defender at the release is not tracked through
        the flight, the release measurement must fall back to the defenders
        that are - otherwise arrival separation to a distant safety is
        subtracted from throw separation to the corner on his hip, and the
        receiver appears to gain tens of yards in a second."""
        tracking, ball_air = simple_play
        # Track only the safety, who is parked 20 yards upfield, and not the
        # corner two yards off the receiver.
        safety_air = ball_air[ball_air["nfl_id"] == 2001].copy()
        safety_air["nfl_id"] = 2002
        air = pd.concat([ball_air[ball_air["nfl_id"] == 1001], safety_air],
                        ignore_index=True)
        row = features.extract_play(tracking, air)

        assert row["coverage_nfl_id"] == 2001          # still charged to the corner
        assert row["separation_at_throw_yd"] == pytest.approx(2.0, abs=1e-6)
        # The comparison basis is the safety, who is further off than the
        # corner the headline number is measured against.
        assert row["separation_at_throw_tracked_yd"] > row["separation_at_throw_yd"] + 3
        assert row["arrival_defender_nfl_id"] == 2002
        # The change is arrival-minus-tracked, not arrival-minus-headline,
        # so it stays the size a second of football can produce.
        assert row["separation_change_yd"] == pytest.approx(
            row["separation_at_arrival_yd"] - row["separation_at_throw_tracked_yd"])
        assert abs(row["separation_change_yd"]) < 5.0

    def test_the_bases_agree_when_the_nearest_defender_is_tracked(self, simple_play):
        row = features.extract_play(*simple_play)
        assert row["separation_at_throw_tracked_yd"] == pytest.approx(
            row["separation_at_throw_yd"], abs=1e-9)

    def test_change_is_missing_when_no_defender_is_tracked(self, simple_play):
        tracking, ball_air = simple_play
        row = features.extract_play(tracking, ball_air[ball_air["nfl_id"] == 1001])
        assert np.isnan(row["separation_at_arrival_yd"])
        assert np.isnan(row["separation_change_yd"])


class TestLandingSpotReachability:
    def test_a_normal_throw_is_reachable(self, simple_play):
        assert features.extract_play(*simple_play)["landing_spot_reachable"] is True

    def test_a_spot_the_receiver_could_never_reach_is_flagged(self):
        """The ball is recorded landing 50 yards away with 0.5 s of flight.
        No receiver covers that, so the spot is not describing this target."""
        n = 12
        rx, ry = straight_path(40.0, 20.0, 90.0, 5.0, n)
        players = [
            make_player(1, "Target", "WR", "Offense", ROLE_TARGET, rx, ry, predict=True),
            make_player(2, "Corner", "CB", "Defense", ROLE_COVERAGE,
                        *straight_path(38.0, 20.0, 90.0, 5.0, n)),
        ]
        air = straight_path(rx[-1], ry[-1], 90.0, 5.0, 5)
        row = features.extract_play(*assemble_play(
            players, ball_land=(95.0, 20.0), air_paths={1: air}))
        assert row["target_dist_to_spot_at_throw_yd"] > 40
        assert row["landing_spot_reachable"] is False


class TestContestCounting:
    def test_defenders_near_the_landing_spot_are_counted(self):
        """Two defenders parked on the ball, one far away: the count is two."""
        n = 12
        rx, ry = straight_path(40.0, 20.0, 90.0, 5.0, n)
        land = (52.0, 20.0)
        near_a = straight_path(51.0, 20.0, 90.0, 0.0, n)
        near_b = straight_path(52.0, 23.0, 90.0, 0.0, n)
        far = straight_path(20.0, 45.0, 90.0, 0.0, n)
        players = [
            make_player(1, "Target", "WR", "Offense", ROLE_TARGET, rx, ry, predict=True),
            make_player(2, "Near A", "CB", "Defense", ROLE_COVERAGE, *near_a),
            make_player(3, "Near B", "FS", "Defense", ROLE_COVERAGE, *near_b),
            make_player(4, "Far", "SS", "Defense", ROLE_COVERAGE, *far),
        ]
        air = straight_path(rx[-1], ry[-1], 90.0, 5.0, 6)
        row = features.extract_play(*assemble_play(
            players, ball_land=land, air_paths={1: air}))
        assert row["defenders_near_spot"] == 2
        assert row["nearest_defender_to_spot_yd"] == pytest.approx(1.0, abs=1e-6)

    def test_position_advantage_is_positive_when_the_defender_is_nearer_the_ball(self):
        n = 12
        rx, ry = straight_path(40.0, 20.0, 90.0, 5.0, n)
        land = (52.0, 20.0)
        inside = straight_path(50.0, 20.0, 90.0, 0.0, n)
        players = [
            make_player(1, "Target", "WR", "Offense", ROLE_TARGET, rx, ry, predict=True),
            make_player(2, "Inside", "CB", "Defense", ROLE_COVERAGE, *inside),
        ]
        air = straight_path(rx[-1], ry[-1], 90.0, 5.0, 6)
        row = features.extract_play(*assemble_play(
            players, ball_land=land, air_paths={1: air}))
        assert row["coverage_position_advantage_yd"] > 0


def test_extract_week_stacks_rows_and_skips_unusable_plays(simple_play):
    tracking, ball_air = simple_play
    second = tracking.copy()
    second["play_id"] = 202
    second_air = ball_air.copy()
    second_air["play_id"] = 202
    # A third play with no flight frames at all, which should be dropped.
    third = tracking.copy()
    third["play_id"] = 303

    frame = features.extract_week(
        pd.concat([tracking, second, third], ignore_index=True),
        pd.concat([ball_air, second_air], ignore_index=True))
    assert sorted(frame["play_id"]) == [101, 202]
    assert frame["separation_at_throw_yd"].notna().all()
