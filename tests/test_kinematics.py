"""Derivatives taken from noisy positions, checked against motion we can solve.

The point of every test here is that the answer is known in closed form, so
a regression shows up as a wrong number rather than as a plot that looks
fine.
"""

import numpy as np
import pytest

from nfl_scouting import kinematics
from nfl_scouting.config import DT


def frames(n):
    return np.arange(n) * DT


class TestDifferentiation:
    """Endpoints matter most here: the throw and the catch are both endpoints."""

    def test_constant_velocity_has_no_acceleration_anywhere(self):
        t = frames(20)
        track = kinematics.build_track(6.0 * t, np.zeros_like(t))
        assert np.allclose(track.speed, 6.0)
        assert np.allclose(track.accel, 0.0, atol=1e-9)

    def test_constant_acceleration_is_exact_at_the_first_and_last_frame(self):
        t = frames(20)
        track = kinematics.build_track(0.5 * 3.0 * t ** 2, np.zeros_like(t))
        assert track.accel[0] == pytest.approx(3.0, abs=1e-6)
        assert track.accel[-1] == pytest.approx(3.0, abs=1e-6)
        assert np.allclose(track.accel, 3.0, atol=1e-6)

    def test_works_on_the_shortest_flight_in_the_data(self):
        """Five frames is the minimum `num_frames_output` the season contains."""
        t = frames(5)
        track = kinematics.build_track(0.5 * 3.0 * t ** 2, np.zeros_like(t))
        assert np.allclose(track.accel, 3.0, atol=1e-6)

    def test_velocity_components_carry_the_right_signs(self):
        t = frames(12)
        track = kinematics.build_track(np.zeros_like(t), -4.0 * t)
        assert np.allclose(track.vy, -4.0)
        assert np.allclose(track.heading, 180.0)

    def test_smoothing_suppresses_jitter_without_moving_the_mean(self):
        rng = np.random.default_rng(3)
        t = frames(40)
        clean = 5.0 * t
        noisy = clean + rng.normal(0, 0.06, t.size)
        smoothed = kinematics.build_track(noisy, np.zeros_like(t))
        raw_speed = np.abs(np.diff(noisy) / DT)
        assert smoothed.speed.std() < raw_speed.std() / 2
        assert smoothed.speed.mean() == pytest.approx(5.0, abs=0.25)

    def test_rejects_mismatched_and_empty_tracks(self):
        with pytest.raises(ValueError):
            kinematics.build_track(np.zeros(4), np.zeros(3))
        with pytest.raises(ValueError):
            kinematics.build_track(np.zeros(0), np.zeros(0))

    def test_single_frame_track_has_no_motion(self):
        track = kinematics.build_track([10.0], [20.0])
        assert track.n_frames == 1
        assert track.duration_s == 0.0
        assert track.path_length == 0.0


class TestPathGeometry:
    def test_path_length_equals_displacement_on_a_straight_line(self):
        t = frames(15)
        track = kinematics.build_track(5.0 * t, np.zeros_like(t))
        assert track.path_length == pytest.approx(track.displacement)

    def test_path_length_exceeds_displacement_on_a_curve(self):
        angle = np.linspace(0, np.pi, 20)
        track = kinematics.build_track(10 * np.cos(angle), 10 * np.sin(angle))
        assert track.path_length > track.displacement * 1.4


class TestChangeOfDirection:
    def test_a_straight_line_is_no_turn_at_all(self):
        t = frames(15)
        cod = kinematics.change_of_direction(
            kinematics.build_track(6.0 * t, np.zeros_like(t)))
        assert cod["cod_total_deg"] == pytest.approx(0.0, abs=1e-6)
        assert cod["cod_net_deg"] == pytest.approx(0.0, abs=1e-6)

    def test_a_right_angle_break_registers_ninety_degrees(self):
        """Ten frames upfield, then ten across: the total turn is the break."""
        speed, n = 6.0, 10
        x = np.concatenate([speed * frames(n), np.full(n, speed * frames(n)[-1])])
        y = np.concatenate([np.zeros(n), speed * frames(n + 1)[1:]])
        cod = kinematics.change_of_direction(kinematics.build_track(x, y))
        assert cod["cod_total_deg"] == pytest.approx(90.0, abs=0.5)
        assert cod["cod_net_deg"] == pytest.approx(90.0, abs=0.5)

    @pytest.mark.parametrize("sharpness_s", [0.1, 0.2, 0.3, 0.5])
    def test_a_realistic_break_is_measured_at_its_true_angle(self, sharpness_s):
        """The quadratic fit must not ring: a 90 degree cut reads 90, however
        fast the receiver makes it. A cubic fit inflates the sharpest of
        these to 101."""
        speed, n = 6.0, 30
        t = frames(n)
        turn = 90.0 / (1.0 + np.exp(-(t - 1.5) / (sharpness_s / 3.33)))
        x = np.cumsum(speed * np.sin(np.radians(turn)) * DT)
        y = np.cumsum(speed * np.cos(np.radians(turn)) * DT)
        cod = kinematics.change_of_direction(kinematics.build_track(x, y))
        assert cod["cod_total_deg"] == pytest.approx(90.0, abs=1.0)

    def test_out_and_back_totals_the_turn_but_nets_out(self):
        """A stutter costs the receiver two turns and gains him no new heading."""
        speed, n = 5.0, 8
        leg = speed * frames(n)
        x = np.concatenate([leg, np.full(n, leg[-1]), leg[-1] + leg[1:]])
        y = np.concatenate([np.zeros(n), leg, np.full(n - 1, leg[-1])])
        cod = kinematics.change_of_direction(kinematics.build_track(x, y))
        assert cod["cod_total_deg"] > 140.0
        assert cod["cod_net_deg"] < 30.0

    def test_a_standing_player_is_not_credited_with_turning(self):
        """Position jitter at a standstill must not read as elite agility."""
        rng = np.random.default_rng(11)
        n = 20
        track = kinematics.build_track(rng.normal(30, 0.03, n), rng.normal(20, 0.03, n))
        assert kinematics.change_of_direction(track)["cod_total_deg"] == 0.0

    def test_turn_rate_normalises_for_window_length(self):
        speed = 6.0
        short = kinematics.build_track(speed * frames(6), np.zeros(6))
        cod = kinematics.change_of_direction(short)
        assert cod["cod_rate_dps"] == pytest.approx(0.0, abs=1e-6)


class TestAccelerationSummary:
    def test_reports_the_speed_at_each_end_and_the_peak(self):
        t = frames(20)
        summary = kinematics.acceleration_summary(
            kinematics.build_track(0.5 * 2.0 * t ** 2, np.zeros_like(t)))
        assert summary["speed_start_yps"] == pytest.approx(0.0, abs=1e-6)
        assert summary["speed_end_yps"] == pytest.approx(2.0 * t[-1], abs=1e-6)
        assert summary["speed_delta_yps"] == pytest.approx(2.0 * t[-1], abs=1e-6)
        assert summary["accel_max_yps2"] == pytest.approx(2.0, abs=1e-6)

    def test_burst_is_the_best_half_second_of_speed_gain(self):
        t = frames(20)
        summary = kinematics.acceleration_summary(
            kinematics.build_track(0.5 * 4.0 * t ** 2, np.zeros_like(t)))
        assert summary["accel_burst_yps2"] == pytest.approx(4.0, abs=0.05)


class TestPursuitToPoint:
    def test_running_straight_at_the_spot_is_perfectly_efficient(self):
        t = frames(20)
        track = kinematics.build_track(5.0 * t, np.zeros_like(t))
        pursuit = kinematics.pursuit_to_point(track, 12.0, 0.0)
        assert pursuit["pursuit_efficiency"] == pytest.approx(1.0, abs=1e-6)
        assert pursuit["mean_off_bearing_deg"] == pytest.approx(0.0, abs=1e-6)
        assert pursuit["closing_speed_yps"] == pytest.approx(5.0, abs=1e-6)

    def test_running_away_closes_negative_ground(self):
        t = frames(20)
        track = kinematics.build_track(-5.0 * t, np.zeros_like(t))
        pursuit = kinematics.pursuit_to_point(track, 12.0, 0.0)
        assert pursuit["dist_closed_yd"] < 0
        assert pursuit["pursuit_efficiency"] == pytest.approx(-1.0, abs=1e-6)
        assert pursuit["mean_off_bearing_deg"] == pytest.approx(180.0, abs=1e-6)

    def test_a_defender_who_redirects_scores_a_positive_correction(self):
        """Starts running away from the ball, turns and chases it."""
        n = 10
        away_x = -5.0 * frames(n)
        back_x = away_x[-1] + 5.0 * frames(n + 1)[1:]
        x = np.concatenate([away_x, back_x])
        track = kinematics.build_track(x, np.zeros_like(x))
        pursuit = kinematics.pursuit_to_point(track, 30.0, 0.0)
        assert pursuit["bearing_correction_deg"] == pytest.approx(180.0, abs=15.0)

    def test_bearing_is_ignored_when_standing_on_the_spot(self):
        """Half a yard away, the bearing is jitter - it must not enter the mean."""
        t = frames(20)
        track = kinematics.build_track(5.0 * t, np.zeros_like(t))
        # The spot is the final position, so the last frames are degenerate.
        pursuit = kinematics.pursuit_to_point(track, float(5.0 * t[-1]), 0.0)
        assert pursuit["mean_off_bearing_deg"] == pytest.approx(0.0, abs=1e-6)
        assert np.isfinite(pursuit["closing_accel_yps2"])

    def test_closing_acceleration_matches_a_known_burst_at_the_spot(self):
        t = frames(20)
        track = kinematics.build_track(0.5 * 2.0 * t ** 2, np.zeros_like(t))
        pursuit = kinematics.pursuit_to_point(track, 60.0, 0.0)
        assert pursuit["closing_accel_yps2"] == pytest.approx(2.0, abs=1e-6)


def test_summarise_is_the_union_of_the_parts():
    t = frames(12)
    track = kinematics.build_track(5.0 * t, np.zeros_like(t))
    summary = kinematics.summarise(track, 20.0, 0.0)
    for key in ("speed_max_yps", "cod_total_deg", "pursuit_efficiency",
                "duration_s", "path_length_yd"):
        assert key in summary
    assert summary["duration_s"] == pytest.approx(1.1)
    # Without a target point the pursuit block is simply absent.
    assert "pursuit_efficiency" not in kinematics.summarise(track)
