"""The angle convention is the thing most likely to be silently wrong."""

import numpy as np
import pytest

from nfl_scouting import geometry


class TestHeadingConvention:
    """Zero degrees is +y and angles run clockwise, matching the tracking file."""

    @pytest.mark.parametrize("dx, dy, expected", [
        (0.0, 1.0, 0.0),      # straight upfield in y
        (1.0, 0.0, 90.0),     # toward increasing x
        (0.0, -1.0, 180.0),
        (-1.0, 0.0, 270.0),
        (1.0, 1.0, 45.0),
    ])
    def test_heading_of_a_displacement(self, dx, dy, expected):
        assert geometry.heading_deg(dx, dy) == pytest.approx(expected)

    def test_heading_is_always_in_zero_to_360(self):
        angles = geometry.heading_deg(
            np.array([-1.0, -0.01, 0.0]), np.array([-1.0, 1.0, -1.0]))
        assert np.all((angles >= 0.0) & (angles < 360.0))

    def test_unit_vector_inverts_heading(self):
        for angle in (0.0, 37.0, 90.0, 184.0, 359.0):
            dx, dy = geometry.unit_vector(angle)
            assert geometry.heading_deg(dx, dy) == pytest.approx(angle, abs=1e-9)

    def test_unit_vector_is_unit_length(self):
        dx, dy = geometry.unit_vector(np.arange(0, 360, 15))
        assert np.allclose(np.hypot(dx, dy), 1.0)


class TestAngleDifference:
    def test_wraps_across_zero(self):
        """350 to 10 degrees is a 20 degree turn, not a 340 degree one."""
        assert geometry.angle_difference_deg(10.0, 350.0) == pytest.approx(20.0)
        assert geometry.angle_difference_deg(350.0, 10.0) == pytest.approx(-20.0)

    def test_result_stays_within_half_a_turn(self):
        rng = np.random.default_rng(0)
        a, b = rng.uniform(-720, 720, 500), rng.uniform(-720, 720, 500)
        diff = geometry.angle_difference_deg(a, b)
        assert np.all((diff > -180.0) & (diff <= 180.0))

    def test_is_antisymmetric(self):
        a, b = 12.0, 275.0
        assert (geometry.angle_difference_deg(a, b)
                == pytest.approx(-geometry.angle_difference_deg(b, a)))


class TestBearingAndProjection:
    def test_bearing_points_from_one_place_to_another(self):
        assert geometry.bearing_deg(0.0, 0.0, 5.0, 0.0) == pytest.approx(90.0)
        assert geometry.bearing_deg(5.0, 0.0, 0.0, 0.0) == pytest.approx(270.0)

    def test_projection_is_full_speed_when_aligned(self):
        # Moving at 4 yd/s toward increasing x, with a bearing of 90 degrees.
        assert geometry.project_onto_bearing(4.0, 0.0, 90.0) == pytest.approx(4.0)

    def test_projection_is_zero_when_perpendicular(self):
        assert geometry.project_onto_bearing(4.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-12)

    def test_projection_is_negative_when_running_away(self):
        assert geometry.project_onto_bearing(4.0, 0.0, 270.0) == pytest.approx(-4.0)


def test_distance_matches_pythagoras():
    assert geometry.distance(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)
    assert np.allclose(
        geometry.distance(np.zeros(3), np.zeros(3), np.array([3.0, 5.0, 8.0]),
                          np.array([4.0, 12.0, 15.0])),
        [5.0, 13.0, 17.0])
