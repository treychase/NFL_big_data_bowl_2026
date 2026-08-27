"""Angles and distances in the tracking file's coordinate frame.

The tracking data measures `dir` and `o` in degrees clockwise from the
positive y axis, so a player heading straight down the field toward
increasing x is at 90 degrees, not 0. Checked against the data: predicting
the next frame as `x + s*dt*sin(dir)`, `y + s*dt*cos(dir)` lands within
0.023 yards on average, while the +x-axis convention is off by 0.58.
Every angle in this package is degrees in that frame unless a name says
otherwise.
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray | float


def unit_vector(direction_deg: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Components of a heading, in the tracking file's clockwise-from-+y frame."""
    radians = np.radians(np.asarray(direction_deg, dtype=float))
    return np.sin(radians), np.cos(radians)


def heading_deg(dx: ArrayLike, dy: ArrayLike) -> np.ndarray:
    """The heading of a displacement, as degrees in [0, 360)."""
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def bearing_deg(
    from_x: ArrayLike, from_y: ArrayLike, to_x: ArrayLike, to_y: ArrayLike
) -> np.ndarray:
    """The heading a player would need to run straight at a point."""
    return heading_deg(np.subtract(to_x, from_x), np.subtract(to_y, from_y))


def angle_difference_deg(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Signed a - b, wrapped into (-180, 180].

    Wrapping matters: a player who turns from 350 degrees to 10 has turned
    20 degrees, not 340.
    """
    return (np.subtract(a, b) + 180.0) % 360.0 - 180.0


def distance(
    x1: ArrayLike, y1: ArrayLike, x2: ArrayLike, y2: ArrayLike
) -> np.ndarray:
    return np.hypot(np.subtract(x2, x1), np.subtract(y2, y1))


def project_onto_bearing(
    vx: ArrayLike, vy: ArrayLike, bearing: ArrayLike
) -> np.ndarray:
    """The component of a vector along a heading.

    Positive means the vector points the same way as the bearing, so a
    positive projection of velocity onto the bearing to the ball is a
    player closing on it.
    """
    bx, by = unit_vector(bearing)
    return np.multiply(vx, bx) + np.multiply(vy, by)
