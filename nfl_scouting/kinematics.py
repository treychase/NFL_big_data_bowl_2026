"""Velocity, acceleration and change of direction from a position track.

The tracking file carries speed, acceleration and direction for every
pre-pass frame, but the ball-in-air frames are positions only. Everything
this package says about what a player did while the ball was in the air is
therefore differentiated from x and y here.

Positions at 10 Hz carry roughly a tenth of a yard of jitter, and
differentiating twice squares that into nonsense. Rather than smooth and
then difference - which drags the endpoints inward and invents acceleration
at exactly the two frames that matter, the throw and the catch - this module
fits a local polynomial to a short window and reads the derivative off the
fit (Savitzky-Golay). That is exact for any motion up to the polynomial's
order, so a player running at a constant speed comes back with zero
acceleration at every frame, and one accelerating steadily comes back with
the right number at the first frame and the last.

Windows are kept short (7 frames, 0.7 s, dropping to 5 or 3 on brief plays)
because a receiver's break is real signal about a third of a second long and
a wider window would flatten the thing being measured.

The fit is quadratic rather than cubic on purpose. A cubic fit rings at a
sharp break - it swings the heading past the new direction and back, which
inflates a 90 degree cut to 101 - while measuring acceleration no better
(both recover 6.58 of a true 7.88 yd/s^2 burst). Quadratic reads every
realistic break, from a tenth of a second to half a second, at exactly 90.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from . import geometry
from .config import DT, MIN_SPEED_FOR_HEADING

MAX_WINDOW = 7
MAX_POLYORDER = 2

# Closer than this to a target point, the bearing to it is set by position
# jitter rather than by geometry, so those frames are left out of the
# off-bearing averages.
MIN_BEARING_DISTANCE = 0.5   # yards


def _window_and_order(n: int) -> tuple[int, int]:
    """The widest smoothing window this many frames can support, and its order."""
    window = min(MAX_WINDOW, n if n % 2 else n - 1)
    if window < 3:
        return 0, 0
    return window, min(MAX_POLYORDER, window - 1)


def differentiate(values: np.ndarray, deriv: int = 1, dt: float = DT) -> np.ndarray:
    """The `deriv`-th time derivative of a sampled signal, same length in and out.

    Uses a local polynomial fit where there are enough frames for one, and
    falls back to finite differences on tracks too short to fit.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if deriv == 0:
        return values.copy()
    if n <= deriv:
        return np.zeros(n)

    window, polyorder = _window_and_order(n)
    if window >= 3 and polyorder >= deriv:
        return savgol_filter(values, window_length=window, polyorder=polyorder,
                             deriv=deriv, delta=dt, mode="interp")
    if deriv == 1:
        return np.gradient(values, dt, edge_order=1)
    return np.zeros(n)


@dataclass(frozen=True)
class Track:
    """One player's path over a window of frames, with derived motion.

    `x`/`y` are the positions as recorded; speed, acceleration and heading
    come from the local polynomial fit. All arrays are the same length as
    the input.
    """

    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    speed: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    accel: np.ndarray
    heading: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.x.size)

    @property
    def duration_s(self) -> float:
        return max(self.n_frames - 1, 0) * DT

    @property
    def path_length(self) -> float:
        """Distance actually run, following every wiggle of the path."""
        if self.n_frames < 2:
            return 0.0
        return float(np.hypot(np.diff(self.x), np.diff(self.y)).sum())

    @property
    def displacement(self) -> float:
        """Straight-line distance from the first frame to the last."""
        if self.n_frames < 2:
            return 0.0
        return float(geometry.distance(self.x[0], self.y[0], self.x[-1], self.y[-1]))


def build_track(x, y, dt: float = DT) -> Track:
    """Differentiate a position track into velocity, acceleration and heading."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must be the same length, got {x.shape} and {y.shape}")
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D track, got shape {x.shape}")
    if x.size == 0:
        raise ValueError("cannot build a track from zero frames")

    vx, vy = differentiate(x, 1, dt), differentiate(y, 1, dt)
    ax, ay = differentiate(x, 2, dt), differentiate(y, 2, dt)
    return Track(
        x=x, y=y, vx=vx, vy=vy, speed=np.hypot(vx, vy),
        ax=ax, ay=ay, accel=np.hypot(ax, ay),
        heading=geometry.heading_deg(vx, vy),
    )


def turn_rates(track: Track, dt: float = DT,
               min_speed: float = MIN_SPEED_FOR_HEADING) -> np.ndarray:
    """Signed degrees per second of heading change, frame to frame.

    Frames where either end of the step is slower than `min_speed` come back
    as NaN: a heading taken from a player standing still is an artefact of
    position jitter, and letting those into a change-of-direction sum would
    rank the slowest players as the twitchiest.
    """
    if track.n_frames < 2:
        return np.zeros(0)
    delta = geometry.angle_difference_deg(track.heading[1:], track.heading[:-1])
    rate = delta / dt
    moving = (track.speed[1:] >= min_speed) & (track.speed[:-1] >= min_speed)
    return np.where(moving, rate, np.nan)


def _nan_safe(fn, values: np.ndarray, default: float = 0.0) -> float:
    """Apply a reducer to the non-NaN part, or return a default if there is none."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return default
    return float(fn(values[~np.isnan(values)]))


def change_of_direction(track: Track, dt: float = DT) -> dict[str, float]:
    """How much a player turned over a window, and how hard.

    - `cod_total_deg`   every degree of turn added up, direction ignored, so
                        a stutter and a single hard break both count for what
                        they cost to execute.
    - `cod_net_deg`     the turn from first heading to last: a receiver who
                        breaks out and comes back scores high on total and
                        near zero on net.
    - `cod_max_rate_dps` the sharpest single turn, the break itself.
    - `cod_rate_dps`    total turn spread over the window, for comparing
                        windows of different lengths.
    """
    rates = turn_rates(track, dt=dt)
    absolute = np.abs(rates)
    total = _nan_safe(lambda v: np.sum(v * dt), absolute)
    duration = track.duration_s
    moving = track.speed >= MIN_SPEED_FOR_HEADING
    if moving.sum() >= 2:
        first, last = np.flatnonzero(moving)[[0, -1]]
        net = abs(float(geometry.angle_difference_deg(
            track.heading[last], track.heading[first])))
    else:
        net = 0.0
    return {
        "cod_total_deg": total,
        "cod_net_deg": net,
        "cod_max_rate_dps": _nan_safe(np.max, absolute),
        "cod_rate_dps": total / duration if duration > 0 else 0.0,
    }


def acceleration_summary(track: Track) -> dict[str, float]:
    """Speed change over a window: how fast, how much faster, how hard.

    `accel_burst_yps2` is the largest gain in speed across any half second in
    the window, which is closer to what a scout means by "he can go" than a
    peak of the differentiated signal.
    """
    speed = track.speed
    burst = 0.0
    lag = int(round(0.5 / DT))
    if speed.size > lag:
        burst = float(np.max((speed[lag:] - speed[:-lag]) / (lag * DT)))
    return {
        "speed_start_yps": float(speed[0]),
        "speed_end_yps": float(speed[-1]),
        "speed_max_yps": float(speed.max()),
        "speed_delta_yps": float(speed[-1] - speed[0]),
        "accel_mean_yps2": float(track.accel.mean()),
        "accel_max_yps2": float(track.accel.max()),
        "accel_burst_yps2": burst,
    }


def pursuit_to_point(track: Track, target_x: float, target_y: float,
                     dt: float = DT) -> dict[str, float]:
    """How directly a player worked toward a fixed point - the ball's landing spot.

    - `dist_*`             distance to the point at each end of the window.
    - `closing_speed_yps`  average rate the gap shut; negative means the
                           point ran away from him.
    - `pursuit_efficiency` ground actually closed over ground covered. One is
                           a straight line at the ball; near zero is a player
                           running hard in the wrong direction.
    - `mean_off_bearing_deg` average angle between where he was going and
                           where the ball was going. This is the number that
                           separates a defender who reacted from one who kept
                           running his drop.
    - `bearing_correction_deg` how much of that error he took out between the
                           first usable frame and the last: the redirect
                           itself, positive when he turned toward the ball.
    - `closing_accel_yps2` mean acceleration projected at the landing spot.

    Frames where the player is standing on the target, or barely moving, are
    left out of the angle summaries: a bearing measured from half a yard away
    is position jitter, not pursuit.
    """
    d_start = float(geometry.distance(track.x[0], track.y[0], target_x, target_y))
    d_end = float(geometry.distance(track.x[-1], track.y[-1], target_x, target_y))
    duration = track.duration_s
    closed = d_start - d_end

    to_spot = geometry.distance(track.x, track.y, target_x, target_y)
    bearing = geometry.bearing_deg(track.x, track.y, target_x, target_y)
    off = np.abs(geometry.angle_difference_deg(track.heading, bearing))
    usable = (track.speed >= MIN_SPEED_FOR_HEADING) & (to_spot >= MIN_BEARING_DISTANCE)
    off_usable = np.where(usable, off, np.nan)

    if usable.sum() >= 2:
        first, last = np.flatnonzero(usable)[[0, -1]]
        correction = float(off[first] - off[last])
    else:
        correction = 0.0

    path = track.path_length
    closing_accel = np.where(to_spot >= MIN_BEARING_DISTANCE,
                             geometry.project_onto_bearing(track.ax, track.ay, bearing),
                             np.nan)

    return {
        "dist_to_spot_start_yd": d_start,
        "dist_to_spot_end_yd": d_end,
        "dist_closed_yd": closed,
        "closing_speed_yps": closed / duration if duration > 0 else 0.0,
        "pursuit_efficiency": closed / path if path > 1e-6 else 0.0,
        "mean_off_bearing_deg": _nan_safe(np.nanmean, off_usable, float("nan")),
        "bearing_correction_deg": correction,
        "closing_accel_yps2": _nan_safe(np.nanmean, closing_accel, 0.0),
    }


def summarise(track: Track, target_x: float | None = None,
              target_y: float | None = None) -> dict[str, float]:
    """Everything this module measures about one window of one player's track."""
    out: dict[str, float] = {
        "n_frames": float(track.n_frames),
        "duration_s": track.duration_s,
        "path_length_yd": track.path_length,
        "displacement_yd": track.displacement,
    }
    out.update(acceleration_summary(track))
    out.update(change_of_direction(track))
    if target_x is not None and target_y is not None:
        out.update(pursuit_to_point(track, target_x, target_y))
    return out
