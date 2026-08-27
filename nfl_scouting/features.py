"""One row per pass attempt, describing the throw and the flight.

Two moments carry almost everything a scout wants out of this data. The
first is the release: where the receiver was, who was on him, how much room
he had. The second is the flight, the half-second to two seconds while the
ball is in the air, which is the only window in the data where you can watch
a receiver and a defender react to the same known destination and compare
what they did about it.

The competition's own framing makes that second window measurable. Every
play carries `ball_land_x` / `ball_land_y`, so the landing spot is known to
the analyst from the first frame of the flight even though it was not known
to the players. Pointing each player's track at that spot is what turns
"he changed direction" into "he changed direction toward the ball".

A note on who is covering whom: the data does not say. This module uses the
defender closest to the targeted receiver at the moment of release, which is
the same proxy Next Gen Stats uses for separation. On plays with a genuine
bracket it will name one of the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import geometry, kinematics
from .config import (
    DT,
    FIELD_WIDTH,
    MAX_RECEIVER_SPEED,
    REACHABILITY_SLACK,
    ROLE_COVERAGE,
    ROLE_OTHER_ROUTE,
    ROLE_PASSER,
    ROLE_TARGET,
)
from .data import PLAY_KEY, _parse_height

# How close to the landing spot a defender counts as being "in the area" at
# the moment of release - roughly the radius a defensive back can cover in
# the time a ball is in the air.
CONTEST_RADIUS_YARDS = 5.0


def throw_state(play_tracking: pd.DataFrame) -> pd.DataFrame:
    """Every player's state on the last pre-pass frame: the release.

    Indexed by `nfl_id`, one row per player.
    """
    if play_tracking.empty:
        raise ValueError("no tracking frames for this play")
    last_frame = play_tracking["frame_id"].max()
    state = play_tracking.loc[play_tracking["frame_id"] == last_frame]
    if state["nfl_id"].duplicated().any():
        raise ValueError("duplicate players on the release frame")
    return state.set_index("nfl_id")


def _role_id(state: pd.DataFrame, role: str) -> int | None:
    """The single player in a role, or None if the play does not have exactly one."""
    matches = state.index[state["player_role"] == role]
    return int(matches[0]) if len(matches) == 1 else None


def build_air_tracks(play_air: pd.DataFrame,
                     state: pd.DataFrame) -> dict[int, kinematics.Track]:
    """A differentiable path through the flight for each predicted player.

    The release-frame position is prepended to the ball-in-air frames. The
    output file starts a tenth of a second after the throw, so without that
    first point every track would be missing the instant the receiver began
    reacting - and short flights would lose a fifth of their length.
    """
    tracks: dict[int, kinematics.Track] = {}
    if play_air.empty:
        return tracks
    for nfl_id, frames in play_air.groupby("nfl_id", sort=False):
        frames = frames.sort_values("frame_id")
        xs, ys = frames["x"].to_numpy(), frames["y"].to_numpy()
        if nfl_id in state.index:
            xs = np.concatenate([[state.at[nfl_id, "x"]], xs])
            ys = np.concatenate([[state.at[nfl_id, "y"]], ys])
        tracks[int(nfl_id)] = kinematics.build_track(xs, ys)
    return tracks


def nearest_player(state: pd.DataFrame, to_x: float, to_y: float,
                   candidates: pd.Index) -> tuple[int | None, float]:
    """Which of `candidates` is closest to a point, and how far away."""
    usable = state.index.intersection(candidates)
    if len(usable) == 0:
        return None, float("nan")
    sub = state.loc[usable]
    dist = geometry.distance(sub["x"].to_numpy(), sub["y"].to_numpy(), to_x, to_y)
    best = int(np.argmin(dist))
    return int(usable[best]), float(dist[best])


def _prefixed(values: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{k}": v for k, v in values.items()}


def _nan_block(prefix: str, keys) -> dict[str, float]:
    return {f"{prefix}{k}": float("nan") for k in keys}


# The kinematic summary keys, so a player with no ball-in-air track still
# produces a row of the same shape rather than a ragged one.
_SUMMARY_KEYS = tuple(
    kinematics.summarise(
        kinematics.build_track(np.arange(6) * 0.5, np.zeros(6)), 10.0, 0.0
    ).keys()
)


def extract_play(play_tracking: pd.DataFrame,
                 play_air: pd.DataFrame) -> dict[str, object] | None:
    """Every number this package derives from one pass attempt.

    Returns None when the play is missing a piece that the rest of the
    pipeline needs - no targeted receiver, or no ball-in-air frames at all.
    """
    state = throw_state(play_tracking)
    target_id = _role_id(state, ROLE_TARGET)
    if target_id is None or play_air.empty:
        return None

    target = state.loc[target_id]
    land_x = float(target["ball_land_x"])
    land_y = float(target["ball_land_y"])
    if not np.isfinite(land_x) or not np.isfinite(land_y):
        return None

    passer_id = _role_id(state, ROLE_PASSER)
    coverage_ids = state.index[state["player_role"] == ROLE_COVERAGE]
    receiver_ids = state.index[state["player_role"].isin((ROLE_TARGET, ROLE_OTHER_ROUTE))]

    # --- the release -------------------------------------------------
    cover_id, separation_throw = nearest_player(
        state, float(target["x"]), float(target["y"]), coverage_ids)

    direction = 1.0 if str(target["play_direction"]).lower() == "right" else -1.0
    los_x = float(target["absolute_yardline_number"])

    row: dict[str, object] = {
        "game_id": int(play_tracking["game_id"].iat[0]),
        "play_id": int(play_tracking["play_id"].iat[0]),
        "play_direction": str(target["play_direction"]),
        "throw_frame": int(state["frame_id"].iat[0]),
        "n_air_frames": int(target["num_frames_output"]),
        "air_time_s": float(target["num_frames_output"]) * DT,
        "ball_land_x": land_x,
        "ball_land_y": land_y,
        "los_x": los_x,

        "target_nfl_id": target_id,
        "target_name": str(target["player_name"]),
        "target_position": str(target["player_position"]),
        "target_height_in": _parse_height(target["player_height"]),
        "target_weight_lb": float(target["player_weight"]),
        "passer_nfl_id": passer_id,
        "passer_name": str(state.at[passer_id, "player_name"]) if passer_id else None,
        "coverage_nfl_id": cover_id,
        "coverage_name": str(state.at[cover_id, "player_name"]) if cover_id else None,
        "coverage_position": str(state.at[cover_id, "player_position"]) if cover_id else None,

        "separation_at_throw_yd": separation_throw,
        "n_route_runners": int(len(receiver_ids)),
        "n_coverage_defenders": int(len(coverage_ids)),
    }

    # Signed so that positive is always toward the offence's end zone,
    # whichever way the play happened to be running.
    row["target_depth_yd"] = direction * (float(target["x"]) - los_x)
    row["air_yards_yd"] = direction * (land_x - los_x)
    row["land_sideline_dist_yd"] = float(min(land_y, FIELD_WIDTH - land_y))
    row["target_sideline_dist_yd"] = float(min(target["y"], FIELD_WIDTH - target["y"]))

    row["target_speed_at_throw_yps"] = float(target["s"])
    row["target_accel_at_throw_yps2"] = float(target["a"])
    row["target_dist_to_spot_at_throw_yd"] = float(
        geometry.distance(target["x"], target["y"], land_x, land_y))

    if passer_id is not None:
        passer = state.loc[passer_id]
        row["throw_distance_yd"] = float(
            geometry.distance(passer["x"], passer["y"], land_x, land_y))
    else:
        row["throw_distance_yd"] = float("nan")

    # How the defence was placed relative to where the ball was going, at the
    # moment it was thrown. `defenders_near_spot` is the contest count.
    if len(coverage_ids):
        cov = state.loc[coverage_ids]
        d_spot = geometry.distance(cov["x"].to_numpy(), cov["y"].to_numpy(),
                                   land_x, land_y)
        row["nearest_defender_to_spot_yd"] = float(d_spot.min())
        row["defenders_near_spot"] = int((d_spot <= CONTEST_RADIUS_YARDS).sum())
    else:
        row["nearest_defender_to_spot_yd"] = float("nan")
        row["defenders_near_spot"] = 0

    if cover_id is not None:
        cover = state.loc[cover_id]
        row["coverage_dist_to_spot_at_throw_yd"] = float(
            geometry.distance(cover["x"], cover["y"], land_x, land_y))
        row["coverage_speed_at_throw_yps"] = float(cover["s"])
        # Positive when the defender is nearer the landing spot than the
        # receiver is: he has inside position on the ball.
        row["coverage_position_advantage_yd"] = (
            row["target_dist_to_spot_at_throw_yd"]
            - row["coverage_dist_to_spot_at_throw_yd"])
        # How far the defender's body was turned away from the ball's path
        # when it was released - the cost he has to pay to get back to it.
        row["coverage_off_bearing_at_throw_deg"] = float(abs(
            geometry.angle_difference_deg(
                cover["dir"],
                geometry.bearing_deg(cover["x"], cover["y"], land_x, land_y))))
    else:
        for key in ("coverage_dist_to_spot_at_throw_yd", "coverage_speed_at_throw_yps",
                    "coverage_position_advantage_yd", "coverage_off_bearing_at_throw_deg"):
            row[key] = float("nan")

    # --- the flight --------------------------------------------------
    tracks = build_air_tracks(play_air, state)

    target_track = tracks.get(target_id)
    if target_track is not None and target_track.n_frames >= 2:
        row.update(_prefixed(
            kinematics.summarise(target_track, land_x, land_y), "rec_"))
    else:
        row.update(_nan_block("rec_", _SUMMARY_KEYS))

    cover_track = tracks.get(cover_id) if cover_id is not None else None
    row["coverage_tracked_in_air"] = cover_track is not None
    if cover_track is not None and cover_track.n_frames >= 2:
        row.update(_prefixed(
            kinematics.summarise(cover_track, land_x, land_y), "cov_"))
    else:
        row.update(_nan_block("cov_", _SUMMARY_KEYS))

    # --- the arrival -------------------------------------------------
    # Separation at the catch point uses whichever tracked defender is
    # actually closest when the ball gets there, which need not be the man
    # who was closest at the throw.
    row.update(_arrival_block(tracks, state, target_id, coverage_ids, land_x, land_y))
    # Negative means the defender closed while the ball was in the air. The
    # subtraction is between two measurements over the same set of tracked
    # defenders; see `_arrival_block` for why it cannot use the headline
    # separation.
    row["separation_change_yd"] = (
        row["separation_at_arrival_yd"] - row["separation_at_throw_tracked_yd"])

    # Whether the recorded landing spot is one this receiver could have got to.
    row["landing_spot_reachable"] = bool(
        row["target_dist_to_spot_at_throw_yd"]
        <= MAX_RECEIVER_SPEED * row["air_time_s"] + REACHABILITY_SLACK)
    return row


def _arrival_block(tracks: dict[int, kinematics.Track], state: pd.DataFrame,
                   target_id: int, coverage_ids: pd.Index, land_x: float,
                   land_y: float) -> dict[str, object]:
    """Separation when the ball arrives, and the release measurement to compare it to.

    Only some defenders are tracked through the flight - about eleven plays
    in a hundred do not include the man who was nearest at the release. That
    makes the headline separation, which is measured against all eleven
    defenders, the wrong thing to subtract from: on those plays the nearest
    *tracked* defender at arrival can be a safety forty yards away, and the
    receiver appears to have gained thirty yards of separation in a second.

    So the change is measured between two numbers taken over the same set of
    players: `separation_at_throw_tracked_yd` restricts the release
    measurement to the defenders who are tracked in the air, and the
    difference against arrival is then like for like.
    """
    empty = {
        "separation_at_arrival_yd": float("nan"),
        "separation_at_throw_tracked_yd": float("nan"),
        "arrival_defender_nfl_id": None,
        "target_dist_to_spot_at_arrival_yd": float("nan"),
    }
    target_track = tracks.get(target_id)
    if target_track is None:
        return empty

    tracked_coverage = [nfl_id for nfl_id in tracks
                        if nfl_id != target_id and nfl_id in coverage_ids]
    tx, ty = float(target_track.x[-1]), float(target_track.y[-1])

    best_id, best_dist = None, float("inf")
    for nfl_id in tracked_coverage:
        track = tracks[nfl_id]
        d = float(geometry.distance(track.x[-1], track.y[-1], tx, ty))
        if d < best_dist:
            best_id, best_dist = nfl_id, d

    if best_id is None:
        block = dict(empty)
        block["target_dist_to_spot_at_arrival_yd"] = float(
            geometry.distance(tx, ty, land_x, land_y))
        return block

    _, separation_throw_tracked = nearest_player(
        state, float(state.at[target_id, "x"]), float(state.at[target_id, "y"]),
        pd.Index(tracked_coverage))

    return {
        "separation_at_arrival_yd": best_dist,
        "separation_at_throw_tracked_yd": separation_throw_tracked,
        "arrival_defender_nfl_id": best_id,
        "target_dist_to_spot_at_arrival_yd": float(
            geometry.distance(tx, ty, land_x, land_y)),
    }


def extract_week(tracking: pd.DataFrame, ball_air: pd.DataFrame) -> pd.DataFrame:
    """Run `extract_play` over a week and stack the rows."""
    air_by_play = dict(tuple(ball_air.groupby(PLAY_KEY, sort=False)))
    empty = ball_air.iloc[:0]
    rows = []
    for key, play_tracking in tracking.groupby(PLAY_KEY, sort=False):
        row = extract_play(play_tracking, air_by_play.get(key, empty))
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)
