"""Pack the artifacts into one payload the dashboard can hold in memory.

The dashboard draws every play in the season, which means the browser needs
the tracking geometry for all 14,000 of them. Sent as JSON objects that is
tens of megabytes; the packing here gets it to a few.

Two ideas do the work. Text repeats - the same 464 receiver names across
14,000 plays - so strings are interned once and referenced by index.
Coordinates do not need to be exact - the field is 120 yards and the
tracking is honest to about a tenth of one - so positions are quantised to
tenths of a yard and packed as 16-bit integers into a single base64 blob
with an offset table, which the page unpacks into one typed array.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .config import (
    COVERAGE_DB_POSITIONS,
    MIN_TARGETS_FOR_DEFENDER,
    MIN_TARGETS_FOR_RECEIVER,
    OPEN_SEPARATION_YARDS,
    SEASON,
)

# Positions are stored as tenths of a yard, which is the precision the
# tracking system actually delivers, and fits any point on the field into an
# int16 with three orders of magnitude to spare.
POSITION_SCALE = 10.0

# The pre-pass route is drawn as a shape, not animated, so every third
# frame traces it perfectly well and costs a third of what every frame does.
ROUTE_STRIDE = 3

KIND_TARGET = 0
KIND_PRIMARY_COVERAGE = 1
KIND_OTHER_COVERAGE = 2

ROLE_CODES = {
    "Targeted Receiver": 0,
    "Defensive Coverage": 1,
    "Passer": 2,
    "Other Route Runner": 3,
}


class Interner:
    """Give every distinct string an index, keeping first-seen order."""

    def __init__(self) -> None:
        self._index: dict[str, int] = {}
        self.values: list[str] = []

    def __call__(self, value) -> int:
        text = "" if value is None or (isinstance(value, float) and np.isnan(value)) \
            else str(value)
        if text not in self._index:
            self._index[text] = len(self.values)
            self.values.append(text)
        return self._index[text]


def _quantise(values) -> np.ndarray:
    """Positions to tenths of a yard, clipped to what an int16 can hold."""
    scaled = np.rint(np.asarray(values, dtype=float) * POSITION_SCALE)
    return np.clip(np.nan_to_num(scaled, nan=0.0), -32768, 32767).astype(np.int16)


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(array.tobytes()).decode("ascii")


def pack_floats(values) -> str:
    """A float column as base64 float32.

    Roughly a third the size of the same numbers written out as JSON text,
    and it carries NaN through natively - so a receiver whose covering
    defender was never tracked stays missing on the page instead of turning
    into a zero that would quietly drag his averages down.
    """
    return _b64(np.asarray(values, dtype="<f4"))


def pack_ints(values, dtype="<i2") -> str:
    """An integer or index column as base64, defaulting to int16."""
    array = np.asarray(values, dtype=float)
    return _b64(np.nan_to_num(array, nan=0.0).astype(dtype))


def build_geometry(plays: pd.DataFrame, weeks=None, season: int = SEASON,
                   root: Path | None = None) -> dict:
    """Per-play tracking, packed into one int16 blob with an offset table.

    Each play's block is laid out as:
        [n_snapshot, n_route_frames, n_flight_players,
         (role, x, y) * n_snapshot,
         (x, y) * n_route_frames,
         (kind, n_frames, (x, y) * n_frames) * n_flight_players]

    The snapshot is every tracked player at the release; the route is the
    targeted receiver's path up to it; the flight blocks are the tracked
    players while the ball is in the air.
    """
    weeks = weeks or sorted(int(w) for w in plays["week"].unique())
    wanted = plays.set_index(["game_id", "play_id"])
    blocks: dict[tuple[int, int], np.ndarray] = {}

    for week in weeks:
        frames = data.load_week(week, season, root)
        air_by_play = dict(tuple(frames.ball_air.groupby(data.PLAY_KEY, sort=False)))

        for key, tracking in frames.tracking.groupby(data.PLAY_KEY, sort=False):
            if key not in wanted.index:
                continue
            play = wanted.loc[key]
            last_frame = tracking["frame_id"].max()
            snapshot = tracking[tracking["frame_id"] == last_frame]

            parts: list[np.ndarray] = []
            header = [len(snapshot)]

            snap = np.empty(len(snapshot) * 3, dtype=np.int16)
            snap[0::3] = [ROLE_CODES.get(r, 3) for r in snapshot["player_role"]]
            snap[1::3] = _quantise(snapshot["x"])
            snap[2::3] = _quantise(snapshot["y"])

            route = tracking[tracking["nfl_id"] == play["target_nfl_id"]].sort_values("frame_id")
            route = route.iloc[::ROUTE_STRIDE]
            route_packed = np.empty(len(route) * 2, dtype=np.int16)
            route_packed[0::2] = _quantise(route["x"])
            route_packed[1::2] = _quantise(route["y"])
            header.append(len(route))

            flight_parts: list[np.ndarray] = []
            air = air_by_play.get(key)
            n_flight = 0
            if air is not None:
                for nfl_id, track in air.groupby("nfl_id", sort=False):
                    track = track.sort_values("frame_id")
                    if nfl_id == play["target_nfl_id"]:
                        kind = KIND_TARGET
                    elif nfl_id == play["coverage_nfl_id"]:
                        kind = KIND_PRIMARY_COVERAGE
                    else:
                        kind = KIND_OTHER_COVERAGE
                    packed = np.empty(len(track) * 2 + 2, dtype=np.int16)
                    packed[0] = kind
                    packed[1] = len(track)
                    packed[2::2] = _quantise(track["x"])
                    packed[3::2] = _quantise(track["y"])
                    flight_parts.append(packed)
                    n_flight += 1
            header.append(n_flight)

            parts.append(np.array(header, dtype=np.int16))
            parts.append(snap)
            parts.append(route_packed)
            parts.extend(flight_parts)
            blocks[key] = np.concatenate(parts)

    order = list(zip(plays["game_id"], plays["play_id"]))
    offsets, chunks, cursor = [], [], 0
    for key in order:
        block = blocks.get(key)
        if block is None:
            offsets.append(-1)
            continue
        offsets.append(cursor)
        chunks.append(block)
        cursor += block.size

    blob = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    return {
        "scale": POSITION_SCALE,
        "offsets": offsets,
        "data": base64.b64encode(blob.astype("<i2").tobytes()).decode("ascii"),
        "n_values": int(blob.size),
    }


# Columns shipped for every play, as columnar arrays.
# Numeric play columns, packed as float32. The decimal in each entry is what
# the page rounds to when it prints the value, not a transport precision.
PLAY_NUMERIC = [
    ("sep", "separation_at_throw_yd", 2),
    ("sepArr", "separation_at_arrival_yd", 2),
    ("sepChg", "separation_change_yd", 2),
    ("airYds", "air_yards_yd", 1),
    ("depth", "target_depth_yd", 1),
    ("airTime", "air_time_s", 1),
    ("throwDist", "throw_distance_yd", 1),
    ("yards", "yards_gained", 0),
    ("cp", "catch_probability", 3),
    ("coe", "catch_over_expected", 3),
    ("landX", "ball_land_x", 1),
    ("landY", "ball_land_y", 1),
    ("losX", "los_x", 1),
    ("recCod", "rec_cod_total_deg", 1),
    ("recBurst", "rec_accel_burst_yps2", 2),
    ("recSpeed", "rec_speed_max_yps", 2),
    ("recEff", "rec_pursuit_efficiency", 3),
    ("covCod", "cov_cod_total_deg", 1),
    ("covCorr", "cov_bearing_correction_deg", 1),
    ("covEff", "cov_pursuit_efficiency", 3),
    ("covOff", "cov_mean_off_bearing_deg", 1),
    ("covClose", "cov_closing_speed_yps", 2),
    ("nearSpot", "nearest_defender_to_spot_yd", 2),
]

PLAY_TEXT = [
    ("recv", "target_name"),
    ("recvPos", "target_position"),
    ("cov", "coverage_name"),
    ("covPos", "coverage_position"),
    ("route", "route_of_targeted_receiver"),
    ("manZone", "team_coverage_man_zone"),
    ("covType", "team_coverage_type"),
    ("off", "possession_team"),
    ("def", "defensive_team"),
]


def build_payload(artifacts, weeks=None, season: int = SEASON,
                  root: Path | None = None, with_geometry: bool = True) -> dict:
    """Everything the dashboard reads, in one JSON-serialisable dict."""
    plays = artifacts.plays.reset_index(drop=True)
    interner = Interner()

    flags = (plays["is_complete"].astype(int)
             + 2 * plays["is_touchdown"].astype(int)
             + 4 * plays["is_interception"].astype(int)
             + 8 * plays["play_direction"].str.lower().eq("right").astype(int))

    play_block: dict[str, object] = {
        "n": len(plays),
        "int16": {
            "week": pack_ints(plays["week"]),
            "quarter": pack_ints(plays["quarter"]),
            "down": pack_ints(plays["down"]),
            "toGo": pack_ints(plays["yards_to_go"]),
            "flags": pack_ints(flags),
        },
        "f32": {},
    }
    for name, column, _ in PLAY_NUMERIC:
        play_block["f32"][name] = pack_floats(plays[column])
    for name, column in PLAY_TEXT:
        play_block["int16"][name] = pack_ints(
            [interner(v) for v in plays[column]])

    payload = {
        "meta": _meta(artifacts),
        "plays": play_block,
        "receivers": _table(artifacts.receivers),
        "defenders": _table(artifacts.defenders),
        "routeCoverage": _table(artifacts.route_coverage),
        "coverageType": _table(artifacts.coverage_type),
        "receiverRoutes": _table(artifacts.receiver_routes),
        "model": {
            "scores": artifacts.summary["model"]["scores"],
            "baselines": artifacts.summary["model"]["baselines"],
            "folds": artifacts.summary["model"]["fold_scores"],
            "calibration": artifacts.summary["model"]["calibration"],
            "importance": _table(artifacts.importance.head(16)),
            "features": artifacts.summary["model"]["n_features"],
            "baseRate": artifacts.summary["model"]["base_rate"],
        },
    }
    if with_geometry:
        payload["geometry"] = build_geometry(plays, weeks, season, root)
    # Interning has to finish before the lookup table is read out.
    payload["strings"] = interner.values
    return payload


def _meta(artifacts) -> dict:
    summary = artifacts.summary
    return {
        "season": SEASON,
        "weeks": summary["weeks"],
        "nPlays": summary["n_plays"],
        "nGames": summary["n_games"],
        "nReceivers": summary["n_receivers"],
        "nDefenders": summary["n_defenders"],
        "completionRate": summary["completion_rate"],
        "meanSeparationThrow": summary["mean_separation_at_throw_yd"],
        "meanSeparationArrival": summary["mean_separation_at_arrival_yd"],
        "coverageTrackedShare": summary["coverage_tracked_in_air_share"],
        "openThreshold": OPEN_SEPARATION_YARDS,
        "minTargetsReceiver": MIN_TARGETS_FOR_RECEIVER,
        "minTargetsDefender": MIN_TARGETS_FOR_DEFENDER,
        "dbPositions": list(COVERAGE_DB_POSITIONS),
    }


def _table(frame: pd.DataFrame) -> list[dict]:
    """A small table as records, with NaN rendered as null and floats trimmed."""
    if frame is None or frame.empty:
        return []
    out = []
    for record in frame.to_dict("records"):
        clean = {}
        for key, value in record.items():
            if isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                clean[key] = None if not np.isfinite(value) else round(float(value), 4)
            elif isinstance(value, (np.bool_, bool)):
                clean[key] = bool(value)
            else:
                clean[key] = value
        out.append(clean)
    return out


def write_payload(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path
