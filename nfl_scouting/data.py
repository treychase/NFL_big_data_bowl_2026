"""Loading the competition files and putting the play context in one frame.

The competition ships three things: an `input` file of tracking frames from
the snap up to the moment the ball leaves the quarterback's hand, an
`output` file of positions for a subset of players while the ball is in the
air, and a play-level `supplementary` file with the route, the coverage and
what happened. This module reads them and nothing else - no derived numbers
live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .config import (
    PASS_RESULTS,
    RESULT_COMPLETE,
    RESULT_INTERCEPTION,
    SEASON,
    WEEKS,
    data_dir,
)

PLAY_KEY = ["game_id", "play_id"]

INPUT_DTYPES = {
    "game_id": "int64",
    "play_id": "int64",
    "nfl_id": "int64",
    "frame_id": "int32",
    "player_to_predict": "bool",
    "absolute_yardline_number": "float64",
    "player_weight": "float64",
    "num_frames_output": "int32",
    "x": "float64",
    "y": "float64",
    "s": "float64",
    "a": "float64",
    "dir": "float64",
    "o": "float64",
    "ball_land_x": "float64",
    "ball_land_y": "float64",
}

OUTPUT_DTYPES = {
    "game_id": "int64",
    "play_id": "int64",
    "nfl_id": "int64",
    "frame_id": "int32",
    "x": "float64",
    "y": "float64",
}


def input_path(week: int, season: int = SEASON, root: Path | None = None) -> Path:
    root = root or data_dir()
    return root / "train" / f"input_{season}_w{week:02d}.csv"


def output_path(week: int, season: int = SEASON, root: Path | None = None) -> Path:
    root = root or data_dir()
    return root / "train" / f"output_{season}_w{week:02d}.csv"


def supplementary_path(root: Path | None = None) -> Path:
    return (root or data_dir()) / "supplementary_data.csv"


@dataclass
class WeekFrames:
    """One week of tracking, both sides of the throw."""

    week: int
    tracking: pd.DataFrame   # pre-pass frames, every player on the field
    ball_air: pd.DataFrame   # ball-in-air frames, the predicted players only

    @property
    def n_plays(self) -> int:
        return int(self.tracking.groupby(PLAY_KEY, sort=False).ngroups)


def load_week(week: int, season: int = SEASON, root: Path | None = None) -> WeekFrames:
    tracking = pd.read_csv(
        input_path(week, season, root), dtype=INPUT_DTYPES, low_memory=False
    )
    ball_air = pd.read_csv(output_path(week, season, root), dtype=OUTPUT_DTYPES)
    tracking = tracking.sort_values(PLAY_KEY + ["nfl_id", "frame_id"])
    ball_air = ball_air.sort_values(PLAY_KEY + ["nfl_id", "frame_id"])
    return WeekFrames(week=week, tracking=tracking, ball_air=ball_air)


def iter_weeks(weeks=WEEKS, season: int = SEASON,
               root: Path | None = None) -> Iterator[WeekFrames]:
    """Weeks one at a time - the full season does not want to be in memory at once."""
    for week in weeks:
        yield load_week(week, season, root)


def available_weeks(season: int = SEASON, root: Path | None = None) -> list[int]:
    return [w for w in WEEKS
            if input_path(w, season, root).exists()
            and output_path(w, season, root).exists()]


def _parse_height(height) -> float:
    """`6-1` into inches. Anything unparseable comes back NaN rather than guessed."""
    if not isinstance(height, str) or "-" not in height:
        return float("nan")
    feet, _, inches = height.partition("-")
    try:
        return float(feet) * 12.0 + float(inches)
    except ValueError:
        return float("nan")


def parse_heights(heights: pd.Series) -> pd.Series:
    return heights.map(_parse_height).astype("float64")


def load_supplementary(root: Path | None = None) -> pd.DataFrame:
    """Play context, with the columns this package needs made usable.

    The file marks a touchdown only inside the play description, so it is
    read out of the text here and only trusted on completions: a description
    that says TOUCHDOWN on an interception is describing the defence
    scoring, which is not a passing touchdown.
    """
    supp = pd.read_csv(supplementary_path(root), low_memory=False)
    supp = supp.rename(columns=str.lower)

    # A column that happens to be entirely blank is read back as floats, and
    # the string accessor refuses it. Pinning the text columns to str keeps
    # a sparse file from taking the loader down.
    for col in ("play_description", "play_nullified_by_penalty", "pass_result"):
        supp[col] = supp[col].astype("str").fillna("")

    supp["is_touchdown"] = (
        supp["play_description"].str.contains("TOUCHDOWN", case=False, na=False)
        & supp["pass_result"].eq(RESULT_COMPLETE)
    )
    supp["is_complete"] = supp["pass_result"].eq(RESULT_COMPLETE)
    supp["is_interception"] = supp["pass_result"].eq(RESULT_INTERCEPTION)
    supp["nullified"] = supp["play_nullified_by_penalty"].str.upper().eq("Y")
    supp["yards_gained"] = pd.to_numeric(supp["yards_gained"], errors="coerce")
    supp["is_man_coverage"] = supp["team_coverage_man_zone"].eq("MAN_COVERAGE")

    for col in ("route_of_targeted_receiver", "team_coverage_man_zone",
                "team_coverage_type", "offense_formation", "receiver_alignment",
                "dropback_type", "pass_location_type"):
        supp[col] = supp[col].fillna("UNKNOWN").astype("str")

    supp["play_action"] = (
        supp["play_action"].astype("str").str.upper().isin(("TRUE", "T", "1", "Y"))
    )
    return supp


def analysis_plays(supp: pd.DataFrame) -> pd.DataFrame:
    """Plays fit to score: a real pass attempt, not wiped out by a flag.

    A nullified play never counted, so letting it into a receiver's target
    total or a defender's passer rating would charge somebody for a snap the
    league itself threw away.
    """
    keep = supp["pass_result"].isin(PASS_RESULTS) & ~supp["nullified"]
    return supp.loc[keep].copy()
