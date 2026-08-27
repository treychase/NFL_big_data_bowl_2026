"""Paths, field geometry and the thresholds every module shares.

Every constant that encodes a judgement call - what counts as "open", how
slow a player has to be moving before his heading stops meaning anything -
lives here rather than being spelled inline, so the assumptions behind a
number on the dashboard can be found in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- repository layout -------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "data"
    / "nfl-big-data-bowl-2026"
    / "114239_nfl_competition_files_published_analytics_final"
)
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts"


def data_dir() -> Path:
    """The competition data directory, overridable for tests and CI."""
    return Path(os.environ.get("BDB2026_DATA_DIR", DEFAULT_DATA_DIR))


def artifact_dir() -> Path:
    return Path(os.environ.get("BDB2026_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR))


# --- tracking and field ------------------------------------------------

FRAME_RATE_HZ = 10.0
DT = 1.0 / FRAME_RATE_HZ

FIELD_LENGTH = 120.0   # yards, goal line to goal line plus both end zones
FIELD_WIDTH = 53.3     # yards, sideline to sideline
END_ZONE_DEPTH = 10.0

SEASON = 2023
WEEKS = tuple(range(1, 19))

# --- roles, as spelled in the tracking file ----------------------------

ROLE_TARGET = "Targeted Receiver"
ROLE_COVERAGE = "Defensive Coverage"
ROLE_PASSER = "Passer"
ROLE_OTHER_ROUTE = "Other Route Runner"

# --- analysis thresholds -----------------------------------------------

# Next Gen Stats calls a receiver "open" at three yards of separation from
# the nearest defender. Keeping the same cut makes these numbers comparable
# to the published ones.
OPEN_SEPARATION_YARDS = 3.0

# A heading derived from two positions a tenth of a second apart is noise
# when a player is barely moving: a 0.05 yd wobble becomes a 45 degree turn.
# Below this speed the frame is dropped from change-of-direction sums.
MIN_SPEED_FOR_HEADING = 1.0   # yards/second

# The fastest NFL players top out a shade over 10 yards a second. A landing
# spot further from the targeted receiver than this speed could cover in the
# ball's flight, plus a yard of slack, is not a spot he was ever going to
# reach - which means the recorded spot is not describing this target, and
# the play is dropped rather than scored as an impossible one.
MAX_RECEIVER_SPEED = 11.0     # yards/second
REACHABILITY_SLACK = 1.0      # yards

# Below this many targets a rate is a coin flip dressed up as a ranking.
MIN_TARGETS_FOR_RECEIVER = 20
MIN_TARGETS_FOR_DEFENDER = 15
MIN_PLAYS_FOR_SPLIT = 10

# Positions treated as coverage defensive backs for the passer-rating table.
COVERAGE_DB_POSITIONS = ("CB", "FS", "SS", "DB", "NB", "S")

# --- pass results ------------------------------------------------------

RESULT_COMPLETE = "C"
RESULT_INCOMPLETE = "I"
RESULT_INTERCEPTION = "IN"
PASS_RESULTS = (RESULT_COMPLETE, RESULT_INCOMPLETE, RESULT_INTERCEPTION)

# --- modelling ---------------------------------------------------------

RANDOM_SEED = 26
N_CV_FOLDS = 5
