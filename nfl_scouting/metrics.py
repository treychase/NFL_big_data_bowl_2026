"""The scouting tables: who gets open, who gives up throws, and against what.

Everything here reduces the one-row-per-target table from `features` down to
a player or a split. Two rules run through all of it.

Volume gates. A rate over four targets is not a skill measurement, so every
table carries its denominator and the pipeline filters on it rather than
silently publishing a 100% catch rate built on one snap.

Charging the right man. Coverage numbers are charged to the defender nearest
the receiver at the release, which on a bracket names one of two defenders
and on a blown zone exchange may name the wrong one. It is a proxy, and the
column is called `targets_covered`, not `targets`, to keep that visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    MIN_PLAYS_FOR_SPLIT,
    MIN_TARGETS_FOR_DEFENDER,
    MIN_TARGETS_FOR_RECEIVER,
    OPEN_SEPARATION_YARDS,
)

# The passer rating components are each capped at 2.375 and floored at zero,
# which is what bounds the whole thing. The ceiling is usually quoted as
# 158.3, but four maxed components give 2.375 * 4 / 6 * 100 = 158.33...,
# so the exact value is computed rather than typed - a bound written a
# third of a point low is a bound that a legitimate line trips over.
_COMPONENT_MIN = 0.0
_COMPONENT_MAX = 2.375
MAX_PASSER_RATING = _COMPONENT_MAX * 4.0 / 6.0 * 100.0


def _clamp(value):
    return np.clip(value, _COMPONENT_MIN, _COMPONENT_MAX)


def passer_rating(attempts, completions, yards, touchdowns, interceptions):
    """The NFL's passer rating, for arrays or scalars.

    Applied to the throws a defender was covering, this is "passer rating
    allowed": what the quarterback's day looked like when he threw at this
    man. Zero attempts gives NaN rather than a zero, because no throws is
    not the same as bad coverage.
    """
    attempts = np.asarray(attempts, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(attempts > 0, attempts, np.nan)
        a = _clamp((np.asarray(completions, float) / safe - 0.30) * 5.0)
        b = _clamp((np.asarray(yards, float) / safe - 3.0) * 0.25)
        c = _clamp(np.asarray(touchdowns, float) / safe * 20.0)
        d = _clamp(2.375 - np.asarray(interceptions, float) / safe * 25.0)
        rating = (a + b + c + d) / 6.0 * 100.0
    return np.where(attempts > 0, rating, np.nan)


def _passer_rating_from_group(group: pd.DataFrame) -> float:
    return float(passer_rating(
        len(group),
        group["is_complete"].sum(),
        group["yards_gained"].sum(),
        group["is_touchdown"].sum(),
        group["is_interception"].sum(),
    ))


def _open_rate(series: pd.Series) -> float:
    """Share of targets with at least the open-receiver threshold of room."""
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float((valid >= OPEN_SEPARATION_YARDS).mean())


# Ball-in-air columns summarised on every player table, so a receiver's
# burst and a defender's redirect ride along with the rate stats.
RECEIVER_AIR_COLUMNS = {
    "rec_speed_max_yps": "air_speed_max_yps",
    "rec_accel_max_yps2": "air_accel_max_yps2",
    "rec_accel_burst_yps2": "air_accel_burst_yps2",
    "rec_speed_delta_yps": "air_speed_delta_yps",
    "rec_cod_total_deg": "air_cod_total_deg",
    "rec_cod_max_rate_dps": "air_cod_max_rate_dps",
    "rec_pursuit_efficiency": "air_pursuit_efficiency",
    "rec_mean_off_bearing_deg": "air_off_bearing_deg",
    "rec_closing_speed_yps": "air_closing_speed_yps",
}

COVERAGE_AIR_COLUMNS = {
    "cov_speed_max_yps": "air_speed_max_yps",
    "cov_accel_max_yps2": "air_accel_max_yps2",
    "cov_accel_burst_yps2": "air_accel_burst_yps2",
    "cov_cod_total_deg": "air_cod_total_deg",
    "cov_cod_max_rate_dps": "air_cod_max_rate_dps",
    "cov_pursuit_efficiency": "air_pursuit_efficiency",
    "cov_mean_off_bearing_deg": "air_off_bearing_deg",
    "cov_bearing_correction_deg": "air_bearing_correction_deg",
    "cov_closing_speed_yps": "air_closing_speed_yps",
    "cov_dist_to_spot_end_yd": "air_dist_to_spot_at_arrival_yd",
}


def _air_means(group: pd.DataFrame, columns: dict[str, str]) -> dict[str, float]:
    return {out: float(group[src].mean(skipna=True))
            for src, out in columns.items() if src in group.columns}


def receiver_table(plays: pd.DataFrame,
                   min_targets: int = MIN_TARGETS_FOR_RECEIVER) -> pd.DataFrame:
    """One row per targeted receiver: separation, production, and flight profile."""
    rows = []
    for (nfl_id, name), group in plays.groupby(
            ["target_nfl_id", "target_name"], sort=False):
        row = {
            "nfl_id": int(nfl_id),
            "player_name": name,
            "position": group["target_position"].mode().iat[0],
            "targets": len(group),
            "catch_rate": float(group["is_complete"].mean()),
            "yards_per_target": float(group["yards_gained"].mean()),
            "touchdowns": int(group["is_touchdown"].sum()),
            "interceptions": int(group["is_interception"].sum()),
            "passer_rating_when_targeted": _passer_rating_from_group(group),
            "mean_air_yards": float(group["air_yards_yd"].mean()),

            "separation_at_throw_yd": float(group["separation_at_throw_yd"].mean()),
            "separation_at_arrival_yd": float(
                group["separation_at_arrival_yd"].mean(skipna=True)),
            "separation_change_yd": float(
                group["separation_change_yd"].mean(skipna=True)),
            "open_rate_at_throw": _open_rate(group["separation_at_throw_yd"]),
            "open_rate_at_arrival": _open_rate(group["separation_at_arrival_yd"]),
        }
        row.update(_air_means(group, RECEIVER_AIR_COLUMNS))
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return (table[table["targets"] >= min_targets]
            .sort_values("targets", ascending=False)
            .reset_index(drop=True))


def defender_table(plays: pd.DataFrame,
                   min_targets: int = MIN_TARGETS_FOR_DEFENDER) -> pd.DataFrame:
    """One row per coverage defender: what quarterbacks got for throwing at him."""
    covered = plays.dropna(subset=["coverage_nfl_id"])
    rows = []
    for (nfl_id, name), group in covered.groupby(
            ["coverage_nfl_id", "coverage_name"], sort=False):
        row = {
            "nfl_id": int(nfl_id),
            "player_name": name,
            "position": group["coverage_position"].mode().iat[0],
            "targets_covered": len(group),
            "completion_rate_allowed": float(group["is_complete"].mean()),
            "yards_per_target_allowed": float(group["yards_gained"].mean()),
            "touchdowns_allowed": int(group["is_touchdown"].sum()),
            "interceptions": int(group["is_interception"].sum()),
            "passer_rating_allowed": _passer_rating_from_group(group),

            "separation_allowed_at_throw_yd": float(
                group["separation_at_throw_yd"].mean()),
            "separation_allowed_at_arrival_yd": float(
                group["separation_at_arrival_yd"].mean(skipna=True)),
            "open_rate_allowed_at_throw": _open_rate(group["separation_at_throw_yd"]),
            "open_rate_allowed_at_arrival": _open_rate(
                group["separation_at_arrival_yd"]),
            "mean_air_yards_allowed": float(group["air_yards_yd"].mean()),
        }
        row.update(_air_means(group, COVERAGE_AIR_COLUMNS))
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return (table[table["targets_covered"] >= min_targets]
            .sort_values("passer_rating_allowed")
            .reset_index(drop=True))


def split_table(plays: pd.DataFrame, by: list[str],
                min_plays: int = MIN_PLAYS_FOR_SPLIT) -> pd.DataFrame:
    """Target outcomes cut by any combination of play-context columns.

    The route-by-coverage table the scouting app leans on is this with
    `by=["route_of_targeted_receiver", "team_coverage_man_zone"]`.
    """
    rows = []
    for keys, group in plays.groupby(by, sort=False, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        row.update({
            "plays": len(group),
            "catch_rate": float(group["is_complete"].mean()),
            "yards_per_target": float(group["yards_gained"].mean()),
            "passer_rating": _passer_rating_from_group(group),
            "interception_rate": float(group["is_interception"].mean()),
            "mean_air_yards": float(group["air_yards_yd"].mean()),
            "separation_at_throw_yd": float(group["separation_at_throw_yd"].mean()),
            "separation_at_arrival_yd": float(
                group["separation_at_arrival_yd"].mean(skipna=True)),
            "separation_change_yd": float(group["separation_change_yd"].mean(skipna=True)),
            "open_rate_at_throw": _open_rate(group["separation_at_throw_yd"]),
            "rec_cod_total_deg": float(group["rec_cod_total_deg"].mean(skipna=True)),
            "rec_accel_burst_yps2": float(group["rec_accel_burst_yps2"].mean(skipna=True)),
            "cov_cod_total_deg": float(group["cov_cod_total_deg"].mean(skipna=True)),
            "cov_pursuit_efficiency": float(
                group["cov_pursuit_efficiency"].mean(skipna=True)),
        })
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return (table[table["plays"] >= min_plays]
            .sort_values("plays", ascending=False)
            .reset_index(drop=True))


def receiver_route_table(plays: pd.DataFrame, min_plays: int = 5) -> pd.DataFrame:
    """Per receiver, per route, against man and against zone.

    This is the table behind "he wins on slants against man but the out route
    disappears against zone", and it is also the one most in need of its
    denominator read alongside it.
    """
    return split_table(
        plays,
        by=["target_nfl_id", "target_name", "route_of_targeted_receiver",
            "team_coverage_man_zone"],
        min_plays=min_plays,
    )
