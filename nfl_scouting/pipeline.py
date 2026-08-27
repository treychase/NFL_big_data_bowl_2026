"""Build every artifact the scouting app reads, from the raw files up.

One entry point, `run`. It extracts per-target features week by week, joins
the play context, cross-validates the catch probability model, aggregates
the scouting tables, and writes the lot to the artifact directory. Feature
extraction is the slow part, so its output is cached to Parquet and reused
unless `refresh` says otherwise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import data, features, metrics, model
from .config import (
    COVERAGE_DB_POSITIONS,
    MIN_TARGETS_FOR_DEFENDER,
    MIN_TARGETS_FOR_RECEIVER,
    N_CV_FOLDS,
    RANDOM_SEED,
    SEASON,
    artifact_dir,
)

log = logging.getLogger("nfl_scouting")

PLAY_TABLE_NAME = "play_features.parquet"

# Play context carried onto every target row.
CONTEXT_COLUMNS = [
    "game_id", "play_id", "season", "week", "game_date", "quarter", "game_clock",
    "down", "yards_to_go", "possession_team", "defensive_team",
    "play_description", "pass_result", "yards_gained", "is_complete",
    "is_touchdown", "is_interception", "is_man_coverage",
    "route_of_targeted_receiver", "team_coverage_man_zone", "team_coverage_type",
    "offense_formation", "receiver_alignment", "pass_location_type",
    "dropback_type", "play_action", "defenders_in_the_box",
    "expected_points_added", "absolute_yardline_number",
]


@dataclass
class Artifacts:
    """Everything `run` produced, in memory, before it hit disk."""

    plays: pd.DataFrame
    receivers: pd.DataFrame
    defenders: pd.DataFrame
    route_coverage: pd.DataFrame
    coverage_type: pd.DataFrame
    receiver_routes: pd.DataFrame
    importance: pd.DataFrame
    cv: model.CrossValidationResult
    summary: dict


def build_play_table(weeks=None, season: int = SEASON,
                     root: Path | None = None) -> pd.DataFrame:
    """One row per pass attempt, tracking features joined to play context."""
    weeks = weeks or data.available_weeks(season, root)
    frames = []
    for week in weeks:
        log.info("extracting week %d", week)
        wk = data.load_week(week, season, root)
        frame = features.extract_week(wk.tracking, wk.ball_air)
        frame["week"] = week
        frames.append(frame)
        log.info("  %d targets", len(frame))

    plays = pd.concat(frames, ignore_index=True)
    supp = data.analysis_plays(data.load_supplementary(root))
    context = supp[[c for c in CONTEXT_COLUMNS if c in supp.columns]]

    merged = plays.merge(context, on=["game_id", "play_id"], how="inner",
                         suffixes=("", "_supp"))
    log.info("%d tracked targets, %d matched to a scorable play",
             len(plays), len(merged))

    # A landing spot the targeted receiver could not have reached is not a
    # spot he was thrown to, whatever the file says. Dropping these keeps
    # them out of the leaderboards, where a physically impossible throw
    # that happened to be completed would sit at the very top of any
    # catch-over-expected list.
    unreachable = ~merged["landing_spot_reachable"]
    if unreachable.any():
        log.info("dropping %d plays whose landing spot the targeted receiver "
                 "could not have reached", int(unreachable.sum()))
    return merged.loc[~unreachable].reset_index(drop=True)


def load_or_build_play_table(out_dir: Path, weeks=None, season: int = SEASON,
                             root: Path | None = None,
                             refresh: bool = False) -> pd.DataFrame:
    cache = out_dir / PLAY_TABLE_NAME
    if cache.exists() and not refresh:
        log.info("reusing cached play table at %s", cache)
        return pd.read_parquet(cache)
    plays = build_play_table(weeks, season, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    plays.to_parquet(cache, index=False)
    return plays


def _summary(plays: pd.DataFrame, cv: model.CrossValidationResult,
             receivers: pd.DataFrame, defenders: pd.DataFrame) -> dict:
    tracked = plays["coverage_tracked_in_air"]
    return {
        # The competition is named for 2026; the tracking in it is the 2023
        # season, and it is the season that belongs on the charts.
        "season": SEASON,
        "weeks": sorted(int(w) for w in plays["week"].unique()),
        "n_plays": int(len(plays)),
        "n_games": int(plays["game_id"].nunique()),
        "n_receivers": int(plays["target_nfl_id"].nunique()),
        "n_defenders": int(plays["coverage_nfl_id"].nunique()),
        "completion_rate": float(plays["is_complete"].mean()),
        "mean_separation_at_throw_yd": float(plays["separation_at_throw_yd"].mean()),
        "mean_separation_at_arrival_yd": float(
            plays["separation_at_arrival_yd"].mean(skipna=True)),
        "mean_separation_change_yd": float(
            plays["separation_change_yd"].mean(skipna=True)),
        "coverage_tracked_in_air_share": float(tracked.mean()),
        "qualified_receivers": int(len(receivers)),
        "qualified_defenders": int(len(defenders)),
        "min_targets_receiver": MIN_TARGETS_FOR_RECEIVER,
        "min_targets_defender": MIN_TARGETS_FOR_DEFENDER,
        "model": {
            "target": model.TARGET_COLUMN,
            "n_features": len(model.FEATURES),
            "n_folds": cv.n_folds,
            "grouped_by": model.GROUP_COLUMN,
            "base_rate": cv.base_rate,
            "scores": cv.scores,
            "baselines": cv.baseline_scores,
            "fold_scores": cv.fold_scores.to_dict("records"),
            "calibration": (cv.calibration.to_dict("records")
                            if cv.calibration is not None else []),
        },
    }


def run(out_dir: Path | None = None, weeks=None, season: int = SEASON,
        root: Path | None = None, refresh: bool = False,
        n_folds: int = N_CV_FOLDS, random_state: int = RANDOM_SEED) -> Artifacts:
    """Build the play table, fit and score the model, write every artifact."""
    out_dir = Path(out_dir or artifact_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    plays = load_or_build_play_table(out_dir, weeks, season, root, refresh)

    log.info("cross-validating catch probability over %d plays", len(plays))
    cv = model.cross_validate(plays, n_folds=n_folds, random_state=random_state)
    plays = model.add_catch_probability(plays, cv.predictions)
    log.info("out-of-fold AUC %.4f, Brier %.4f", cv.scores["roc_auc"], cv.scores["brier"])

    estimator = model.fit_full(plays, random_state)
    importance = model.feature_importance(plays, estimator, random_state=random_state)

    receivers = _with_cpoe(metrics.receiver_table(plays), plays, "target_nfl_id")
    defenders = _with_cpoe(metrics.defender_table(plays), plays, "coverage_nfl_id",
                           allowed=True)
    defenders["is_defensive_back"] = defenders["position"].isin(COVERAGE_DB_POSITIONS)

    route_coverage = metrics.split_table(
        plays, ["route_of_targeted_receiver", "team_coverage_man_zone"])
    coverage_type = metrics.split_table(
        plays, ["route_of_targeted_receiver", "team_coverage_type"])
    receiver_routes = metrics.receiver_route_table(plays)

    summary = _summary(plays, cv, receivers, defenders)

    artifacts = Artifacts(
        plays=plays, receivers=receivers, defenders=defenders,
        route_coverage=route_coverage, coverage_type=coverage_type,
        receiver_routes=receiver_routes, importance=importance,
        cv=cv, summary=summary,
    )
    _write(artifacts, out_dir)
    return artifacts


def _with_cpoe(table: pd.DataFrame, plays: pd.DataFrame, key: str,
               allowed: bool = False) -> pd.DataFrame:
    """Add catch rate over expected, from the out-of-fold predictions.

    For a defender this is signed the other way round and named `_allowed`:
    a negative number means quarterbacks completed fewer throws at him than
    the geometry of those throws deserved.
    """
    if table.empty:
        return table
    grouped = plays.groupby(key).agg(
        expected=("catch_probability", "mean"),
        actual=("is_complete", "mean"),
    )
    grouped["cpoe"] = grouped["actual"] - grouped["expected"]
    suffix = "_allowed" if allowed else ""
    merged = table.merge(
        grouped[["expected", "cpoe"]].rename(columns={
            "expected": f"expected_catch_rate{suffix}",
            "cpoe": f"catch_rate_over_expected{suffix}",
        }),
        left_on="nfl_id", right_index=True, how="left",
    )
    return merged


def _write(artifacts: Artifacts, out_dir: Path) -> None:
    tables = {
        "play_features": artifacts.plays,
        "receiver_scouting": artifacts.receivers,
        "defender_coverage": artifacts.defenders,
        "route_by_coverage": artifacts.route_coverage,
        "route_by_coverage_type": artifacts.coverage_type,
        "receiver_route_splits": artifacts.receiver_routes,
        "model_feature_importance": artifacts.importance,
        "model_fold_scores": artifacts.cv.fold_scores,
        "model_calibration": artifacts.cv.calibration,
    }
    for name, table in tables.items():
        if table is None:
            continue
        table.to_csv(out_dir / f"{name}.csv", index=False)
    artifacts.plays.to_parquet(out_dir / PLAY_TABLE_NAME, index=False)
    (out_dir / "summary.json").write_text(json.dumps(artifacts.summary, indent=2))
    log.info("wrote %d artifacts to %s", len(tables) + 2, out_dir)
