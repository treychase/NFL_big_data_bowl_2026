"""Catch probability: how likely a throw was to be completed, at release.

What the model is allowed to see is the whole design question here. The
tracking data makes it trivially easy to build a model that looks
spectacular and says nothing - put the receiver's distance to the landing
spot at the moment the ball arrives into the features and the thing will hit
0.95 AUC by learning that a receiver standing on the ball caught it.

So the cut is the release. The model sees where everyone was and how they
were moving when the ball left the quarterback's hand, plus where the ball
was going and how long it would be in the air, and nothing that happened
afterwards. `BALL_IN_AIR_COLUMNS` names the flight-derived columns
explicitly and `assert_no_leakage` fails the build if one reaches the
feature list, because this is the kind of mistake that is invisible in the
metrics - it just makes them better.

Ball destination and flight time are known at release to the analyst but not
to the players, which makes this a completion probability conditional on the
throw - the same framing as the league's own - rather than a model of the
quarterback's decision.

Validation is grouped by game. Plays from one game share a quarterback, a
secondary, a stadium and a weather, so a random split would let the model
recognise the game rather than the throw and would flatter every number in
the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from .config import N_CV_FOLDS, RANDOM_SEED

TARGET_COLUMN = "is_complete"
GROUP_COLUMN = "game_id"

# Geometry and motion at the moment the ball is released.
NUMERIC_FEATURES = [
    "air_yards_yd",
    "air_time_s",
    "throw_distance_yd",
    "target_depth_yd",
    "land_sideline_dist_yd",
    "target_sideline_dist_yd",
    "separation_at_throw_yd",
    "target_speed_at_throw_yps",
    "target_accel_at_throw_yps2",
    "target_dist_to_spot_at_throw_yd",
    "coverage_dist_to_spot_at_throw_yd",
    "coverage_speed_at_throw_yps",
    "coverage_position_advantage_yd",
    "coverage_off_bearing_at_throw_deg",
    "nearest_defender_to_spot_yd",
    "defenders_near_spot",
    "n_route_runners",
    "n_coverage_defenders",
    "target_height_in",
    "target_weight_lb",
    "down",
    "yards_to_go",
]

CATEGORICAL_FEATURES = [
    "route_of_targeted_receiver",
    "team_coverage_man_zone",
    "team_coverage_type",
    "offense_formation",
    "receiver_alignment",
    "pass_location_type",
    "target_position",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Anything derived from the ball-in-air frames. None of it may be a feature.
BALL_IN_AIR_PREFIXES = ("rec_", "cov_")
BALL_IN_AIR_COLUMNS = frozenset({
    "separation_at_arrival_yd",
    "separation_change_yd",
    "target_dist_to_spot_at_arrival_yd",
    "arrival_defender_nfl_id",
    "coverage_tracked_in_air",
    "yards_gained",
    "is_touchdown",
    "is_interception",
    "pass_result",
    "expected_points_added",
    "pre_penalty_yards_gained",
})


class LeakageError(AssertionError):
    """Raised when a feature describes something that happened after the throw."""


def assert_no_leakage(features=FEATURES) -> None:
    """Fail loudly if a post-release column has crept into the feature list."""
    offenders = sorted(
        f for f in features
        if f in BALL_IN_AIR_COLUMNS or f.startswith(BALL_IN_AIR_PREFIXES)
    )
    if offenders:
        raise LeakageError(
            "these features describe what happened after the ball was thrown "
            f"and cannot be used to predict it: {offenders}"
        )


def build_estimator(random_state: int = RANDOM_SEED) -> Pipeline:
    """Gradient boosted trees over the release-time features.

    Trees are the right shape for this: completion probability turns sharply
    on separation and air yards and the interaction between them, and a
    boosted tree finds that without being told. Categories go in as ordered
    codes because the trees split on them directly rather than reading the
    order as a magnitude.
    """
    assert_no_leakage()
    return Pipeline([
        ("encode", ColumnTransformer(
            transformers=[
                ("numeric", "passthrough", NUMERIC_FEATURES),
                ("categorical", OneHotEncoder(handle_unknown="ignore",
                                              sparse_output=False,
                                              min_frequency=25),
                 CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )),
        ("model", HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=random_state,
        )),
    ])


def _baselines(random_state: int = RANDOM_SEED) -> dict[str, Any]:
    """What the boosted model has to beat to have earned its complexity."""
    return {
        "base_rate": DummyClassifier(strategy="prior"),
        "logistic": Pipeline([
            ("encode", ColumnTransformer([
                ("numeric", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), NUMERIC_FEATURES),
                ("categorical", OneHotEncoder(handle_unknown="ignore",
                                              sparse_output=False,
                                              min_frequency=25),
                 CATEGORICAL_FEATURES),
            ], remainder="drop")),
            ("model", LogisticRegression(max_iter=2000, C=1.0,
                                         random_state=random_state)),
        ]),
    }


def score_predictions(y_true, y_prob) -> dict[str, float]:
    """The four numbers that matter for a probability, not a label.

    Accuracy is reported last and trusted least: at a 69% completion rate a
    model that says "complete" every time already scores 0.69.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "accuracy": float(((y_prob >= 0.5).astype(int) == y_true).mean()),
    }


@dataclass
class CrossValidationResult:
    """Out-of-fold predictions and the scores computed from them."""

    predictions: pd.Series
    scores: dict[str, float]
    fold_scores: pd.DataFrame
    baseline_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    calibration: pd.DataFrame | None = None
    n_plays: int = 0
    n_folds: int = 0
    base_rate: float = float("nan")


def cross_validate(plays: pd.DataFrame, n_folds: int = N_CV_FOLDS,
                   random_state: int = RANDOM_SEED,
                   with_baselines: bool = True) -> CrossValidationResult:
    """Out-of-fold catch probability for every play, grouped by game.

    Every play gets a prediction from a model that never saw its game, so
    the predictions are safe to subtract from outcomes downstream - which is
    the whole point of producing them, since catch rate over expected is only
    meaningful if the expectation was formed without the answer.
    """
    assert_no_leakage()
    frame = plays.dropna(subset=[TARGET_COLUMN, GROUP_COLUMN]).copy()
    X = frame[FEATURES]
    y = frame[TARGET_COLUMN].astype(int).to_numpy()
    groups = frame[GROUP_COLUMN].to_numpy()

    n_folds = min(n_folds, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_folds)
    oof = np.full(len(frame), np.nan)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        estimator = build_estimator(random_state)
        estimator.fit(X.iloc[train_idx], y[train_idx])
        probability = estimator.predict_proba(X.iloc[test_idx])[:, 1]
        oof[test_idx] = probability
        fold_rows.append({"fold": fold, "n_test": len(test_idx),
                          **score_predictions(y[test_idx], probability)})

    predictions = pd.Series(oof, index=frame.index, name="catch_probability")
    result = CrossValidationResult(
        predictions=predictions,
        scores=score_predictions(y, oof),
        fold_scores=pd.DataFrame(fold_rows),
        n_plays=len(frame),
        n_folds=n_folds,
        base_rate=float(y.mean()),
    )

    fraction, mean_predicted = calibration_curve(y, oof, n_bins=10, strategy="quantile")
    result.calibration = pd.DataFrame(
        {"mean_predicted": mean_predicted, "observed": fraction})

    if with_baselines:
        for name, estimator in _baselines(random_state).items():
            baseline_oof = np.full(len(frame), np.nan)
            for train_idx, test_idx in splitter.split(X, y, groups):
                clone = _baselines(random_state)[name]
                clone.fit(X.iloc[train_idx], y[train_idx])
                baseline_oof[test_idx] = clone.predict_proba(X.iloc[test_idx])[:, 1]
            result.baseline_scores[name] = score_predictions(y, baseline_oof)

    return result


def fit_full(plays: pd.DataFrame, random_state: int = RANDOM_SEED) -> Pipeline:
    """Refit on every play, for scoring throws the cross-validation never saw."""
    assert_no_leakage()
    frame = plays.dropna(subset=[TARGET_COLUMN])
    estimator = build_estimator(random_state)
    estimator.fit(frame[FEATURES], frame[TARGET_COLUMN].astype(int))
    return estimator


def feature_importance(plays: pd.DataFrame, estimator: Pipeline,
                       n_repeats: int = 5,
                       random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """Permutation importance, measured as damage done to log loss.

    Permutation rather than split counts, because split counts reward a
    continuous feature for having many possible thresholds rather than for
    carrying information.
    """
    frame = plays.dropna(subset=[TARGET_COLUMN])
    result = permutation_importance(
        estimator, frame[FEATURES], frame[TARGET_COLUMN].astype(int),
        scoring="neg_log_loss", n_repeats=n_repeats,
        random_state=random_state, n_jobs=1,
    )
    return (pd.DataFrame({
        "feature": FEATURES,
        "importance": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True))


def add_catch_probability(plays: pd.DataFrame,
                          predictions: pd.Series) -> pd.DataFrame:
    """Attach catch probability and the over-expected residual to the play table.

    `catch_over_expected` is the residual a scout actually wants: positive
    means the receiver caught balls the geometry said were hard.
    """
    out = plays.copy()
    out["catch_probability"] = predictions
    out["catch_over_expected"] = (
        out[TARGET_COLUMN].astype(float) - out["catch_probability"])
    return out
