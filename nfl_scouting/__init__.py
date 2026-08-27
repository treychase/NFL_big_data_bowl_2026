"""Scouting tools for the NFL Big Data Bowl 2026 tracking data.

Separation and coverage from the release frame, receiver and defender
kinematics while the ball is in the air, and a catch probability model that
is not allowed to see the flight it is predicting.
"""

from . import config, data, export, features, geometry, kinematics, metrics, model, pipeline

__all__ = [
    "config", "data", "export", "features", "geometry",
    "kinematics", "metrics", "model", "pipeline",
]

__version__ = "1.0.0"
