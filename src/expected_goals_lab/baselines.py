"""The three pre-registered baselines. Fitting sees training rows only.

* ``constant_train_rate``  the goal fraction of the training shots, for every shot.
* ``distance_only``        logistic regression on ``distance_m``.
* ``distance_plus_angle``  logistic regression on ``distance_m`` and ``angle_rad``.

Logistic models are ``Pipeline(StandardScaler, LogisticRegression(C=1.0, solver="lbfgs",
max_iter=1000))``: the library's default L2 penalty, fixed in advance and not tuned. The scaler is
inside the pipeline, so its mean and scale come from the rows passed to ``fit`` and nothing else.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from expected_goals_lab.dataset import Partition, feature_matrix
from expected_goals_lab.geometry import Pitch, shot_geometry

CONSTANT, DISTANCE_ONLY, DISTANCE_PLUS_ANGLE = (
    "constant_train_rate",
    "distance_only",
    "distance_plus_angle",
)
MODEL_FEATURES = {
    CONSTANT: (),
    DISTANCE_ONLY: ("distance_m",),
    DISTANCE_PLUS_ANGLE: ("distance_m", "angle_rad"),
}
LOGISTIC_PARAMS = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(**LOGISTIC_PARAMS)),
        ]
    )


@dataclass
class ConstantRate:
    rate: float = float("nan")

    def fit(self, y: np.ndarray) -> "ConstantRate":
        self.rate = float(np.mean(y))
        return self


@dataclass
class FittedModel:
    name: str
    features: tuple[str, ...]
    estimator: object  # ConstantRate or a fitted Pipeline

    def predict_proba(self, partition: Partition) -> np.ndarray:
        if not self.features:
            return np.full(len(partition), self.estimator.rate)
        return self.estimator.predict_proba(feature_matrix(partition, self.features))[:, 1]

    def coefficients(self) -> dict | None:
        """Standardised and original-unit coefficients of a logistic model."""
        if not self.features:
            return None
        scaler, logit = self.estimator.named_steps["scale"], self.estimator.named_steps["logit"]
        standardised = logit.coef_[0]
        original = standardised / scaler.scale_
        intercept = float(logit.intercept_[0] - np.sum(standardised * scaler.mean_ / scaler.scale_))
        return {
            "features": self.features,
            "standardised": dict(zip(self.features, map(float, standardised), strict=True)),
            "per_unit": dict(zip(self.features, map(float, original), strict=True)),
            "intercept_standardised": float(logit.intercept_[0]),
            "intercept_per_unit": intercept,
            "train_mean": dict(zip(self.features, map(float, scaler.mean_), strict=True)),
            "train_scale": dict(zip(self.features, map(float, scaler.scale_), strict=True)),
        }


def fit_baselines(
    train: Partition,
    pipeline_factory: Callable[[], Pipeline] = make_pipeline,
    fit_split: str = "train",
) -> dict[str, FittedModel]:
    """Fit all three candidates on the training partition. Refuses any other split.

    ``fit_split`` is "train" in S03-S05. The S07 final evaluation passes "development" for the
    combined train + validation partition; the test split is never accepted.
    """
    if fit_split == "test" or train.split != fit_split:
        raise ValueError(f"baselines are fitted on the {fit_split} split only, not {train.split!r}")
    models = {}
    for name, features in MODEL_FEATURES.items():
        if not features:
            models[name] = FittedModel(name, features, ConstantRate().fit(train.y))
            continue
        pipeline = pipeline_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit(feature_matrix(train, features), train.y)
        models[name] = FittedModel(name, features, pipeline)
    return models


def with_pitch(partition: Partition, pitch: Pitch) -> Partition:
    """The same shots with distance and angle recomputed from raw percentages on another pitch."""
    geometry = [
        shot_geometry(int(x), int(y), pitch)
        for x, y in zip(partition.x_pct, partition.y_pct, strict=True)
    ]
    return replace(
        partition,
        features={
            "distance_m": np.array([g.distance_m for g in geometry]),
            "angle_rad": np.array([g.angle_rad for g in geometry]),
        },
    )
