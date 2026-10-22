"""The two S04 candidates, fixed in docs/NONLINEAR_PROTOCOL.md. No parameter search.

* ``nonlinear_location``            cubic spline bases of ``distance_m`` and ``angle_rad``
                                    (separately), standardised, then logistic regression.
* ``nonlinear_location_body_part``  the same spline blocks plus a one-hot ``body_part_class``.

Every component (spline knots, scalers, one-hot categories, logistic coefficients) is learned in
``fit`` on the training rows passed in. Validation is scored with ``predict_proba`` only.
"""

import warnings
from collections.abc import Callable

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from expected_goals_lab.baselines import LOGISTIC_PARAMS
from expected_goals_lab.candidate_data import BODY_PART_VOCABULARY, CandidatePartition
from expected_goals_lab.dataset import feature_matrix

NONLINEAR_LOCATION = "nonlinear_location"
NONLINEAR_LOCATION_BODY_PART = "nonlinear_location_body_part"
CONTINUOUS_FEATURES = ("distance_m", "angle_rad")
SPLINE_PARAMS = {
    "n_knots": 5,
    "degree": 3,
    "knots": "quantile",
    "extrapolation": "constant",
    "include_bias": False,
}
# Column positions in the design matrix: distance, angle, then (candidate B only) body part.
DISTANCE_COLUMN, ANGLE_COLUMN, BODY_PART_COLUMN_INDEX = 0, 1, 2


def _spline_block() -> Pipeline:
    return Pipeline(
        [
            ("spline", SplineTransformer(**SPLINE_PARAMS)),
            ("scale", StandardScaler()),
        ]
    )


def _location_transformers() -> list[tuple]:
    return [
        ("distance", _spline_block(), [DISTANCE_COLUMN]),
        ("angle", _spline_block(), [ANGLE_COLUMN]),
    ]


def make_nonlinear_location() -> Pipeline:
    return Pipeline(
        [
            ("features", ColumnTransformer(_location_transformers())),
            ("logit", LogisticRegression(**LOGISTIC_PARAMS)),
        ]
    )


def make_nonlinear_location_body_part() -> Pipeline:
    body = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [*_location_transformers(), ("body_part", body, [BODY_PART_COLUMN_INDEX])]
                ),
            ),
            ("logit", LogisticRegression(**LOGISTIC_PARAMS)),
        ]
    )


FACTORIES: dict[str, Callable[[], Pipeline]] = {
    NONLINEAR_LOCATION: make_nonlinear_location,
    NONLINEAR_LOCATION_BODY_PART: make_nonlinear_location_body_part,
}
USES_BODY_PART = {NONLINEAR_LOCATION: False, NONLINEAR_LOCATION_BODY_PART: True}


def design_matrix(part: CandidatePartition, with_body_part: bool) -> np.ndarray:
    """Object array: distance, angle (via S03's allowlisted accessor) and optionally body part."""
    continuous = feature_matrix(part.base, CONTINUOUS_FEATURES)
    columns = [continuous[:, 0], continuous[:, 1]]
    if with_body_part:
        columns.append(part.body_part)
    matrix = np.empty((len(part), len(columns)), dtype=object)
    for i, column in enumerate(columns):
        matrix[:, i] = column
    return matrix


def check_probabilities(p: np.ndarray, name: str) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError(f"{name}: predicted probabilities must be finite and within [0, 1]")
    return p


class FittedCandidate:
    def __init__(self, name: str, pipeline: Pipeline):
        self.name = name
        self.pipeline = pipeline

    def predict_proba(self, part: CandidatePartition) -> np.ndarray:
        x = design_matrix(part, USES_BODY_PART[self.name])
        return check_probabilities(self.pipeline.predict_proba(x)[:, 1], self.name)


def fit_candidates(
    train: CandidatePartition,
    factories: dict[str, Callable[[], Pipeline]] | None = None,
    fit_split: str = "train",
) -> dict[str, FittedCandidate]:
    """Fit the candidates on the training partition. Refuses any other split.

    ``fit_split`` is "train" in S04-S05. The S07 final evaluation passes "development" for the
    combined train + validation partition; the test split is never accepted.
    """
    if fit_split == "test" or train.split != fit_split:
        raise ValueError(
            f"candidates are fitted on the {fit_split} split only, not {train.split!r}"
        )
    if len(set(train.base.y.tolist())) < 2:
        raise ValueError("the training cohort must contain both outcome classes")
    present = set(train.body_part.tolist())
    if present != set(BODY_PART_VOCABULARY):
        raise ValueError(
            f"training body-part classes {sorted(present)} differ from the expected "
            f"{sorted(BODY_PART_VOCABULARY)}"
        )
    factories = factories or FACTORIES
    fitted = {}
    for name, factory in factories.items():
        pipeline = factory()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit(design_matrix(train, USES_BODY_PART[name]), train.base.y)
        fitted[name] = FittedCandidate(name, pipeline)
    return fitted
