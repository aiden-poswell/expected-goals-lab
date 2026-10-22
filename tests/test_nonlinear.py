import dataclasses
import inspect

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from expected_goals_lab import evaluate_nonlinear, nonlinear
from expected_goals_lab.candidate_data import BODY_PART_VOCABULARY, load_body_parts
from expected_goals_lab.dataset import load_partitions, read_identity
from expected_goals_lab.nonlinear import (
    FACTORIES,
    NONLINEAR_LOCATION,
    NONLINEAR_LOCATION_BODY_PART,
    SPLINE_PARAMS,
    check_probabilities,
    design_matrix,
    fit_candidates,
)
from expected_goals_lab.splits import assign_date_splits, split_of_events
from tests.s04_helpers import write_s04_dataset


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    csv_path, manifest_path, _ = write_s04_dataset(tmp_path_factory.mktemp("d"))
    identity = read_identity(csv_path, manifest_path)
    plan = assign_date_splits(identity.dates.tolist())
    mapping = split_of_events(identity, plan)
    return load_body_parts(csv_path, load_partitions(csv_path, mapping), mapping)


def test_candidate_designs_are_exactly_the_protocol(parts):
    a = FACTORIES[NONLINEAR_LOCATION]()
    b = FACTORIES[NONLINEAR_LOCATION_BODY_PART]()
    for pipeline in (a, b):
        features: ColumnTransformer = pipeline.named_steps["features"]
        by_name = {name: transformer for name, transformer, _ in features.transformers}
        for block in ("distance", "angle"):
            spline = by_name[block].named_steps["spline"]
            assert isinstance(spline, SplineTransformer)
            got = {k: v for k, v in spline.get_params().items() if k in SPLINE_PARAMS}
            assert got == {
                "n_knots": 5,
                "degree": 3,
                "knots": "quantile",
                "extrapolation": "constant",
                "include_bias": False,
            }
            assert isinstance(by_name[block].named_steps["scale"], StandardScaler)
        logit = pipeline.named_steps["logit"]
        assert isinstance(logit, LogisticRegression)
        params = logit.get_params()
        assert (params["C"], params["solver"], params["max_iter"]) == (1.0, "lbfgs", 1000)
    assert [n for n, _, _ in a.named_steps["features"].transformers] == ["distance", "angle"]
    body = {n: t for n, t, _ in b.named_steps["features"].transformers}["body_part"]
    assert isinstance(body, OneHotEncoder) and body.handle_unknown == "ignore"
    assert body.sparse_output is False


def test_fitted_widths_are_six_columns_per_spline_block_plus_three_classes(parts):
    fitted = fit_candidates(parts["train"])
    a_width = (
        fitted[NONLINEAR_LOCATION]
        .pipeline.named_steps["features"]
        .transform(design_matrix(parts["validation"], False))
        .shape[1]
    )
    b_width = (
        fitted[NONLINEAR_LOCATION_BODY_PART]
        .pipeline.named_steps["features"]
        .transform(design_matrix(parts["validation"], True))
        .shape[1]
    )
    assert (a_width, b_width) == (12, 15)


def test_no_search_machinery_is_used():
    for module in (nonlinear, evaluate_nonlinear):
        source = inspect.getsource(module)
        for banned in (
            "GridSearchCV",
            "RandomizedSearchCV",
            "cross_val",
            "HalvingGrid",
            "LogisticRegressionCV",
        ):
            assert banned not in source, f"{banned} in {module.__name__}"


class Recorder:
    """Shared log of every fit call made by the spy components."""

    def __init__(self):
        self.fits: list[tuple[str, int, np.ndarray]] = []


def spy_factories(recorder: Recorder):
    class SpySpline(SplineTransformer):
        def fit(self, X, y=None, sample_weight=None):
            recorder.fits.append(("spline", len(X), np.array(X, copy=True)))
            return super().fit(X, y, sample_weight)

    class SpyScaler(StandardScaler):
        def fit(self, X, y=None, sample_weight=None):
            recorder.fits.append(("scaler", len(X), np.array(X, copy=True)))
            return super().fit(X, y, sample_weight)

    class SpyOneHot(OneHotEncoder):
        def fit(self, X, y=None):
            recorder.fits.append(("onehot", len(X), np.array(X, copy=True)))
            return super().fit(X, y)

    class SpyLogit(LogisticRegression):
        def fit(self, X, y, sample_weight=None):
            recorder.fits.append(("logit", len(X), np.array(y, copy=True)))
            return super().fit(X, y, sample_weight)

    def block():
        spline = SpySpline(
            n_knots=5, degree=3, knots="quantile", extrapolation="constant", include_bias=False
        )
        return Pipeline([("spline", spline), ("scale", SpyScaler())])

    def location():
        return [("distance", block(), [0]), ("angle", block(), [1])]

    def make_a():
        return Pipeline(
            [
                ("features", ColumnTransformer(location())),
                ("logit", SpyLogit(C=1.0, solver="lbfgs", max_iter=1000)),
            ]
        )

    def make_b():
        body = SpyOneHot(handle_unknown="ignore", sparse_output=False)
        return Pipeline(
            [
                ("features", ColumnTransformer([*location(), ("body_part", body, [2])])),
                ("logit", SpyLogit(C=1.0, solver="lbfgs", max_iter=1000)),
            ]
        )

    return {NONLINEAR_LOCATION: make_a, NONLINEAR_LOCATION_BODY_PART: make_b}


def test_splines_scalers_onehot_and_model_fit_on_training_rows_only(parts):
    train, validation = parts["train"], parts["validation"]
    recorder = Recorder()

    fitted = fit_candidates(train, spy_factories(recorder))

    n = len(train)
    assert recorder.fits and all(rows == n for _, rows, _ in recorder.fits)
    kinds = [k for k, _, _ in recorder.fits]
    # A: 2 splines + 2 scalers + 1 logit; B: 2 splines + 2 scalers + 1 one-hot + 1 logit
    assert sorted(kinds) == sorted(["spline"] * 4 + ["scaler"] * 4 + ["onehot"] + ["logit"] * 2)
    onehot_input = next(x for k, _, x in recorder.fits if k == "onehot")
    assert onehot_input[:, 0].tolist() == train.body_part.tolist()  # train body parts, nothing else
    distance_inputs = [x for k, _, x in recorder.fits if k == "spline"][0]
    assert np.array_equal(distance_inputs[:, 0], train.base.features["distance_m"])
    labels = [y for k, _, y in recorder.fits if k == "logit"]
    assert all(np.array_equal(y, train.base.y) for y in labels)

    before = len(recorder.fits)
    for model in fitted.values():
        model.predict_proba(validation)  # scoring validation may only transform / predict
    assert len(recorder.fits) == before


def test_a_validation_only_class_or_range_cannot_leak_into_the_fit(parts):
    train = parts["train"]
    pipeline = fit_candidates(train)[NONLINEAR_LOCATION_BODY_PART].pipeline
    encoder = pipeline.named_steps["features"].named_transformers_["body_part"]
    assert sorted(encoder.categories_[0].tolist()) == sorted(BODY_PART_VOCABULARY)
    distance_scaler = (
        pipeline.named_steps["features"].named_transformers_["distance"].named_steps["scale"]
    )
    assert distance_scaler.n_samples_seen_ == len(train)


def test_training_must_contain_every_body_part_class_and_both_outcomes(parts):
    train = parts["train"]
    keep = train.body_part != "head_or_body"
    missing_class = dataclasses.replace(train, body_part=train.body_part.copy())
    missing_class.body_part[~keep] = "right_foot"  # no head_or_body left in train
    with pytest.raises(ValueError, match="differ from the expected"):
        fit_candidates(missing_class)

    one_class = dataclasses.replace(
        train, base=dataclasses.replace(train.base, y=np.zeros_like(train.base.y))
    )
    with pytest.raises(ValueError, match="both outcome classes"):
        fit_candidates(one_class)


def test_only_the_train_split_can_be_fitted(parts):
    with pytest.raises(ValueError, match="train split only"):
        fit_candidates(parts["validation"])


def test_non_convergence_is_a_hard_error(parts):
    def impatient():
        return Pipeline(
            [
                ("features", ColumnTransformer([("d", StandardScaler(), [0])])),
                ("logit", LogisticRegression(max_iter=1)),
            ]
        )

    with pytest.raises(ConvergenceWarning):
        fit_candidates(parts["train"], {NONLINEAR_LOCATION: impatient})


@pytest.mark.parametrize(
    "bad",
    [
        np.array([0.2, np.nan]),
        np.array([0.2, 1.0001]),
        np.array([-0.1, 0.5]),
        np.array([[0.1, 0.2]]),
    ],
)
def test_invalid_probabilities_are_rejected(bad):
    with pytest.raises(ValueError, match="finite and within"):
        check_probabilities(bad, "m")


def test_valid_probabilities_include_the_closed_endpoints():
    assert check_probabilities(np.array([0.0, 0.5, 1.0]), "m").tolist() == [0.0, 0.5, 1.0]


def test_candidate_predictions_are_finite_probabilities(parts):
    fitted = fit_candidates(parts["train"])
    for part in (parts["train"], parts["validation"]):
        for model in fitted.values():
            p = model.predict_proba(part)
            assert p.shape == (len(part),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_design_matrix_uses_only_allowlisted_continuous_features(parts):
    x = design_matrix(parts["train"], with_body_part=False)
    assert x.shape == (len(parts["train"]), 2)
    x3 = design_matrix(parts["train"], with_body_part=True)
    assert x3.shape == (len(parts["train"]), 3) and set(x3[:, 2]) <= set(BODY_PART_VOCABULARY)
