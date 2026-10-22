import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from expected_goals_lab.baselines import (
    CONSTANT,
    DISTANCE_ONLY,
    DISTANCE_PLUS_ANGLE,
    MODEL_FEATURES,
    fit_baselines,
    make_pipeline,
    with_pitch,
)
from expected_goals_lab.dataset import (
    FeatureNotAllowedError,
    feature_matrix,
    load_partitions,
    read_identity,
)
from expected_goals_lab.geometry import Pitch
from expected_goals_lab.shot_table import FEATURE_ALLOWLIST
from expected_goals_lab.splits import assign_date_splits, split_of_events
from tests.s03_helpers import write_dataset


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    csv_path, manifest_path, _ = write_dataset(tmp_path_factory.mktemp("d"))
    identity = read_identity(csv_path, manifest_path)
    plan = assign_date_splits(identity.dates.tolist())
    return load_partitions(csv_path, split_of_events(identity, plan))


def test_constant_baseline_equals_the_training_goal_rate_exactly(parts):
    train, validation = parts["train"], parts["validation"]
    models = fit_baselines(train)

    expected = int(train.y.sum()) / len(train)
    assert models[CONSTANT].estimator.rate == expected  # exact, not approximate
    predictions = models[CONSTANT].predict_proba(validation)
    assert predictions.shape == (len(validation),) and (predictions == expected).all()
    assert expected != validation.y.mean()  # the validation rate plays no part


def test_models_use_only_allowlisted_features():
    assert set().union(*MODEL_FEATURES.values()) <= set(FEATURE_ALLOWLIST)
    assert MODEL_FEATURES[DISTANCE_ONLY] == ("distance_m",)
    assert MODEL_FEATURES[DISTANCE_PLUS_ANGLE] == ("distance_m", "angle_rad")


def test_fitting_refuses_any_split_but_train(parts):
    with pytest.raises(ValueError, match="train split only"):
        fit_baselines(parts["validation"])


class SpyScaler(StandardScaler):
    seen = []

    def fit(self, X, y=None, **kwargs):
        SpyScaler.seen.append(("scaler.fit", np.array(X, copy=True)))
        return super().fit(X, y, **kwargs)


class SpyLogit(LogisticRegression):
    seen = []

    def fit(self, X, y, **kwargs):
        SpyLogit.seen.append(("logit.fit", np.array(X, copy=True), np.array(y, copy=True)))
        return super().fit(X, y, **kwargs)


def spy_pipeline():
    return Pipeline([("scale", SpyScaler()), ("logit", SpyLogit(C=1.0, max_iter=1000))])


def test_scaler_and_logistic_model_see_only_training_rows(parts):
    train, validation = parts["train"], parts["validation"]
    SpyScaler.seen, SpyLogit.seen = [], []

    models = fit_baselines(train, pipeline_factory=spy_pipeline)

    assert len(SpyScaler.seen) == 2 and len(SpyLogit.seen) == 2  # one per logistic candidate
    full = np.column_stack([train.features["distance_m"], train.features["angle_rad"]])
    for _, x in SpyScaler.seen:
        assert x.shape[0] == len(train)
        assert np.array_equal(x, full[:, : x.shape[1]])  # exactly the training rows
    for _, x, y in SpyLogit.seen:
        assert x.shape[0] == len(train) == len(y) and np.array_equal(y, train.y)
    scaler = models[DISTANCE_PLUS_ANGLE].estimator.named_steps["scale"]
    assert scaler.mean_ == pytest.approx(full.mean(axis=0))
    pooled = np.concatenate([train.features["distance_m"], validation.features["distance_m"]])
    assert scaler.mean_[0] != pytest.approx(pooled.mean())  # validation rows never touched it
    # Scoring validation transforms but never refits.
    before = (len(SpyScaler.seen), len(SpyLogit.seen))
    models[DISTANCE_PLUS_ANGLE].predict_proba(validation)
    assert (len(SpyScaler.seen), len(SpyLogit.seen)) == before


def test_fitted_models_learn_the_direction_built_into_the_synthetic_labels(parts):
    models = fit_baselines(parts["train"])
    c = models[DISTANCE_ONLY].coefficients()
    assert c["standardised"]["distance_m"] < 0 and c["per_unit"]["distance_m"] < 0
    probabilities = models[DISTANCE_ONLY].predict_proba(parts["validation"])
    near = parts["validation"].features["distance_m"] < 10
    assert probabilities[near].mean() > probabilities[~near].mean()
    assert models[CONSTANT].coefficients() is None


def test_per_unit_coefficients_reproduce_the_pipeline_probabilities(parts):
    model = fit_baselines(parts["train"])[DISTANCE_PLUS_ANGLE]
    c = model.coefficients()
    validation = parts["validation"]
    z = c["intercept_per_unit"] + sum(
        c["per_unit"][f] * validation.features[f] for f in c["features"]
    )
    assert 1 / (1 + np.exp(-z)) == pytest.approx(model.predict_proba(validation), abs=1e-9)


def test_non_convergence_is_an_error_not_a_silent_fit(parts):
    def impatient():
        return Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(max_iter=1))])

    with pytest.raises(ConvergenceWarning):
        fit_baselines(parts["train"], pipeline_factory=impatient)


def test_default_pipeline_is_the_documented_one():
    steps = dict(make_pipeline().named_steps)
    assert isinstance(steps["scale"], StandardScaler)
    params = steps["logit"].get_params()
    assert (params["C"], params["solver"], params["max_iter"]) == (1.0, "lbfgs", 1000)


def test_with_pitch_changes_features_only(parts):
    validation = parts["validation"]
    other = with_pitch(validation, Pitch(length_m=100.0, width_m=64.0))

    assert np.array_equal(other.y, validation.y) and np.array_equal(
        other.match_ids, validation.match_ids
    )
    assert not np.allclose(other.features["distance_m"], validation.features["distance_m"])
    assert set(other.features) == {"distance_m", "angle_rad"}
    same = with_pitch(validation, Pitch())
    assert np.allclose(same.features["distance_m"], validation.features["distance_m"], atol=1e-4)
    with pytest.raises(FeatureNotAllowedError):
        feature_matrix(other, ["x_pct"])
