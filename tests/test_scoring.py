"""Scoring rules and the bootstrap, checked against numbers worked out by hand."""

import math

import numpy as np
import pytest

from expected_goals_lab import scoring
from expected_goals_lab.scoring import (
    Interval,
    auc,
    brier_score,
    calibration_bins,
    cluster_bootstrap_differences,
    cluster_bootstrap_draws,
    log_loss,
    match_weights,
    score,
)


def test_log_loss_and_brier_match_hand_calculation():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.2, 0.6, 0.4])
    by_hand = -(math.log(0.9) + math.log(0.8) + math.log(0.6) + math.log(0.6)) / 4
    assert log_loss(y, p) == pytest.approx(by_hand, abs=1e-12)
    assert by_hand == pytest.approx(0.33753883, abs=1e-8)  # 1.3501553 / 4
    assert brier_score(y, p) == pytest.approx((0.01 + 0.04 + 0.16 + 0.16) / 4, abs=1e-12)  # 0.0925


def test_constant_prediction_log_loss_is_the_binary_entropy():
    y = np.array([1] * 3 + [0] * 7)
    r = 0.3
    entropy = -(r * math.log(r) + (1 - r) * math.log(1 - r))
    assert log_loss(y, np.full(10, r)) == pytest.approx(entropy, abs=1e-12)
    assert brier_score(y, np.full(10, r)) == pytest.approx(r * (1 - r), abs=1e-12)


def test_auc_matches_known_values_including_ties():
    assert auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)  # textbook example
    assert auc([1, 0, 1, 0], [0.9, 0.2, 0.6, 0.4]) == pytest.approx(1.0)
    assert auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)  # all ties
    assert auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.0)


def test_probabilities_of_exactly_zero_or_one_are_clipped_not_infinite():
    loss = log_loss([1, 0], [0.0, 1.0])
    assert math.isfinite(loss)
    # about 34.5: the clip bound 1 - 1e-15 is not exactly representable, hence the tolerance
    assert loss == pytest.approx(-math.log(scoring.PROB_CLIP), abs=1e-3)


def test_score_bundle_reports_sample_size_and_goal_count():
    s = score(np.array([1, 0, 0, 1, 0]), np.array([0.7, 0.2, 0.3, 0.6, 0.1]))
    assert (s.n, s.goals) == (5, 2)
    assert 0 < s.log_loss < 1 and 0 < s.brier < 1 and s.auc == pytest.approx(1.0)


def test_match_weights_are_deterministic_and_resample_the_right_number_of_matches():
    w = match_weights(7, 50, seed=3)
    assert w.shape == (50, 7)
    assert (w.sum(axis=1) == 7).all()
    assert np.array_equal(w, match_weights(7, 50, seed=3))
    assert not np.array_equal(w, match_weights(7, 50, seed=4))


def test_bootstrap_draws_whole_matches_not_individual_shots():
    # Match A has 1 shot with loss 0; match B has 3 shots each with loss 1. A resample of two
    # matches can only be (A,A)=0, (A,B)=3/4 or (B,B)=1. Shot-level resampling would produce
    # many other values (e.g. 1/3, 2/3).
    match_ids = np.array([1, 2, 2, 2])
    terms = {"a": np.array([0.0, 1.0, 1.0, 1.0]), "b": np.zeros(4)}

    draws = cluster_bootstrap_draws(match_ids, terms, resamples=400, seed=1)

    assert set(np.round(draws["a"], 6).tolist()) <= {0.0, 0.75, 1.0}
    assert set(np.round(draws["a"], 6).tolist()) == {0.0, 0.75, 1.0}  # all three occur
    assert np.all(draws["b"] == 0)


def test_bootstrap_is_deterministic_and_preserves_the_point_estimate():
    rng = np.random.default_rng(0)
    match_ids = np.repeat(np.arange(30), 5)
    terms = {"a": rng.random(150), "b": rng.random(150)}
    first = cluster_bootstrap_differences(match_ids, terms, [("a", "b")], resamples=200, seed=9)
    again = cluster_bootstrap_differences(match_ids, terms, [("a", "b")], resamples=200, seed=9)
    other = cluster_bootstrap_differences(match_ids, terms, [("a", "b")], resamples=200, seed=10)

    iv = first[("a", "b")]
    assert iv == again[("a", "b")] and iv != other[("a", "b")]
    assert iv.estimate == pytest.approx(terms["a"].mean() - terms["b"].mean())
    assert iv.low <= iv.estimate <= iv.high


def test_a_single_match_has_no_resampling_variation():
    # With one cluster, a shot-level bootstrap would still vary; a match-level one cannot.
    match_ids = np.zeros(20, dtype=int)
    rng = np.random.default_rng(2)
    terms = {"a": rng.random(20), "b": rng.random(20)}
    iv = cluster_bootstrap_differences(match_ids, terms, [("a", "b")], resamples=100, seed=1)[
        ("a", "b")
    ]
    assert iv.low == pytest.approx(iv.estimate) and iv.high == pytest.approx(iv.estimate)


def test_interval_is_wider_when_matches_differ_more():
    n, per = 40, 5
    match_ids = np.repeat(np.arange(n), per)
    steady = {"a": np.full(n * per, 1.0), "b": np.full(n * per, 0.5)}
    rng = np.random.default_rng(5)
    effect = np.repeat(rng.normal(0.5, 0.4, n), per)
    noisy = {"a": 0.5 + effect, "b": np.full(n * per, 0.5)}
    narrow = cluster_bootstrap_differences(match_ids, steady, [("a", "b")], 300, 1)[("a", "b")]
    wide = cluster_bootstrap_differences(match_ids, noisy, [("a", "b")], 300, 1)[("a", "b")]
    assert narrow.high - narrow.low == pytest.approx(0.0, abs=1e-12)
    assert wide.high - wide.low > 0.1
    assert isinstance(wide, Interval)


def test_calibration_bins_are_equal_frequency_and_exhaustive():
    rng = np.random.default_rng(4)
    p = rng.random(103)
    y = (rng.random(103) < p).astype(int)

    bins = calibration_bins(y, p, n_bins=10)

    sizes = [b["n"] for b in bins]
    assert sum(sizes) == 103 and max(sizes) - min(sizes) <= 1
    assert sum(b["goals"] for b in bins) == y.sum()
    means = [b["mean_prediction"] for b in bins]
    assert means == sorted(means)
    assert all(0 <= b["observed_rate"] <= 1 for b in bins)
