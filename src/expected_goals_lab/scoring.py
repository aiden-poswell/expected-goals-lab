"""Proper scoring rules, a match-cluster bootstrap and calibration bins.

Everything here takes labels and predictions as arguments. It never reads files, so it cannot
reach the sealed test split on its own.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

PROB_CLIP = 1e-15
BOOTSTRAP_SEED = 20180513
BOOTSTRAP_RESAMPLES = 1000


def log_loss_terms(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-shot negative log likelihood, with probabilities clipped away from 0 and 1."""
    p = np.clip(np.asarray(p, dtype=float), PROB_CLIP, 1 - PROB_CLIP)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def brier_terms(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def log_loss(y, p) -> float:
    return float(np.mean(log_loss_terms(y, p)))


def brier_score(y, p) -> float:
    return float(np.mean(brier_terms(y, p)))


def auc(y, p) -> float:
    """ROC AUC; 0.5 for constant predictions (all ties)."""
    return float(roc_auc_score(y, p))


@dataclass(frozen=True)
class Scores:
    n: int
    goals: int
    log_loss: float
    brier: float
    auc: float


def score(y, p) -> Scores:
    y = np.asarray(y)
    return Scores(
        n=int(len(y)),
        goals=int(y.sum()),
        log_loss=log_loss(y, p),
        brier=brier_score(y, p),
        auc=auc(y, p),
    )


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def match_weights(n_matches: int, resamples: int, seed: int) -> np.ndarray:
    """How often each match is drawn in each resample.

    A resample draws ``n_matches`` matches with replacement, so the counts of a row follow a
    multinomial with equal probabilities and sum to ``n_matches``. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_matches, np.full(n_matches, 1.0 / n_matches), size=resamples)


def cluster_bootstrap_draws(
    match_ids: np.ndarray,
    terms: dict[str, np.ndarray],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    """Mean per-shot loss of each model in each match-level resample.

    ``terms`` holds a per-shot loss for each model, aligned with ``match_ids``. A resample draws
    whole matches with replacement and keeps every shot of each drawn match, so the within-match
    dependence is preserved. The same resamples are used for every model.
    """
    unique, inverse = np.unique(match_ids, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    sums = {name: np.bincount(inverse, weights=t) for name, t in terms.items()}
    weights = match_weights(len(unique), resamples, seed)
    denominator = weights @ counts
    return {name: (weights @ s) / denominator for name, s in sums.items()}


def cluster_bootstrap_differences(
    match_ids: np.ndarray,
    terms: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[tuple[str, str], Interval]:
    """95% percentile intervals for mean(terms[a]) - mean(terms[b]), resampling whole matches."""
    boot = cluster_bootstrap_draws(match_ids, terms, resamples, seed)
    out = {}
    for a, b in pairs:
        estimate = float(terms[a].mean() - terms[b].mean())
        draws = boot[a] - boot[b]
        low, high = np.percentile(draws, [2.5, 97.5])
        out[(a, b)] = Interval(estimate, float(low), float(high))
    return out


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Equal-frequency bins on the predicted probability (ties broken by sort order)."""
    order = np.argsort(p, kind="stable")
    bins = []
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        bins.append(
            {
                "n": int(len(chunk)),
                "mean_prediction": float(np.mean(p[chunk])),
                "observed_rate": float(np.mean(y[chunk])),
                "goals": int(np.sum(y[chunk])),
            }
        )
    return bins
