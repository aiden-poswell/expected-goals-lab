"""Validation diagnostics for the selected model and the benchmark (docs/DIAGNOSTICS_PROTOCOL.md).

Pure functions over arrays that the guarded S03/S04 loaders already produced. Nothing here reads
files, so nothing here can reach the test window. Every result is an aggregate: no event ID, match
ID or shot-level value is returned.
"""

import math

import numpy as np

from expected_goals_lab.build_shots import wilson
from expected_goals_lab.candidate_data import BODY_PART_VOCABULARY
from expected_goals_lab.dataset import ContractError
from expected_goals_lab.geometry import shot_geometry
from expected_goals_lab.scoring import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    auc,
    brier_score,
    brier_terms,
    cluster_bootstrap_differences,
    log_loss,
    log_loss_terms,
)

MIN_CLASS_FOR_AUC = 10  # goals and non-goals each, for a league AUC
SPARSE_BELOW = 30  # shots in a group, or training shots near an illustrated position
CLOSE_RANGE_M = 12.0
NEIGHBOURHOOD_M = 1.5
NEIGHBOURHOOD_DEG = 5.0
# Fixed before any result was seen (docs/DIAGNOSTICS_PROTOCOL.md, section 6): attack-relative
# percentages, converted with the S02 geometry.
ILLUSTRATIVE_LOCATIONS = {
    "central, penalty-spot area": (90, 50),
    "central, edge of the box": (84, 50),
    "wide inside the box": (90, 35),
    "central, long range": (75, 50),
}
CLASS_LABELS = {
    "right_foot": "right foot",
    "left_foot": "left foot",
    "head_or_body": "head/body (tag 403)",
}


def _rate(goals: int, n: int) -> float | None:
    return goals / n if n else None


def _interval(goals: int, n: int) -> tuple[float, float] | None:
    """Wilson interval clamped into [0, 1] and onto the observed rate (float-noise safe)."""
    if not n:
        return None
    low, high = wilson(goals, n)
    rate = goals / n
    return max(0.0, min(low, rate)), min(1.0, max(high, rate))


def league_rows(
    part,
    p_reference: np.ndarray,
    p_selected: np.ndarray,
    names: dict[int, str],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict]:
    """One row per known league, validation rows only (``part`` is the validation partition)."""
    unknown = sorted(set(part.competition_ids.tolist()) - set(names))
    if unknown:
        raise ContractError(f"validation rows belong to leagues without a verified name: {unknown}")
    rows = []
    for competition_id, name in names.items():
        mask = part.competition_ids == competition_id
        n = int(mask.sum())
        y = part.y[mask]
        goals = int(y.sum())
        row = {
            "competition_id": int(competition_id),
            "name": name,
            "shots": n,
            "goals": goals,
            "matches": int(len(set(part.match_ids[mask].tolist()))),
            "conversion": _rate(goals, n),
            "reference": None,
            "selected": None,
            "diff_log_loss": None,
            "diff_brier": None,
            "interval_log_loss": None,
            "interval_brier": None,
        }
        if n:
            for key, p in (("reference", p_reference[mask]), ("selected", p_selected[mask])):
                row[key] = {
                    "log_loss": log_loss(y, p),
                    "brier": brier_score(y, p),
                    # AUC needs enough of both outcomes to mean anything
                    "auc": auc(y, p) if min(goals, n - goals) >= MIN_CLASS_FOR_AUC else None,
                }
            row["diff_log_loss"] = row["selected"]["log_loss"] - row["reference"]["log_loss"]
            row["diff_brier"] = row["selected"]["brier"] - row["reference"]["brier"]
            if row["matches"] >= 2:
                match_ids = part.match_ids[mask]
                for metric, fn in (("log_loss", log_loss_terms), ("brier", brier_terms)):
                    terms = {
                        "selected": fn(y, p_selected[mask]),
                        "reference": fn(y, p_reference[mask]),
                    }
                    interval = cluster_bootstrap_differences(
                        match_ids, terms, [("selected", "reference")], resamples, seed
                    )[("selected", "reference")]
                    row[f"interval_{metric}"] = (interval.low, interval.high)
        rows.append(row)
    if sum(r["shots"] for r in rows) != len(part) or sum(r["goals"] for r in rows) != int(
        part.y.sum()
    ):
        raise ContractError("league denominators do not add up to the validation totals")
    return rows


def league_summary(rows: list[dict]) -> dict:
    """In how many leagues the selected model has the lower score (descriptive count only)."""
    usable = [r for r in rows if r["diff_log_loss"] is not None]
    return {
        "leagues": len(usable),
        "lower_log_loss": sum(1 for r in usable if r["diff_log_loss"] < 0),
        "lower_brier": sum(1 for r in usable if r["diff_brier"] < 0),
    }


def calibration_rows(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Equal-frequency bins with Wilson 95% intervals for the observed rate."""
    p = np.asarray(p, dtype=float)
    if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
        raise ContractError("predicted probabilities must be finite and within [0, 1]")
    order = np.argsort(p, kind="stable")
    rows = []
    for chunk in np.array_split(order, n_bins):
        if not len(chunk):
            continue
        n, goals = int(len(chunk)), int(y[chunk].sum())
        low, high = _interval(goals, n)
        rows.append(
            {
                "shots": n,
                "goals": goals,
                "mean_prediction": float(p[chunk].mean()),
                "observed": goals / n,
                "low": low,
                "high": high,
            }
        )
    if sum(r["shots"] for r in rows) != len(y) or sum(r["goals"] for r in rows) != int(np.sum(y)):
        raise ContractError("calibration bins do not add up to the validation totals")
    return rows


def overall_tally(y: np.ndarray, p: np.ndarray) -> dict:
    goals, n = int(np.sum(y)), int(len(y))
    expected = float(np.sum(p))
    return {
        "shots": n,
        "goals": goals,
        "expected_goals": expected,
        "ratio": goals / expected if expected else None,
        "mean_prediction": expected / n if n else None,
        "conversion": goals / n if n else None,
    }


def bins_covering_diagonal(rows: list[dict]) -> int:
    """Bins whose 95% interval for the observed rate contains the bin's mean prediction."""
    return sum(1 for r in rows if r["low"] <= r["mean_prediction"] <= r["high"])


def bins_missing_diagonal(rows: list[dict]) -> list[tuple[int, float]]:
    """(bin number, gap) for bins whose interval excludes the mean prediction; gap in "
    "probability."""
    out = []
    for i, r in enumerate(rows, 1):
        if r["mean_prediction"] < r["low"]:
            out.append((i, r["low"] - r["mean_prediction"]))
        elif r["mean_prediction"] > r["high"]:
            out.append((i, r["mean_prediction"] - r["high"]))
    return out


def raw_body_part_rows(body_part: np.ndarray, y: np.ndarray, distance: np.ndarray) -> list[dict]:
    """Per body-part class: shots, goals, conversion, spread of shot distance, share at close "
    "range."""
    rows = []
    for cls in BODY_PART_VOCABULARY:
        mask = np.asarray(body_part) == cls
        n, goals = int(mask.sum()), int(y[mask].sum())
        d = distance[mask]
        interval = _interval(goals, n)
        rows.append(
            {
                "class": cls,
                "label": CLASS_LABELS[cls],
                "shots": n,
                "goals": goals,
                "conversion": _rate(goals, n),
                "low": interval[0] if interval else None,
                "high": interval[1] if interval else None,
                "median_distance": float(np.median(d)) if n else None,
                "q25_distance": float(np.percentile(d, 25)) if n else None,
                "q75_distance": float(np.percentile(d, 75)) if n else None,
                "close_share": float(np.mean(d <= CLOSE_RANGE_M)) if n else None,
                "close_shots": int(np.sum(d <= CLOSE_RANGE_M)),
                "sparse": n < SPARSE_BELOW,
            }
        )
    return rows


def adjusted_by_class(
    body_part: np.ndarray, y: np.ndarray, p_reference: np.ndarray, p_selected: np.ndarray
) -> list[dict]:
    """Observed conversion beside the mean prediction of each model, per body-part class."""
    rows = []
    for cls in BODY_PART_VOCABULARY:
        mask = np.asarray(body_part) == cls
        n = int(mask.sum())
        rows.append(
            {
                "class": cls,
                "label": CLASS_LABELS[cls],
                "shots": n,
                "observed": float(y[mask].mean()) if n else None,
                "mean_reference": float(p_reference[mask].mean()) if n else None,
                "mean_selected": float(p_selected[mask].mean()) if n else None,
            }
        )
    return rows


def illustrative_predictions(pipeline, train, locations: dict | None = None) -> list[dict]:
    """The selected model's predicted probability per class at fixed positions.

    These are model predictions, not observations. For each position the training shots of each
    class within the neighbourhood (1.5 m of distance and 5 degrees of angle) are counted; fewer
    than ``SPARSE_BELOW`` marks that class at that position as sparse (extrapolated).
    """
    locations = locations or ILLUSTRATIVE_LOCATIONS
    distance = train.base.features["distance_m"]
    angle_deg = np.degrees(train.base.features["angle_rad"])
    out = []
    for name, (x_pct, y_pct) in locations.items():
        geometry = shot_geometry(x_pct, y_pct)
        near = (np.abs(distance - geometry.distance_m) <= NEIGHBOURHOOD_M) & (
            np.abs(angle_deg - geometry.angle_deg) <= NEIGHBOURHOOD_DEG
        )
        cells = []
        for cls in BODY_PART_VOCABULARY:
            x = np.empty((1, 3), dtype=object)
            x[0] = [geometry.distance_m, geometry.angle_rad, cls]
            probability = float(pipeline.predict_proba(x)[0, 1])
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ContractError("illustrative probability is not a finite value in [0, 1]")
            n_near = int((near & (train.body_part == cls)).sum())
            cells.append(
                {
                    "class": cls,
                    "label": CLASS_LABELS[cls],
                    "probability": probability,
                    "training_shots_nearby": n_near,
                    "sparse": n_near < SPARSE_BELOW,
                }
            )
        out.append(
            {
                "location": name,
                "x_pct": x_pct,
                "y_pct": y_pct,
                "distance_m": geometry.distance_m,
                "angle_deg": geometry.angle_deg,
                "cells": cells,
            }
        )
    return out
