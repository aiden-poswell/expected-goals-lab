"""Pure diagnostics functions, checked against small hand-countable cases."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from expected_goals_lab import diagnostics
from expected_goals_lab.dataset import ContractError
from expected_goals_lab.diagnostics import (
    SPARSE_BELOW,
    adjusted_by_class,
    bins_covering_diagonal,
    bins_missing_diagonal,
    calibration_rows,
    illustrative_predictions,
    league_rows,
    league_summary,
    overall_tally,
    raw_body_part_rows,
)

NAMES = {1: "League One", 2: "League Two", 3: "League Three"}


class FakePartition:
    def __init__(self, competition_ids, match_ids, y):
        self.competition_ids = np.array(competition_ids)
        self.match_ids = np.array(match_ids)
        self.y = np.array(y)

    def __len__(self):
        return len(self.y)


def partition(per_league: dict[int, tuple[int, int]], matches_per_league: int = 4):
    """per_league: id -> (shots, goals); goals come first within each league."""
    comp, match, y = [], [], []
    for league, (shots, goals) in per_league.items():
        for i in range(shots):
            comp.append(league)
            match.append(league * 100 + i % matches_per_league)
            y.append(1 if i < goals else 0)
    return FakePartition(comp, match, y)


def flat_predictions(part, p):
    return np.full(len(part), p)


def test_league_rows_denominators_and_independent_log_loss():
    part = partition({1: (40, 10), 2: (30, 3), 3: (20, 0)})
    ref, sel = flat_predictions(part, 0.25), flat_predictions(part, 0.2)

    rows = league_rows(part, ref, sel, NAMES, resamples=50)

    assert [r["name"] for r in rows] == list(NAMES.values())
    assert [(r["shots"], r["goals"]) for r in rows] == [(40, 10), (30, 3), (20, 0)]
    assert sum(r["shots"] for r in rows) == len(part)
    first = rows[0]
    by_hand = -(10 * math.log(0.25) + 30 * math.log(0.75)) / 40
    assert first["reference"]["log_loss"] == pytest.approx(by_hand)
    by_hand_selected = -(10 * math.log(0.2) + 30 * math.log(0.8)) / 40
    assert first["diff_log_loss"] == pytest.approx(by_hand_selected - by_hand)
    assert first["diff_brier"] == pytest.approx(
        (10 * 0.8**2 + 30 * 0.2**2) / 40 - (10 * 0.75**2 + 30 * 0.25**2) / 40
    )
    assert first["conversion"] == 0.25 and first["matches"] == 4


def test_auc_needs_enough_of_both_outcomes_and_zero_goal_leagues_do_not_divide_by_zero():
    part = partition({1: (60, 20), 2: (30, 3), 3: (20, 0)})
    rng = np.random.default_rng(0)
    p = rng.random(len(part))

    rows = league_rows(part, p, p, NAMES, resamples=20)

    assert rows[0]["reference"]["auc"] is not None  # 20 goals, 40 non-goals
    assert rows[1]["reference"]["auc"] is None  # only 3 goals
    assert rows[2]["reference"]["auc"] is None and rows[2]["conversion"] == 0.0  # no goals
    assert all(math.isfinite(rows[2]["reference"][k]) for k in ("log_loss", "brier"))


def test_a_league_with_no_shots_gets_a_row_of_nones():
    part = partition({1: (40, 10), 2: (30, 3)})
    rows = league_rows(part, flat_predictions(part, 0.2), flat_predictions(part, 0.2), NAMES, 20)
    empty = rows[2]
    assert (empty["shots"], empty["goals"], empty["matches"]) == (0, 0, 0)
    assert (
        empty["conversion"] is None
        and empty["reference"] is None
        and empty["diff_log_loss"] is None
    )


def test_a_single_match_league_gets_no_interval():
    part = partition({1: (40, 10), 2: (30, 3), 3: (20, 2)}, matches_per_league=1)
    rows = league_rows(part, flat_predictions(part, 0.2), flat_predictions(part, 0.1), NAMES, 20)
    assert all(r["interval_log_loss"] is None and r["diff_log_loss"] is not None for r in rows)


def test_unknown_league_in_the_data_is_a_hard_error():
    part = partition({1: (10, 1), 9: (10, 1)})
    with pytest.raises(ContractError, match="without a verified name"):
        league_rows(part, flat_predictions(part, 0.1), flat_predictions(part, 0.1), NAMES, 10)


def test_league_summary_counts_only_leagues_with_results():
    part = partition({1: (40, 10), 2: (30, 3)})
    better = np.where(part.y == 1, 0.5, 0.05)  # selected clearly better than a flat reference
    rows = league_rows(part, flat_predictions(part, 0.2), better, NAMES, 20)
    summary = league_summary(rows)
    assert summary == {"leagues": 2, "lower_log_loss": 2, "lower_brier": 2}


def test_calibration_bins_are_exhaustive_and_intervals_bracket_the_rate():
    rng = np.random.default_rng(1)
    p = rng.random(237) * 0.5
    y = (rng.random(237) < p).astype(int)

    rows = calibration_rows(y, p)

    assert sum(r["shots"] for r in rows) == 237 and sum(r["goals"] for r in rows) == y.sum()
    assert [r["mean_prediction"] for r in rows] == sorted(r["mean_prediction"] for r in rows)
    assert all(r["low"] <= r["observed"] <= r["high"] for r in rows)
    assert len(rows) == 10 and max(r["shots"] for r in rows) - min(r["shots"] for r in rows) <= 1


def test_calibration_with_fewer_shots_than_bins_skips_empty_bins():
    rows = calibration_rows(np.array([0, 1, 0, 0, 1]), np.array([0.1, 0.8, 0.2, 0.3, 0.6]))
    assert len(rows) == 5 and all(r["shots"] == 1 for r in rows)


@pytest.mark.parametrize("bad", [np.nan, 1.2, -0.1, np.inf])
def test_calibration_rejects_invalid_probabilities(bad):
    with pytest.raises(ContractError, match="finite and within"):
        calibration_rows(np.array([0, 1, 0]), np.array([0.1, bad, 0.3]))


def test_overall_tally_and_bin_coverage_counts():
    y = np.array([1, 0, 0, 0])
    p = np.array([0.5, 0.25, 0.125, 0.125])
    tally = overall_tally(y, p)
    assert tally["expected_goals"] == pytest.approx(1.0) and tally["goals"] == 1
    assert tally["ratio"] == pytest.approx(1.0) and tally["mean_prediction"] == pytest.approx(0.25)

    rows = [
        {"low": 0.1, "high": 0.3, "mean_prediction": 0.2},  # inside
        {"low": 0.1, "high": 0.3, "mean_prediction": 0.35},  # above the interval by 0.05
        {"low": 0.1, "high": 0.3, "mean_prediction": 0.04},  # below by 0.06
    ]
    assert bins_covering_diagonal(rows) == 1
    missing = bins_missing_diagonal(rows)
    assert [i for i, _ in missing] == [2, 3]
    assert [g for _, g in missing] == pytest.approx([0.05, 0.06])


def classes(spec):
    labels, goals, distance = [], [], []
    for cls, shots, scored, d in spec:
        labels += [cls] * shots
        goals += [1] * scored + [0] * (shots - scored)
        distance += [d] * shots
    return np.array(labels, dtype=object), np.array(goals), np.array(distance, dtype=float)


def test_raw_body_part_rows_denominators_and_sparse_classes():
    body, y, distance = classes(
        [("right_foot", 100, 10, 20.0), ("left_foot", 50, 5, 18.0), ("head_or_body", 20, 4, 8.0)]
    )
    rows = {r["class"]: r for r in raw_body_part_rows(body, y, distance)}

    assert [rows[c]["shots"] for c in ("right_foot", "left_foot", "head_or_body")] == [100, 50, 20]
    assert rows["right_foot"]["conversion"] == 0.1 and not rows["right_foot"]["sparse"]
    assert rows["head_or_body"]["sparse"] and rows["head_or_body"]["shots"] < SPARSE_BELOW
    assert rows["head_or_body"]["close_share"] == 1.0 and rows["right_foot"]["close_shots"] == 0
    assert rows["head_or_body"]["label"] == "head/body (tag 403)"  # never "header"


def test_an_absent_class_is_reported_not_divided_by_zero():
    body, y, distance = classes([("right_foot", 40, 4, 15.0)])
    rows = {r["class"]: r for r in raw_body_part_rows(body, y, distance)}
    assert rows["left_foot"]["shots"] == 0 and rows["left_foot"]["conversion"] is None
    assert rows["left_foot"]["low"] is None and rows["left_foot"]["median_distance"] is None
    assert rows["left_foot"]["sparse"]
    adjusted = {
        r["class"]: r for r in adjusted_by_class(body, y, np.full(40, 0.1), np.full(40, 0.1))
    }
    assert (
        adjusted["head_or_body"]["observed"] is None and adjusted["right_foot"]["observed"] == 0.1
    )


def test_adjusted_by_class_reports_each_models_mean_prediction():
    body, y, _ = classes([("right_foot", 4, 1, 1.0), ("head_or_body", 2, 1, 1.0)])
    ref = np.array([0.2, 0.2, 0.2, 0.2, 0.6, 0.6])
    sel = np.array([0.3, 0.3, 0.1, 0.1, 0.4, 0.4])
    rows = {r["class"]: r for r in adjusted_by_class(body, y, ref, sel)}
    assert rows["right_foot"]["mean_reference"] == pytest.approx(0.2)
    assert rows["right_foot"]["mean_selected"] == pytest.approx(0.2)
    assert rows["head_or_body"]["mean_reference"] == pytest.approx(0.6)
    assert rows["head_or_body"]["observed"] == 0.5


class StubPipeline:
    def __init__(self, value):
        self.value = value

    def predict_proba(self, x):
        assert x.shape == (1, 3) and x[0, 2] in diagnostics.CLASS_LABELS
        return np.array([[1 - self.value, self.value]])


def stub_train(n_per_class: int):
    rng = np.random.default_rng(3)
    distance = np.concatenate([rng.uniform(5, 30, n_per_class) for _ in range(3)])
    angle = np.radians(rng.uniform(5, 60, 3 * n_per_class))
    body = np.array(["right_foot", "left_foot", "head_or_body"], dtype=object).repeat(n_per_class)
    base = SimpleNamespace(features={"distance_m": distance, "angle_rad": angle})
    return SimpleNamespace(base=base, body_part=body)


def test_illustrations_give_probabilities_and_flag_sparse_neighbourhoods():
    rows = illustrative_predictions(StubPipeline(0.2), stub_train(5000))
    assert [r["location"] for r in rows] == list(diagnostics.ILLUSTRATIVE_LOCATIONS)
    for row in rows:
        assert len(row["cells"]) == 3
        for cell in row["cells"]:
            assert cell["probability"] == 0.2
            assert cell["sparse"] == (cell["training_shots_nearby"] < SPARSE_BELOW)
    tiny = illustrative_predictions(StubPipeline(0.2), stub_train(5))
    assert all(cell["sparse"] for row in tiny for cell in row["cells"])  # 5 shots cannot be dense


def test_a_non_finite_illustrative_probability_is_a_hard_error():
    with pytest.raises(ContractError, match="finite value in"):
        illustrative_predictions(StubPipeline(float("nan")), stub_train(100))


def test_zero_goal_bins_keep_an_interval_that_brackets_the_observed_rate():
    # Regression: a Wilson lower bound a hair above 0 made an error bar negative (plot crash).
    rows = calibration_rows(np.zeros(100, dtype=int), np.linspace(0.01, 0.2, 100))
    assert all(r["goals"] == 0 and r["observed"] == 0.0 for r in rows)
    assert all(r["low"] <= r["observed"] <= r["high"] for r in rows)
    assert all(r["low"] >= 0.0 for r in rows)
