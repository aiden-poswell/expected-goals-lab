"""End-to-end S03 behaviour on synthetic data, with emphasis on the sealed test split."""

import json
from pathlib import Path

import numpy as np
import pytest

from expected_goals_lab import baselines, evaluate_baselines
from expected_goals_lab.baselines import CONSTANT, DISTANCE_ONLY, DISTANCE_PLUS_ANGLE
from expected_goals_lab.dataset import OPEN_COLUMNS, iter_csv_rows, read_identity
from expected_goals_lab.evaluate_baselines import evaluate, render_report
from expected_goals_lab.scoring import Interval
from expected_goals_lab.shot_table import COLUMNS
from expected_goals_lab.splits import assign_date_splits
from tests.s03_helpers import column_index, rewrite_csv, write_dataset

SENSITIVE = set(OPEN_COLUMNS) | {"is_goal"}
RESAMPLES = 60


def held_out_dates(csv_path: Path, manifest_path: Path) -> set[str]:
    identity = read_identity(csv_path, manifest_path)
    plan = assign_date_splits(identity.dates.tolist())
    return {d for d, s in plan.date_split.items() if s == "test"}


class GuardedRow(dict):
    """A CSV row that raises if a feature, location or target column of a test-date row is read."""

    def __init__(self, row, sealed, log):
        super().__init__(row)
        self.sealed, self.log = sealed, log

    def __getitem__(self, key):
        if self.sealed and key in SENSITIVE:
            self.log.append(key)
            raise AssertionError(f"sealed test row: column {key!r} was read")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if self.sealed and key in SENSITIVE:
            self.log.append(key)
            raise AssertionError(f"sealed test row: column {key!r} was read")
        return super().get(key, default)


def strip(text: str) -> str:
    """Drop the lines that legitimately differ between runs or inputs."""
    skipped = ("Generated", "Input:", "Checks passed")
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith(skipped))


def guarded_reader(test_dates, log):
    def rows(path):
        for row in iter_csv_rows(path):
            yield GuardedRow(row, row["match_datetime_utc"][:10] in test_dates, log)

    return rows


@pytest.fixture
def dataset(tmp_path):
    return write_dataset(tmp_path / "in")


def test_the_whole_evaluation_never_reads_a_sealed_test_label_or_feature(dataset, tmp_path):
    csv_path, manifest_path, _ = dataset
    test_dates = held_out_dates(csv_path, manifest_path)
    log: list[str] = []

    evaluation = evaluate(
        csv_path, manifest_path, tmp_path / "out", RESAMPLES, rows=guarded_reader(test_dates, log)
    )

    assert log == []  # no sensitive column of a test row was ever subscripted
    assert set(evaluation.scores) == {"train", "validation"}
    assert set(evaluation.partitions) == {"train", "validation"}
    assert not (set(evaluation.partitions["train"].dates.tolist()) & test_dates)
    assert not (set(evaluation.partitions["validation"].dates.tolist()) & test_dates)


def test_the_spy_would_catch_a_leak(dataset):
    # Negative control: if code did read a test row's label, the guard must fire.
    csv_path, manifest_path, _ = dataset
    test_dates = held_out_dates(csv_path, manifest_path)
    log: list[str] = []
    rows = list(guarded_reader(test_dates, log)(csv_path))
    test_row = next(r for r in rows if r.sealed)
    for column in ("is_goal", "distance_m", "angle_rad", "x_pct"):
        with pytest.raises(AssertionError, match="sealed"):
            test_row[column]
    assert test_row["event_id"] and test_row["match_id"]  # identity columns remain readable
    assert sorted(set(log)) == ["angle_rad", "distance_m", "is_goal", "x_pct"]


def test_results_do_not_depend_on_test_labels_or_features(dataset, tmp_path):
    csv_path, manifest_path, _ = dataset
    clean = evaluate(csv_path, manifest_path, tmp_path / "clean", RESAMPLES)
    test_dates = held_out_dates(csv_path, manifest_path)

    def garble(lines):
        out = [lines[0]]
        for line in lines[1:]:
            cells = line.split(",") if line else []
            if cells and cells[column_index("match_datetime_utc")][:10] in test_dates:
                for name in ("is_goal", "distance_m", "angle_rad", "x_pct", "y_pct"):
                    cells[column_index(name)] = "NOT-A-NUMBER"
            out.append(",".join(cells) if cells else line)
        return out

    rewrite_csv(csv_path, manifest_path, garble)
    garbled = evaluate(csv_path, manifest_path, tmp_path / "garbled", RESAMPLES)

    for split in ("train", "validation"):
        for name in (CONSTANT, DISTANCE_ONLY, DISTANCE_PLUS_ANGLE):
            assert garbled.scores[split][name] == clean.scores[split][name]
    assert garbled.intervals == clean.intervals
    assert garbled.sensitivity == clean.sensitivity
    assert strip(render_report(garbled, csv_path)) == strip(render_report(clean, csv_path))


def test_quarantined_rows_never_reach_a_model(dataset, tmp_path):
    csv_path, manifest_path, rows = dataset
    evaluation = evaluate(csv_path, manifest_path, tmp_path / "out", RESAMPLES)
    quarantined = {r.event_id for r in rows if not r.eligible}
    used = set()
    for part in evaluation.partitions.values():
        used |= set(part.event_ids.tolist())
    assert quarantined and not used & quarantined
    total = sum(len(p) for p in evaluation.partitions.values())
    assert total + evaluation.split_summary["test"]["rows"] == sum(r.eligible for r in rows)


def test_only_training_rows_are_used_to_fit_in_the_real_evaluation_path(
    dataset, tmp_path, monkeypatch
):
    csv_path, manifest_path, _ = dataset
    seen = []

    class SpyScaler(baselines.StandardScaler):
        def fit(self, X, y=None, **kw):
            seen.append(np.array(X, copy=True))
            return super().fit(X, y, **kw)

    def spying_factory():
        return baselines.Pipeline(
            [("scale", SpyScaler()), ("logit", baselines.LogisticRegression(C=1.0, max_iter=1000))]
        )

    real_fit = baselines.fit_baselines
    monkeypatch.setattr(
        evaluate_baselines, "fit_baselines", lambda train: real_fit(train, spying_factory)
    )
    evaluation = evaluate(csv_path, manifest_path, tmp_path / "out", RESAMPLES)

    n_train = len(evaluation.partitions["train"])
    # two logistic fits for the reference pitch plus two per alternative pitch, all on train rows
    assert len(seen) == 2 * (1 + len(evaluate_baselines.ALTERNATIVE_PITCHES))
    assert all(x.shape[0] == n_train for x in seen)


def test_report_is_reproducible_apart_from_the_timestamp(dataset, tmp_path):
    csv_path, manifest_path, _ = dataset
    first = render_report(evaluate(csv_path, manifest_path, tmp_path / "a", RESAMPLES), csv_path)
    second = render_report(evaluate(csv_path, manifest_path, tmp_path / "b", RESAMPLES), csv_path)
    assert strip(first) == strip(second)
    assert (tmp_path / "a" / "wyscout_split_assignment.csv").read_bytes() == (
        tmp_path / "b" / "wyscout_split_assignment.csv"
    ).read_bytes()


def test_report_states_what_it_does_and_does_not_contain(dataset, tmp_path):
    csv_path, manifest_path, _ = dataset
    ev = evaluate(csv_path, manifest_path, tmp_path / "out", RESAMPLES)
    text = render_report(ev, csv_path)

    assert "sealed" in text and "Validation only" in text
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7."):
        assert heading in text
    assert "10.6084/m9.figshare.c.4415000.v5" in text and "CC BY 4.0" in text
    # The sealed split appears only as counts and dates, never with goals or a rate.
    test_summary = ev.split_summary["test"]
    assert "goals" not in test_summary and "rate" not in test_summary
    test_lines = [ln for ln in text.splitlines() if ln.startswith("| test ")]
    assert (
        len(test_lines) == 1 and test_lines[0].count("|") == 7  # split, 2 dates, 3 counts
    )  # split, dates, dates, matches, shots
    split_manifest = json.loads((tmp_path / "out" / "wyscout_split_manifest.json").read_text())
    assert "is_goal" not in json.dumps(split_manifest["splits"])


def test_goal_line_shots_are_counted_in_open_splits_only(dataset, tmp_path):
    csv_path, manifest_path, _ = dataset
    ev = evaluate(csv_path, manifest_path, tmp_path / "out", RESAMPLES)
    assert set(ev.goal_line) == {"train", "validation"}


def test_league_results_shown_only_when_samples_suffice(dataset, tmp_path, monkeypatch):
    csv_path, manifest_path, _ = dataset
    default = evaluate(csv_path, manifest_path, tmp_path / "a", RESAMPLES)
    assert not any(e["shown"] for e in default.by_league.values())  # synthetic leagues are small
    assert "not shown" in render_report(default, csv_path)

    monkeypatch.setattr(evaluate_baselines, "MIN_LEAGUE_SHOTS", 20)
    monkeypatch.setattr(evaluate_baselines, "MIN_LEAGUE_GOALS", 2)
    shown = evaluate(csv_path, manifest_path, tmp_path / "b", RESAMPLES)
    assert all(e["shown"] and "intervals" in e for e in shown.by_league.values())
    assert sum(e["n"] for e in shown.by_league.values()) == len(shown.partitions["validation"])


def test_sensitivity_covers_both_alternative_pitches_and_leaves_the_reference_alone(
    dataset, tmp_path
):
    csv_path, manifest_path, _ = dataset
    ev = evaluate(csv_path, manifest_path, tmp_path / "out", RESAMPLES)
    assert set(ev.sensitivity) == {"100x64", "110x70"}
    assert ev.sensitivity["100x64"][CONSTANT] == ev.scores["validation"][CONSTANT]  # unaffected
    assert ev.sensitivity["100x64"][DISTANCE_ONLY] != ev.scores["validation"][DISTANCE_ONLY]


# --- CLI ----------------------------------------------------------------------------------


def run_cli(csv_path, tmp_path, *extra):
    return evaluate_baselines.main(
        [
            "--csv",
            str(csv_path),
            "--report",
            str(tmp_path / "report.md"),
            "--out-dir",
            str(tmp_path / "split"),
            "--resamples",
            str(RESAMPLES),
            *extra,
        ]
    )


def test_cli_success_writes_report_and_split_files(dataset, tmp_path, capsys):
    csv_path, _, _ = dataset
    assert run_cli(csv_path, tmp_path) == 0
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "split" / "wyscout_split_assignment.csv").exists()
    assert "Baseline validation (S03)" in capsys.readouterr().out


def test_cli_missing_input_exits_2_and_writes_nothing(tmp_path, capsys):
    assert run_cli(tmp_path / "absent.csv", tmp_path) == 2
    assert not (tmp_path / "report.md").exists()
    assert "build_shots" in capsys.readouterr().err


def test_cli_fails_closed_on_hash_mismatch_and_corrupt_manifest(dataset, tmp_path, capsys):
    csv_path, manifest_path, _ = dataset
    csv_path.write_bytes(csv_path.read_bytes() + b"\n")
    assert run_cli(csv_path, tmp_path) == 1
    assert "SHA-256" in capsys.readouterr().err
    assert not (tmp_path / "report.md").exists()

    manifest_path.write_text("{broken", encoding="utf-8")
    assert run_cli(csv_path, tmp_path) == 1
    assert "manifest" in capsys.readouterr().err
    assert not (tmp_path / "report.md").exists()


def test_columns_constant_is_what_the_contract_checks():
    assert COLUMNS[0] == "event_id" and COLUMNS[-1] == "flags"


def test_angle_summary_describes_each_case_from_the_intervals():
    k = (DISTANCE_PLUS_ANGLE, DISTANCE_ONLY)
    gains = {
        (DISTANCE_PLUS_ANGLE, CONSTANT): Interval(-0.0500, -0.06, -0.04),
        (DISTANCE_ONLY, CONSTANT): Interval(-0.0450, -0.05, -0.04),
    }
    clear, unclear = Interval(-0.002, -0.004, -0.0005), Interval(-0.002, -0.004, 0.0002)

    both = evaluate_baselines._angle_summary({**gains, k: clear}, {**gains, k: clear})
    split = evaluate_baselines._angle_summary({**gains, k: unclear}, {**gains, k: clear})
    neither = evaluate_baselines._angle_summary({**gains, k: unclear}, {**gains, k: unclear})

    assert "both intervals exclude zero" in both
    assert "disagree" in split and "only the Brier score interval excludes zero" in split
    assert "neither interval excludes zero" in neither
    assert "90% of the log-loss gain" in both  # 0.0450 / 0.0500
