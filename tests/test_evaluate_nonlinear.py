"""End-to-end S04 behaviour on synthetic data, with emphasis on the seal and fail-closed paths."""

import json
from pathlib import Path

import numpy as np
import pytest

from expected_goals_lab import evaluate_baselines, evaluate_nonlinear
from expected_goals_lab.baselines import DISTANCE_PLUS_ANGLE
from expected_goals_lab.dataset import ContractError
from expected_goals_lab.evaluate_nonlinear import (
    MODELS,
    PAIRS,
    REFERENCE,
    A,
    B,
    is_fragile,
    reading,
    render_report,
)
from expected_goals_lab.evaluate_nonlinear import (
    evaluate_nonlinear as run,
)
from expected_goals_lab.scoring import Interval, brier_terms, log_loss_terms
from tests.s03_helpers import column_index, rewrite_csv
from tests.s04_helpers import (
    SENSITIVE,
    guarded_reader,
    held_out_dates,
    s04_rows,
    write_s04_dataset,
)

RESAMPLES = 60


@pytest.fixture
def dataset(tmp_path):
    return write_s04_dataset(tmp_path / "in")


def call(csv_path, manifest_path, tmp_path, **kwargs):
    kwargs.setdefault("s03_report", tmp_path / "no_s03_report.md")
    kwargs.setdefault("protocol", tmp_path / "no_protocol.md")
    return run(csv_path, manifest_path, tmp_path / "out", RESAMPLES, **kwargs)


def strip(text: str) -> str:
    skipped = ("Generated", "- Input table", "- Split assignment", "- Git commit")
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith(skipped))


# --- S03 is unchanged -----------------------------------------------------------------------


def test_reference_equals_s03_exactly_and_reproduces_its_report(dataset, tmp_path):
    csv_path, manifest_path, _ = dataset
    s03_dir = tmp_path / "s03"
    s03 = evaluate_baselines.evaluate(csv_path, manifest_path, s03_dir, RESAMPLES)
    s03_report = tmp_path / "baseline_validation.md"
    s03_report.write_text(evaluate_baselines.render_report(s03, csv_path), encoding="utf-8")

    ev = run(csv_path, manifest_path, s03_dir, RESAMPLES, s03_report, tmp_path / "p.md")

    assert ev.scores["validation"][REFERENCE] == s03.scores["validation"][DISTANCE_PLUS_ANGLE]
    assert ev.scores["train"][REFERENCE] == s03.scores["train"][DISTANCE_PLUS_ANGLE]
    results = dict(ev.checks)
    assert (
        results[
            "Regenerated S03 reference equals the committed S03 report (to its printed precision)"
        ]
        == "pass"
    )
    assert (
        results["Split assignment equals the one S03 recorded (SHA-256 and row counts)"] == "pass"
    )
    assert ev.assignment_sha == s03.assignment_sha  # the S03 assignment checksum is unchanged


def test_a_reference_that_no_longer_matches_the_s03_report_stops_the_run(dataset, tmp_path):
    csv_path, manifest_path, _ = dataset
    s03 = evaluate_baselines.evaluate(csv_path, manifest_path, tmp_path / "s03", RESAMPLES)
    text = evaluate_baselines.render_report(s03, csv_path)
    tampered = text.replace(
        "| distance_plus_angle |",
        "| distance_plus_angle | 1 | 1 | 9.9999 | 9.99999 | 0.999 | X |",
        1,
    )
    report = tmp_path / "r.md"
    report.write_text(tampered, encoding="utf-8")
    with pytest.raises(ContractError, match="differs from the S03 report"):
        run(csv_path, manifest_path, tmp_path / "s03", RESAMPLES, report, tmp_path / "p.md")


def test_a_changed_split_assignment_stops_the_run(dataset, tmp_path):
    csv_path, manifest_path, _ = dataset
    out = tmp_path / "s03"
    evaluate_baselines.evaluate(csv_path, manifest_path, out, RESAMPLES)
    recorded = json.loads((out / "wyscout_split_manifest.json").read_text(encoding="utf-8"))
    recorded["assignment_csv_sha256"] = "0" * 64
    (out / "wyscout_split_manifest.json").write_text(json.dumps(recorded), encoding="utf-8")
    with pytest.raises(ContractError, match="differs from the one S03 recorded"):
        run(csv_path, manifest_path, out, RESAMPLES, tmp_path / "none.md", tmp_path / "p.md")


def test_every_eligible_event_is_in_one_unchanged_partition(dataset, tmp_path):
    csv_path, manifest_path, rows = dataset
    ev = call(csv_path, manifest_path, tmp_path)
    train, validation = ev.parts["train"].base, ev.parts["validation"].base
    assert not set(train.event_ids.tolist()) & set(validation.event_ids.tolist())
    assert max(train.dates.tolist()) < min(validation.dates.tolist())
    total = len(train) + len(validation) + ev.split_summary["test"]["rows"]
    assert total == sum(r.eligible for r in rows)
    assert all(r.eligible for r in rows if r.event_id in set(train.event_ids.tolist()))


# --- the seal -------------------------------------------------------------------------------


def test_full_s04_path_never_reads_any_attribute_of_a_sealed_test_row(dataset, tmp_path):
    csv_path, manifest_path, _ = dataset
    log: list[str] = []
    reader = guarded_reader(held_out_dates(csv_path, manifest_path), log)

    ev = call(csv_path, manifest_path, tmp_path, rows=reader)

    assert log == []
    assert set(ev.scores) == {"train", "validation"} and set(ev.parts) == {"train", "validation"}


def test_trap_would_fire_on_every_sensitive_column(dataset):
    csv_path, manifest_path, _ = dataset
    log: list[str] = []
    row = next(
        r
        for r in guarded_reader(held_out_dates(csv_path, manifest_path), log)(csv_path)
        if r.sealed
    )
    for column in sorted(SENSITIVE):
        with pytest.raises(AssertionError, match="sealed"):
            row[column]
    assert sorted(set(log)) == sorted(SENSITIVE)


def test_altering_test_rows_cannot_change_any_validation_output(dataset, tmp_path):
    csv_path, manifest_path, _ = dataset
    clean = call(csv_path, manifest_path, tmp_path / "clean")
    test_dates = held_out_dates(csv_path, manifest_path)

    def alter(lines):
        out = [lines[0]]
        for line in lines[1:]:
            cells = line.split(",") if line else []
            if cells and cells[column_index("match_datetime_utc")][:10] in test_dates:
                for name in (
                    "is_goal",
                    "distance_m",
                    "angle_rad",
                    "x_pct",
                    "y_pct",
                    "body_part_class",
                ):
                    cells[column_index(name)] = "GARBAGE"
            out.append(",".join(cells) if cells else line)
        return out

    rewrite_csv(csv_path, manifest_path, alter)
    altered = call(csv_path, manifest_path, tmp_path / "altered")

    assert altered.provenance["input_csv_sha256"] != clean.provenance["input_csv_sha256"]
    assert altered.scores == clean.scores
    assert altered.intervals == clean.intervals
    assert altered.sensitivity == clean.sensitivity
    assert altered.calibration == clean.calibration
    assert altered.body_part_table == clean.body_part_table
    assert strip(render_report(altered, csv_path)) == strip(render_report(clean, csv_path))


# --- fixed designs and no hidden selection ---------------------------------------------------


def test_every_validation_shot_is_scored_in_the_main_results(dataset, tmp_path):
    csv_path, manifest_path, _ = dataset
    ev = call(csv_path, manifest_path, tmp_path)
    n_val = len(ev.parts["validation"])
    assert all(
        ev.scores["validation"][m].n == n_val for m in MODELS
    )  # no validation-based exclusion


def test_paired_differences_match_an_independent_calculation(dataset, tmp_path):
    csv_path, manifest_path, _ = dataset
    ev = call(csv_path, manifest_path, tmp_path)
    # Recompute point estimates from scores of the same model objects' predictions: the mean
    # log loss and Brier of each model are scored independently of the bootstrap code.
    for a, b in PAIRS:
        for metric, key in (("log_loss", "log_loss"), ("brier", "brier")):
            expected = getattr(ev.scores["validation"][a], key) - getattr(
                ev.scores["validation"][b], key
            )
            assert ev.intervals[metric][(a, b)].estimate == pytest.approx(expected, abs=1e-12)
            iv = ev.intervals[metric][(a, b)]
            assert iv.low <= iv.high


# --- goal-line sensitivity -------------------------------------------------------------------


def test_goal_line_sensitivity_changes_only_the_reporting_subset(tmp_path):
    base_rows = s04_rows()
    eligible = [r.event_id for r in base_rows if r.eligible]
    # make a few late (validation-window) shots and a few early (train) shots goal-line shots
    late = tuple(eligible[-300:-295]) + tuple(eligible[-700:-698])
    early = tuple(eligible[10:14])
    csv_path, manifest_path, _ = write_s04_dataset(
        tmp_path / "in", goal_line_event_ids=late + early
    )

    ev = call(csv_path, manifest_path, tmp_path)

    validation = ev.parts["validation"].base
    n_goal_line = int((validation.x_pct == 100).sum())
    assert n_goal_line > 0 and ev.sensitivity["removed"] == n_goal_line
    assert all(s.n == len(validation) for s in ev.scores["validation"].values())  # main scores: all
    assert all(s.n == len(validation) - n_goal_line for s in ev.sensitivity["scores"].values())
    assert ev.sensitivity["removed_goals"] == int(validation.y[validation.x_pct == 100].sum())
    assert (
        ev.parts["train"].base.x_pct == 100
    ).sum() > 0  # train keeps its goal-line rows (no refit)


def test_sensitivity_does_not_refit_anything(dataset, tmp_path, monkeypatch):
    csv_path, manifest_path, _ = dataset
    calls = []
    real = evaluate_nonlinear.fit_candidates
    monkeypatch.setattr(
        evaluate_nonlinear, "fit_candidates", lambda train: calls.append(1) or real(train)
    )
    real_baseline = evaluate_nonlinear.fit_baselines
    monkeypatch.setattr(
        evaluate_nonlinear, "fit_baselines", lambda train: calls.append(2) or real_baseline(train)
    )
    call(csv_path, manifest_path, tmp_path)
    assert calls == [2, 1]  # the reference and the candidates are each fitted exactly once


# --- reading rule ----------------------------------------------------------------------------


def iv(estimate, low, high):
    return Interval(estimate, low, high)


def test_reading_rule_matches_the_protocol():
    assert reading(iv(-0.01, -0.02, -0.005), iv(-0.003, -0.005, 0.0), "ref").startswith(
        "improvement over ref"
    )
    # log loss improves and its interval excludes zero, but Brier points the other way
    assert reading(iv(-0.01, -0.02, -0.005), iv(0.001, -0.001, 0.003), "ref").startswith("no clear")
    assert "disagree" in reading(iv(-0.01, -0.02, -0.005), iv(0.001, -0.001, 0.003), "ref")
    # interval includes zero
    assert "includes zero" in reading(iv(-0.01, -0.02, 0.002), iv(-0.003, -0.005, 0.0), "ref")
    assert reading(iv(0.01, 0.004, 0.02), iv(0.002, 0.0, 0.004), "ref").startswith("worse than ref")


def test_fragile_means_nearest_edge_within_a_quarter_of_the_estimate():
    assert is_fragile(iv(-0.0040, -0.0080, -0.0005))  # 0.0005 < 0.25 * 0.0040
    assert not is_fragile(iv(-0.0040, -0.0080, -0.0030))
    assert not is_fragile(iv(-0.0040, -0.0080, 0.0010))  # includes zero: not "excluding" at all


# --- report honesty --------------------------------------------------------------------------


def test_report_statements_about_the_test_window_and_the_body_part_class(dataset, tmp_path):
    csv_path, manifest_path, _ = dataset
    text = render_report(call(csv_path, manifest_path, tmp_path), csv_path)

    assert "has not been modelled or scored" in text
    assert "row-level labels/features have not been accessed" in text
    assert "aggregate goal count is indirectly inferable" in text
    assert "no model is selected for final testing" in text
    for claim in (
        "outcome-blind",
        "entirely sealed",
        "completely sealed",
        "never read",
        "statistically significant",
    ):
        lowered = text.lower()
        # the word may only appear in a negation of the claim
        assert claim not in lowered or "nothing is called statistically significant" in lowered
    assert "not a confirmed header" in text
    assert "tag-derived" in text
    # the test window appears nowhere with goals, a rate, or any score
    assert "| test" not in text


def test_summary_record_holds_only_aggregates(dataset, tmp_path):
    csv_path, manifest_path, _ = dataset
    out = tmp_path / "run"
    code = evaluate_nonlinear.main(
        [
            "--csv",
            str(csv_path),
            "--report",
            str(tmp_path / "r.md"),
            "--out-dir",
            str(out),
            "--s03-report",
            str(tmp_path / "none.md"),
            "--protocol",
            str(tmp_path / "none2.md"),
            "--resamples",
            str(RESAMPLES),
        ]
    )
    assert code == 0
    record = json.loads((out / evaluate_nonlinear.SUMMARY_NAME).read_text(encoding="utf-8"))
    assert set(record) == {
        "provenance",
        "checks",
        "validation_scores",
        "train_scores",
        "intervals",
        "resamples",
        "seed",
    }
    assert "event_id" not in json.dumps(record) and "test" not in record["validation_scores"]


# --- fail closed -----------------------------------------------------------------------------


def cli(csv_path, tmp_path, *extra):
    return evaluate_nonlinear.main(
        [
            "--csv",
            str(csv_path),
            "--report",
            str(tmp_path / "report.md"),
            "--out-dir",
            str(tmp_path / "o"),
            "--s03-report",
            str(tmp_path / "none.md"),
            "--protocol",
            str(tmp_path / "none2.md"),
            "--resamples",
            str(RESAMPLES),
            *extra,
        ]
    )


def test_cli_success_then_each_failure_writes_nothing(dataset, tmp_path, capsys):
    csv_path, manifest_path, _ = dataset
    assert cli(csv_path, tmp_path) == 0
    assert "Nonlinear location and body-part validation (S04)" in capsys.readouterr().out
    (tmp_path / "report.md").unlink()

    assert cli(tmp_path / "absent.csv", tmp_path) == 2
    assert "build_shots" in capsys.readouterr().err

    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text("{broken", encoding="utf-8")
    assert cli(csv_path, tmp_path) == 1
    manifest_path.write_text(manifest_text, encoding="utf-8")

    csv_path.write_bytes(csv_path.read_bytes() + b"\n")
    assert cli(csv_path, tmp_path) == 1
    assert "SHA-256" in capsys.readouterr().err
    assert not (tmp_path / "report.md").exists()


@pytest.mark.parametrize(
    ("column", "bad", "fragment"),
    [
        ("body_part_class", "unknown", "body_part_class"),
        ("body_part_class", "header", "body_part_class"),
        ("distance_m", "abc", "unreadable"),
        ("distance_m", "nan", "non-finite"),
        ("is_goal", "2", "expected 0 or 1"),
    ],
)
def test_bad_values_in_open_rows_fail_closed(dataset, tmp_path, capsys, column, bad, fragment):
    csv_path, manifest_path, _ = dataset
    first_train = call(csv_path, manifest_path, tmp_path / "probe").parts["train"].base.event_ids[0]

    def corrupt(lines):
        out = []
        for line in lines:
            cells = line.split(",")
            if cells[0] == str(int(first_train)):
                cells[column_index(column)] = bad
            out.append(",".join(cells))
        return out

    rewrite_csv(csv_path, manifest_path, corrupt)
    assert cli(csv_path, tmp_path) == 1
    assert fragment in capsys.readouterr().err
    assert not (tmp_path / "report.md").exists()


def test_duplicate_event_ids_fail_closed(dataset, tmp_path, capsys):
    csv_path, manifest_path, _ = dataset
    rewrite_csv(csv_path, manifest_path, lambda lines: [*lines[:3], lines[1], *lines[3:]])
    assert cli(csv_path, tmp_path) == 1
    assert "duplicate event_id" in capsys.readouterr().err


def test_single_class_validation_cohort_fails_closed(dataset, tmp_path, capsys):
    csv_path, manifest_path, _ = dataset
    first = call(csv_path, manifest_path, tmp_path / "probe")
    validation_ids = {str(int(e)) for e in first.parts["validation"].base.event_ids}

    def no_goals(lines):
        out = []
        for line in lines:
            cells = line.split(",")
            if cells[0] in validation_ids:
                cells[column_index("is_goal")] = "0"
            out.append(",".join(cells))
        return out

    rewrite_csv(csv_path, manifest_path, no_goals)
    assert cli(csv_path, tmp_path) == 1
    assert "single outcome class" in capsys.readouterr().err


def test_candidate_probabilities_helpers_are_importable_and_consistent():
    p = np.array([0.1, 0.9])
    y = np.array([0, 1])
    assert log_loss_terms(y, p).mean() == pytest.approx(-np.log(0.9))
    assert brier_terms(y, p).mean() == pytest.approx(0.01)
    assert Path(evaluate_nonlinear.DEFAULT_REPORT).name == "nonlinear_validation.md"
    assert (A, B) == ("nonlinear_location", "nonlinear_location_body_part")


# --- reporting additions made after the first run (see the protocol, section 12) -------------


def test_verdict_names_the_category_of_the_reading():
    better = (iv(-0.01, -0.02, -0.005), iv(-0.003, -0.005, 0.0))
    unclear = (iv(-0.01, -0.02, 0.002), iv(-0.003, -0.005, 0.0))
    worse = (iv(0.01, 0.004, 0.02), iv(0.002, 0.0, 0.004))
    assert evaluate_nonlinear.verdict(*better) == "improvement"
    assert evaluate_nonlinear.verdict(*unclear) == "no clear improvement"
    assert evaluate_nonlinear.verdict(*worse) == "worse"


def test_sensitivity_flags_only_the_comparisons_whose_reading_changes():
    from types import SimpleNamespace

    better = iv(-0.01, -0.02, -0.005)
    unclear = iv(-0.01, -0.02, 0.002)
    brier = iv(-0.003, -0.005, 0.0)
    main = {
        "log_loss": {PAIRS[0]: unclear, PAIRS[1]: better, PAIRS[2]: better},
        "brier": {p: brier for p in PAIRS},
    }
    reduced = {
        "log_loss": {PAIRS[0]: better, PAIRS[1]: better, PAIRS[2]: better},
        "brier": {p: brier for p in PAIRS},
    }
    ev = SimpleNamespace(intervals=main, sensitivity={"removed": 2, "intervals": reduced})

    flags = evaluate_nonlinear.sensitivity_flags(ev)

    assert len(flags) == 1 and PAIRS[0][0] in flags[0] and PAIRS[0][1] in flags[0]
    assert "depends on those few records" in flags[0]
    ev_same = SimpleNamespace(intervals=main, sensitivity={"removed": 2, "intervals": main})
    assert evaluate_nonlinear.sensitivity_flags(ev_same) == []


def test_report_lists_fitted_body_part_terms_and_the_departures(dataset, tmp_path):
    csv_path, manifest_path, _ = dataset
    ev = call(csv_path, manifest_path, tmp_path)
    assert [row[0] for row in ev.body_part_terms] == sorted(evaluate_nonlinear.BODY_PART_VOCABULARY)
    text = render_report(ev, csv_path)
    assert "Fitted body-part terms (train)" in text
    assert "nothing about heading as a skill" in " ".join(text.split())
    assert "Two reporting additions were made after the first run" in text


# --- integration on the real S02 table (skipped only when it is not present) -----------------

REAL_CSV = Path("data/processed/wyscout_regular_shots.csv")
REAL_MANIFEST = REAL_CSV.with_suffix(".manifest.json")
REAL_PRESENT = REAL_CSV.exists() and REAL_MANIFEST.exists()


@pytest.mark.skipif(not REAL_PRESENT, reason="processed S02 table absent; run build_shots first")
def test_real_table_through_the_trap_passes_every_reproduction_check():
    log: list[str] = []
    reader = guarded_reader(held_out_dates(REAL_CSV, REAL_MANIFEST), log)

    ev = run(REAL_CSV, REAL_MANIFEST, REAL_CSV.parent, 100, rows=reader)

    assert log == []  # no attribute of any sealed test row was touched
    assert len(ev.checks) == 3 and all(outcome == "pass" for _, outcome in ev.checks)
    assert ev.scores["validation"][REFERENCE].n == 5509
    assert set(ev.parts) == {"train", "validation"}
