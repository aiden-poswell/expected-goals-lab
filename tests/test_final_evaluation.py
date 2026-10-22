"""S06: the frozen final-evaluation procedure, exercised only on synthetic fixtures.

No test here opens a real held-out row. The single real-data test is the metadata-only preflight,
which hashes bytes and reads manifests, and it is skipped when the local data are absent.
"""

import csv
import dataclasses
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from expected_goals_lab import final_evaluation as fe
from expected_goals_lab import scoring
from expected_goals_lab.baselines import make_pipeline
from expected_goals_lab.dataset import SealedSplitError
from expected_goals_lab.final_lock import REAL_LOCKED
from expected_goals_lab.nonlinear import FACTORIES
from tests.s06_helpers import (
    INVENTORY_FILE,
    NeverRead,
    PhasedReader,
    build_fixture,
    flip_test_labels,
    result_files,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return build_fixture(tmp_path_factory.mktemp("s06"))


@pytest.fixture
def fresh(tmp_path):
    return build_fixture(tmp_path)


def run(fx, **kwargs):
    return fe.execute_final_evaluation(
        fx.locked,
        fx.paths,
        execute=True,
        confirm_final_test=True,
        resamples=fx.resamples,
        **kwargs,
    )


def watch_opens(monkeypatch, watched):
    """Record how each watched path is opened."""
    opened = []
    real = Path.open

    def spy(self, mode="r", *args, **kwargs):
        if self in watched:
            opened.append((self.name, mode))
        return real(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    return opened


def forbid_csv_parsing(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("csv parsing during a metadata-only run")

    monkeypatch.setattr(csv, "DictReader", boom)
    monkeypatch.setattr(csv, "reader", boom)


# --- the metadata-only dry run never touches a held-out record ---------------------------------


def test_dry_run_passes_and_says_the_test_is_not_opened(fx, capsys):
    reader = NeverRead()
    code = fe.main([], locked=fx.locked, paths=fx.paths, rows=reader)
    out = capsys.readouterr().out
    assert code == 0
    assert "FINAL TEST NOT YET OPENED" in out and "PREFLIGHT PASSED" in out
    assert "[FAIL]" not in out
    assert reader.calls == 0
    assert result_files(fx) == []


def test_explicit_dry_run_flag_is_the_same(fx, capsys):
    assert fe.main(["--dry-run"], locked=fx.locked, paths=fx.paths, rows=NeverRead()) == 0
    assert "FINAL TEST NOT YET OPENED" in capsys.readouterr().out


def test_dry_run_opens_shot_and_assignment_files_only_as_bytes(fx, monkeypatch):
    opened = watch_opens(monkeypatch, {fx.paths.csv, fx.paths.assignment})
    forbid_csv_parsing(monkeypatch)
    assert fe.preflight(fx.locked, fx.paths).ok
    assert opened, "the byte hashes should have opened both files"
    assert {mode for _, mode in opened} == {"rb"}, opened


def test_dry_run_survives_a_shot_table_that_cannot_be_parsed(fresh, monkeypatch):
    """A sentinel: replace the CSV by garbage that would crash any parser, with a matching lock."""
    garbage = b"\xff\xfe not,a,csv\x00\x00"
    fresh.paths.csv.write_bytes(garbage)
    from expected_goals_lab.dataset import sha256_bytes_of

    new_sha = sha256_bytes_of(fresh.paths.csv)
    manifest = json.loads(fresh.paths.csv_manifest.read_text(encoding="utf-8"))
    manifest["csv_sha256"] = new_sha
    fresh.paths.csv_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    split = json.loads(fresh.paths.split_manifest.read_text(encoding="utf-8"))
    split["input_csv_sha256"] = new_sha
    fresh.paths.split_manifest.write_text(json.dumps(split), encoding="utf-8")
    locked = dataclasses.replace(
        fresh.locked,
        csv_sha256=new_sha,
        csv_manifest_sha256=sha256_bytes_of(fresh.paths.csv_manifest),
        split_manifest_sha256=sha256_bytes_of(fresh.paths.split_manifest),
    )
    forbid_csv_parsing(monkeypatch)
    assert fe.preflight(locked, fresh.paths).ok


# --- execution needs both confirmations, fails closed, and is gated for real data --------------


@pytest.mark.parametrize(
    "argv",
    [["--execute"], ["--confirm-final-test"], ["--dry-run", "--execute", "--confirm-final-test"]],
)
def test_execute_needs_both_flags_and_does_nothing_otherwise(fresh, argv, capsys):
    reader = NeverRead()
    assert fe.main(argv, locked=fresh.locked, paths=fresh.paths, rows=reader) == 2
    assert reader.calls == 0
    assert result_files(fresh) == []
    assert capsys.readouterr().err


def test_programmatic_execute_refuses_without_both_confirmations(fresh):
    for execute, confirm in [(True, False), (False, True), (False, False)]:
        with pytest.raises(fe.ExecutionRefused):
            fe.execute_final_evaluation(
                fresh.locked, fresh.paths, execute=execute, confirm_final_test=confirm
            )
    assert result_files(fresh) == []


def test_real_data_execution_is_closed_in_s06(tmp_path, capsys):
    """Deliberately fails when the gate is opened: S07 must then update this test knowingly."""
    assert fe.REAL_EXECUTION_ENABLED is False
    paths = fe.default_paths(tmp_path)
    reader = NeverRead()
    code = fe.main(
        ["--execute", "--confirm-final-test"], locked=REAL_LOCKED, paths=paths, rows=reader
    )
    assert code == 2 and reader.calls == 0
    assert "disabled" in capsys.readouterr().err
    assert not (tmp_path / "reports").exists()


MUTATIONS = {
    "shot_table": lambda fx: fx.paths.csv.write_bytes(fx.paths.csv.read_bytes() + b" "),
    "s02_manifest": lambda fx: fx.paths.csv_manifest.write_bytes(
        fx.paths.csv_manifest.read_bytes() + b" "
    ),
    "assignment": lambda fx: fx.paths.assignment.write_bytes(
        fx.paths.assignment.read_bytes() + b"\n"
    ),
    "split_manifest": lambda fx: fx.paths.split_manifest.write_bytes(
        fx.paths.split_manifest.read_bytes() + b" "
    ),
    "protocol": lambda fx: fx.paths.protocol.write_text("changed\n", encoding="utf-8"),
    "decision": lambda fx: fx.paths.decision.write_text("changed\n", encoding="utf-8"),
    "lockfile": lambda fx: fx.paths.lockfile.write_text("version = 2\n", encoding="utf-8"),
    "pyproject": lambda fx: fx.paths.pyproject.write_text("[x]\n", encoding="utf-8"),
    "inventory_file": lambda fx: fx.inventory.write_text("VALUE = 2\n", encoding="utf-8"),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_a_single_changed_file_fails_closed_with_no_results(fresh, name):
    MUTATIONS[name](fresh)
    reader = NeverRead()
    assert not fe.preflight(fresh.locked, fresh.paths).ok
    with pytest.raises(fe.PreflightFailure):
        run(fresh, rows=reader)
    assert reader.calls == 0
    assert result_files(fresh) == []
    assert fe.main(["--dry-run"], locked=fresh.locked, paths=fresh.paths) == 1


def test_a_split_manifest_with_one_changed_count_fails_closed(fresh):
    recorded = json.loads(fresh.paths.split_manifest.read_text(encoding="utf-8"))
    recorded["splits"]["test"]["matches"] += 1
    fresh.paths.split_manifest.write_text(json.dumps(recorded), encoding="utf-8")
    from expected_goals_lab.dataset import sha256_bytes_of

    relocked = dataclasses.replace(
        fresh.locked, split_manifest_sha256=sha256_bytes_of(fresh.paths.split_manifest)
    )
    failed = [c.name for c in fe.preflight(relocked, fresh.paths).failures()]
    assert failed == ["Split manifest counts and date boundaries equal the locked values"]


@pytest.mark.parametrize(
    "change",
    [
        {"csv_sha256": "0" * 64},
        {"csv_manifest_sha256": "0" * 64},
        {"assignment_sha256": "0" * 64},
        {"split_manifest_sha256": "0" * 64},
        {"protocol_sha256": "0" * 64},
        {"primary_recipe_sha256": "0" * 64},
        {"benchmark_recipe_sha256": "0" * 64},
        {"packages": {"scikit-learn": "0.0.0"}},
        {"bootstrap_seed": 1},
        {"bootstrap_resamples": 999},
        {"split_rows": {"train": 1, "validation": 1, "test": 1}},
        {"boundaries": {"train": ("a", "b"), "validation": ("c", "d"), "test": ("e", "f")}},
        {"file_hashes": {INVENTORY_FILE: "0" * 64}},
    ],
    ids=lambda c: next(iter(c)),
)
def test_a_single_changed_lock_value_fails_closed(fresh, change):
    locked = dataclasses.replace(fresh.locked, **change)
    reader = NeverRead()
    assert not fe.preflight(locked, fresh.paths).ok
    with pytest.raises(fe.PreflightFailure):
        fe.execute_final_evaluation(
            locked, fresh.paths, execute=True, confirm_final_test=True, rows=reader
        )
    assert reader.calls == 0 and result_files(fresh) == []


def test_a_drifted_recomputed_split_stops_before_any_row_is_loaded(fresh, monkeypatch):
    """Files and hashes all agree, but the recomputed split differs from the locked one."""
    real = fe.assign_date_splits

    def drifted(dates):
        plan = real(dates)
        first_test = min(d for d, s in plan.date_split.items() if s == "test")
        moved = {**plan.date_split, first_test: "validation"}
        return dataclasses.replace(plan, date_split=moved)

    monkeypatch.setattr(fe, "assign_date_splits", drifted)
    state, log = {"open": False}, []
    reader = PhasedReader(fresh.test_dates, state, log)
    with pytest.raises(fe.PreflightFailure):
        run(fresh, rows=reader)
    assert log == []
    assert not fresh.paths.report.exists() and not fresh.paths.summary.exists()
    entries = fe.read_log(fresh.paths.log)
    assert [e["outcome"] for e in entries] == ["started", "failed"]
    assert entries[-1]["error_type"] == "PreflightFailure"


# --- the access order: nothing sensitive before the models are fitted and verified --------------


def test_test_rows_are_read_only_after_development_models_are_fitted(fresh, monkeypatch):
    state, trap = {"open": False}, []
    events = []
    reader = PhasedReader(fresh.test_dates, state, trap)

    real_baselines, real_candidates = fe.fit_baselines, fe.fit_candidates
    real_loader = fe.load_final_test_partition

    def baselines(train, *args, **kwargs):
        events.append(("fit_baselines", train.split, set(train.dates.tolist())))
        return real_baselines(train, *args, **kwargs)

    def candidates(train, *args, **kwargs):
        events.append(("fit_candidates", train.split, set(train.base.dates.tolist())))
        return real_candidates(train, *args, **kwargs)

    def loader(*args, **kwargs):
        state["open"] = True
        events.append(("open_test",))
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(fe, "fit_baselines", baselines)
    monkeypatch.setattr(fe, "fit_candidates", candidates)
    monkeypatch.setattr(fe, "load_final_test_partition", loader)
    ev = run(fresh, rows=reader)

    assert trap == [], "a sensitive column of a held-out row was read before the test was opened"
    assert [e[0] for e in events] == ["fit_baselines", "fit_candidates", "open_test"]
    for _, split, dates in events[:2]:
        assert split == "development"
        assert dates.isdisjoint(fresh.test_dates)
    assert ev.events == [
        "identity_verified",
        "development_loaded",
        "development_models_fitted",
        "test_opened",
        "test_loaded",
        "test_scored",
        "outputs_written",
    ]


def test_models_are_fitted_on_development_data_only(fresh):
    """Inverting every held-out label changes observed goals but no prediction or expected goal."""
    original = run(fresh, rows=PhasedReader(fresh.test_dates, {"open": True}, []))
    other = build_fixture(
        fresh.paths.root.parent / "flipped", flip_test_labels(fresh.rows, fresh.test_dates)
    )
    flipped = run(other, rows=PhasedReader(other.test_dates, {"open": True}, []))
    assert original.development == flipped.development
    assert original.test["goals"] + flipped.test["goals"] == original.test["shots"]
    for model in original.scores:
        assert original.scores[model].expected_goals == pytest.approx(
            flipped.scores[model].expected_goals, rel=1e-12
        )
    assert original.scores[fe.PRIMARY].log_loss != flipped.scores[fe.PRIMARY].log_loss


def test_scores_equal_an_independent_refit_on_train_plus_validation(fresh):
    # tolerance: the shot table stores features to a fixed number of decimals
    ev = run(fresh, rows=PhasedReader(fresh.test_dates, {"open": True}, []))
    eligible = [r for r in fresh.rows if r.eligible]
    dev = [r for r in eligible if r.match_datetime_utc[:10] not in fresh.test_dates]
    test = [r for r in eligible if r.match_datetime_utc[:10] in fresh.test_dates]

    def location(rows):
        return np.array([[r.distance_m, r.angle_rad] for r in rows])

    def with_body(rows):
        matrix = np.empty((len(rows), 3), dtype=object)
        for i, r in enumerate(rows):
            matrix[i] = [r.distance_m, r.angle_rad, r.body_part_class]
        return matrix

    y_dev = np.array([r.is_goal for r in dev])
    y_test = np.array([r.is_goal for r in test])
    benchmark = make_pipeline().fit(location(dev), y_dev)
    primary = FACTORIES[fe.PRIMARY]().fit(with_body(dev), y_dev)
    p_benchmark = benchmark.predict_proba(location(test))[:, 1]
    p_primary = primary.predict_proba(with_body(test))[:, 1]

    assert ev.test["shots"] == len(test) and ev.test["goals"] == int(y_test.sum())
    assert ev.development["shots"] == len(dev) and ev.development["goals"] == int(y_dev.sum())
    assert ev.scores[fe.BENCHMARK].log_loss == pytest.approx(
        scoring.log_loss(y_test, p_benchmark), rel=1e-6
    )
    assert ev.scores[fe.PRIMARY].log_loss == pytest.approx(
        scoring.log_loss(y_test, p_primary), rel=1e-6
    )
    assert ev.scores[fe.PRIMARY].brier == pytest.approx(
        scoring.brier_score(y_test, p_primary), rel=1e-6
    )
    assert ev.scores[fe.REFERENCE_ONLY].expected_goals == pytest.approx(
        y_dev.mean() * len(test), rel=1e-12
    )
    assert ev.test["first_date"] == min(fresh.test_dates)
    assert ev.test["last_date"] == max(fresh.test_dates)


def test_a_failed_model_probability_check_stops_without_results(fresh, monkeypatch):
    from expected_goals_lab.baselines import FittedModel

    monkeypatch.setattr(FittedModel, "predict_proba", lambda self, part: np.full(len(part), 1.5))
    with pytest.raises(ValueError):
        run(fresh, rows=PhasedReader(fresh.test_dates, {"open": True}, []))
    assert not fresh.paths.report.exists()
    assert fe.read_log(fresh.paths.log)[-1]["outcome"] == "failed"


# --- one-shot log -------------------------------------------------------------------------------


def test_a_completed_evaluation_cannot_be_rerun(fresh, capsys):
    reader = PhasedReader(fresh.test_dates, {"open": True}, [])
    run(fresh, rows=reader)
    log_before = fresh.paths.log.read_text(encoding="utf-8")
    assert not fe.preflight(fresh.locked, fresh.paths).ok
    with pytest.raises(fe.PreflightFailure):
        run(fresh, rows=reader)
    assert fresh.paths.log.read_text(encoding="utf-8") == log_before
    assert "FINAL TEST ALREADY EXECUTED" in (
        fe.format_preflight(fe.preflight(fresh.locked, fresh.paths), fresh.locked, fresh.paths)
    )


def test_a_retry_after_a_failure_needs_a_recorded_reason(fresh, monkeypatch):
    reader = PhasedReader(fresh.test_dates, {"open": True}, [])
    real = fe.fit_candidates
    monkeypatch.setattr(fe, "fit_candidates", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        run(fresh, rows=reader)
    monkeypatch.setattr(fe, "fit_candidates", real)
    with pytest.raises(fe.ExecutionRefused, match="retry-reason"):
        run(fresh, rows=reader)
    with pytest.raises(fe.ExecutionRefused):
        run(fresh, rows=reader, retry_reason="   ")
    run(fresh, rows=reader, retry_reason="fixed a demonstrated execution fault")

    entries = fe.read_log(fresh.paths.log)
    assert [e["outcome"] for e in entries] == ["started", "failed", "started", "completed"]
    assert entries[1]["error_type"] == "OSError"
    assert entries[2]["retry_reason"] == "fixed a demonstrated execution fault"
    allowed = {
        "started_utc", "commit", "uncommitted_changes", "protocol_sha256", "retry_reason",
        "previous_outcome", "stale_outputs", "outcome", "error_type", "report_sha256",
        "summary_sha256",
    }  # fmt: skip
    assert all(set(e) <= allowed for e in entries), "the log must hold outcomes only"


def test_cli_writes_aggregate_files_and_states_they_are_not_for_reselection(fresh, capsys):
    reader = PhasedReader(fresh.test_dates, {"open": True}, [])
    code = fe.main(
        ["--execute", "--confirm-final-test"],
        locked=fresh.locked,
        paths=fresh.paths,
        rows=reader,
        resamples=fresh.resamples,
    )
    assert code == 0
    assert "not for model reselection" in capsys.readouterr().out
    assert fresh.paths.report.exists() and fresh.paths.summary.exists()
    summary = json.loads(fresh.paths.summary.read_text(encoding="utf-8"))
    assert set(summary) == {
        "provenance", "development", "test", "scores", "intervals", "calibration", "leagues",
    }  # fmt: skip


# --- scoring, by hand ---------------------------------------------------------------------------


Y = np.array([1, 0, 0, 1])
P = np.array([0.8, 0.1, 0.4, 0.5])


def test_hand_calculated_scores_and_expected_goals():
    match_ids = np.array([1, 1, 2, 2])
    scores, intervals, valid = fe.score_models(Y, {fe.PRIMARY: P, fe.BENCHMARK: P}, match_ids, 20)
    s = scores[fe.PRIMARY]
    assert s.log_loss == pytest.approx(
        (-math.log(0.8) - math.log(0.9) - math.log(0.6) - math.log(0.5)) / 4, abs=1e-12
    )
    assert s.log_loss == pytest.approx(0.383119, abs=1e-6)
    assert s.brier == pytest.approx((0.04 + 0.01 + 0.16 + 0.25) / 4, abs=1e-12)
    assert s.auc == 1.0
    assert (s.shots, s.goals) == (4, 2)
    assert s.expected_goals == pytest.approx(1.8, abs=1e-12)
    assert valid and intervals["log_loss"].estimate == 0.0


def test_auc_ties_and_single_class():
    assert fe.safe_auc(np.array([1, 0]), np.array([0.5, 0.5])) == 0.5
    assert fe.safe_auc(np.array([1, 0, 1, 0]), np.array([0.2, 0.8, 0.9, 0.1])) == 0.75
    assert fe.safe_auc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])) is None
    assert fe.safe_auc(np.array([1, 1]), np.array([0.1, 0.2])) is None
    scores, *_ = fe.score_models(
        np.zeros(4, dtype=int), {m: P for m in (fe.PRIMARY, fe.BENCHMARK)}, np.arange(4)
    )
    assert scores[fe.PRIMARY].auc is None and scores[fe.PRIMARY].goals == 0


def test_probability_bounds_are_enforced_and_extremes_stay_finite():
    from expected_goals_lab.nonlinear import check_probabilities

    for bad in ([0.2, 1.2], [-0.1, 0.5], [np.nan, 0.5], [np.inf, 0.1]):
        with pytest.raises(ValueError):
            check_probabilities(np.array(bad), "m")
    y, p = np.array([1, 0]), np.array([1.0, 0.0])
    scores, *_ = fe.score_models(y, {fe.PRIMARY: p, fe.BENCHMARK: p}, np.array([1, 2]), 10)
    assert math.isfinite(scores[fe.PRIMARY].log_loss) and scores[fe.PRIMARY].log_loss < 1e-12
    assert scores[fe.PRIMARY].brier == 0.0


def test_the_bootstrap_resamples_matches_not_shots():
    """Two matches of 3 and 1 shots: a match-level draw can only be one of three weightings."""
    y = np.array([1, 0, 0, 1])
    match_ids = np.array([7, 7, 7, 9])
    primary = np.array([0.7, 0.2, 0.2, 0.6])
    benchmark = np.array([0.3, 0.3, 0.3, 0.3])
    _, intervals, valid = fe.score_models(
        y, {fe.PRIMARY: primary, fe.BENCHMARK: benchmark}, match_ids, 400
    )
    diff = scoring.log_loss_terms(y, primary) - scoring.log_loss_terms(y, benchmark)
    achievable = [diff[:3].mean(), diff[3:].mean(), diff.mean()]
    iv = intervals["log_loss"]
    assert valid
    assert iv.estimate == pytest.approx(diff.mean())
    for edge in (iv.low, iv.high):
        assert any(edge == pytest.approx(a) for a in achievable), (edge, achievable)
    # a shot-level bootstrap of 4 shots would give a continuum of values: not what happened
    draws = scoring.cluster_bootstrap_draws(
        match_ids,
        {"a": scoring.log_loss_terms(y, primary), "b": scoring.log_loss_terms(y, benchmark)},
        400,
        scoring.BOOTSTRAP_SEED,
    )
    assert len(np.unique(np.round(draws["a"] - draws["b"], 12))) <= 3


def test_one_match_has_no_interval_and_no_reading():
    _, intervals, valid = fe.score_models(
        Y, {fe.PRIMARY: P, fe.BENCHMARK: P}, np.array([5, 5, 5, 5]), 20
    )
    assert not valid and intervals == {"log_loss": None, "brier": None}


# --- the sparse-league path and the report ------------------------------------------------------


@pytest.fixture(scope="module")
def sparse(tmp_path_factory):
    """Only two leagues present, so three leagues have no shots at all."""
    from tests.s04_helpers import s04_rows

    rows = [dataclasses.replace(r, match_id=r.match_id + 8_000_000_000) for r in s04_rows()]
    assert {r.competition_id for r in rows} <= {364, 412}
    fx = build_fixture(tmp_path_factory.mktemp("sparse"), rows)
    ev = run(fx, rows=PhasedReader(fx.test_dates, {"open": True}, []))
    return fx, ev


def test_empty_leagues_are_reported_without_error(sparse):
    fx, ev = sparse
    empty = [r for r in ev.leagues if r["shots"] == 0]
    assert len(empty) == 3
    assert all(r["benchmark"] is None and r["diff_log_loss"] is None for r in empty)
    assert sum(r["shots"] for r in ev.leagues) == ev.test["shots"]
    text = fe.render_final_report(ev, fx.locked, fx.paths.csv)
    assert "n/a" in text


def test_the_report_has_the_predetermined_sections_and_attribution(sparse):
    fx, ev = sparse
    text = fe.render_final_report(ev, fx.locked, fx.paths.csv)
    for heading in (
        "## Result", "## Expected versus observed goals", "## Calibration",
        "## By league (descriptive only)", "## Limits", "## Identity",
    ):  # fmt: skip
        assert heading in text
    for required in (
        "CC BY 4.0", "doi:10.6084/m9.figshare.c.4415000.v5", "doi:10.1038/s41597-019-0247-7",
        "Wyscout", "not a basis for choosing, tuning or changing", fx.locked.protocol_sha256,
        "the choice is not revisited", "inferable",
    ):  # fmt: skip
        assert required in text, required
    assert not re.search(r"\b[78]\d{9}\b", text), "an event or match identifier leaked"
    assert "player" not in text.lower().replace("same teams and players", "")


def test_the_report_counts_come_from_the_evaluation_not_the_source(sparse):
    fx, ev = sparse
    bumped = dataclasses.replace(ev, test={**ev.test, "shots": 123456, "goals": 4321})
    text = fe.render_final_report(bumped, fx.locked, fx.paths.csv)
    assert "123,456" in text and "4,321" in text
    assert "6,653" not in text and "28,263" not in text


def test_the_report_reads_correctly_when_the_primary_loses(sparse):
    fx, ev = sparse
    worse = {
        "log_loss": scoring.Interval(0.0100, 0.0050, 0.0150),
        "brier": scoring.Interval(0.0030, 0.0010, 0.0050),
    }
    text = fe.render_final_report(dataclasses.replace(ev, intervals=worse), fx.locked, fx.paths.csv)
    assert f"worse than {fe.BENCHMARK}" in text
    assert "improvement over" not in text
    assert "no clear improvement" not in text


def test_the_report_reads_correctly_when_the_primary_wins_or_is_unclear(sparse):
    fx, ev = sparse
    better = {
        "log_loss": scoring.Interval(-0.0100, -0.0150, -0.0050),
        "brier": scoring.Interval(-0.0030, -0.0050, -0.0010),
    }
    text = fe.render_final_report(
        dataclasses.replace(ev, intervals=better), fx.locked, fx.paths.csv
    )
    assert f"improvement over {fe.BENCHMARK}" in text
    unclear = {
        "log_loss": scoring.Interval(-0.0010, -0.0050, 0.0030),
        "brier": scoring.Interval(-0.0003, -0.0010, 0.0004),
    }
    text = fe.render_final_report(
        dataclasses.replace(ev, intervals=unclear), fx.locked, fx.paths.csv
    )
    assert "no clear improvement" in text


def test_the_report_survives_missing_intervals_and_single_class_auc(sparse):
    fx, ev = sparse
    scores = {k: dataclasses.replace(v, auc=None) for k, v in ev.scores.items()}
    none = {"log_loss": None, "brier": None}
    text = fe.render_final_report(
        dataclasses.replace(ev, scores=scores, intervals=none, intervals_valid=False),
        fx.locked,
        fx.paths.csv,
    )
    assert "No interval was computed" in text and "no reading is made" in text and "n/a" in text


# --- the real protocol and lock: consistency, and the real (metadata-only) preflight ------------


def test_protocol_document_and_lock_agree():
    protocol = (REPO / "docs/FINAL_TEST_PROTOCOL.md").read_text(encoding="utf-8")
    assert fe.normalised_sha256(REPO / "docs/FINAL_TEST_PROTOCOL.md") == REAL_LOCKED.protocol_sha256
    values = [
        REAL_LOCKED.csv_sha256, REAL_LOCKED.csv_manifest_sha256, REAL_LOCKED.assignment_sha256,
        REAL_LOCKED.split_manifest_sha256, REAL_LOCKED.primary_recipe_sha256,
        REAL_LOCKED.benchmark_recipe_sha256, *REAL_LOCKED.file_hashes.values(),
    ]  # fmt: skip
    for value in values:
        assert re.fullmatch(r"[0-9a-f]{64}", value)
        assert value in protocol
    for split, (first, last) in REAL_LOCKED.boundaries.items():
        assert first in protocol and last in protocol
        assert f"{REAL_LOCKED.split_rows[split]:,}" in protocol
    assert set(REAL_LOCKED.file_hashes) == set(fe.FROZEN_FILES)
    assert str(REAL_LOCKED.bootstrap_seed) in protocol
    assert "FINAL TEST NOT YET OPENED" in protocol
    assert "CC BY 4.0" in protocol and "10.6084/m9.figshare.c.4415000.v5" in protocol


def test_lock_matches_the_values_published_in_earlier_slices():
    from expected_goals_lab import evaluate_nonlinear as s04

    assert REAL_LOCKED.csv_sha256 == s04.PUBLISHED_CSV_SHA
    assert REAL_LOCKED.assignment_sha256 == s04.PUBLISHED_ASSIGNMENT_SHA
    assert REAL_LOCKED.split_rows == s04.PUBLISHED_SPLIT_ROWS
    assert sum(REAL_LOCKED.split_rows.values()) == REAL_LOCKED.eligible_rows
    assert (REAL_LOCKED.bootstrap_seed, REAL_LOCKED.bootstrap_resamples) == (
        scoring.BOOTSTRAP_SEED,
        scoring.BOOTSTRAP_RESAMPLES,
    )


def test_fitted_recipes_equal_unfitted_recipes():
    """The execution compares fitted pipelines with the locked hash; fitting must not change it."""
    rng = np.random.default_rng(0)
    x = rng.uniform(1, 30, size=(200, 2))
    y = (rng.random(200) < 0.2).astype(int)
    fitted = make_pipeline().fit(x, y)
    assert fe.recipe_sha256(fitted) == fe.recipe_sha256(make_pipeline())
    body = np.empty((200, 3), dtype=object)
    body[:, :2] = x
    body[:, 2] = rng.choice(["right_foot", "left_foot", "head_or_body"], 200)
    primary = FACTORIES[fe.PRIMARY]().fit(body, y)
    assert fe.recipe_sha256(primary) == fe.recipe_sha256(FACTORIES[fe.PRIMARY]())
    assert fe.recipe_sha256(primary) != fe.recipe_sha256(FACTORIES["nonlinear_location"]())


def test_authorisation_is_the_only_way_to_open_the_test():
    with pytest.raises(SealedSplitError):
        fe.FinalTestAuthorisation("x")
    with pytest.raises(SealedSplitError):
        fe.FinalTestAuthorisation("x", _proof=object())


REAL_DATA = REPO / "data/processed/wyscout_regular_shots.csv"


@pytest.mark.skipif(not REAL_DATA.exists(), reason="local Wyscout shot table not present")
def test_real_metadata_only_preflight_passes_without_parsing_rows(monkeypatch):
    paths = fe.default_paths(REPO)
    opened = watch_opens(monkeypatch, {paths.csv, paths.assignment})
    forbid_csv_parsing(monkeypatch)
    report = fe.preflight(REAL_LOCKED, paths)
    # This public snapshot is not byte-identical to the private revision the lock was written for
    # (provider-neutral imports, packaging metadata), so the pyproject and inventory checks are
    # expected to fail, and the real log records a completed evaluation.
    # See docs/REPRODUCIBILITY.md.
    allowed = {
        "Frozen implementation inventory is unchanged",
        "pyproject hash equals the locked value",
        "Execution log is readable and records no completed evaluation",
    }
    failed = {c.name for c in report.failures()}
    assert failed <= allowed, [(c.name, c.detail) for c in report.failures()]
    assert {mode for _, mode in opened} == {"rb"}
