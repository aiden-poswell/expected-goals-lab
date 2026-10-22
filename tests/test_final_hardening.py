"""S06A: frozen bootstrap settings, a trustworthy completion record, descriptive-only leagues.

Synthetic data only. Nothing here reads a real held-out row: the real lock appears only with a
reader that must never be called, and the real-data gate stays closed.
"""

import dataclasses
import inspect
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from expected_goals_lab import final_evaluation as fe
from expected_goals_lab import scoring
from expected_goals_lab.dataset import ContractError, Partition
from expected_goals_lab.final_lock import REAL_LOCKED
from tests.s06_helpers import NeverRead, PhasedReader, build_fixture, result_files

REPO = Path(__file__).resolve().parents[1]
ALLOWED_LOG_KEYS = {
    "started_utc", "commit", "uncommitted_changes", "protocol_sha256", "retry_reason",
    "previous_outcome", "stale_outputs", "outcome", "error_type", "report_sha256",
    "summary_sha256",
}  # fmt: skip


@pytest.fixture
def fresh(tmp_path):
    return build_fixture(tmp_path)


def reader_for(fx):
    return PhasedReader(fx.test_dates, {"open": True}, [])


def run(fx, **kwargs):
    kwargs.setdefault("resamples", fx.resamples)
    kwargs.setdefault("rows", reader_for(fx))
    return fe.execute_final_evaluation(
        fx.locked, fx.paths, execute=True, confirm_final_test=True, **kwargs
    )


def outcomes(fx):
    return [e["outcome"] for e in fe.read_log(fx.paths.log)]


def no_temporary_files(fx):
    return not list(fx.paths.report.parent.glob("*.tmp")) and not list(
        fx.paths.summary.parent.glob("*.tmp")
    )


# --- A. the bootstrap settings are frozen --------------------------------------------------------


@pytest.mark.parametrize("flag", ["--resamples", "--resample", "--seed"])
def test_the_command_line_has_no_bootstrap_override(fresh, flag, capsys):
    reader = NeverRead()
    with pytest.raises(SystemExit) as stop:
        fe.main(
            ["--execute", "--confirm-final-test", flag, "5"],
            locked=fresh.locked,
            paths=fresh.paths,
            rows=reader,
        )
    assert stop.value.code == 2 and reader.calls == 0 and result_files(fresh) == []
    assert "unrecognized" in capsys.readouterr().err


def test_flag_abbreviations_are_not_accepted(fresh):
    with pytest.raises(SystemExit):
        fe.main(["--exec", "--confirm-final-test"], locked=fresh.locked, paths=fresh.paths)


@pytest.mark.parametrize("change", [{"resamples": 999}, {"resamples": 60}, {"seed": 1}])
def test_the_real_lock_refuses_changed_bootstrap_settings_before_anything(tmp_path, change):
    reader = NeverRead()
    paths = fe.default_paths(tmp_path)
    with pytest.raises(fe.ExecutionRefused, match="frozen"):
        fe.execute_final_evaluation(
            REAL_LOCKED, paths, execute=True, confirm_final_test=True, rows=reader, **change
        )
    assert reader.calls == 0 and not (tmp_path / "reports").exists()


def test_the_real_lock_accepts_its_own_values_but_the_gate_stays_closed(tmp_path):
    reader = NeverRead()
    with pytest.raises(fe.ExecutionRefused, match="disabled"):
        fe.execute_final_evaluation(
            REAL_LOCKED,
            fe.default_paths(tmp_path),
            execute=True,
            confirm_final_test=True,
            rows=reader,
            resamples=REAL_LOCKED.bootstrap_resamples,
            seed=REAL_LOCKED.bootstrap_seed,
        )
    assert reader.calls == 0 and fe.REAL_EXECUTION_ENABLED is False


@pytest.fixture
def real_like(fresh, monkeypatch):
    """The synthetic lock standing in for the real one, with the gate open: tests the real path."""
    monkeypatch.setattr(fe, "REAL_LOCKED", fresh.locked)
    monkeypatch.setattr(fe, "REAL_EXECUTION_ENABLED", True)
    return fresh


@pytest.mark.parametrize("change", [{"resamples": 999}, {"resamples": 60}, {"seed": 1}])
def test_a_real_like_lock_refuses_overrides_before_reading_rows_or_logging(real_like, change):
    reader = NeverRead()
    with pytest.raises(fe.ExecutionRefused, match="frozen"):
        fe.execute_final_evaluation(
            real_like.locked,
            real_like.paths,
            execute=True,
            confirm_final_test=True,
            rows=reader,
            **change,
        )
    assert reader.calls == 0 and result_files(real_like) == []


def test_a_real_like_run_uses_the_locked_bootstrap_settings(real_like, monkeypatch):
    calls = []
    real = fe.cluster_bootstrap_differences

    def spy(match_ids, terms, pairs, resamples, seed):
        calls.append((resamples, seed))
        return real(match_ids, terms, pairs, resamples, seed)

    monkeypatch.setattr(fe, "cluster_bootstrap_differences", spy)
    ev = fe.execute_final_evaluation(
        real_like.locked,
        real_like.paths,
        execute=True,
        confirm_final_test=True,
        rows=reader_for(real_like),
    )
    locked = real_like.locked
    assert calls == [(locked.bootstrap_resamples, locked.bootstrap_seed)] * 2
    assert ev.provenance["bootstrap"] == {
        "seed": locked.bootstrap_seed,
        "resamples": locked.bootstrap_resamples,
    }
    assert (locked.bootstrap_resamples, locked.bootstrap_seed) == (1000, 20180513)


def test_a_synthetic_lock_may_use_fewer_resamples_for_speed(fresh, monkeypatch):
    calls = []
    real = fe.cluster_bootstrap_differences
    monkeypatch.setattr(
        fe,
        "cluster_bootstrap_differences",
        lambda m, t, p, r, s: calls.append(r) or real(m, t, p, r, s),
    )
    run(fresh, resamples=25)
    assert calls == [25, 25]


# --- B. completion means both outputs exist -------------------------------------------------------


def test_success_writes_both_outputs_and_completed_comes_last(fresh, monkeypatch):
    seen = []
    real = fe.append_log

    def spy(path, entry):
        seen.append((entry["outcome"], fresh.paths.report.exists(), fresh.paths.summary.exists()))
        return real(path, entry)

    monkeypatch.setattr(fe, "append_log", spy)
    ev = run(fresh)
    assert seen == [("started", False, False), ("completed", True, True)]
    entries = fe.read_log(fresh.paths.log)
    assert [e["outcome"] for e in entries] == ["started", "completed"]
    done = entries[-1]
    assert done["report_sha256"] == fe.sha256_text(fresh.paths.report.read_text(encoding="utf-8"))
    assert done["summary_sha256"] == fe.sha256_text(fresh.paths.summary.read_text(encoding="utf-8"))
    assert ev.events[-1] == "outputs_written" and no_temporary_files(fresh)
    assert fe.log_status(entries) == "completed"


def test_a_rendering_failure_never_records_completion(fresh, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("render")

    monkeypatch.setattr(fe, "render_final_report", boom)
    with pytest.raises(RuntimeError):
        run(fresh)
    assert outcomes(fresh) == ["started", "failed"]
    assert result_files(fresh) == [fresh.paths.log] and no_temporary_files(fresh)
    assert fe.read_log(fresh.paths.log)[-1]["error_type"] == "RuntimeError"


def test_a_summary_serialisation_failure_never_records_completion(fresh, monkeypatch):
    monkeypatch.setattr(fe, "summary_json", lambda ev: {"bad": object()})
    with pytest.raises(TypeError):
        run(fresh)
    assert outcomes(fresh) == ["started", "failed"]
    assert not fresh.paths.report.exists() and not fresh.paths.summary.exists()


@pytest.mark.parametrize("failing_replace", [1, 2])
def test_a_failure_moving_either_output_into_place_leaves_no_output(
    fresh, monkeypatch, failing_replace
):
    real, count = os.replace, []

    def replace(src, dst):
        count.append(dst)
        if len(count) == failing_replace:
            raise OSError("disk full")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(OSError):
        run(fresh)
    assert outcomes(fresh) == ["started", "failed"]
    assert not fresh.paths.report.exists() and not fresh.paths.summary.exists()
    assert no_temporary_files(fresh)


@pytest.mark.parametrize("failing_call", [1, 2])
def test_a_failure_creating_either_temporary_file_leaves_no_output(
    fresh, monkeypatch, failing_call
):
    real, count = tempfile.mkstemp, []

    def mkstemp(*args, **kwargs):
        count.append(1)
        if len(count) == failing_call:
            raise OSError("no space")
        return real(*args, **kwargs)

    monkeypatch.setattr(tempfile, "mkstemp", mkstemp)
    with pytest.raises(OSError):
        run(fresh)
    assert outcomes(fresh) == ["started", "failed"]
    assert not fresh.paths.report.exists() and not fresh.paths.summary.exists()
    assert no_temporary_files(fresh)


def test_a_failure_recording_completion_removes_the_unconfirmed_outputs(fresh, monkeypatch):
    real = fe.append_log

    def append(path, entry):
        if entry["outcome"] == "completed":
            raise OSError("log write failed")
        return real(path, entry)

    monkeypatch.setattr(fe, "append_log", append)
    with pytest.raises(OSError):
        run(fresh)
    assert outcomes(fresh) == ["started", "failed"]
    assert not fresh.paths.report.exists() and not fresh.paths.summary.exists()


def test_a_failed_attempt_needs_a_reason_and_a_retry_matches_a_clean_run(
    fresh, tmp_path_factory, monkeypatch
):
    monkeypatch.setattr(fe, "render_final_report", lambda *a, **k: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        run(fresh)
    monkeypatch.undo()
    with pytest.raises(fe.ExecutionRefused, match="retry-reason"):
        run(fresh)
    assert outcomes(fresh) == ["started", "failed"], "a refused retry must not log anything"
    retried = run(fresh, retry_reason="rendering bug fixed")
    entries = fe.read_log(fresh.paths.log)
    assert [e["outcome"] for e in entries] == ["started", "failed", "started", "completed"]
    assert entries[2]["retry_reason"] == "rendering bug fixed"
    assert entries[2]["previous_outcome"] == "failed"
    clean_fx = build_fixture(tmp_path_factory.mktemp("clean"))
    clean = run(clean_fx)
    assert retried.scores == clean.scores and retried.leagues == clean.leagues


def test_an_interrupted_attempt_is_not_silently_retried(fresh, monkeypatch, capsys):
    real = fe.fit_candidates

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(fe, "fit_candidates", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(fresh)
    monkeypatch.setattr(fe, "fit_candidates", real)
    assert outcomes(fresh) == ["started"] and not fresh.paths.report.exists()
    assert fe.log_status(fe.read_log(fresh.paths.log)) == "interrupted"

    reader = NeverRead()
    with pytest.raises(fe.ExecutionRefused, match="interrupted"):
        run(fresh, rows=reader)
    for blank in (None, "", "   "):
        with pytest.raises(fe.ExecutionRefused):
            run(fresh, rows=reader, retry_reason=blank)
    assert reader.calls == 0 and outcomes(fresh) == ["started"]
    code = fe.main(
        ["--execute", "--confirm-final-test"], locked=fresh.locked, paths=fresh.paths, rows=reader
    )
    assert code == 2 and "interrupted" in capsys.readouterr().err
    assert fe.main([], locked=fresh.locked, paths=fresh.paths) == 0
    assert "INTERRUPTED" in capsys.readouterr().out

    run(fresh, retry_reason="interrupted by operator Ctrl-C")
    entries = fe.read_log(fresh.paths.log)
    assert [e["outcome"] for e in entries] == ["started", "started", "completed"]
    assert entries[1]["previous_outcome"] == "interrupted"
    assert entries[1]["retry_reason"] == "interrupted by operator Ctrl-C"


@pytest.mark.parametrize("which", ["report", "summary", "both"])
def test_existing_outputs_without_a_completed_entry_are_not_overwritten_silently(fresh, which):
    stale = {"report": [fresh.paths.report], "summary": [fresh.paths.summary]}.get(
        which, [fresh.paths.report, fresh.paths.summary]
    )
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("older content", encoding="utf-8")
    reader = NeverRead()
    with pytest.raises(fe.ExecutionRefused, match="without a completed log entry"):
        run(fresh, rows=reader)
    assert reader.calls == 0 and not fresh.paths.log.exists()
    assert all(p.read_text(encoding="utf-8") == "older content" for p in stale)

    run(fresh, retry_reason="inspected the leftover files; they are from a crashed attempt")
    started = fe.read_log(fresh.paths.log)[0]
    assert sorted(started["stale_outputs"]) == sorted(p.name for p in stale)
    assert "older content" not in fresh.paths.report.read_text(encoding="utf-8")
    assert fe.log_status(fe.read_log(fresh.paths.log)) == "completed"


def test_a_completed_evaluation_cannot_be_overwritten_or_rerun(fresh, capsys):
    run(fresh)
    before = {p: p.read_bytes() for p in (fresh.paths.report, fresh.paths.summary, fresh.paths.log)}
    reader = NeverRead()
    for reason in (None, "I would like another look"):
        with pytest.raises(fe.PreflightFailure, match="completed"):
            run(fresh, rows=reader, retry_reason=reason)
    assert reader.calls == 0
    assert before == {p: p.read_bytes() for p in before}
    assert fe.main(["--execute", "--confirm-final-test"], locked=fresh.locked,
                   paths=fresh.paths, rows=reader) == 1  # fmt: skip
    assert "ALREADY EXECUTED" in "\n".join(
        [fe.format_preflight(fe.preflight(fresh.locked, fresh.paths), fresh.locked, fresh.paths)]
    )


@pytest.mark.parametrize("line", ["{not json", '{"outcome": "weird"}', "[1, 2]"])
def test_a_corrupt_log_fails_closed(fresh, line, capsys):
    fresh.paths.log.parent.mkdir(parents=True, exist_ok=True)
    fresh.paths.log.write_text(line + "\n", encoding="utf-8")
    reader = NeverRead()
    with pytest.raises(fe.LogCorrupt):
        fe.read_log(fresh.paths.log)
    with pytest.raises(fe.PreflightFailure):
        run(fresh, rows=reader)
    assert fe.main([], locked=fresh.locked, paths=fresh.paths) == 1
    assert "UNREADABLE" in capsys.readouterr().out
    assert reader.calls == 0 and not fresh.paths.report.exists()


def test_the_log_holds_outcomes_only(fresh, monkeypatch):
    monkeypatch.setattr(fe, "render_final_report", lambda *a, **k: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        run(fresh)
    monkeypatch.undo()
    run(fresh, retry_reason="fixed a demonstrated rendering fault")
    entries = fe.read_log(fresh.paths.log)
    assert all(set(e) <= ALLOWED_LOG_KEYS for e in entries)
    raw = fresh.paths.log.read_text(encoding="utf-8")
    assert not re.search(r"\b[78]\d{9}\b", raw), "an event or match identifier leaked"
    assert "probab" not in raw and "is_goal" not in raw


# --- C. per-league results are descriptive only ------------------------------------------


def keys_of(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from keys_of(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from keys_of(item)


def test_final_league_outputs_have_no_intervals_and_no_per_league_bootstrap(fresh, monkeypatch):
    calls = []
    real = fe.cluster_bootstrap_differences

    def spy(match_ids, terms, pairs, resamples, seed):
        calls.append(len(match_ids))
        return real(match_ids, terms, pairs, resamples, seed)

    monkeypatch.setattr(fe, "cluster_bootstrap_differences", spy)
    monkeypatch.setattr(scoring, "cluster_bootstrap_differences", spy)
    ev = run(fresh)
    assert calls == [ev.test["shots"]] * 2, "only the overall comparison may be bootstrapped"
    assert not hasattr(fe, "league_rows")
    leagues_keys = set(keys_of(ev.leagues))
    assert not [k for k in leagues_keys if "interval" in k or k in ("low", "high")]
    summary = json.loads(fresh.paths.summary.read_text(encoding="utf-8"))
    assert not [k for k in keys_of(summary["leagues"]) if "interval" in k]
    assert set(summary["intervals"]) == {"log_loss", "brier"}, "the overall intervals remain"
    assert sum(r["shots"] for r in ev.leagues) == ev.test["shots"]
    assert sum(r["goals"] for r in ev.leagues) == ev.test["goals"]
    assert sum(r["matches"] for r in ev.leagues) == ev.test["matches"]
    text = fresh.paths.report.read_text(encoding="utf-8")
    assert "neither computed nor reported" in text and "AUC benchmark" in text


def partition(competition, y, matches=None):
    n = len(y)
    return Partition(
        split="test",
        event_ids=np.arange(n),
        match_ids=np.asarray(matches if matches is not None else np.arange(n) // 2),
        competition_ids=np.asarray(competition),
        dates=np.array(["2018-05-01"] * n),
        features={"distance_m": np.ones(n), "angle_rad": np.ones(n)},
        x_pct=np.zeros(n, dtype=int),
        y_pct=np.zeros(n, dtype=int),
        y=np.asarray(y, dtype=np.int8),
    )


def test_league_descriptives_handle_empty_single_class_and_sparse_leagues():
    names = {364: "A", 412: "B", 426: "C", 524: "D"}
    y = [0] * 6 + ([1, 0] * 10) + [1, 0, 1, 0, 1, 0]
    competition = [364] * 6 + [412] * 20 + [426] * 6
    part = partition(competition, y)
    rng = np.random.default_rng(3)
    p_benchmark, p_primary = rng.uniform(0.05, 0.6, len(y)), rng.uniform(0.05, 0.6, len(y))
    rows = {r["name"]: r for r in fe.league_descriptives(part, p_benchmark, p_primary, names)}

    assert rows["D"]["shots"] == 0 and rows["D"]["benchmark"] is None
    assert rows["D"]["diff_log_loss"] is None and rows["D"]["matches"] == 0
    assert rows["A"]["goals"] == 0
    assert rows["A"]["benchmark"]["auc"] is None  # one outcome class only
    assert rows["C"]["benchmark"]["auc"] is None  # 3 goals, below the minimum class size
    assert rows["B"]["goals"] == 10 and rows["B"]["benchmark"]["auc"] is not None
    mask = np.array(competition) == 412
    yb = np.array(y)[mask]
    assert rows["B"]["benchmark"]["auc"] == scoring.auc(yb, p_benchmark[mask])
    assert rows["B"]["diff_log_loss"] == pytest.approx(
        scoring.log_loss(yb, p_primary[mask]) - scoring.log_loss(yb, p_benchmark[mask])
    )
    assert rows["B"]["diff_brier"] == pytest.approx(
        scoring.brier_score(yb, p_primary[mask]) - scoring.brier_score(yb, p_benchmark[mask])
    )
    assert sum(r["shots"] for r in rows.values()) == len(y)
    assert not [k for k in keys_of(list(rows.values())) if "interval" in k]


def test_league_descriptives_reject_unnamed_leagues():
    part = partition([364, 999], [0, 1])
    with pytest.raises(ContractError):
        fe.league_descriptives(part, np.array([0.1, 0.2]), np.array([0.1, 0.2]), {364: "A"})


def test_s05_league_diagnostics_are_unchanged():
    """The validation-stage function is untouched and still computes its own intervals."""
    from expected_goals_lab import diagnostics

    assert "interval_log_loss" in inspect.getsource(diagnostics.league_rows)
    assert fe.REAL_LOCKED.file_hashes["src/expected_goals_lab/diagnostics.py"] == (
        fe.normalised_sha256(REPO / "src/expected_goals_lab/diagnostics.py")
    )


# --- D. protocol wording --------------------------------------------------------------------------


def test_protocol_states_the_s06a_rules():
    text = (REPO / "docs/FINAL_TEST_PROTOCOL.md").read_text(encoding="utf-8")
    for phrase in (
        "scoring occasion",
        "not to a single pass over the shot table",
        "neither computed nor reported",
        "refuses any different value before a held-out row is read",
        "`completed` is appended only after both aggregate files",
        "supplies a retry reason",
        "not permission to change any model",
    ):
        assert phrase in text, phrase
    assert "the test partition read, once" not in text
    assert dataclasses.is_dataclass(REAL_LOCKED)
