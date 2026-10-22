"""Frozen final held-out evaluation: metadata-only preflight (S06) and one-shot execution (S07).

    uv run python -m expected_goals_lab.final_evaluation               # preflight only
    uv run python -m expected_goals_lab.final_evaluation --dry-run     # same

The execution mode needs BOTH ``--execute`` and ``--confirm-final-test``, and is switched off for
the real data in this repository until the S07 review (``REAL_EXECUTION_ENABLED``). Protocol:
docs/FINAL_TEST_PROTOCOL.md.

Preflight reads no shot row. It hashes files byte for byte and reads manifests and published
counts. Execution, whose procedure is exercised only on synthetic fixtures in S06, runs in this
fixed order, stopping at the first failure:

1. both confirmations present, bootstrap settings equal to the locked ones, real-data gate
   open, preflight passes, execution log and existing outputs allow a run;
2. split identity recomputed from the identity columns and compared with the locked values;
3. development partition (original train + validation) loaded through the guarded loaders;
4. the three frozen recipes refitted on the development partition only, and their parameters
   verified against the locked recipe hashes;
5. only then is the sealed test partition read, through a ``FinalTestAuthorisation``;
6. scores, paired match-cluster bootstrap, expected goals, calibration and descriptive league
   tables (no per-league intervals);
7. both aggregate output files are written to temporary files and moved into place, and only
   then is a ``completed`` entry appended to the execution log.

Results are aggregates only. They are not a basis for choosing, tuning or changing any model.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from expected_goals_lab import scoring
from expected_goals_lab.baselines import CONSTANT, DISTANCE_PLUS_ANGLE, fit_baselines, make_pipeline
from expected_goals_lab.candidate_data import (
    CandidatePartition,
    load_body_parts,
    load_final_test_body_parts,
)
from expected_goals_lab.common import table
from expected_goals_lab.dataset import (
    DEFAULT_CSV,
    OPEN_SPLITS,
    ContractError,
    FinalTestAuthorisation,
    Partition,
    RowReader,
    SealedSplitError,
    check_manifest,
    iter_csv_rows,
    load_final_test_partition,
    load_manifest,
    load_partitions,
    read_identity,
    sha256_bytes_of,
)
from expected_goals_lab.diagnostics import MIN_CLASS_FOR_AUC, calibration_rows
from expected_goals_lab.evaluate_baselines import LEAGUE_NAMES
from expected_goals_lab.evaluate_nonlinear import git_state, reading
from expected_goals_lab.final_lock import REAL_LOCKED, LockedIdentities
from expected_goals_lab.nonlinear import (
    FACTORIES,
    NONLINEAR_LOCATION_BODY_PART,
    check_probabilities,
    fit_candidates,
)
from expected_goals_lab.scoring import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    Interval,
    brier_terms,
    cluster_bootstrap_differences,
    log_loss_terms,
)
from expected_goals_lab.splits import (
    assign_date_splits,
    split_of_events,
    summarise,
    verify_split,
    write_assignment,
)

PRIMARY, BENCHMARK = NONLINEAR_LOCATION_BODY_PART, DISTANCE_PLUS_ANGLE
REFERENCE_ONLY = CONSTANT  # predeclared: included, fitted on development labels only
DEVELOPMENT, TEST = "development", "test"
# Switched on only by the reviewed S07 change; until then real-data execution is refused.
REAL_EXECUTION_ENABLED = False

FROZEN_FILES = (
    "pyproject.toml",
    "uv.lock",
    "docs/MODEL_SELECTION_DECISION.md",
    "src/expected_goals_lab/geometry.py",
    "src/expected_goals_lab/wyscout.py",
    "src/expected_goals_lab/wyscout_shots.py",
    "src/expected_goals_lab/shot_table.py",
    "src/expected_goals_lab/shot_diagnostics.py",
    "src/expected_goals_lab/build_shots.py",
    "src/expected_goals_lab/dataset.py",
    "src/expected_goals_lab/splits.py",
    "src/expected_goals_lab/baselines.py",
    "src/expected_goals_lab/scoring.py",
    "src/expected_goals_lab/candidate_data.py",
    "src/expected_goals_lab/nonlinear.py",
    "src/expected_goals_lab/evaluate_baselines.py",
    "src/expected_goals_lab/evaluate_nonlinear.py",
    "src/expected_goals_lab/diagnostics.py",
    "src/expected_goals_lab/audit.py",
)
LOCKED_PACKAGES = ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl")


class PreflightFailure(RuntimeError):
    """A frozen identity, hash, recipe or environment check failed. Stop and request review."""


class ExecutionRefused(RuntimeError):
    """Execution is not permitted: missing confirmation, closed gate, or a prior completed run."""


class LogCorrupt(ExecutionRefused):
    """The execution log cannot be read as a sequence of known outcomes. Fail closed."""


@dataclass(frozen=True)
class Paths:
    root: Path
    csv: Path
    csv_manifest: Path
    assignment: Path
    split_manifest: Path
    protocol: Path
    decision: Path
    lockfile: Path
    pyproject: Path
    report: Path
    log: Path
    summary: Path


def default_paths(root: Path = Path(".")) -> Paths:
    processed = root / DEFAULT_CSV.parent
    return Paths(
        root=root,
        csv=root / DEFAULT_CSV,
        csv_manifest=(root / DEFAULT_CSV).with_suffix(".manifest.json"),
        assignment=processed / "wyscout_split_assignment.csv",
        split_manifest=processed / "wyscout_split_manifest.json",
        protocol=root / "docs/FINAL_TEST_PROTOCOL.md",
        decision=root / "docs/MODEL_SELECTION_DECISION.md",
        lockfile=root / "uv.lock",
        pyproject=root / "pyproject.toml",
        report=root / "reports/final_test_evaluation.md",
        log=root / "reports/final_test_execution_log.jsonl",
        summary=processed / "final_test_summary.json",
    )


# --- hashing, recipes, environment ---------------------------------------------------------------


def normalised_sha256(path: Path) -> str:
    """SHA-256 of a text file with LF line endings, so a CRLF checkout hashes identically."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _describe(obj) -> object:
    """A canonical, comparable description of a (fitted or unfitted) estimator tree."""
    if isinstance(obj, Pipeline):
        return {"pipeline": [[name, _describe(step)] for name, step in obj.steps]}
    if isinstance(obj, ColumnTransformer):
        fitted = getattr(obj, "transformers_", None)
        return {
            "columns": [
                [name, _describe(transformer), list(columns)]
                for name, transformer, columns in (fitted or obj.transformers)
                if transformer != "drop"
            ],
            "remainder": str(obj.remainder),
        }
    return {"class": type(obj).__name__, "params": obj.get_params(deep=False)}


def recipe_sha256(pipeline) -> str:
    canonical = json.dumps(_describe(pipeline), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def locked_package_versions(lockfile: Path) -> dict[str, str]:
    packages = tomllib.loads(lockfile.read_text(encoding="utf-8")).get("package", [])
    return {p["name"]: p["version"] for p in packages if p["name"] in LOCKED_PACKAGES}


def installed_package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in LOCKED_PACKAGES}


OUTCOMES = ("started", "failed", "completed")


def read_log(path: Path) -> list[dict]:
    """The log entries in order. A line that is not a known outcome makes the log corrupt."""
    if not path.exists():
        return []
    entries = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LogCorrupt(f"execution log line {number} is not valid JSON") from exc
        if not isinstance(entry, dict) or entry.get("outcome") not in OUTCOMES:
            raise LogCorrupt(f"execution log line {number} has no known outcome")
        entries.append(entry)
    return entries


def log_status(entries: list[dict]) -> str:
    """ "none", "completed", "failed" or "interrupted" (a started attempt with no outcome)."""
    if any(e["outcome"] == "completed" for e in entries):
        return "completed"
    if not entries:
        return "none"
    return "failed" if entries[-1]["outcome"] == "failed" else "interrupted"


def append_log(path: Path, entry: dict) -> None:
    """Append one line and force it to disk, so a crash cannot lose an earlier entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_outputs_atomically(outputs: list[tuple[Path, str]]) -> None:
    """Write every output to a temporary file beside its target, then move each into place.

    Either all targets are replaced or none is left behind: on any failure the temporary files
    are removed and a target already replaced in this call is deleted again.
    """
    pending: list[tuple[Path, Path]] = []
    replaced: list[Path] = []
    try:
        for target, text in outputs:
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                dir=target.parent, prefix=f"{target.name}.", suffix=".tmp"
            )
            pending.append((Path(name), target))
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, target in pending:
            os.replace(temporary, target)
            replaced.append(target)
    except BaseException:
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)
        for target in replaced:
            target.unlink(missing_ok=True)
        raise


# --- metadata-only preflight ---------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class PreflightReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]


def preflight(locked: LockedIdentities, paths: Paths) -> PreflightReport:
    """Verify every frozen identity without reading a single shot row.

    Shot data are touched only by ``sha256_bytes_of`` (a binary byte hash). The split assignment
    file is likewise only hashed. Everything else is JSON, text or version metadata.
    """
    report = PreflightReport()

    def add(name: str, ok: bool, detail: str = "") -> None:
        report.checks.append(Check(name, bool(ok), detail))

    required = [
        paths.csv,
        paths.csv_manifest,
        paths.assignment,
        paths.split_manifest,
        paths.protocol,
        paths.decision,
        paths.lockfile,
        paths.pyproject,
    ]
    missing = [str(p) for p in required if not p.exists()]
    add("Required files present", not missing, f"missing: {missing}" if missing else "all present")
    if missing:
        return report

    csv_sha = sha256_bytes_of(paths.csv)
    add("Shot table byte hash equals the locked SHA-256", csv_sha == locked.csv_sha256, csv_sha)
    manifest_sha = sha256_bytes_of(paths.csv_manifest)
    add(
        "S02 manifest file hash equals the locked value",
        manifest_sha == locked.csv_manifest_sha256,
        manifest_sha,
    )
    try:
        manifest = load_manifest(paths.csv_manifest)
        check_manifest(manifest)
        add(
            "S02 manifest schema, roles, allowlist and target are as frozen",
            manifest["csv_sha256"] == locked.csv_sha256
            and manifest["rows"] == locked.rows
            and manifest["eligible_rows"] == locked.eligible_rows,
            f"rows {manifest['rows']}, eligible {manifest['eligible_rows']}",
        )
    except (ContractError, KeyError, ValueError) as exc:
        add("S02 manifest schema, roles, allowlist and target are as frozen", False, str(exc))

    assignment_sha = sha256_bytes_of(paths.assignment)
    add(
        "Split assignment file hash equals the locked value",
        assignment_sha == locked.assignment_sha256,
        assignment_sha,
    )
    split_sha = sha256_bytes_of(paths.split_manifest)
    add(
        "Split manifest file hash equals the locked value",
        split_sha == locked.split_manifest_sha256,
        split_sha,
    )
    try:
        recorded = json.loads(paths.split_manifest.read_text(encoding="utf-8"))
        splits = recorded["splits"]
        same = (
            recorded["assignment_csv_sha256"] == locked.assignment_sha256
            and recorded["input_csv_sha256"] == locked.csv_sha256
            and recorded["input_rows"] == locked.rows
            and recorded["input_eligible_rows"] == locked.eligible_rows
        )
        for split in ("train", "validation", "test"):
            same = (
                same
                and splits[split]["rows"] == locked.split_rows[split]
                and splits[split]["matches"] == locked.split_matches[split]
                and splits[split]["dates"] == locked.split_dates[split]
                and (splits[split]["first_date"], splits[split]["last_date"])
                == tuple(locked.boundaries[split])
            )
        add("Split manifest counts and date boundaries equal the locked values", same)
        ordered = (
            splits["train"]["last_date"] < splits["validation"]["first_date"]
            and splits["validation"]["last_date"] < splits["test"]["first_date"]
        )
        add(
            "Split windows are strictly ordered in time (no date, hence no match, in two splits)",
            ordered,
        )
        add(
            "Split sizes add up to the eligible rows",
            sum(splits[s]["rows"] for s in splits) == locked.eligible_rows,
        )
    except (KeyError, ValueError, TypeError) as exc:
        add("Split manifest counts and date boundaries equal the locked values", False, repr(exc))

    add(
        "Protocol file hash equals the locked value",
        normalised_sha256(paths.protocol) == locked.protocol_sha256,
    )
    add(
        "Model decision file hash equals the locked value",
        normalised_sha256(paths.decision) == locked.decision_sha256,
    )
    add(
        "Dependency lockfile hash equals the locked value",
        normalised_sha256(paths.lockfile) == locked.lockfile_sha256,
    )
    add(
        "pyproject hash equals the locked value",
        normalised_sha256(paths.pyproject) == locked.pyproject_sha256,
    )

    pinned = locked_package_versions(paths.lockfile)
    installed = installed_package_versions()
    add(
        "Installed packages equal the versions pinned in uv.lock and in the lock",
        pinned == locked.packages == installed,
        f"installed {installed}",
    )

    drifted = []
    for relative, expected in locked.file_hashes.items():
        path = paths.root / relative
        if not path.exists() or normalised_sha256(path) != expected:
            drifted.append(relative)
    add(
        "Frozen implementation inventory is unchanged",
        not drifted,
        f"changed: {drifted}" if drifted else f"{len(locked.file_hashes)} files",
    )

    primary, benchmark = recipe_sha256(FACTORIES[PRIMARY]()), recipe_sha256(make_pipeline())
    add(
        "Primary model recipe equals the locked recipe",
        primary == locked.primary_recipe_sha256,
        primary,
    )
    add(
        "Benchmark model recipe equals the locked recipe",
        benchmark == locked.benchmark_recipe_sha256,
        benchmark,
    )
    add(
        "Bootstrap seed and resamples equal the locked values",
        (BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES)
        == (locked.bootstrap_seed, locked.bootstrap_resamples),
        f"seed {BOOTSTRAP_SEED}, resamples {BOOTSTRAP_RESAMPLES}",
    )

    try:
        status = log_status(read_log(paths.log))
    except LogCorrupt as exc:
        add("Execution log is readable and records no completed evaluation", False, str(exc))
    else:
        add(
            "Execution log is readable and records no completed evaluation",
            status != "completed",
            "a completed evaluation already exists" if status == "completed" else status,
        )
    return report


def format_preflight(report: PreflightReport, locked: LockedIdentities, paths: Paths) -> str:
    lines = ["FINAL EVALUATION PREFLIGHT (metadata only: no shot row was read)", ""]
    for check in report.checks:
        mark = "PASS" if check.ok else "FAIL"
        detail = f"  [{check.detail}]" if check.detail and len(check.detail) < 90 else ""
        lines.append(f"[{mark}] {check.name}{detail}")
    gate = "OPEN" if REAL_EXECUTION_ENABLED else "CLOSED (S06: preflight only)"
    lines += [
        "",
        f"Real-data execution gate: {gate}",
        "Published windows (first to last UTC date): "
        + ", ".join(f"{s} {b[0]} to {b[1]}" for s, b in locked.boundaries.items()),
        f"PREFLIGHT {'PASSED' if report.ok else 'FAILED'}",
    ]
    try:
        status = log_status(read_log(paths.log))
    except LogCorrupt:
        status = "corrupt"
    if status == "completed":
        lines.append("FINAL TEST ALREADY EXECUTED")
    elif status == "corrupt":
        lines.append("EXECUTION LOG UNREADABLE: review required before any execution")
    elif status in ("failed", "interrupted"):
        lines.append(
            f"A PREVIOUS ATTEMPT {status.upper()}: a recorded --retry-reason is required to retry"
        )
    else:
        lines.append("FINAL TEST NOT YET OPENED")
    return "\n".join(lines)


# --- execution -----------------------------------------------------------------------------------


def require_confirmations(execute: bool, confirm_final_test: bool) -> None:
    if not (execute and confirm_final_test):
        raise ExecutionRefused("final-test execution needs both --execute and --confirm-final-test")


def combine_partitions(parts: list[Partition], split: str) -> Partition:
    cat = np.concatenate
    return Partition(
        split=split,
        event_ids=cat([p.event_ids for p in parts]),
        match_ids=cat([p.match_ids for p in parts]),
        competition_ids=cat([p.competition_ids for p in parts]),
        dates=cat([p.dates for p in parts]),
        features={k: cat([p.features[k] for p in parts]) for k in parts[0].features},
        x_pct=cat([p.x_pct for p in parts]),
        y_pct=cat([p.y_pct for p in parts]),
        y=cat([p.y for p in parts]),
    )


def combine_candidates(parts: list[CandidatePartition], split: str) -> CandidatePartition:
    return CandidatePartition(
        base=combine_partitions([p.base for p in parts], split),
        body_part=np.concatenate([p.body_part for p in parts]),
    )


@dataclass(frozen=True)
class FinalScores:
    shots: int
    goals: int
    log_loss: float
    brier: float
    auc: float | None  # None when the cohort has one outcome class only
    expected_goals: float


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    return scoring.auc(y, p) if 0 < int(np.sum(y)) < len(y) else None


def score_models(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    match_ids: np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, FinalScores], dict[str, Interval | None], bool]:
    """Scores per model plus paired match-cluster intervals for primary minus benchmark.

    Intervals resample whole matches, never shots. They are None (and the flag False) when
    fewer than two matches are available, because a one-cluster bootstrap has no variation.
    """
    scores = {
        name: FinalScores(
            shots=int(len(y)),
            goals=int(np.sum(y)),
            log_loss=scoring.log_loss(y, p),
            brier=scoring.brier_score(y, p),
            auc=safe_auc(y, p),
            expected_goals=float(np.sum(p)),
        )
        for name, p in predictions.items()
    }
    valid = len(set(match_ids.tolist())) >= 2
    intervals: dict[str, Interval | None] = {"log_loss": None, "brier": None}
    if valid:
        for metric, fn in (("log_loss", log_loss_terms), ("brier", brier_terms)):
            terms = {PRIMARY: fn(y, predictions[PRIMARY]), BENCHMARK: fn(y, predictions[BENCHMARK])}
            intervals[metric] = cluster_bootstrap_differences(
                match_ids, terms, [(PRIMARY, BENCHMARK)], resamples, seed
            )[(PRIMARY, BENCHMARK)]
    return scores, intervals, valid


def league_descriptives(
    part: Partition, p_benchmark: np.ndarray, p_primary: np.ndarray, names: dict[int, str]
) -> list[dict]:
    """Per-league counts and point scores, descriptive only.

    No bootstrap, interval or test is computed for a league (protocol section 5). AUC is given
    only with at least ``MIN_CLASS_FOR_AUC`` goals and as many non-goals.
    """
    unknown = sorted(set(part.competition_ids.tolist()) - set(names))
    if unknown:
        raise ContractError(f"test rows belong to leagues without a verified name: {unknown}")
    rows = []
    for competition_id, name in names.items():
        mask = part.competition_ids == competition_id
        n, y = int(mask.sum()), part.y[mask]
        goals = int(y.sum())
        row = {
            "competition_id": int(competition_id),
            "name": name,
            "shots": n,
            "goals": goals,
            "matches": len(set(part.match_ids[mask].tolist())),
            "benchmark": None,
            "primary": None,
            "diff_log_loss": None,
            "diff_brier": None,
        }
        if n:
            for key, p in (("benchmark", p_benchmark[mask]), ("primary", p_primary[mask])):
                row[key] = {
                    "log_loss": scoring.log_loss(y, p),
                    "brier": scoring.brier_score(y, p),
                    "auc": safe_auc(y, p) if min(goals, n - goals) >= MIN_CLASS_FOR_AUC else None,
                }
            row["diff_log_loss"] = row["primary"]["log_loss"] - row["benchmark"]["log_loss"]
            row["diff_brier"] = row["primary"]["brier"] - row["benchmark"]["brier"]
        rows.append(row)
    if sum(r["shots"] for r in rows) != len(part) or sum(r["goals"] for r in rows) != int(
        part.y.sum()
    ):
        raise ContractError("league denominators do not add up to the test totals")
    return rows


@dataclass
class FinalEvaluation:
    provenance: dict
    development: dict
    test: dict
    scores: dict[str, FinalScores]
    intervals: dict[str, Interval | None]
    intervals_valid: bool
    calibration: dict[str, list[dict]]
    leagues: list[dict]
    attempts: list[dict]
    events: list[str] = field(default_factory=list)  # order of milestones, for audit and tests


def _assert_locked_split(identity, plan, locked: LockedIdentities) -> dict:
    """Recompute the split from identity columns and compare it with the locked values."""
    verify_split(identity, plan)
    with tempfile.TemporaryDirectory() as scratch:
        assignment_sha = write_assignment(Path(scratch) / "assignment.csv", identity, plan)
    if assignment_sha != locked.assignment_sha256:
        raise PreflightFailure("the recomputed split assignment differs from the locked one")
    summary = summarise(identity, plan)
    for split in ("train", "validation", "test"):
        got = (summary[split]["rows"], summary[split]["matches"], summary[split]["dates"])
        want = (locked.split_rows[split], locked.split_matches[split], locked.split_dates[split])
        bounds = (summary[split]["first_date"], summary[split]["last_date"])
        if got != want or bounds != tuple(locked.boundaries[split]):
            raise PreflightFailure(f"{split} counts or dates differ from the locked split")
    return summary


def execute_final_evaluation(
    locked: LockedIdentities,
    paths: Paths,
    *,
    execute: bool,
    confirm_final_test: bool,
    retry_reason: str | None = None,
    rows: RowReader = iter_csv_rows,
    resamples: int | None = None,
    seed: int | None = None,
) -> FinalEvaluation:
    """The one-time evaluation. Never called against real data in S06 or S06A.

    The bootstrap always uses ``locked.bootstrap_resamples`` and ``locked.bootstrap_seed``.
    ``resamples`` and ``seed`` exist only so synthetic tests can run quickly; against the real
    lock any different value is refused before anything is read or logged.

    On success both aggregate files exist and the log ends with ``completed``. On failure no
    ``completed`` entry is written and any output written by this attempt is removed.
    """
    require_confirmations(execute, confirm_final_test)
    if locked is REAL_LOCKED and (
        (resamples is not None and resamples != locked.bootstrap_resamples)
        or (seed is not None and seed != locked.bootstrap_seed)
    ):
        raise ExecutionRefused(
            "the bootstrap seed and number of resamples are frozen for the real evaluation"
        )
    if locked is REAL_LOCKED and not REAL_EXECUTION_ENABLED:
        raise ExecutionRefused("real-data execution is disabled until the S07 review")
    resamples = locked.bootstrap_resamples if resamples is None else resamples
    seed = locked.bootstrap_seed if seed is None else seed
    report = preflight(locked, paths)
    if not report.ok:
        failed = "; ".join(c.name for c in report.failures())
        raise PreflightFailure(f"preflight failed, stopping before any test access: {failed}")

    status = log_status(read_log(paths.log))
    if status == "completed":
        raise ExecutionRefused(
            "a final evaluation was already completed; a rerun is not new evidence"
        )
    reason = (retry_reason or "").strip()
    stale = [p.name for p in (paths.report, paths.summary) if p.exists()]
    if status == "failed" and not reason:
        raise ExecutionRefused("a previous attempt failed: a retry needs --retry-reason")
    if status == "interrupted" and not reason:
        raise ExecutionRefused(
            "a previous attempt started but recorded no outcome (interrupted): diagnose it, "
            "then retry with --retry-reason"
        )
    if stale and not reason:
        raise ExecutionRefused(
            f"output files {stale} exist without a completed log entry: "
            "inspect them, then retry with --retry-reason"
        )
    state = git_state()
    attempt = {
        "started_utc": f"{datetime.now(UTC):%Y-%m-%d %H:%M:%S}",
        "commit": state["commit"],
        "uncommitted_changes": state["uncommitted_changes"],
        "protocol_sha256": locked.protocol_sha256,
        "retry_reason": reason or None,
        "previous_outcome": status if status != "none" else None,
        "stale_outputs": stale,
    }
    append_log(paths.log, {**attempt, "outcome": "started"})
    events: list[str] = []
    try:
        evaluation = _run(locked, paths, rows, resamples, seed, events, state)
        evaluation.attempts = read_log(paths.log)
        report_text = render_final_report(evaluation, locked, paths.csv)
        summary_text = json.dumps(summary_json(evaluation), indent=2, sort_keys=True) + "\n"
        write_outputs_atomically([(paths.report, report_text), (paths.summary, summary_text)])
        events.append("outputs_written")
        append_log(
            paths.log,
            {
                **attempt,
                "outcome": "completed",
                "report_sha256": sha256_text(report_text),
                "summary_sha256": sha256_text(summary_text),
            },
        )
    except Exception as exc:
        if "outputs_written" in events:  # written, but not recorded as complete: remove them
            for output in (paths.report, paths.summary):
                output.unlink(missing_ok=True)
        append_log(paths.log, {**attempt, "outcome": "failed", "error_type": type(exc).__name__})
        raise
    return evaluation


def _run(locked, paths, rows, resamples, seed, events, state) -> FinalEvaluation:
    identity = read_identity(paths.csv, paths.csv_manifest, rows)
    if identity.csv_sha256 != locked.csv_sha256:
        raise PreflightFailure("the shot table hash differs from the locked one")
    plan = assign_date_splits(identity.dates.tolist())
    _assert_locked_split(identity, plan, locked)
    split_of = split_of_events(identity, plan)
    events.append("identity_verified")

    base = load_partitions(paths.csv, split_of, OPEN_SPLITS, rows)
    parts = load_body_parts(paths.csv, base, split_of, OPEN_SPLITS, rows)
    development = combine_candidates([parts["train"], parts["validation"]], DEVELOPMENT)
    test_event_ids = {e for e, s in split_of.items() if s == TEST}
    if test_event_ids & set(development.base.event_ids.tolist()):
        raise PreflightFailure("development and test partitions share an event")
    if set(development.base.dates.tolist()) & {d for d, s in plan.date_split.items() if s == TEST}:
        raise PreflightFailure("development and test partitions share a calendar date")
    events.append("development_loaded")

    dev_base = development.base
    reference = fit_baselines(dev_base, fit_split=DEVELOPMENT)
    primary = fit_candidates(development, {PRIMARY: FACTORIES[PRIMARY]}, fit_split=DEVELOPMENT)
    if recipe_sha256(primary[PRIMARY].pipeline) != locked.primary_recipe_sha256:
        raise PreflightFailure("the fitted primary pipeline differs from the locked recipe")
    if recipe_sha256(reference[BENCHMARK].estimator) != locked.benchmark_recipe_sha256:
        raise PreflightFailure("the fitted benchmark pipeline differs from the locked recipe")
    events.append("development_models_fitted")

    authorisation = FinalTestAuthorisation.grant(True, True, locked.protocol_sha256)
    events.append("test_opened")
    test_base = load_final_test_partition(paths.csv, split_of, authorisation, rows)
    test = load_final_test_body_parts(paths.csv, test_base, split_of, authorisation, rows)
    if set(test.base.event_ids.tolist()) != test_event_ids or len(test) != locked.split_rows[TEST]:
        raise PreflightFailure("the test partition does not match the locked split")
    events.append("test_loaded")

    predictions = {
        PRIMARY: primary[PRIMARY].predict_proba(test),
        BENCHMARK: check_probabilities(reference[BENCHMARK].predict_proba(test.base), BENCHMARK),
        REFERENCE_ONLY: check_probabilities(reference[CONSTANT].predict_proba(test.base), CONSTANT),
    }
    y, match_ids = test.base.y, test.base.match_ids
    scores, intervals, valid = score_models(y, predictions, match_ids, resamples, seed)
    leagues = league_descriptives(
        test.base, predictions[BENCHMARK], predictions[PRIMARY], LEAGUE_NAMES
    )
    calibration = {
        PRIMARY: calibration_rows(y, predictions[PRIMARY]),
        BENCHMARK: calibration_rows(y, predictions[BENCHMARK]),
    }
    events.append("test_scored")
    dates = sorted(set(test.base.dates.tolist()))
    return FinalEvaluation(
        provenance={
            **state,
            "csv_sha256": identity.csv_sha256,
            "assignment_sha256": locked.assignment_sha256,
            "protocol_sha256": locked.protocol_sha256,
            "lockfile_sha256": locked.lockfile_sha256,
            "scikit_learn": sklearn.__version__,
            "bootstrap": {"seed": seed, "resamples": resamples},
        },
        development={
            "shots": len(dev_base),
            "goals": int(dev_base.y.sum()),
            "matches": len(set(dev_base.match_ids.tolist())),
            "dates": len(set(dev_base.dates.tolist())),
            "goal_rate": float(dev_base.y.mean()),
        },
        test={
            "shots": len(test),
            "goals": int(y.sum()),
            "matches": len(set(match_ids.tolist())),
            "first_date": dates[0],
            "last_date": dates[-1],
            "dates": len(dates),
        },
        scores=scores,
        intervals=intervals,
        intervals_valid=valid,
        calibration=calibration,
        leagues=leagues,
        attempts=[],
        events=events,
    )


# --- report (aggregates only; wording follows the numbers) ----------------------------------------


def _num(x: float | None, digits: int) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _signed(x: float, digits: int) -> str:
    return f"{x:+.{digits}f}"


def primary_reading(ev: FinalEvaluation) -> str:
    """The pre-stated reading rule (protocol section 5), whichever way the numbers fall."""
    ll, br = ev.intervals["log_loss"], ev.intervals["brier"]
    if ll is None or br is None:
        return "no interval available (fewer than two test matches): no reading is made"
    return reading(ll, br, BENCHMARK)


def render_final_report(ev: FinalEvaluation, locked: LockedIdentities, csv_path: Path) -> str:
    s = ev.scores
    out: list[str] = []
    add = out.extend
    primary, benchmark, constant = s[PRIMARY], s[BENCHMARK], s[REFERENCE_ONLY]
    add(
        [
            "# Final held-out test evaluation",
            "",
            "**These results come from the single pre-registered final evaluation "
            f"(`docs/FINAL_TEST_PROTOCOL.md`, SHA-256 `{locked.protocol_sha256}`). "
            "They are not a basis for choosing, tuning or changing any model, feature, cohort "
            "or date.** Rerunning unchanged code is a reproducibility check, not new "
            "independent evidence.",
            "",
            "The test window was held out from scoring, not from all outcome information: "
            "its aggregate goal count was already inferable from published cohort totals. "
            "Model selection was exploratory and used the validation window. This is one "
            "season, five leagues, one chronological cut.",
            "",
            f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Data: Pappalardo & Massucco "
            "(2019), *Soccer match event dataset*, figshare, CC BY 4.0, "
            "doi:10.6084/m9.figshare.c.4415000.v5; Pappalardo et al. (2019), Scientific Data "
            "6:236, doi:10.1038/s41597-019-0247-7; data collected by Wyscout; processed for "
            "this project.",
            "",
            "## Identity",
            "",
            table(
                ["Item", "Value"],
                [
                    [
                        "Shot table",
                        f"`{csv_path.as_posix()}` SHA-256 `{ev.provenance['csv_sha256']}`",
                    ],
                    ["Split assignment SHA-256", f"`{ev.provenance['assignment_sha256']}`"],
                    ["Lockfile SHA-256", f"`{ev.provenance['lockfile_sha256']}`"],
                    [
                        "Git commit",
                        f"`{ev.provenance['commit']}`"
                        + (
                            " (uncommitted changes)" if ev.provenance["uncommitted_changes"] else ""
                        ),
                    ],
                    ["scikit-learn", ev.provenance["scikit_learn"]],
                    [
                        "Bootstrap",
                        f"{ev.provenance['bootstrap']['resamples']} match-level resamples, "
                        f"seed {ev.provenance['bootstrap']['seed']}",
                    ],
                ],
            ),
            "",
            "Development (original train + validation, used only to fit): "
            f"{ev.development['shots']:,} shots, {ev.development['goals']:,} goals, "
            f"{ev.development['matches']:,} matches on {ev.development['dates']} dates. "
            f"Test: {ev.test['shots']:,} shots, {ev.test['goals']:,} goals, "
            f"{ev.test['matches']:,} matches on {ev.test['dates']} dates "
            f"({ev.test['first_date']} to {ev.test['last_date']}).",
            "",
            "## Result",
            "",
            f"Primary comparison (`{PRIMARY}` versus `{BENCHMARK}`): {primary_reading(ev)}.",
            "",
            table(
                ["Model", "Shots", "Goals", "Log loss", "Brier", "ROC AUC", "Expected goals"],
                [
                    [
                        name,
                        sc.shots,
                        sc.goals,
                        _num(sc.log_loss, 4),
                        _num(sc.brier, 5),
                        _num(sc.auc, 3),
                        _num(sc.expected_goals, 1),
                    ]
                    for name, sc in (
                        (PRIMARY, primary),
                        (BENCHMARK, benchmark),
                        ("constant_development_rate (reference only)", constant),
                    )
                ],
            ),
            "",
            "Lower log loss and Brier score are better; higher AUC is better. Log loss is the "
            "primary metric, Brier the co-primary supporting score, AUC secondary.",
            "",
        ]
    )
    ll, br = ev.intervals["log_loss"], ev.intervals["brier"]
    if ll is not None and br is not None:
        add(
            [
                "Paired match-cluster bootstrap, primary minus benchmark (exploratory 95% "
                "percentile intervals; resampling whole matches):",
                "",
                table(
                    ["Metric", "Difference", "95% interval"],
                    [
                        [
                            "Log loss",
                            _signed(ll.estimate, 4),
                            f"[{_signed(ll.low, 4)}, {_signed(ll.high, 4)}]",
                        ],
                        [
                            "Brier",
                            _signed(br.estimate, 5),
                            f"[{_signed(br.low, 5)}, {_signed(br.high, 5)}]",
                        ],
                    ],
                ),
                "",
            ]
        )
    else:
        add(["No interval was computed because fewer than two test matches were available.", ""])
    add(
        [
            "## Expected versus observed goals",
            "",
            table(
                ["Model", "Expected goals", "Observed goals", "Observed / expected"],
                [
                    [
                        name,
                        _num(sc.expected_goals, 1),
                        sc.goals,
                        _num(sc.goals / sc.expected_goals if sc.expected_goals else None, 3),
                    ]
                    for name, sc in (
                        (PRIMARY, primary),
                        (BENCHMARK, benchmark),
                        ("constant_development_rate", constant),
                    )
                ],
            ),
            "",
            "## Calibration (ten equal-frequency bins; no calibration fitted)",
            "",
            "The Wilson intervals treat shots as independent and so understate uncertainty.",
            "",
        ]
    )
    for name in (PRIMARY, BENCHMARK):
        add(
            [
                f"**{name}**",
                "",
                table(
                    ["Bin", "Shots", "Goals", "Mean prediction", "Observed rate", "95% interval"],
                    [
                        [
                            i + 1,
                            b["shots"],
                            b["goals"],
                            f"{b['mean_prediction']:.4f}",
                            f"{b['observed']:.4f}",
                            f"[{b['low']:.4f}, {b['high']:.4f}]",
                        ]
                        for i, b in enumerate(ev.calibration[name])
                    ],
                ),
                "",
            ]
        )
    add(
        [
            "## By league (descriptive only)",
            "",
            "Differences are primary minus benchmark (negative favours the primary). Per-league "
            "bootstrap intervals are neither computed nor reported, and no threshold or "
            "significance statement is attached; denominators are shown. AUC appears only with "
            "at least 10 goals and 10 non-goals.",
            "",
            table(
                [
                    "League",
                    "Shots",
                    "Goals",
                    "Matches",
                    "LL benchmark",
                    "LL primary",
                    "LL diff",
                    "Brier benchmark",
                    "Brier primary",
                    "Brier diff",
                    "AUC benchmark",
                    "AUC primary",
                ],
                [
                    [
                        r["name"],
                        r["shots"],
                        r["goals"],
                        r["matches"],
                        _num(r["benchmark"] and r["benchmark"]["log_loss"], 4),
                        _num(r["primary"] and r["primary"]["log_loss"], 4),
                        _num(r["diff_log_loss"], 4),
                        _num(r["benchmark"] and r["benchmark"]["brier"], 5),
                        _num(r["primary"] and r["primary"]["brier"], 5),
                        _num(r["diff_brier"], 5),
                        _num(r["benchmark"] and r["benchmark"]["auc"], 3),
                        _num(r["primary"] and r["primary"]["auc"], 3),
                    ]
                    for r in ev.leagues
                ],
            ),
            "",
            "## Limits",
            "",
            "- One season (2017/18), five leagues, one chronological cut; no claim about "
            "other seasons or leagues.",
            "- The same teams and players appear in development and test.",
            "- Labels and the body-part tag come from human provider tagging; Wyscout tag 403 is "
            '"head/body", not a confirmed header.',
            "- The match bootstrap ignores dependence between matches of one team or league "
            "and the selection of the primary model among candidates.",
            "- Known data issues are frozen in the protocol: one match with a score "
            "discrepancy is quarantined, and shots recorded very far from the attacked goal "
            "(x below 20% of pitch length) are quarantined as suspect.",
            "- If the primary model did worse than the benchmark, that is the result; the "
            "choice is not revisited.",
            "",
            "## Execution record",
            "",
            f"{len(ev.attempts)} execution-log entries existed when this report was rendered. "
            "The `completed` entry is appended only after this report and its JSON summary are "
            "both in place; retries carry a recorded reason in "
            "`reports/final_test_execution_log.jsonl`.",
        ]
    )
    return "\n".join(out) + "\n"


def summary_json(ev: FinalEvaluation) -> dict:
    """Aggregates only."""
    return {
        "provenance": ev.provenance,
        "development": ev.development,
        "test": ev.test,
        "scores": {k: vars(v) for k, v in ev.scores.items()},
        "intervals": {k: (vars(v) if v else None) for k, v in ev.intervals.items()},
        "calibration": ev.calibration,
        "leagues": ev.leagues,
    }


# --- command line ------------------------------------------------------------


def main(
    argv: list[str] | None = None,
    *,
    locked: LockedIdentities | None = None,
    paths: Paths | None = None,
    rows: RowReader = iter_csv_rows,
    resamples: int | None = None,
) -> int:
    """Command line. ``resamples`` is a synthetic-test hook; the real lock refuses any change."""
    parser = argparse.ArgumentParser(
        description="Final held-out evaluation: preflight or execute.", allow_abbrev=False
    )
    parser.add_argument("--dry-run", action="store_true", help="metadata-only preflight (default)")
    parser.add_argument(
        "--execute", action="store_true", help="run the final evaluation (S07 only)"
    )
    parser.add_argument("--confirm-final-test", action="store_true", help="second required switch")
    parser.add_argument(
        "--retry-reason", default=None, help="required to retry after a failed attempt"
    )
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    locked = locked or REAL_LOCKED
    paths = paths or default_paths()

    if args.dry_run and (args.execute or args.confirm_final_test):
        print(
            "error: --dry-run cannot be combined with --execute or --confirm-final-test",
            file=sys.stderr,
        )
        return 2
    if not (args.execute or args.confirm_final_test):
        report = preflight(locked, paths)
        print(format_preflight(report, locked, paths))
        return 0 if report.ok else 1

    try:
        execute_final_evaluation(
            locked,
            paths,
            execute=args.execute,
            confirm_final_test=args.confirm_final_test,
            retry_reason=args.retry_reason,
            rows=rows,
            resamples=resamples,
        )
    except (ExecutionRefused, SealedSplitError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except (PreflightFailure, ContractError, ValueError, RuntimeError, Warning, OSError) as exc:
        print(
            f"STOP: {type(exc).__name__}: {exc}. Request review; do not change data or dates.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Final evaluation written to {paths.report}. These results are not for model reselection."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
