"""Fully synthetic fixture for the S06 final-evaluation tests. Invented values; no provider data.

``build_fixture`` writes a miniature of the real file layout (shot table, manifests, assignment,
protocol, lockfile, one inventory file) plus a matching ``LockedIdentities``. Nothing here touches
the real Wyscout files.
"""

import dataclasses
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

from expected_goals_lab import final_evaluation as fe
from expected_goals_lab.baselines import make_pipeline
from expected_goals_lab.dataset import read_identity, sha256_bytes_of
from expected_goals_lab.final_lock import LockedIdentities
from expected_goals_lab.nonlinear import FACTORIES
from expected_goals_lab.scoring import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
from expected_goals_lab.splits import assign_date_splits, split_manifest, write_assignment
from tests.s03_helpers import write_dataset
from tests.s04_helpers import GuardedRow
from tests.s05_helpers import s05_rows

PROTOCOL_TEXT = "# Synthetic protocol\nFINAL TEST NOT YET OPENED\n"
DECISION_TEXT = "# Synthetic decision\n"
PYPROJECT_TEXT = '[project]\nname = "synthetic"\n'
INVENTORY_FILE = "src/inventory_file.py"
INVENTORY_TEXT = "VALUE = 1\n"


def lockfile_text() -> str:
    blocks = [
        f'[[package]]\nname = "{name}"\nversion = "{importlib.metadata.version(name)}"\n'
        for name in fe.LOCKED_PACKAGES
    ]
    return "version = 1\n\n" + "\n".join(blocks)


def build_fixture(folder: Path, rows=None, resamples: int = 60) -> SimpleNamespace:
    folder.mkdir(parents=True, exist_ok=True)
    csv_path, manifest_path, rows = write_dataset(folder, rows if rows is not None else s05_rows())
    paths = fe.Paths(
        root=folder,
        csv=csv_path,
        csv_manifest=manifest_path,
        assignment=folder / "assignment.csv",
        split_manifest=folder / "split_manifest.json",
        protocol=folder / "protocol.md",
        decision=folder / "decision.md",
        lockfile=folder / "uv.lock",
        pyproject=folder / "pyproject.toml",
        report=folder / "reports" / "final_test_evaluation.md",
        log=folder / "reports" / "final_test_execution_log.jsonl",
        summary=folder / "final_test_summary.json",
    )
    identity = read_identity(csv_path, manifest_path)
    plan = assign_date_splits(identity.dates.tolist())
    assignment_sha = write_assignment(paths.assignment, identity, plan)
    split = split_manifest(identity, plan, assignment_sha)
    paths.split_manifest.write_text(json.dumps(split, indent=2), encoding="utf-8")
    paths.protocol.write_text(PROTOCOL_TEXT, encoding="utf-8", newline="")
    paths.decision.write_text(DECISION_TEXT, encoding="utf-8", newline="")
    paths.pyproject.write_text(PYPROJECT_TEXT, encoding="utf-8", newline="")
    paths.lockfile.write_text(lockfile_text(), encoding="utf-8", newline="")
    inventory = folder / INVENTORY_FILE
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text(INVENTORY_TEXT, encoding="utf-8", newline="")

    splits = split["splits"]
    names = ("train", "validation", "test")
    locked = LockedIdentities(
        csv_sha256=sha256_bytes_of(csv_path),
        csv_manifest_sha256=sha256_bytes_of(manifest_path),
        assignment_sha256=assignment_sha,
        split_manifest_sha256=sha256_bytes_of(paths.split_manifest),
        rows=identity.n_rows,
        eligible_rows=identity.n_eligible,
        split_rows={s: splits[s]["rows"] for s in names},
        split_matches={s: splits[s]["matches"] for s in names},
        split_dates={s: splits[s]["dates"] for s in names},
        boundaries={s: (splits[s]["first_date"], splits[s]["last_date"]) for s in names},
        protocol_sha256=fe.normalised_sha256(paths.protocol),
        decision_sha256=fe.normalised_sha256(paths.decision),
        lockfile_sha256=fe.normalised_sha256(paths.lockfile),
        pyproject_sha256=fe.normalised_sha256(paths.pyproject),
        packages=fe.locked_package_versions(paths.lockfile),
        file_hashes={INVENTORY_FILE: fe.normalised_sha256(inventory)},
        primary_recipe_sha256=fe.recipe_sha256(FACTORIES[fe.PRIMARY]()),
        benchmark_recipe_sha256=fe.recipe_sha256(make_pipeline()),
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    test_dates = {d for d, s in plan.date_split.items() if s == "test"}
    return SimpleNamespace(
        paths=paths,
        locked=locked,
        rows=rows,
        plan=plan,
        test_dates=test_dates,
        resamples=resamples,
        inventory=inventory,
    )


def result_files(fx) -> list[Path]:
    """Every file an evaluation could write: none may exist after a refused or dry run."""
    return [p for p in (fx.paths.report, fx.paths.summary, fx.paths.log) if p.exists()]


class PhasedReader:
    """Row reader that raises if a sensitive column of a held-out-date row is read before the
    test is deliberately opened (``state["open"]``)."""

    def __init__(self, test_dates, state, log):
        self.test_dates, self.state, self.log = test_dates, state, log
        self.calls = 0

    def __call__(self, path):
        from expected_goals_lab.dataset import iter_csv_rows

        self.calls += 1
        for row in iter_csv_rows(path):
            sealed = row["match_datetime_utc"][:10] in self.test_dates and not self.state["open"]
            yield GuardedRow(row, sealed, self.log)


class NeverRead:
    """A row reader that must not be called at all."""

    def __init__(self):
        self.calls = 0

    def __call__(self, path):
        self.calls += 1
        raise AssertionError("a row reader was called")


def flip_test_labels(rows, test_dates):
    """The same rows with every held-out-date label inverted. Features are untouched."""
    return [
        dataclasses.replace(r, is_goal=1 - r.is_goal)
        if r.eligible and r.match_datetime_utc[:10] in test_dates
        else r
        for r in rows
    ]
