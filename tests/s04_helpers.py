"""Synthetic data and the test-row trap for S04 tests. Invented values only."""

import dataclasses
from pathlib import Path

from expected_goals_lab.candidate_data import BODY_PART_COLUMN, BODY_PART_VOCABULARY
from expected_goals_lab.dataset import OPEN_COLUMNS, iter_csv_rows, read_identity
from expected_goals_lab.shot_table import ShotRow
from expected_goals_lab.splits import assign_date_splits
from tests.s03_helpers import synthetic_rows, write_dataset

# Every column a sealed test row must never reveal in S04: S03's feature, location and target
# columns plus the new candidate feature.
SENSITIVE = set(OPEN_COLUMNS) | {"is_goal", BODY_PART_COLUMN}


def s04_rows(seed: int = 7, goal_line_event_ids: tuple[int, ...] = (), **kwargs) -> list[ShotRow]:
    """S03's synthetic rows with all three body-part classes and optional goal-line shots."""
    rows = []
    for row in synthetic_rows(seed=seed, **kwargs):
        changes = {"body_part_class": BODY_PART_VOCABULARY[row.event_id % 3]}
        if row.event_id in goal_line_event_ids and row.eligible:
            changes.update(
                x_pct=100,
                y_pct=50,
                x_m=105.0,
                y_m=34.0,
                distance_m=0.0,
                angle_rad=3.141592653589793,
                angle_deg=180.0,
                flags="ON_GOAL_LINE",
            )
        rows.append(dataclasses.replace(row, **changes))
    return rows


def write_s04_dataset(folder: Path, **kwargs):
    return write_dataset(folder, s04_rows(**kwargs))


def held_out_dates(csv_path: Path, manifest_path: Path) -> set[str]:
    identity = read_identity(csv_path, manifest_path)
    plan = assign_date_splits(identity.dates.tolist())
    return {d for d, s in plan.date_split.items() if s == "test"}


class GuardedRow(dict):
    """A CSV row that raises if a sensitive column of a sealed test-date row is read."""

    def __init__(self, row, sealed, log):
        super().__init__(row)
        self.sealed, self.log = sealed, log

    def _check(self, key):
        if self.sealed and key in SENSITIVE:
            self.log.append(key)
            raise AssertionError(f"sealed test row: column {key!r} was read")

    def __getitem__(self, key):
        self._check(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._check(key)
        return super().get(key, default)


def guarded_reader(test_dates, log):
    def rows(path):
        for row in iter_csv_rows(path):
            yield GuardedRow(row, row["match_datetime_utc"][:10] in test_dates, log)

    return rows
