"""Synthetic five-league data and committed-report generators for S05 tests. Invented values."""

import dataclasses
from pathlib import Path

from expected_goals_lab import evaluate_baselines, evaluate_nonlinear
from expected_goals_lab.evaluate_baselines import LEAGUE_NAMES
from tests.s03_helpers import write_dataset
from tests.s04_helpers import s04_rows

LEAGUE_IDS = list(LEAGUE_NAMES)
EVENT_OFFSET, MATCH_OFFSET = 7_000_000_000, 8_000_000_000  # ten-digit IDs: easy to grep for


def s05_rows(**kwargs):
    """S04 rows spread over the five real league IDs, with ten-digit event and match IDs."""
    rows = []
    for row in s04_rows(**kwargs):
        league = LEAGUE_IDS[row.match_id % len(LEAGUE_IDS)]
        rows.append(
            dataclasses.replace(
                row,
                event_id=row.event_id + EVENT_OFFSET,
                match_id=row.match_id + MATCH_OFFSET,
                competition_id=league,
                competition=LEAGUE_NAMES[league],
            )
        )
    return rows


def write_s05_dataset(folder: Path, rows=None):
    return write_dataset(folder, rows if rows is not None else s05_rows())


def write_committed_reports(csv_path: Path, manifest_path: Path, folder: Path, resamples: int = 60):
    """Run S03 and S04 on the data and write their reports, as 'the committed reports' to match."""
    s03 = evaluate_baselines.evaluate(csv_path, manifest_path, folder / "split", resamples)
    s03_report = folder / "baseline_validation.md"
    s03_report.write_text(evaluate_baselines.render_report(s03, csv_path), encoding="utf-8")
    s04 = evaluate_nonlinear.evaluate_nonlinear(
        csv_path, manifest_path, folder / "split", resamples, s03_report, folder / "none.md"
    )
    s04_report = folder / "nonlinear_validation.md"
    s04_report.write_text(evaluate_nonlinear.render_report(s04, csv_path), encoding="utf-8")
    return s03_report, s04_report, folder / "split"
