"""Synthetic S02-shaped datasets for S03 tests. Invented values; no provider data.

Rows follow the real column contract (``shot_table.COLUMNS``). Labels are drawn from an
arbitrary logistic function of distance so that a fitted model has something to find.
"""

import datetime
import json
import math
from pathlib import Path

import numpy as np

from expected_goals_lab.dataset import sha256_bytes_of
from expected_goals_lab.geometry import shot_geometry
from expected_goals_lab.shot_table import (
    CANDIDATE_FEATURES,
    COLUMN_ROLES,
    COLUMNS,
    FEATURE_ALLOWLIST,
    SCHEMA_VERSION,
    TARGET,
    ShotRow,
    write_csv,
)

COMPETITIONS = {364: "English first division", 412: "French first division"}


def synthetic_rows(
    n_dates: int = 40,
    matches_per_date: int = 3,
    shots_per_match: int = 12,
    seed: int = 7,
    quarantined_per_match: int = 1,
) -> list[ShotRow]:
    rng = np.random.default_rng(seed)
    rows: list[ShotRow] = []
    event_id, match_id = 1000, 500
    start = datetime.date(2017, 8, 1)
    for d in range(n_dates):
        date = (start + datetime.timedelta(days=d)).isoformat()
        for m in range(matches_per_date):
            match_id += 1
            competition_id = list(COMPETITIONS)[m % len(COMPETITIONS)]
            for s in range(shots_per_match + quarantined_per_match):
                event_id += 1
                bad = s >= shots_per_match
                x = int(rng.integers(3, 8) if bad else rng.integers(55, 101))
                y = int(rng.integers(5, 96))
                g = shot_geometry(x, y)
                p = 1 / (1 + math.exp(-(1.0 - 0.17 * g.distance_m)))
                goal = int(rng.random() < p)
                rows.append(
                    ShotRow(
                        event_id=event_id,
                        match_id=match_id,
                        competition_id=competition_id,
                        competition=COMPETITIONS[competition_id],
                        season_id=1,
                        match_datetime_utc=f"{date} 15:00:00",
                        team_id=10 + (s % 2),
                        opponent_team_id=10 + ((s + 1) % 2),
                        side="home" if s % 2 == 0 else "away",
                        player_id=100 + s,
                        period="1H" if s % 2 == 0 else "2H",
                        event_sec=float(60 + s),
                        x_pct=x,
                        y_pct=y,
                        x_m=g.x_m,
                        y_m=g.y_m,
                        distance_m=g.distance_m,
                        angle_rad=g.angle_rad,
                        angle_deg=g.angle_deg,
                        body_part_class="right_foot",
                        is_goal=goal,
                        eligible=0 if bad else 1,
                        exclusion_reasons="SUSPECT_POSITION" if bad else "",
                        primary_exclusion="SUSPECT_POSITION" if bad else "",
                        flags="OWN_HALF_POSITION" if bad else "",
                    )
                )
    return rows


def manifest_for(csv_path: Path, rows: list[ShotRow] | int, eligible: int | None = None) -> dict:
    n_rows = rows if isinstance(rows, int) else len(rows)
    n_eligible = eligible if eligible is not None else sum(r.eligible for r in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "columns": list(COLUMNS),
        "rows": n_rows,
        "eligible_rows": n_eligible,
        "csv_sha256": sha256_bytes_of(csv_path),
        "column_roles": COLUMN_ROLES,
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "candidate_features": list(CANDIDATE_FEATURES),
        "target": TARGET,
    }


def write_dataset(
    folder: Path, rows: list[ShotRow] | None = None
) -> tuple[Path, Path, list[ShotRow]]:
    rows = rows if rows is not None else synthetic_rows()
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "shots.csv"
    write_csv(rows, csv_path)
    manifest_path = folder / "shots.manifest.json"
    manifest_path.write_text(json.dumps(manifest_for(csv_path, rows)), encoding="utf-8")
    return csv_path, manifest_path, rows


def rewrite_csv(csv_path: Path, manifest_path: Path, transform) -> None:
    """Apply ``transform(list_of_lines) -> list_of_lines`` to the CSV and refresh its hash."""
    lines = csv_path.read_text(encoding="utf-8").split("\n")
    csv_path.write_text("\n".join(transform(lines)), encoding="utf-8", newline="")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["csv_sha256"] = sha256_bytes_of(csv_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def set_manifest(manifest_path: Path, **changes) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(changes)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def column_index(name: str) -> int:
    return COLUMNS.index(name)
