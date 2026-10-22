"""Inputs for the S08 portfolio figures. Pure functions; no plotting and no model fitting here.

Everything in this module is either read from frozen S07 artefacts, from the validation window
through the guarded S03/S04 loaders, or from the provider's match metadata. It inspects no
held-out feature or outcome (the shot table is still traversed for identity columns, as in
S03-S05) and never calls the final-evaluation code. Rules: docs/S08_VISUAL_SPEC.md.
"""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expected_goals_lab.dataset import RowReader, iter_csv_rows
from expected_goals_lab.final_lock import REAL_LOCKED, LockedIdentities
from expected_goals_lab.geometry import REFERENCE_PITCH, Pitch, to_metres

S07_SUMMARY = Path("results/final_test_summary.json")  # published aggregate, not raw data
S07_LOG = Path("reports/final_test_execution_log.jsonl")

PRIMARY_KEY, BENCHMARK_KEY, CONSTANT_KEY = (
    "nonlinear_location_body_part",
    "distance_plus_angle",
    "constant_train_rate",
)
MIN_MATCH_SHOTS = 10  # both teams together, eligible regular shots
SUPPORT_MIN_SHOTS = 25  # training shots of the class near a grid cell, to draw it
SUPPORT_HALF_WINDOW_M = 3.0
BODY_PART_LABELS = {
    "right_foot": "Right foot",
    "left_foot": "Left foot",
    "head_or_body": "Head / other body",
}


class FigureInputError(RuntimeError):
    """A frozen input is missing, altered or inconsistent. The figure is not built."""


MISSING_SHOT_TABLE = (
    "The shot table {path} does not exist. It is built locally from the public Figshare "
    "collection and is never stored in this repository. Build it first:\n"
    "    uv run python -m expected_goals_lab.build_shots\n"
    "(about 78 MB are downloaded once and checked against pinned checksums; see README.md)."
)


def require_shot_table(csv_path: Path) -> None:
    """Fail with instructions, not a traceback, when the locally built shot table is absent."""
    if not csv_path.exists():
        raise FigureInputError(MISSING_SHOT_TABLE.format(path=csv_path))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- Figure 1: the frozen S07 summary  --------------------------------------------------------


def load_frozen_summary(
    summary_path: Path = S07_SUMMARY,
    log_path: Path = S07_LOG,
    locked: LockedIdentities = REAL_LOCKED,
) -> dict:
    """Read the S07 summary after proving it is the one the completed log recorded.

    The summary's SHA-256 must equal ``summary_sha256`` of the single ``completed`` log entry and
    its protocol hash must equal the locked one. Nothing is recomputed.
    """
    for path in (summary_path, log_path):
        if not path.exists():
            raise FigureInputError(f"S07 artefact {path} is missing; it is not recreated here")
    entries = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line
    ]
    completed = [e for e in entries if e.get("outcome") == "completed"]
    if len(completed) != 1:
        raise FigureInputError(f"expected exactly one completed S07 entry, found {len(completed)}")
    if sha256_file(summary_path) != completed[0].get("summary_sha256"):
        raise FigureInputError("the S07 summary does not match the hash in the execution log")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("provenance", {}).get("protocol_sha256") != locked.protocol_sha256:
        raise FigureInputError("the S07 summary was not produced under the locked protocol")
    needed = {PRIMARY_KEY, BENCHMARK_KEY, CONSTANT_KEY}
    if not needed <= set(summary.get("scores", {})) or not {"log_loss", "brier"} <= set(
        summary.get("intervals", {})
    ):
        raise FigureInputError("the S07 summary lacks the expected scores or intervals")
    return summary


def headline(summary: dict) -> dict:
    """Numbers for Figure 1 and the story, taken from the summary; only arithmetic is added."""
    scores = summary["scores"]
    primary, benchmark = scores[PRIMARY_KEY], scores[BENCHMARK_KEY]

    def relative(metric: str) -> float:
        return (primary[metric] - benchmark[metric]) / benchmark[metric]

    return {
        "shots": summary["test"]["shots"],
        "goals": summary["test"]["goals"],
        "matches": summary["test"]["matches"],
        "first_date": summary["test"]["first_date"],
        "last_date": summary["test"]["last_date"],
        "scores": scores,
        "intervals": summary["intervals"],
        "relative_log_loss": relative("log_loss"),
        "relative_brier": relative("brier"),
        "auc_gain": primary["auc"] - benchmark["auc"],
    }


# --- coordinates  -----------------------------------------------------------------------------


def to_display(
    x_pct: np.ndarray, y_pct: np.ndarray, side: str, pitch: Pitch = REFERENCE_PITCH
) -> tuple[np.ndarray, np.ndarray]:
    """Metre positions on a full pitch drawn with the home team attacking the right-hand goal.

    Wyscout positions are attacker-relative (x = 100% is the goal being attacked). The home team
    is drawn as recorded; the away team is rotated 180 degrees so that it attacks the left goal.
    """
    if side not in ("home", "away"):
        raise FigureInputError(f"unknown side {side!r}")
    x_pct, y_pct = np.asarray(x_pct, dtype=float), np.asarray(y_pct, dtype=float)
    if side == "away":
        x_pct, y_pct = 100.0 - x_pct, 100.0 - y_pct
    return x_pct * pitch.length_m / 100.0, y_pct * pitch.width_m / 100.0


def pct_to_metres(x_pct: np.ndarray, y_pct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = [to_metres(float(x), float(y)) for x, y in zip(x_pct, y_pct, strict=True)]
    return np.array([p[0] for p in points]), np.array([p[1] for p in points])


# --- case-study match ----------------------------------------------------------------


def read_team_columns(
    csv_path: Path, event_ids: set[int], rows: RowReader = iter_csv_rows
) -> dict[int, tuple[int, str]]:
    """Team id and side for the requested events only.

    For every other row (including every held-out one) just ``event_id`` is read.
    """
    found: dict[int, tuple[int, str]] = {}
    for row in rows(csv_path):
        event_id = int(row["event_id"])
        if event_id in event_ids:
            found[event_id] = (int(row["team_id"]), row["side"])
    missing = event_ids - set(found)
    if missing:
        raise FigureInputError(f"{len(missing)} requested events are not in the shot table")
    return found


def select_showcase_match(shots: list[tuple[int, str, int]]) -> int:
    """The case-study match (docs/S08_VISUAL_SPEC.md, section 5).

    ``shots`` holds ``(match_id, date, team_id)`` for every eligible validation shot. Only those
    three facts are used: never an outcome, a prediction or a plot. A match qualifies with at
    least one shot for each of two teams and at least ``MIN_MATCH_SHOTS`` in all; the earliest
    date wins, then the lowest match id.
    """
    per_match: dict[int, dict] = defaultdict(lambda: {"date": None, "teams": defaultdict(int)})
    for match_id, date, team_id in shots:
        entry = per_match[match_id]
        if entry["date"] not in (None, date):
            raise FigureInputError(f"match {match_id} has more than one date")
        entry["date"] = date
        entry["teams"][team_id] += 1
    qualifying = [
        (entry["date"], match_id)
        for match_id, entry in per_match.items()
        if len(entry["teams"]) == 2 and sum(entry["teams"].values()) >= MIN_MATCH_SHOTS
    ]
    if not qualifying:
        raise FigureInputError("no validation match satisfies the selection rule")
    return min(qualifying)[1]


@dataclass(frozen=True)
class MatchInfo:
    match_id: int
    date_utc: str
    home_id: int
    away_id: int
    home_name: str
    away_name: str
    home_score: int
    away_score: int


def parse_match_info(record: dict) -> MatchInfo:
    """Names and final score from the provider's match record, cross-checked against its teams."""
    label = record.get("label", "")
    try:
        teams_part, score_part = label.rsplit(", ", 1)
        home_name, away_name = teams_part.split(" - ")
        label_home, label_away = (int(s) for s in score_part.split(" - "))
    except ValueError as exc:
        raise FigureInputError(f"match label {label!r} has an unexpected format") from exc
    sides = {int(t): d for t, d in record.get("teamsData", {}).items()}
    homes = [t for t, d in sides.items() if d.get("side") == "home"]
    aways = [t for t, d in sides.items() if d.get("side") == "away"]
    if len(sides) != 2 or len(homes) != 1 or len(aways) != 1:
        raise FigureInputError(
            f"match {record.get('wyId')}: teamsData does not hold one home, one away"
        )
    home_id, away_id = homes[0], aways[0]
    if (sides[home_id]["score"], sides[away_id]["score"]) != (label_home, label_away):
        raise FigureInputError(
            f"match {record.get('wyId')}: label score {label_home}-{label_away} disagrees with "
            "the per-team scores, so the home/away order of the label cannot be trusted"
        )
    return MatchInfo(
        match_id=int(record["wyId"]),
        date_utc=str(record["dateutc"]),
        home_id=home_id,
        away_id=away_id,
        home_name=home_name,
        away_name=away_name,
        home_score=label_home,
        away_score=label_away,
    )


def team_totals(team_ids: np.ndarray, p: np.ndarray, y: np.ndarray) -> dict[int, dict]:
    """Regular shots, goals and summed xG per team."""
    return {
        int(t): {
            "shots": int((team_ids == t).sum()),
            "goals": int(y[team_ids == t].sum()),
            "xg": float(p[team_ids == t].sum()),
        }
        for t in sorted(set(team_ids.tolist()))
    }


# --- Figure 2: grid of positions with training support  ---------------------------------------


def support_counts(
    train_x_m: np.ndarray,
    train_y_m: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    half_window_m: float = SUPPORT_HALF_WINDOW_M,
) -> np.ndarray:
    """For each grid cell (rows = y, columns = x), training shots within +-half_window in x and y.

    ``grid_x`` and ``grid_y`` are cell centres on a regular grid.
    """
    counts = np.empty((len(grid_y), len(grid_x)), dtype=int)
    for j, y in enumerate(grid_y):
        in_row = np.abs(train_y_m - y) <= half_window_m
        xs = np.sort(train_x_m[in_row])
        lo = np.searchsorted(xs, grid_x - half_window_m, side="left")
        hi = np.searchsorted(xs, grid_x + half_window_m, side="right")
        counts[j] = hi - lo
    return counts
