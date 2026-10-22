"""Chronological train / validation / test assignment by UTC calendar date.

Protocol (fixed before any model is fitted; see docs/BASELINE_PROTOCOL.md):

* Take the distinct UTC calendar dates of matches that have at least one eligible shot.
* Sort them. With N dates, the earliest ``(70 N + 50) // 100`` go to train, the dates up to
  ``(85 N + 50) // 100`` go to validation, and the rest go to test. This is integer arithmetic,
  i.e. 70% and 85% cumulative cut points rounded half up, so there is no floating-point
  ambiguity. If any split would receive no date (N below 6) it is an error.
* Every match, and therefore every shot, inherits the split of its date. Two matches on the same
  date are always in the same split, so no split boundary falls inside a day.
* Dates are strings ``YYYY-MM-DD``, which sort chronologically. They are distinct, so there are
  no ties to break; matches on one date are ordered by match ID wherever an order is needed.
* Shots are never shuffled or split individually.

This module reads no labels and no features. ``python -m expected_goals_lab.splits`` writes the
assignment and its manifest without touching a target.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expected_goals_lab.dataset import (
    DEFAULT_CSV,
    ContractError,
    Identity,
    read_identity,
    sha256_bytes_of,
)

SPLITS = ("train", "validation", "test")
TRAIN_PERCENT, VALIDATION_CUMULATIVE_PERCENT = 70, 85
ASSIGNMENT_NAME = "wyscout_split_assignment.csv"
SPLIT_MANIFEST_NAME = "wyscout_split_manifest.json"


@dataclass(frozen=True)
class SplitPlan:
    date_split: dict[str, str]  # 'YYYY-MM-DD' -> split name
    boundaries: dict[str, dict]  # split -> {first_date, last_date, n_dates}


def assign_date_splits(dates) -> SplitPlan:
    ordered = sorted(set(dates))
    n = len(ordered)
    n_train = (TRAIN_PERCENT * n + 50) // 100
    n_train_val = (VALIDATION_CUMULATIVE_PERCENT * n + 50) // 100
    if n_train < 1 or n_train_val - n_train < 1 or n - n_train_val < 1:
        raise ValueError(f"{n} dates are too few to give every split at least one date")
    date_split = {}
    for i, date in enumerate(ordered):
        date_split[date] = "train" if i < n_train else "validation" if i < n_train_val else "test"
    boundaries = {}
    for split in SPLITS:
        in_split = [d for d in ordered if date_split[d] == split]
        boundaries[split] = {
            "first_date": in_split[0],
            "last_date": in_split[-1],
            "n_dates": len(in_split),
        }
    return SplitPlan(date_split=date_split, boundaries=boundaries)


def split_of_events(identity: Identity, plan: SplitPlan) -> dict[int, str]:
    return {
        int(event_id): plan.date_split[str(date)]
        for event_id, date in zip(identity.event_ids, identity.dates, strict=True)
    }


def verify_split(identity: Identity, plan: SplitPlan) -> None:
    """Exhaustive, partition-wide checks. Raises ContractError on any violation."""
    split_of = split_of_events(identity, plan)
    if len(split_of) != identity.n_eligible or set(split_of.values()) - set(SPLITS):
        raise ContractError("every eligible event must have exactly one valid split")
    match_split: dict[int, set[str]] = {}
    date_split: dict[str, set[str]] = {}
    for event_id, match_id, date in zip(
        identity.event_ids, identity.match_ids, identity.dates, strict=True
    ):
        split = split_of[int(event_id)]
        match_split.setdefault(int(match_id), set()).add(split)
        date_split.setdefault(str(date), set()).add(split)
    if any(len(s) != 1 for s in match_split.values()):
        raise ContractError("a match appears in more than one split")
    if any(len(s) != 1 for s in date_split.values()):
        raise ContractError("a calendar date appears in more than one split")
    train, validation, test = (plan.boundaries[s] for s in SPLITS)
    if not (
        train["last_date"] < validation["first_date"]
        and validation["last_date"] < test["first_date"]
    ):
        raise ContractError("splits are not strictly ordered in time")


def summarise(identity: Identity, plan: SplitPlan) -> dict[str, dict]:
    """Rows, matches, dates and rows per league for each split. No labels, no features."""
    split_of = split_of_events(identity, plan)
    summary = {}
    for split in SPLITS:
        mask = np.array([split_of[int(e)] == split for e in identity.event_ids], dtype=bool)
        summary[split] = {
            "rows": int(mask.sum()),
            "matches": int(len(set(identity.match_ids[mask].tolist()))),
            "dates": int(len(set(identity.dates[mask].tolist()))),
            "rows_by_competition": {
                int(k): int(v)
                for k, v in sorted(Counter(identity.competition_ids[mask].tolist()).items())
            },
            "matches_by_competition": {
                int(c): len(
                    {
                        m
                        for m, cc in zip(
                            identity.match_ids[mask], identity.competition_ids[mask], strict=True
                        )
                        if cc == c
                    }
                )
                for c in sorted(set(identity.competition_ids[mask].tolist()))
            },
            **plan.boundaries[split],
        }
    return summary


def write_assignment(path: Path, identity: Identity, plan: SplitPlan) -> str:
    """Event-level assignment CSV (identity and split only). Returns its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    split_of = split_of_events(identity, plan)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["event_id", "match_id", "competition_id", "match_date_utc", "split"])
        order = sorted(
            range(identity.n_eligible),
            key=lambda i: (identity.dates[i], identity.match_ids[i], identity.event_ids[i]),
        )
        for i in order:
            writer.writerow(
                [
                    identity.event_ids[i],
                    identity.match_ids[i],
                    identity.competition_ids[i],
                    identity.dates[i],
                    split_of[int(identity.event_ids[i])],
                ]
            )
    return sha256_bytes_of(path)


def split_manifest(identity: Identity, plan: SplitPlan, assignment_sha256: str) -> dict:
    return {
        "protocol": {
            "unit": "UTC calendar date of the match, over matches with an eligible shot",
            "train_percent_of_dates": TRAIN_PERCENT,
            "validation_cumulative_percent_of_dates": VALIDATION_CUMULATIVE_PERCENT,
            "rounding": "half up on integer arithmetic",
        },
        "input_csv_sha256": identity.csv_sha256,
        "input_rows": identity.n_rows,
        "input_eligible_rows": identity.n_eligible,
        "assignment_csv_sha256": assignment_sha256,
        "splits": summarise(identity, plan),
        "label_statistics": "none recorded here; row-level test labels are not accessed by S03",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the chronological split assignment only.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    manifest_path = args.manifest or args.csv.with_suffix(".manifest.json")
    out_dir = args.out_dir or args.csv.parent
    try:
        identity = read_identity(args.csv, manifest_path)
        plan = assign_date_splits(identity.dates.tolist())
        verify_split(identity, plan)
    except FileNotFoundError as exc:
        print(
            f"error: {exc} not found. Run `uv run python -m expected_goals_lab.build_shots` first.",
            file=sys.stderr,
        )
        return 2
    except (ContractError, ValueError) as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 1
    sha = write_assignment(out_dir / ASSIGNMENT_NAME, identity, plan)
    manifest = split_manifest(identity, plan, sha)
    (out_dir / SPLIT_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
