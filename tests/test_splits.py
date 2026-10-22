import csv
import datetime
import json

import numpy as np
import pytest

from expected_goals_lab import splits
from expected_goals_lab.dataset import ContractError, Identity, read_identity
from expected_goals_lab.splits import (
    assign_date_splits,
    split_manifest,
    split_of_events,
    summarise,
    verify_split,
    write_assignment,
)
from tests.s03_helpers import write_dataset


def dates(n, start=datetime.date(2017, 8, 1)):
    return [(start + datetime.timedelta(days=i)).isoformat() for i in range(n)]


def counts(plan):
    c = {"train": 0, "validation": 0, "test": 0}
    for s in plan.date_split.values():
        c[s] += 1
    return c


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (178, (125, 26, 27)),  # the real corpus
        (100, (70, 15, 15)),
        (10, (7, 2, 1)),
        (6, (4, 1, 1)),
        (20, (14, 3, 3)),
    ],
)
def test_integer_cut_points_round_half_up(n, expected):
    plan = assign_date_splits(dates(n))
    assert tuple(counts(plan).values()) == expected


@pytest.mark.parametrize("n", [0, 1, 3, 4, 5])
def test_too_few_dates_to_fill_every_split_is_an_error(n):
    with pytest.raises(ValueError):
        assign_date_splits(dates(n))


def test_assignment_is_chronological_and_boundaries_are_exact():
    plan = assign_date_splits(dates(178))
    b = plan.boundaries
    assert (b["train"]["first_date"], b["train"]["last_date"]) == ("2017-08-01", "2017-12-03")
    assert b["train"]["last_date"] < b["validation"]["first_date"] <= b["validation"]["last_date"]
    assert b["validation"]["last_date"] < b["test"]["first_date"]
    ordered = sorted(plan.date_split)
    labels = [plan.date_split[d] for d in ordered]
    assert labels == sorted(
        labels, key=["train", "validation", "test"].index
    )  # never goes backwards


def test_input_order_and_duplicates_do_not_matter():
    d = dates(60)
    shuffled = list(reversed(d)) + d[:10]
    assert assign_date_splits(shuffled).date_split == assign_date_splits(d).date_split


def make_identity(match_dates: list[str], shots_per_match: int = 3) -> Identity:
    event, match, comp, day = [], [], [], []
    e = 1
    for m, date in enumerate(match_dates, 1):
        for _ in range(shots_per_match):
            event.append(e)
            match.append(m)
            comp.append(364 if m % 2 else 412)
            day.append(date)
            e += 1
    return Identity(
        np.array(event),
        np.array(match),
        np.array(comp),
        np.array(day, dtype="U10"),
        len(event),
        len(event),
        "0" * 64,
        {},
    )


def test_matches_on_the_same_date_always_share_a_split_even_at_a_boundary():
    # 10 dates: train 7, validation 2, test 1. Put three matches on the last training date and
    # two on the first validation date: the boundary must not cut through a day.
    ds = dates(10)
    match_dates = [
        ds[0],
        ds[1],
        ds[2],
        ds[3],
        ds[4],
        ds[5],
        ds[6],
        ds[6],
        ds[6],
        ds[7],
        ds[7],
        ds[8],
        ds[9],
    ]
    identity = make_identity(match_dates)
    plan = assign_date_splits(identity.dates.tolist())

    verify_split(identity, plan)
    by_match = {}
    for event, match in zip(identity.event_ids, identity.match_ids, strict=True):
        by_match.setdefault(int(match), set()).add(split_of_events(identity, plan)[int(event)])
    assert all(len(v) == 1 for v in by_match.values())
    assert plan.date_split[ds[6]] == "train" and plan.date_split[ds[7]] == "validation"


def test_every_event_and_match_in_exactly_one_split_and_counts_add_up():
    identity = make_identity(dates(40))
    plan = assign_date_splits(identity.dates.tolist())
    summary = summarise(identity, plan)

    assert sum(s["rows"] for s in summary.values()) == identity.n_eligible
    assert sum(s["matches"] for s in summary.values()) == len(set(identity.match_ids.tolist()))
    assert sum(s["dates"] for s in summary.values()) == 40
    for s in summary.values():
        assert sum(s["rows_by_competition"].values()) == s["rows"]


def test_verify_split_rejects_a_plan_that_breaks_time_order_or_shares_a_date():
    identity = make_identity(dates(20))
    plan = assign_date_splits(identity.dates.tolist())
    verify_split(identity, plan)

    swapped = dict(plan.date_split)
    first, last = sorted(swapped)[0], sorted(swapped)[-1]
    swapped[first], swapped[last] = swapped[last], swapped[first]
    bad_boundaries = {k: dict(v) for k, v in plan.boundaries.items()}
    bad_boundaries["train"]["last_date"] = last  # train now ends after validation begins
    with pytest.raises(ContractError, match="strictly ordered"):
        verify_split(identity, type(plan)(swapped, bad_boundaries))

    unknown = dict(plan.date_split)
    unknown[sorted(unknown)[0]] = "holdout"
    with pytest.raises(ContractError, match="exactly one valid split"):
        verify_split(identity, type(plan)(unknown, plan.boundaries))


def test_assignment_files_are_deterministic_and_carry_no_labels(tmp_path):
    csv_path, manifest_path, _ = write_dataset(tmp_path / "in")
    identity = read_identity(csv_path, manifest_path)
    plan = assign_date_splits(identity.dates.tolist())

    sha_a = write_assignment(tmp_path / "a.csv", identity, plan)
    sha_b = write_assignment(tmp_path / "b.csv", identity, plan)
    assert sha_a == sha_b and (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()

    with (tmp_path / "a.csv").open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "event_id",
            "match_id",
            "competition_id",
            "match_date_utc",
            "split",
        ]
        assigned = list(reader)
    assert len(assigned) == identity.n_eligible
    assert len({r["event_id"] for r in assigned}) == len(assigned)

    manifest = split_manifest(identity, plan, sha_a)
    text = json.dumps(manifest)
    assert "is_goal" not in text.replace("test labels", "")  # no label statistics recorded
    assert set(manifest["splits"]["test"]) == {
        "rows",
        "matches",
        "dates",
        "rows_by_competition",
        "matches_by_competition",
        "first_date",
        "last_date",
        "n_dates",
    }


def test_split_only_cli_writes_files_and_reports_missing_inputs(tmp_path, capsys):
    csv_path, manifest_path, _ = write_dataset(tmp_path / "in")
    out = tmp_path / "out"

    assert splits.main(["--csv", str(csv_path), "--out-dir", str(out)]) == 0
    assert (out / splits.ASSIGNMENT_NAME).exists() and (out / splits.SPLIT_MANIFEST_NAME).exists()
    capsys.readouterr()

    assert splits.main(["--csv", str(tmp_path / "missing.csv"), "--out-dir", str(out)]) == 2
    assert "build_shots" in capsys.readouterr().err

    csv_path.write_bytes(csv_path.read_bytes() + b"\n")
    assert splits.main(["--csv", str(csv_path), "--out-dir", str(out)]) == 1
    assert "contract error" in capsys.readouterr().err
