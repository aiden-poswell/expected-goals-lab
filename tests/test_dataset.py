import json

import numpy as np
import pytest

from expected_goals_lab.dataset import (
    OPEN_SPLITS,
    ContractError,
    FeatureNotAllowedError,
    SealedSplitError,
    feature_matrix,
    load_partitions,
    read_identity,
)
from expected_goals_lab.splits import assign_date_splits, split_of_events
from tests.s03_helpers import column_index, rewrite_csv, set_manifest, write_dataset


@pytest.fixture
def dataset(tmp_path):
    return write_dataset(tmp_path)


def test_valid_contract_returns_only_eligible_identity(dataset):
    csv_path, manifest_path, rows = dataset

    identity = read_identity(csv_path, manifest_path)

    assert identity.n_rows == len(rows)
    assert identity.n_eligible == sum(r.eligible for r in rows)
    eligible_ids = {r.event_id for r in rows if r.eligible}
    assert set(identity.event_ids.tolist()) == eligible_ids  # quarantined rows never admitted
    assert identity.dates.dtype.kind == "U" and identity.dates[0] == "2017-08-01"


def test_missing_files_raise_file_not_found(dataset, tmp_path):
    csv_path, manifest_path, _ = dataset
    with pytest.raises(FileNotFoundError):
        read_identity(tmp_path / "nope.csv", manifest_path)
    with pytest.raises(FileNotFoundError):
        read_identity(csv_path, tmp_path / "nope.json")


def test_changed_csv_fails_the_hash_check(dataset):
    csv_path, manifest_path, _ = dataset
    csv_path.write_bytes(csv_path.read_bytes() + b"\n")
    with pytest.raises(ContractError, match="SHA-256"):
        read_identity(csv_path, manifest_path)


@pytest.mark.parametrize("content", ["{not json", "[]", ""])
def test_corrupt_manifest_fails_closed(dataset, content):
    csv_path, manifest_path, _ = dataset
    manifest_path.write_text(content, encoding="utf-8")
    with pytest.raises(ContractError):
        read_identity(csv_path, manifest_path)


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"columns": ["event_id"]},
        {"feature_allowlist": ["distance_m", "angle_rad", "body_part_class"]},
        {"column_roles": {"event_id": "feature_core"}},
        {"target": "is_miss"},
        {"candidate_features": []},
    ],
)
def test_manifest_must_match_the_schema_and_roles_s02_defines(dataset, changes):
    csv_path, manifest_path, _ = dataset
    set_manifest(manifest_path, **changes)
    with pytest.raises(ContractError, match="manifest field"):
        read_identity(csv_path, manifest_path)


def test_row_and_eligible_counts_must_match_the_manifest(dataset):
    csv_path, manifest_path, rows = dataset
    set_manifest(manifest_path, rows=len(rows) + 1)
    with pytest.raises(ContractError, match="rows, the manifest says"):
        read_identity(csv_path, manifest_path)
    set_manifest(manifest_path, rows=len(rows), eligible_rows=1)
    with pytest.raises(ContractError, match="eligible rows"):
        read_identity(csv_path, manifest_path)


def test_header_must_match_exactly(dataset):
    csv_path, manifest_path, _ = dataset
    rewrite_csv(
        csv_path, manifest_path, lambda lines: [lines[0].replace("event_id", "id"), *lines[1:]]
    )
    with pytest.raises(ContractError, match="header"):
        read_identity(csv_path, manifest_path)


def test_duplicate_event_ids_fail(dataset):
    csv_path, manifest_path, _ = dataset

    def duplicate(lines):
        return [*lines[:-1][:3], lines[1], *lines[3:]]  # row 1 appears twice, one row replaced

    rewrite_csv(csv_path, manifest_path, duplicate)
    with pytest.raises(ContractError, match="duplicate event_id"):
        read_identity(csv_path, manifest_path)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("eligible", "2", "expected 0 or 1"),
        ("eligible", "", "expected 0 or 1"),
        ("exclusion_reasons", "SUSPECT_POSITION", "reasons"),  # eligible row with a reason
        ("event_id", "abc", "not an integer"),
    ],
)
def test_invalid_eligibility_and_ids_fail(dataset, column, value, message):
    csv_path, manifest_path, _ = dataset
    index = column_index(column)

    def corrupt(lines):
        cells = lines[1].split(",")
        cells[index] = value
        return [lines[0], ",".join(cells), *lines[2:]]

    rewrite_csv(csv_path, manifest_path, corrupt)
    with pytest.raises(ContractError, match=message):
        read_identity(csv_path, manifest_path)


def test_quarantined_row_without_a_reason_fails(dataset):
    csv_path, manifest_path, rows = dataset
    first_bad = next(i for i, r in enumerate(rows) if not r.eligible) + 1

    def blank_reason(lines):
        cells = lines[first_bad].split(",")
        cells[column_index("exclusion_reasons")] = ""
        return [*lines[:first_bad], ",".join(cells), *lines[first_bad + 1 :]]

    rewrite_csv(csv_path, manifest_path, blank_reason)
    with pytest.raises(ContractError, match="eligible=0"):
        read_identity(csv_path, manifest_path)


def test_a_match_with_two_dates_fails(dataset):
    csv_path, manifest_path, rows = dataset

    def split_date(lines):
        cells = lines[2].split(",")
        cells[column_index("match_datetime_utc")] = "2030-01-01 15:00:00"
        return [lines[0], lines[1], ",".join(cells), *lines[3:]]

    rewrite_csv(csv_path, manifest_path, split_date)
    with pytest.raises(ContractError, match="more than one date"):
        read_identity(csv_path, manifest_path)


def open_partitions(dataset):
    csv_path, manifest_path, _ = dataset
    identity = read_identity(csv_path, manifest_path)
    plan = assign_date_splits(identity.dates.tolist())
    return csv_path, identity, plan, load_partitions(csv_path, split_of_events(identity, plan))


def test_partitions_hold_only_open_split_eligible_rows(dataset):
    csv_path, identity, plan, parts = open_partitions(dataset)

    assert set(parts) == set(OPEN_SPLITS)
    train_dates, val_dates = (
        set(parts["train"].dates.tolist()),
        set(parts["validation"].dates.tolist()),
    )
    assert max(train_dates) < min(val_dates)
    assert not (
        set(parts["train"].event_ids.tolist()) & set(parts["validation"].event_ids.tolist())
    )
    test_ids = {
        int(e)
        for e, d in zip(identity.event_ids, identity.dates, strict=True)
        if plan.date_split[str(d)] == "test"
    }
    loaded = set(parts["train"].event_ids.tolist()) | set(parts["validation"].event_ids.tolist())
    assert not loaded & test_ids and test_ids
    assert set(parts["train"].y.tolist()) <= {0, 1}


def test_requesting_the_test_split_raises(dataset):
    csv_path, identity, plan, _ = open_partitions(dataset)
    with pytest.raises(SealedSplitError, match="sealed"):
        load_partitions(csv_path, split_of_events(identity, plan), splits=("train", "test"))


def test_unreadable_labels_or_features_on_open_rows_fail(dataset):
    csv_path, manifest_path, _ = dataset
    identity = read_identity(csv_path, manifest_path)
    plan = assign_date_splits(identity.dates.tolist())
    mapping = split_of_events(identity, plan)
    first_train_id = next(int(e) for e in identity.event_ids if mapping[int(e)] == "train")
    index_of = {
        name: column_index(name) for name in ("is_goal", "distance_m", "angle_rad", "x_pct")
    }
    lines = csv_path.read_text(encoding="utf-8").split("\n")
    target_line = next(
        i for i, ln in enumerate(lines) if i and ln.split(",")[0] == str(first_train_id)
    )
    for column, bad, message in [
        ("is_goal", "2", "expected 0 or 1"),
        ("distance_m", "nan", "non-finite"),
        ("distance_m", "-1.0", "valid range"),
        ("angle_rad", "4.0", "valid range"),
        ("x_pct", "150", "outside 0-100"),
        ("distance_m", "abc", "unreadable"),
    ]:
        text = csv_path.read_text(encoding="utf-8").split("\n")
        cells = text[target_line].split(",")
        cells[index_of[column]] = bad
        text[target_line] = ",".join(cells)
        bad_csv = csv_path.with_name(f"bad_{column}_{bad}.csv")
        bad_csv.write_text("\n".join(text), encoding="utf-8", newline="")
        with pytest.raises(ContractError, match=message):
            load_partitions(bad_csv, mapping)


@pytest.mark.parametrize(
    "name",
    [
        "is_goal",
        "event_id",
        "match_id",
        "team_id",
        "player_id",
        "competition_id",
        "eligible",
        "exclusion_reasons",
        "primary_exclusion",
        "flags",
        "body_part_class",
        "x_pct",
        "y_pct",
        "x_m",
        "angle_deg",
        "period",
        "event_sec",
        "side",
        "match_datetime_utc",
        "provider_xg",
        "tags",
    ],
)
def test_feature_allowlist_rejects_everything_but_distance_and_angle(dataset, name):
    parts = open_partitions(dataset)[3]
    with pytest.raises(FeatureNotAllowedError):
        feature_matrix(parts["train"], ["distance_m", name])
    with pytest.raises(FeatureNotAllowedError):
        feature_matrix(parts["train"], [name])


def test_allowed_features_and_empty_request(dataset):
    parts = open_partitions(dataset)[3]
    x = feature_matrix(parts["train"], ["distance_m", "angle_rad"])
    assert x.shape == (len(parts["train"]), 2) and np.isfinite(x).all()
    with pytest.raises(FeatureNotAllowedError):
        feature_matrix(parts["train"], [])


def test_manifest_json_is_not_modified_by_reading(dataset):
    csv_path, manifest_path, _ = dataset
    before = manifest_path.read_text(encoding="utf-8")
    read_identity(csv_path, manifest_path)
    assert manifest_path.read_text(encoding="utf-8") == before
    assert json.loads(before)["target"] == "is_goal"
