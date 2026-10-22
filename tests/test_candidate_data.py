import dataclasses

import numpy as np
import pytest

from expected_goals_lab.baselines import MODEL_FEATURES, fit_baselines
from expected_goals_lab.candidate_data import (
    BODY_PART_VOCABULARY,
    CandidatePartition,
    load_body_parts,
)
from expected_goals_lab.dataset import (
    ContractError,
    FeatureNotAllowedError,
    SealedSplitError,
    feature_matrix,
    iter_csv_rows,
    load_partitions,
    read_identity,
)
from expected_goals_lab.shot_table import CANDIDATE_FEATURES, FEATURE_ALLOWLIST
from expected_goals_lab.splits import assign_date_splits, split_of_events
from tests.s03_helpers import column_index, rewrite_csv
from tests.s04_helpers import GuardedRow, guarded_reader, held_out_dates, write_s04_dataset


@pytest.fixture
def dataset(tmp_path):
    return write_s04_dataset(tmp_path)


def prepared(dataset):
    csv_path, manifest_path, _ = dataset
    identity = read_identity(csv_path, manifest_path)
    plan = assign_date_splits(identity.dates.tolist())
    mapping = split_of_events(identity, plan)
    return csv_path, mapping, load_partitions(csv_path, mapping)


def test_body_part_is_joined_by_event_id_for_open_splits_only(dataset):
    csv_path, mapping, base = prepared(dataset)
    rows = dataset[2]
    truth = {r.event_id: r.body_part_class for r in rows}

    parts = load_body_parts(csv_path, base, mapping)

    assert set(parts) == {"train", "validation"}
    for split, part in parts.items():
        assert isinstance(part, CandidatePartition) and part.split == split
        assert len(part.body_part) == len(part.base)
        assert [truth[int(e)] for e in part.base.event_ids] == part.body_part.tolist()
        assert set(part.body_part.tolist()) <= set(BODY_PART_VOCABULARY)


def test_test_split_cannot_be_requested(dataset):
    csv_path, mapping, base = prepared(dataset)
    with pytest.raises(SealedSplitError, match="sealed"):
        load_body_parts(csv_path, base, mapping, splits=("train", "test"))
    with pytest.raises(SealedSplitError):
        load_body_parts(csv_path, base, mapping, splits=("test",))


def test_no_attribute_of_a_test_row_is_read_by_the_body_part_loader(dataset):
    csv_path, manifest_path, _ = dataset
    log: list[str] = []
    reader = guarded_reader(held_out_dates(csv_path, manifest_path), log)
    identity = read_identity(csv_path, manifest_path, reader)
    plan = assign_date_splits(identity.dates.tolist())
    mapping = split_of_events(identity, plan)
    base = load_partitions(csv_path, mapping, rows=reader)

    parts = load_body_parts(csv_path, base, mapping, rows=reader)

    assert log == [] and set(parts) == {"train", "validation"}


def test_negative_control_the_trap_fires_on_a_sealed_body_part_read(dataset):
    csv_path, manifest_path, _ = dataset
    log: list[str] = []
    rows = list(guarded_reader(held_out_dates(csv_path, manifest_path), log)(csv_path))
    sealed = next(r for r in rows if r.sealed)
    assert isinstance(sealed, GuardedRow)
    with pytest.raises(AssertionError, match="sealed"):
        sealed["body_part_class"]
    with pytest.raises(AssertionError, match="sealed"):
        sealed.get("body_part_class")
    assert sealed["event_id"] and log == ["body_part_class", "body_part_class"]


def test_unexplained_categories_fail_closed(dataset, tmp_path):
    csv_path, manifest_path, rows = dataset
    _, mapping, base = prepared(dataset)
    train_id = int(base["train"].event_ids[0])
    index = column_index("body_part_class")
    for bad in ("unknown", "header", ""):

        def corrupt(lines, bad=bad):
            out = []
            for line in lines:
                cells = line.split(",")
                if cells[0] == str(train_id):
                    cells[index] = bad
                out.append(",".join(cells))
            return out

        rewrite_csv(csv_path, manifest_path, corrupt)
        with pytest.raises(ContractError, match="body_part_class"):
            load_body_parts(csv_path, base, mapping)


def test_join_must_be_exact_no_missing_no_extra_no_duplicates(dataset):
    csv_path, mapping, base = prepared(dataset)
    some_train = int(base["train"].event_ids[0])
    # an event of the partition that the mapping no longer assigns to an open split
    missing = {e: s for e, s in mapping.items() if e != some_train}
    with pytest.raises(ContractError, match="no body-part value"):
        load_body_parts(csv_path, base, missing)
    # a row the partitions do not contain
    short = {"train": base["train"], "validation": base["validation"]}
    trimmed = dataclasses.replace(
        base["train"],
        event_ids=base["train"].event_ids[1:],
        match_ids=base["train"].match_ids[1:],
        competition_ids=base["train"].competition_ids[1:],
        dates=base["train"].dates[1:],
        features={k: v[1:] for k, v in base["train"].features.items()},
        x_pct=base["train"].x_pct[1:],
        y_pct=base["train"].y_pct[1:],
        y=base["train"].y[1:],
    )
    short["train"] = trimmed
    with pytest.raises(ContractError, match="do not belong"):
        load_body_parts(csv_path, short, mapping)


def test_duplicate_rows_in_the_stream_are_rejected(dataset):
    csv_path, mapping, base = prepared(dataset)

    def doubled(path):
        for row in iter_csv_rows(path):
            yield row
            yield row

    with pytest.raises(ContractError, match="more than once"):
        load_body_parts(csv_path, base, mapping, rows=doubled)


def test_s03_allowlist_and_baselines_are_untouched_by_s04(dataset):
    _, _, base = prepared(dataset)
    assert FEATURE_ALLOWLIST == ("distance_m", "angle_rad")
    assert CANDIDATE_FEATURES == ("body_part_class",)
    assert set().union(*MODEL_FEATURES.values()) == {"distance_m", "angle_rad"}
    with pytest.raises(FeatureNotAllowedError):
        feature_matrix(base["train"], ["body_part_class"])
    with pytest.raises(FeatureNotAllowedError):
        feature_matrix(base["train"], ["distance_m", "body_part_class"])
    models = fit_baselines(base["train"])  # S03 code path still works on the same partition
    assert set(models) == {"constant_train_rate", "distance_only", "distance_plus_angle"}
    assert not hasattr(base["train"], "body_part")  # the S03 partition carries no body part


def test_partitions_alignment_survives_numpy_dtypes(dataset):
    csv_path, mapping, base = prepared(dataset)
    parts = load_body_parts(csv_path, base, mapping)
    assert parts["train"].body_part.dtype == object
    assert np.array_equal(parts["train"].base.event_ids, base["train"].event_ids)
