"""S06: the narrow, token-gated route to the sealed split, and the ``fit_split`` keyword.

The public S03/S04 loaders must keep refusing the test split exactly as before; only a
``FinalTestAuthorisation`` obtained with both confirmations opens it. Synthetic data only.
"""

import dataclasses

import numpy as np
import pytest

from expected_goals_lab.baselines import fit_baselines
from expected_goals_lab.candidate_data import (
    load_body_parts,
    load_final_test_body_parts,
)
from expected_goals_lab.dataset import (
    FinalTestAuthorisation,
    SealedSplitError,
    load_final_test_partition,
    load_partitions,
    read_identity,
)
from expected_goals_lab.final_evaluation import combine_candidates, combine_partitions
from expected_goals_lab.nonlinear import fit_candidates
from expected_goals_lab.splits import assign_date_splits, split_of_events
from tests.s04_helpers import GuardedRow, held_out_dates, write_s04_dataset


@pytest.fixture
def data(tmp_path):
    csv_path, manifest_path, rows = write_s04_dataset(tmp_path)
    identity = read_identity(csv_path, manifest_path)
    plan = assign_date_splits(identity.dates.tolist())
    split_of = split_of_events(identity, plan)
    return csv_path, manifest_path, rows, split_of


def granted():
    return FinalTestAuthorisation.grant(True, True, "protocol-hash")


def test_the_public_loaders_still_refuse_the_test_split(data):
    csv_path, _, _, split_of = data
    with pytest.raises(SealedSplitError):
        load_partitions(csv_path, split_of, ("test",))
    with pytest.raises(SealedSplitError):
        load_partitions(csv_path, split_of, ("train", "test"))
    parts = load_partitions(csv_path, split_of)
    with pytest.raises(SealedSplitError):
        load_body_parts(csv_path, parts, split_of, ("test",))


@pytest.mark.parametrize("execute,confirm", [(True, False), (False, True), (False, False)])
def test_the_authorisation_needs_both_confirmations(execute, confirm):
    with pytest.raises(SealedSplitError):
        FinalTestAuthorisation.grant(execute, confirm, "protocol-hash")


def test_truthy_but_not_true_confirmations_are_refused():
    with pytest.raises(SealedSplitError):
        FinalTestAuthorisation.grant(1, "yes", "protocol-hash")


def test_the_authorisation_cannot_be_constructed_directly():
    with pytest.raises(SealedSplitError):
        FinalTestAuthorisation("protocol-hash")
    with pytest.raises(SealedSplitError):
        FinalTestAuthorisation("protocol-hash", _proof=True)
    token = granted()
    with pytest.raises(SealedSplitError):
        dataclasses.replace(token, _proof=None)


def test_the_test_loaders_reject_anything_but_an_authorisation(data):
    csv_path, _, _, split_of = data
    for fake in (None, "granted", True, object()):
        with pytest.raises(SealedSplitError):
            load_final_test_partition(csv_path, split_of, fake)
    parts = load_partitions(csv_path, split_of)
    with pytest.raises(SealedSplitError):
        load_final_test_body_parts(csv_path, parts["validation"], split_of, None)


def test_an_authorised_load_returns_exactly_the_test_rows(data):
    csv_path, _, rows, split_of = data
    test = load_final_test_partition(csv_path, split_of, granted())
    expected = [r for r in rows if r.eligible and split_of[r.event_id] == "test"]
    assert test.split == "test"
    assert sorted(test.event_ids.tolist()) == sorted(r.event_id for r in expected)
    assert int(test.y.sum()) == sum(r.is_goal for r in expected)
    with_body = load_final_test_body_parts(csv_path, test, split_of, granted())
    by_event = {r.event_id: r.body_part_class for r in expected}
    assert [by_event[e] for e in with_body.base.event_ids.tolist()] == with_body.body_part.tolist()


def test_the_open_splits_never_include_test_rows_even_when_authorisation_exists(data):
    csv_path, _, _, split_of = data
    granted()  # holding a token changes nothing about the public loader
    parts = load_partitions(csv_path, split_of)
    assert set(parts) == {"train", "validation"}
    test_events = {e for e, s in split_of.items() if s == "test"}
    for part in parts.values():
        assert test_events.isdisjoint(part.event_ids.tolist())


def test_loading_the_open_splits_never_reads_a_sensitive_test_column(data):
    csv_path, manifest_path, _, split_of = data
    trap = []
    dates = held_out_dates(csv_path, manifest_path)

    def reader(path):
        from expected_goals_lab.dataset import iter_csv_rows

        for row in iter_csv_rows(path):
            yield GuardedRow(row, row["match_datetime_utc"][:10] in dates, trap)

    parts = load_partitions(csv_path, split_of, rows=reader)
    load_body_parts(csv_path, parts, split_of, rows=reader)
    assert trap == []


def test_combined_development_partitions_keep_alignment(data):
    csv_path, _, _, split_of = data
    parts = load_body_parts(csv_path, load_partitions(csv_path, split_of), split_of)
    dev = combine_candidates([parts["train"], parts["validation"]], "development")
    assert dev.split == "development"
    assert len(dev) == len(parts["train"]) + len(parts["validation"])
    assert len(dev.body_part) == len(dev.base.y) == len(dev.base.features["distance_m"])
    expected = np.concatenate([parts["train"].base.event_ids, parts["validation"].base.event_ids])
    assert dev.base.event_ids.tolist() == expected.tolist()
    base = combine_partitions([parts["train"].base, parts["validation"].base], "development")
    assert base.event_ids.tolist() == expected.tolist()


def test_fit_split_defaults_to_train_and_never_accepts_test(data):
    csv_path, _, _, split_of = data
    parts = load_body_parts(csv_path, load_partitions(csv_path, split_of), split_of)
    train, validation = parts["train"], parts["validation"]
    assert fit_baselines(train.base)
    assert fit_candidates(train)
    with pytest.raises(ValueError):
        fit_baselines(validation.base)
    with pytest.raises(ValueError):
        fit_candidates(validation)
    dev = combine_candidates([train, validation], "development")
    with pytest.raises(ValueError):
        fit_baselines(dev.base)  # default fit_split is "train"
    with pytest.raises(ValueError):
        fit_candidates(dev)
    assert fit_baselines(dev.base, fit_split="development")
    assert fit_candidates(dev, fit_split="development")
    # the test split is refused even if a partition is relabelled as such
    mislabeled = dataclasses.replace(dev.base, split="test")
    with pytest.raises(ValueError):
        fit_baselines(mislabeled, fit_split="test")
    with pytest.raises(ValueError):
        fit_candidates(dataclasses.replace(dev, base=mislabeled), fit_split="test")
    # a partition fitted under the wrong name is refused, in both directions
    with pytest.raises(ValueError):
        fit_baselines(train.base, fit_split="development")
