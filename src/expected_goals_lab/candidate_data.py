"""Guarded access to the S04 candidate feature ``body_part_class``.

S03 deliberately exposes only ``distance_m`` and ``angle_rad``; ``dataset.FEATURE_ALLOWLIST`` is
left exactly as it was and S03's baselines still reject any other column. S04 needs one more
S02 column. This module reads it under the same two-pass rule as S03:

* only for eligible rows of the requested **open** splits (train, validation);
* never for the test split: asking for it raises ``SealedSplitError``, and for test rows no
  column other than ``event_id`` is subscripted;
* joined to the S03 partitions by ``event_id``, with every partition event required to appear
  exactly once and no other event admitted.

``body_part_class`` is a Wyscout tag-derived class. ``head_or_body`` comes from tag 403
("head/body") and is not a confirmed header.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expected_goals_lab.dataset import (
    OPEN_SPLITS,
    SEALED_SPLIT,
    ContractError,
    FinalTestAuthorisation,
    Partition,
    RowReader,
    SealedSplitError,
    iter_csv_rows,
)

BODY_PART_COLUMN = "body_part_class"
BODY_PART_VOCABULARY = ("right_foot", "left_foot", "head_or_body")


@dataclass(frozen=True)
class CandidatePartition:
    """An S03 open-split partition plus the guarded body-part class, aligned row by row."""

    base: Partition
    body_part: np.ndarray  # str, one of BODY_PART_VOCABULARY

    @property
    def split(self) -> str:
        return self.base.split

    def __len__(self) -> int:
        return len(self.base)


def load_body_parts(
    csv_path: Path,
    partitions: dict[str, Partition],
    split_of_event: dict[int, str],
    splits: Iterable[str] = OPEN_SPLITS,
    rows: RowReader = iter_csv_rows,
) -> dict[str, CandidatePartition]:
    """Attach ``body_part_class`` to the requested open partitions, or fail closed."""
    wanted = tuple(splits)
    if SEALED_SPLIT in wanted:
        raise SealedSplitError(
            "the test split is sealed in S04: its body-part classes (and every other attribute) "
            "are not read"
        )
    unknown = [s for s in wanted if s not in OPEN_SPLITS]
    if unknown:
        raise ValueError(f"unknown splits {unknown}")
    return _join_body_parts(csv_path, partitions, split_of_event, wanted, rows)


def load_final_test_body_parts(
    csv_path: Path,
    test_partition: Partition,
    split_of_event: dict[int, str],
    authorisation: FinalTestAuthorisation,
    rows: RowReader = iter_csv_rows,
) -> CandidatePartition:
    """Attach ``body_part_class`` to the sealed test partition. Requires an authorisation (S07)."""
    if not isinstance(authorisation, FinalTestAuthorisation):
        raise SealedSplitError("reading the test split needs a FinalTestAuthorisation")
    joined = _join_body_parts(
        csv_path, {SEALED_SPLIT: test_partition}, split_of_event, (SEALED_SPLIT,), rows
    )
    return joined[SEALED_SPLIT]


def _join_body_parts(
    csv_path: Path,
    partitions: dict[str, Partition],
    split_of_event: dict[int, str],
    wanted: tuple[str, ...],
    rows: RowReader,
) -> dict[str, CandidatePartition]:
    """Shared join. Public callers are responsible for the seal; see the loaders above."""
    missing = [s for s in wanted if s not in partitions]
    if missing:
        raise ContractError(f"no partition supplied for splits {missing}")

    found: dict[int, str] = {}
    for row in rows(csv_path):
        event_id = int(row["event_id"])
        if split_of_event.get(event_id) not in wanted:
            continue  # test or quarantined: nothing else about the row is read
        if event_id in found:
            raise ContractError(f"event {event_id} appears more than once in the body-part join")
        value = row[BODY_PART_COLUMN]
        if value not in BODY_PART_VOCABULARY:
            raise ContractError(
                f"event {event_id}: {BODY_PART_COLUMN} is {value!r}, expected one of "
                f"{BODY_PART_VOCABULARY}"
            )
        found[event_id] = value

    expected_ids = set()
    result = {}
    for split in wanted:
        base = partitions[split]
        ids = base.event_ids.tolist()
        expected_ids.update(ids)
        absent = [e for e in ids if e not in found]
        if absent:
            raise ContractError(
                f"{len(absent)} {split} events have no body-part value, e.g. {absent[:3]}"
            )
        result[split] = CandidatePartition(
            base=base, body_part=np.array([found[e] for e in ids], dtype=object)
        )
    extra = set(found) - expected_ids
    if extra:
        raise ContractError(f"{len(extra)} body-part rows do not belong to the supplied partitions")
    return result
