"""Contract between the S02 shot table and the S03 modelling code, and the test-set seal.

The S02 CSV and its manifest are the only inputs. Nothing here re-reads provider events.

Reading happens in two passes so that the sealed test split can be kept out of reach by
construction, not by convention:

1. ``read_identity`` validates the whole file and reads **only identity columns**
   (event, match, league, UTC date, eligibility). That is enough to assign every match to a split.
2. ``load_partitions`` then parses features and the goal label **only for rows of the requested
   splits**. Asking for the test split raises ``SealedSplitError``. For test rows the feature,
   location and target columns are never read at all.

The SHA-256 check necessarily reads every byte of the file; it does not interpret any label.
Label-domain checks run on the open splits only. S02's own integrity checks already covered the
whole table.
"""

import csv
import hashlib
import json
import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from expected_goals_lab.shot_table import (
    CANDIDATE_FEATURES,
    COLUMN_ROLES,
    COLUMNS,
    FEATURE_ALLOWLIST,
    SCHEMA_VERSION,
    TARGET,
)

DEFAULT_CSV = Path("data/processed/wyscout_regular_shots.csv")
OPEN_SPLITS = ("train", "validation")
SEALED_SPLIT = "test"
IDENTITY_COLUMNS = (
    "event_id",
    "match_id",
    "competition_id",
    "match_datetime_utc",
    "eligible",
    "exclusion_reasons",
)
# Read only for open-split rows. ``x_pct``/``y_pct`` are location sources, used for the
# pitch-size sensitivity check; they are never model inputs.
OPEN_COLUMNS = (*FEATURE_ALLOWLIST, "x_pct", "y_pct", TARGET)


class ContractError(ValueError):
    """The shot table or manifest does not match what S02 promised."""


class SealedSplitError(RuntimeError):
    """Code tried to read the labels or features of the sealed test split."""


class FeatureNotAllowedError(ValueError):
    """A column outside the approved feature allowlist was requested as a model input."""


RowReader = Callable[[Path], Iterable[dict]]


def iter_csv_rows(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def sha256_bytes_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Identity:
    """Eligible shots, identity columns only. No target, no features."""

    event_ids: np.ndarray
    match_ids: np.ndarray
    competition_ids: np.ndarray
    dates: np.ndarray  # 'YYYY-MM-DD' (UTC) as strings
    n_rows: int
    n_eligible: int
    csv_sha256: str
    manifest: dict = field(repr=False)


def load_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"manifest {path} is unreadable or not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ContractError(f"manifest {path} is not a JSON object")
    return manifest


def check_manifest(manifest: dict) -> None:
    """The manifest must describe exactly the schema and roles this code was written for."""
    expected = {
        "schema_version": SCHEMA_VERSION,
        "columns": list(COLUMNS),
        "column_roles": COLUMN_ROLES,
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "candidate_features": list(CANDIDATE_FEATURES),
        "target": TARGET,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ContractError(
                f"manifest field {key!r} is {manifest.get(key)!r}, expected {value!r}"
            )
    for key in ("csv_sha256", "rows", "eligible_rows"):
        if key not in manifest:
            raise ContractError(f"manifest has no {key!r}")


def read_identity(csv_path: Path, manifest_path: Path, rows: RowReader = iter_csv_rows) -> Identity:
    """Validate the contract and return the identity of every eligible shot.

    Only ``IDENTITY_COLUMNS`` are read from data rows. Quarantined rows are validated for their
    flags and then discarded: they never reach modelling.
    """
    manifest = load_manifest(manifest_path)
    check_manifest(manifest)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    actual_sha = sha256_bytes_of(csv_path)
    if actual_sha != manifest["csv_sha256"]:
        raise ContractError(
            f"{csv_path} has SHA-256 {actual_sha}, the manifest says {manifest['csv_sha256']}"
        )
    with csv_path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), None)
    if header != list(COLUMNS):
        raise ContractError(f"CSV header {header} does not match the expected columns")

    seen: set[int] = set()
    n_rows = 0
    kept: list[tuple[int, int, int, str]] = []
    match_date: dict[int, str] = {}
    for row in rows(csv_path):
        n_rows += 1
        try:
            event_id = int(row["event_id"])
        except ValueError as exc:
            raise ContractError(
                f"row {n_rows}: event_id {row['event_id']!r} is not an integer"
            ) from exc
        if event_id in seen:
            raise ContractError(f"duplicate event_id {event_id}")
        seen.add(event_id)
        eligible, reasons = row["eligible"], row["exclusion_reasons"]
        if eligible not in ("0", "1"):
            raise ContractError(f"event {event_id}: eligible is {eligible!r}, expected 0 or 1")
        if (eligible == "1") != (reasons == ""):
            raise ContractError(f"event {event_id}: eligible={eligible} but reasons={reasons!r}")
        if eligible == "0":
            continue  # quarantined rows are never admitted
        try:
            match_id, competition_id = int(row["match_id"]), int(row["competition_id"])
        except ValueError as exc:
            raise ContractError(
                f"event {event_id}: match or competition id is not an integer"
            ) from exc
        date = row["match_datetime_utc"][:10]
        if len(date) != 10 or date[4] != "-" or date[7] != "-":
            raise ContractError(
                f"event {event_id}: unusable datetime {row['match_datetime_utc']!r}"
            )
        if match_date.setdefault(match_id, date) != date:
            raise ContractError(f"match {match_id} has more than one date")
        kept.append((event_id, match_id, competition_id, date))

    if n_rows != manifest["rows"]:
        raise ContractError(f"CSV has {n_rows} rows, the manifest says {manifest['rows']}")
    if len(kept) != manifest["eligible_rows"]:
        raise ContractError(
            f"CSV has {len(kept)} eligible rows, the manifest says {manifest['eligible_rows']}"
        )
    return Identity(
        event_ids=np.array([k[0] for k in kept], dtype=np.int64),
        match_ids=np.array([k[1] for k in kept], dtype=np.int64),
        competition_ids=np.array([k[2] for k in kept], dtype=np.int64),
        dates=np.array([k[3] for k in kept], dtype="U10"),
        n_rows=n_rows,
        n_eligible=len(kept),
        csv_sha256=actual_sha,
        manifest=manifest,
    )


@dataclass(frozen=True)
class Partition:
    """Eligible shots of one open split, with features and labels."""

    split: str
    event_ids: np.ndarray
    match_ids: np.ndarray
    competition_ids: np.ndarray
    dates: np.ndarray
    features: dict[str, np.ndarray]  # only FEATURE_ALLOWLIST columns
    x_pct: np.ndarray  # location sources, for the pitch-size sensitivity check only
    y_pct: np.ndarray
    y: np.ndarray  # int8 goal labels

    def __len__(self) -> int:
        return len(self.y)


def feature_matrix(partition: Partition, names: Iterable[str]) -> np.ndarray:
    """Model inputs for ``names``. Only the approved allowlist may be requested."""
    names = tuple(names)
    bad = [n for n in names if n not in FEATURE_ALLOWLIST]
    if bad or not names:
        raise FeatureNotAllowedError(
            f"features {bad or names} are not on the approved allowlist {FEATURE_ALLOWLIST}"
        )
    return np.column_stack([partition.features[n] for n in names])


_GRANTED = object()  # module-private proof that ``FinalTestAuthorisation.grant`` was used


@dataclass(frozen=True)
class FinalTestAuthorisation:
    """The only capability that unlocks the sealed test split (S07 final evaluation).

    It can be obtained only through ``grant`` with both explicit confirmations, and constructing
    it directly fails. Possession is checked by ``load_final_test_partition`` and by the
    body-part loader's test counterpart. Nothing in S01-S06 grants it against real data.
    """

    protocol_sha256: str
    _proof: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._proof is not _GRANTED:
            raise SealedSplitError("a FinalTestAuthorisation can only be created with grant()")

    @classmethod
    def grant(cls, execute: bool, confirm_final_test: bool, protocol_sha256: str):
        if not (execute is True and confirm_final_test is True):
            raise SealedSplitError(
                "the sealed test split needs both --execute and --confirm-final-test"
            )
        return cls(protocol_sha256=protocol_sha256, _proof=_GRANTED)


def load_partitions(
    csv_path: Path,
    split_of_event: dict[int, str],
    splits: Iterable[str] = OPEN_SPLITS,
    rows: RowReader = iter_csv_rows,
) -> dict[str, Partition]:
    """Parse features and labels for the requested splits only; never for the sealed test."""
    wanted = tuple(splits)
    if SEALED_SPLIT in wanted:
        raise SealedSplitError(
            "the test split is sealed in S03: its labels and features are not read"
        )
    unknown = [s for s in wanted if s not in OPEN_SPLITS]
    if unknown:
        raise ValueError(f"unknown splits {unknown}")
    return _parse_partitions(csv_path, split_of_event, wanted, rows)


def load_final_test_partition(
    csv_path: Path,
    split_of_event: dict[int, str],
    authorisation: FinalTestAuthorisation,
    rows: RowReader = iter_csv_rows,
) -> Partition:
    """Parse the sealed test split. Requires a ``FinalTestAuthorisation`` (S07 only)."""
    if not isinstance(authorisation, FinalTestAuthorisation):
        raise SealedSplitError("reading the test split needs a FinalTestAuthorisation")
    return _parse_partitions(csv_path, split_of_event, (SEALED_SPLIT,), rows)[SEALED_SPLIT]


def _parse_partitions(
    csv_path: Path,
    split_of_event: dict[int, str],
    wanted: tuple[str, ...],
    rows: RowReader,
) -> dict[str, Partition]:
    """Shared parser. Public callers are responsible for the seal; see the two loaders above."""
    collected = {s: [] for s in wanted}
    for row in rows(csv_path):
        split = split_of_event.get(int(row["event_id"]))
        if split not in collected:
            continue  # test or quarantined: no other column is touched
        label = row[TARGET]
        if label not in ("0", "1"):
            raise ContractError(f"event {row['event_id']}: {TARGET} is {label!r}, expected 0 or 1")
        try:
            distance, angle = float(row["distance_m"]), float(row["angle_rad"])
            x_pct, y_pct = int(row["x_pct"]), int(row["y_pct"])
        except ValueError as exc:
            raise ContractError(f"event {row['event_id']}: unreadable numeric field") from exc
        if not (math.isfinite(distance) and math.isfinite(angle)):
            raise ContractError(f"event {row['event_id']}: non-finite distance or angle")
        if distance < 0 or not (0 <= angle <= math.pi + 1e-9):
            raise ContractError(
                f"event {row['event_id']}: distance/angle outside their valid range"
            )
        if not (0 <= x_pct <= 100 and 0 <= y_pct <= 100):
            raise ContractError(f"event {row['event_id']}: position outside 0-100")
        collected[split].append(
            (
                int(row["event_id"]),
                int(row["match_id"]),
                int(row["competition_id"]),
                row["match_datetime_utc"][:10],
                distance,
                angle,
                x_pct,
                y_pct,
                int(label),
            )
        )

    partitions = {}
    for split, items in collected.items():
        columns = list(zip(*items, strict=True)) if items else [[]] * 9
        partitions[split] = Partition(
            split=split,
            event_ids=np.array(columns[0], dtype=np.int64),
            match_ids=np.array(columns[1], dtype=np.int64),
            competition_ids=np.array(columns[2], dtype=np.int64),
            dates=np.array(columns[3], dtype="U10"),
            features={
                "distance_m": np.array(columns[4], dtype=float),
                "angle_rad": np.array(columns[5], dtype=float),
            },
            x_pct=np.array(columns[6], dtype=np.int64),
            y_pct=np.array(columns[7], dtype=np.int64),
            y=np.array(columns[8], dtype=np.int8),
        )
    return partitions
