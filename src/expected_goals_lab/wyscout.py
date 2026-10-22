"""Read the Pappalardo-Massucco Wyscout open event dataset from Figshare, with a local cache.

Source: Figshare collection "Soccer match event dataset", DOI 10.6084/m9.figshare.c.4415000.v5.
Paper: Pappalardo et al. (2019), Scientific Data 6:236, https://doi.org/10.1038/s41597-019-0247-7.
Licence: CC BY 4.0 (stated on every Figshare article below and in the paper). See
docs/WYSCOUT_DATA_AUDIT.md for attribution requirements and what was and was not verified.

Each file is pinned by Figshare file ID, size and MD5, all read from the Figshare API on
2026-10-20 (article versions listed below were the latest at that time). A cached file is
reused only if its size and MD5 match; a download that does not match is discarded.
"""

import csv
import hashlib
import http.client
import io
import json
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from expected_goals_lab.common import DataAccessError

DEFAULT_CACHE_DIR = Path("data/cache/wyscout")
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/{file_id}"

COLLECTION_DOI = "10.6084/m9.figshare.c.4415000.v5"
COLLECTION_URL = "https://doi.org/10.6084/m9.figshare.c.4415000"
PAPER_DOI = "10.1038/s41597-019-0247-7"
LICENCE = "CC BY 4.0"
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"


@dataclass(frozen=True)
class FigshareFile:
    name: str
    article_id: int
    article_version: int
    file_id: int
    size: int
    md5: str


EVENTS = FigshareFile(
    "events.zip", 7770599, 1, 14464685, 77323413, "7c20e8647e7eda58d7838a0c7b1ec6ab"
)
MATCHES = FigshareFile(
    "matches.zip", 7770422, 1, 14464622, 645097, "51d80beb17480919f69a53a0152c2d71"
)
COMPETITIONS = FigshareFile(
    "competitions.json", 7765316, 4, 15073685, 1209, "3dc210a4805dda5337b0ff9f7eaa407a"
)
EVENT_NAMES = FigshareFile(
    "eventid2name.csv", 11743836, 1, 21385245, 1001, "46daf16100ece0c743eedc9adcfea162"
)
TAG_NAMES = FigshareFile(
    "tags2name.csv", 11743818, 1, 21385239, 1754, "e7acb14918d00e40c80a898b1da8fc39"
)
ALL_FILES = (EVENTS, MATCHES, COMPETITIONS, EVENT_NAMES, TAG_NAMES)


class WyscoutSource:
    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR, offline: bool = False):
        self.cache_dir = Path(cache_dir)
        self.offline = offline  # never touch the network; fail if a file is not cached
        self.downloads: list[str] = []  # names of files fetched by this instance

    def path(self, file: FigshareFile) -> Path:
        """Local path of a verified copy of ``file``, downloading it if necessary."""
        target = self.cache_dir / file.name
        if target.exists():
            if _matches(target, file):
                return target
            target.unlink()  # truncated or altered: do not trust it
        if self.offline:
            raise DataAccessError(
                f"{file.name} is not in the cache ({self.cache_dir}) and --offline was given. "
                "Run without --offline once to download it."
            )
        self._download(file, target)
        self.downloads.append(file.name)
        return target

    def competitions(self) -> list[dict]:
        return json.loads(self.path(COMPETITIONS).read_text(encoding="utf-8"))

    def matches(self) -> list[dict]:
        """All match records (one JSON file per competition inside matches.zip)."""
        records: list[dict] = []
        with zipfile.ZipFile(self.path(MATCHES)) as archive:
            for name in sorted(n for n in archive.namelist() if n.endswith(".json")):
                records.extend(json.loads(archive.read(name).decode("utf-8")))
        return records

    def events_members(self) -> list[str]:
        with zipfile.ZipFile(self.path(EVENTS)) as archive:
            return sorted(n for n in archive.namelist() if n.endswith(".json"))

    def iter_events(self, member: str) -> Iterator[dict]:
        """Yield the events of one competition file inside events.zip, one at a time.

        Each file is a single JSON array of 150-190 MB. Decoding it with ``json.loads`` would
        hold every event in memory at once; this holds the text and one decoded event.
        """
        with zipfile.ZipFile(self.path(EVENTS)) as archive:
            text = archive.read(member).decode("utf-8")
        decoder = json.JSONDecoder()
        position = text.index("[") + 1
        while True:
            while text[position] in " \t\r\n,":
                position += 1
            if text[position] == "]":
                return
            event, position = decoder.raw_decode(text, position)
            yield event

    def event_names(self) -> list[dict]:
        return _read_csv(self.path(EVENT_NAMES))

    def tag_names(self) -> list[dict]:
        return _read_csv(self.path(TAG_NAMES))

    @staticmethod
    def _download(file: FigshareFile, target: Path, timeout: float = 120.0) -> None:
        url = DOWNLOAD_URL.format(file_id=file.file_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        digest = hashlib.md5()
        try:
            with (
                urllib.request.urlopen(url, timeout=timeout) as response,
                partial.open("wb") as out,
            ):
                while chunk := response.read(1 << 20):
                    digest.update(chunk)
                    out.write(chunk)
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            raise DataAccessError(f"HTTP {exc.code} for {url}") from exc
        except (OSError, http.client.HTTPException) as exc:
            partial.unlink(missing_ok=True)
            raise DataAccessError(
                f"Could not download {file.name} from {url} ({exc}). "
                "If you are offline, run once with network access to fill the cache."
            ) from exc
        if partial.stat().st_size != file.size or digest.hexdigest() != file.md5:
            actual = f"{partial.stat().st_size} bytes, md5 {digest.hexdigest()}"
            partial.unlink()
            raise DataAccessError(
                f"{file.name} failed verification: expected {file.size} bytes, md5 {file.md5}; "
                f"received {actual}. The published file may have changed."
            )
        partial.replace(target)


def _matches(path: Path, file: FigshareFile) -> bool:
    if path.stat().st_size != file.size:
        return False
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest() == file.md5


def _read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
