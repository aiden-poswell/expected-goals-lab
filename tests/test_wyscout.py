import hashlib
import io
import json
import urllib.error
import zipfile
from pathlib import Path

import pytest

from expected_goals_lab import wyscout
from expected_goals_lab.common import DataAccessError
from expected_goals_lab.wyscout import FigshareFile, WyscoutSource


def figshare_file(name: str, content: bytes, **overrides) -> FigshareFile:
    values = dict(
        name=name,
        article_id=1,
        article_version=1,
        file_id=42,
        size=len(content),
        md5=hashlib.md5(content).hexdigest(),
    )
    return FigshareFile(**{**values, **overrides})


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fail_if_called(*args, **kwargs):
    raise AssertionError("network was used")


def test_verified_cache_is_used_without_network(tmp_path, monkeypatch):
    content = b'[{"wyId": 1}]'
    file = figshare_file("competitions.json", content)
    (tmp_path / file.name).write_bytes(content)
    monkeypatch.setattr(wyscout.urllib.request, "urlopen", fail_if_called)

    source = WyscoutSource(tmp_path)

    assert source.path(file) == tmp_path / file.name
    assert source.downloads == []


def test_download_is_verified_cached_and_not_repeated(tmp_path, monkeypatch):
    content = b"a,b\n1,2\n"
    file = figshare_file("tags2name.csv", content)
    urls = []

    def fake_urlopen(url, timeout):
        urls.append(url)
        return FakeResponse(content)

    monkeypatch.setattr(wyscout.urllib.request, "urlopen", fake_urlopen)
    source = WyscoutSource(tmp_path)

    first = source.path(file)
    second = source.path(file)

    assert first == second and first.read_bytes() == content
    assert urls == ["https://ndownloader.figshare.com/files/42"]
    assert source.downloads == ["tags2name.csv"]
    assert not list(tmp_path.glob("*.part"))


def test_download_with_wrong_checksum_is_rejected_and_leaves_nothing(tmp_path, monkeypatch):
    file = figshare_file("events.zip", b"expected content")
    monkeypatch.setattr(
        wyscout.urllib.request, "urlopen", lambda url, timeout: FakeResponse(b"expected contenX")
    )

    with pytest.raises(DataAccessError, match="failed verification"):
        WyscoutSource(tmp_path).path(file)
    assert list(tmp_path.iterdir()) == []


def test_altered_cache_file_is_discarded_and_offline_fails(tmp_path, monkeypatch):
    file = figshare_file("matches.zip", b"genuine bytes")
    (tmp_path / file.name).write_bytes(b"truncated")
    monkeypatch.setattr(wyscout.urllib.request, "urlopen", fail_if_called)

    with pytest.raises(DataAccessError, match="--offline"):
        WyscoutSource(tmp_path, offline=True).path(file)
    assert not (tmp_path / file.name).exists()


def test_offline_without_cache_never_touches_the_network(tmp_path, monkeypatch):
    monkeypatch.setattr(wyscout.urllib.request, "urlopen", fail_if_called)

    with pytest.raises(DataAccessError, match="not in the cache"):
        WyscoutSource(tmp_path, offline=True).competitions()


def test_network_failure_is_an_understandable_error(tmp_path, monkeypatch):
    def offline(url, timeout):
        raise urllib.error.URLError("getaddrinfo failed")

    monkeypatch.setattr(wyscout.urllib.request, "urlopen", offline)

    with pytest.raises(DataAccessError, match="offline"):
        WyscoutSource(tmp_path).path(figshare_file("competitions.json", b"[]"))
    assert list(tmp_path.iterdir()) == []


def test_http_error_names_the_status(tmp_path, monkeypatch):
    def missing(url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(wyscout.urllib.request, "urlopen", missing)

    with pytest.raises(DataAccessError, match="HTTP 404"):
        WyscoutSource(tmp_path).path(figshare_file("competitions.json", b"[]"))


def test_pinned_file_table_is_consistent():
    assert {f.name for f in wyscout.ALL_FILES} == {
        "events.zip",
        "matches.zip",
        "competitions.json",
        "eventid2name.csv",
        "tags2name.csv",
    }
    assert all(len(f.md5) == 32 and f.size > 0 for f in wyscout.ALL_FILES)


def write_zip(path: Path, members: dict[str, str]) -> bytes:
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)
    return path.read_bytes()


def test_iter_events_streams_pretty_printed_and_empty_arrays(tmp_path, monkeypatch):
    events = [{"id": 1, "tags": [{"id": 101}]}, {"id": 2, "tags": []}, {"id": 3, "note": "é"}]
    content = write_zip(
        tmp_path / "events.zip",
        {
            "events_Test.json": json.dumps(events, indent=2, ensure_ascii=False),
            "events_Empty.json": "[ ]",
            "README.txt": "ignored",
        },
    )
    monkeypatch.setattr(wyscout, "EVENTS", figshare_file("events.zip", content))
    source = WyscoutSource(tmp_path, offline=True)

    assert source.events_members() == ["events_Empty.json", "events_Test.json"]
    assert list(source.iter_events("events_Test.json")) == events
    assert list(source.iter_events("events_Empty.json")) == []


def test_matches_flatten_all_competition_files(tmp_path, monkeypatch):
    content = write_zip(
        tmp_path / "matches.zip",
        {"matches_A.json": json.dumps([{"wyId": 1}]), "matches_B.json": json.dumps([{"wyId": 2}])},
    )
    monkeypatch.setattr(wyscout, "MATCHES", figshare_file("matches.zip", content))

    assert [m["wyId"] for m in WyscoutSource(tmp_path, offline=True).matches()] == [1, 2]
