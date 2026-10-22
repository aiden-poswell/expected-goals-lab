"""Repository hygiene: narrow, fast, offline checks that forbidden file kinds are absent.

This is not a secret scanner and not a legal clearance review. It checks a short, explicit list
of path patterns (dataset archives, raw Wyscout event/match files, processed shot tables,
serialized data, common credential-file names, nested Git histories) against the files this
repository actually contains, and a short grep for personal machine paths in tracked text files.
It explicitly allow-lists the few data-derived artefacts this project does publish; see
docs/DATA_LICENSE_AND_ATTRIBUTION.md for what those are and why. Passing this test is evidence
against a specific, narrow list of mistakes, not a guarantee that no sensitive content exists
anywhere in the repository.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories never walked even without Git metadata present (environments/caches/tooling, not
# repository content). Matters only for the no-.git fallback in tracked_files() below.
EXCLUDED_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}

# Published, reviewed artefacts that would otherwise match a forbidden pattern below. Each one is
# a deliberate publication decision recorded in docs/DATA_LICENSE_AND_ATTRIBUTION.md, not an
# oversight this test should hide.
ALLOWED = {
    "results/final_test_summary.json",
    "reports/final_test_execution_log.jsonl",
}

# (pattern, human label). Matched against the POSIX-style repository-relative path.
FORBIDDEN_PATTERNS = [
    (r"\.(zip|7z|tar|tar\.gz|tgz|gz|part)$", "archive file"),
    (r"(^|/)events_.*\.json$", "raw Wyscout event file"),
    (r"(^|/)matches_.*\.json$", "raw Wyscout match file"),
    (r"(^|/)(players|teams)\.json$", "raw Wyscout reference file"),
    (r"\.csv$", "processed or raw tabular data file"),
    (r"\.(parquet|pkl)$", "serialized data file"),
    (r"\.(key|pem)$", "key or credential file"),
    (r"(^|/)\.env(\..*)?$", "local environment or secrets file"),
    (r"(^|/)\.git$", "nested Git history"),
]

# Checked only in tracked text files, as a plain substring search (not a full secret scan).
#
# Built from short parts, not as complete literals: this file is itself a tracked text file and
# so is itself scanned by test_no_personal_machine_paths_in_tracked_text_files below. Writing a
# marker as one whole-string literal anywhere in this file, including in a comment, would make
# the file match itself on every run, in this checkout and in CI, regardless of any real personal
# path anywhere else. The runtime value of PERSONAL_PATH_MARKERS is exactly the same three
# strings as before; only how they are spelled in the source changes, so real matches in other
# files are unaffected.
_SEP, _BACKSLASH, _USERS, _HOME = "/", chr(92), "Users", "home"
PERSONAL_PATH_MARKERS = (
    "C:" + _BACKSLASH + _USERS + _BACKSLASH,
    _SEP + _USERS + _SEP,
    _SEP + _HOME + _SEP,
)
TEXT_SUFFIXES = {".py", ".md", ".toml", ".cff", ".json", ".jsonl", ".txt", ".yml", ".yaml"}


def personal_path_hits(text: str) -> list[str]:
    """Which entries of PERSONAL_PATH_MARKERS occur in ``text``. A pure function, so it can be
    exercised directly on synthetic text without needing a file on disk."""
    return [marker for marker in PERSONAL_PATH_MARKERS if marker in text]


def tracked_files(root: Path = REPO_ROOT) -> list[str]:
    """Git-tracked paths when ``root`` is itself a Git worktree; otherwise a manual walk.

    Git metadata is trusted only when ``root`` has its own ``.git``: running ``git ls-files``
    from a directory with no ``.git`` of its own but sitting beneath some unrelated enclosing
    Git repository does not fail, it succeeds and silently reports an empty, meaningless list
    (nothing in that outer repository's index is under this path), which would make every check
    below vacuously pass instead of actually checking anything. The manual fallback also covers
    a plain extracted source ZIP, which has no ``.git`` at all, while never descending into
    ``.venv`` or other excluded directories.
    """
    if (root / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            files = [line for line in result.stdout.splitlines() if line]
            if files:
                return files
        except (OSError, subprocess.CalledProcessError, FileNotFoundError):
            pass
    found = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        found.append(path.relative_to(root).as_posix())
    return found


@pytest.fixture(scope="module")
def files() -> list[str]:
    return tracked_files()


def forbidden_matches(paths: list[str]) -> list[tuple[str, str]]:
    """(path, label) for every path matching a forbidden pattern and not explicitly allowed."""
    found = []
    for path in paths:
        if path in ALLOWED:
            continue
        for pattern, label in FORBIDDEN_PATTERNS:
            if re.search(pattern, path):
                found.append((path, label))
                break
    return found


def test_no_forbidden_file_kinds_are_tracked(files):
    violations = forbidden_matches(files)
    assert violations == [], f"forbidden files present: {violations}"


def test_no_data_directory_is_tracked(files):
    assert not any(f == "data" or f.startswith("data/") for f in files)


def test_no_nested_git_directory_exists():
    for path in REPO_ROOT.rglob(".git"):
        if path == REPO_ROOT / ".git":
            continue
        pytest.fail(f"unexpected nested .git at {path}")


def test_no_personal_machine_paths_in_tracked_text_files(files):
    hits = []
    for rel in files:
        path = REPO_ROOT / rel
        if path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits += [(rel, marker) for marker in personal_path_hits(text)]
    assert hits == [], f"possible personal machine path(s): {hits}"


def test_gitignore_excludes_the_raw_data_directory_and_common_credential_names():
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/" in text, ".gitignore should exclude the local data/ directory"
    for needle in (".env", "*.key", "*.pem"):
        assert needle in text, f".gitignore does not mention {needle}"


def test_tracked_files_does_not_silently_return_empty_without_local_git_metadata(tmp_path):
    """A directory with no .git of its own (a plain extracted source tree) must still be walked,
    even though an enclosing, unrelated checkout elsewhere on disk may have its own .git."""
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1", encoding="utf-8")
    assert set(tracked_files(tmp_path)) == {"README.md", "src/a.py"}


def test_tracked_files_fallback_skips_excluded_directories(tmp_path):
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_text("", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("", encoding="utf-8")
    assert tracked_files(tmp_path) == ["README.md"]


# --- positive regression: the checker itself behaves as intended --------------------------------


def test_the_check_rejects_a_synthetic_forbidden_path():
    """A deliberately planted forbidden-looking path is caught, proving the check is not inert."""
    violations = forbidden_matches(
        ["data/processed/wyscout_regular_shots.csv", "events_England.json", "secrets/aws.pem"]
    )
    assert len(violations) == 3
    assert {label for _, label in violations} == {
        "processed or raw tabular data file",
        "raw Wyscout event file",
        "key or credential file",
    }


def test_the_check_does_not_reject_an_allowed_published_artefact():
    assert forbidden_matches(["results/final_test_summary.json"]) == []
    assert forbidden_matches(["reports/final_test_execution_log.jsonl"]) == []
    assert forbidden_matches(["reports/shot_origin_audit.md"]) == []
    assert forbidden_matches(["figures/portfolio/fig1_heldout_evidence.png"]) == []
    assert forbidden_matches(["figures/portfolio/fig1_heldout_evidence.svg"]) == []


def test_personal_path_detection_catches_realistic_synthetic_examples():
    """The marker-from-parts rewrite must not have weakened detection of a real personal path.

    These example paths are themselves built from parts for the same reason the markers are: any
    of the three marker strings written directly in this file's source, even inside a comment or
    docstring, would make the self-scan above match this very line, in every checkout. Composing
    text at call time, instead of writing it as a string literal, keeps that substring out of the
    file bytes that get scanned while still exercising the real detection logic end to end.
    """
    windows_path = "C:" + _BACKSLASH + _USERS + _BACKSLASH + "someone" + _BACKSLASH + "project"
    mac_path = _SEP + _USERS + _SEP + "someone" + _SEP + "project"
    linux_path = _SEP + _HOME + _SEP + "someone" + _SEP + "project"
    assert personal_path_hits(windows_path) != []
    assert personal_path_hits(mac_path) != []
    assert personal_path_hits(linux_path) != []
    assert personal_path_hits("reports/shot_origin_audit.md has no such path") == []


def test_personal_path_markers_are_not_spelled_out_verbatim_in_this_files_own_source():
    """Guards the fix itself: if a future edit reintroduces a marker as a whole-string literal,
    this file would again match itself, in this checkout and in CI. Checked directly against the
    file's own bytes, independent of pytest's or Git's view of "tracked"."""
    source = Path(__file__).read_text(encoding="utf-8")
    assert personal_path_hits(source) == []


def test_this_check_does_not_overclaim_its_own_scope():
    """The module docstring must keep stating what this test is not, so it is never cited as more
    than it is."""
    assert __doc__ is not None
    assert "not a secret scanner" in __doc__
    assert "not a legal clearance" in __doc__
