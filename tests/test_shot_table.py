"""Tests of the canonical shot table on small invented fixtures shaped like the real files."""

import csv
import itertools
import json
import math
from pathlib import Path

import pytest

from expected_goals_lab import build_shots, shot_table
from expected_goals_lab.shot_table import (
    COLUMNS,
    INVALID_PERIOD_OR_TIME,
    INVALID_POSITION,
    LABEL_UNKNOWN,
    MATCH_SCORE_MISMATCH,
    SUSPECT_POSITION,
    IntegrityError,
    build_table,
    check_integrity,
    make_row,
    process_match,
    write_csv,
)
from expected_goals_lab.wyscout import DEFAULT_CACHE_DIR, WyscoutSource
from expected_goals_lab.wyscout_shots import match_team_ids, parse_shot, shot_kind
from tests.wyscout_helpers import (
    AWAY,
    AWAY_PLAYER,
    CORNER,
    FREE_KICK_SHOT,
    HOME,
    PENALTY,
    SAVE_ATTEMPT,
    TOUCH,
    make_event,
    make_match,
    make_shot,
)


class FakeSource:
    """Duck-typed stand-in for WyscoutSource: files of events, already grouped by member."""

    def __init__(self, matches, members):
        self._matches = matches
        self._members = members
        self.downloads = []

    def competitions(self):
        return [
            {"wyId": 364, "name": "Test League", "type": "club"},
            {"wyId": 102, "name": "Test Cup", "type": "international"},
        ]

    def matches(self):
        return self._matches

    def events_members(self):
        return sorted(self._members)

    def iter_events(self, member):
        return iter(self._members[member])


def goal_scenario_events(match_id=5000, base=0):
    """Home 4 - 1 Away: one goal each from a regular shot, a free kick, a penalty and an own
    goal (all home), plus a goal scored directly from a corner (away)."""
    kw = {"match_id": match_id}
    return [
        make_shot(base + 1, HOME, tags=(101, 402), sec=100.0, **kw),
        make_event(base + 2, AWAY, SAVE_ATTEMPT, (101,), sec=101.0, player_id=AWAY_PLAYER, **kw),
        make_shot(base + 3, HOME, kind=FREE_KICK_SHOT, tags=(101, 402), sec=200.0, **kw),
        make_shot(base + 4, HOME, kind=PENALTY, tags=(101, 402), sec=300.0, **kw),
        make_shot(base + 5, AWAY, tags=(401, 1802), sec=400.0, player_id=AWAY_PLAYER, **kw),
        make_event(base + 6, AWAY, TOUCH, (102,), sec=500.0, player_id=AWAY_PLAYER, **kw),
        make_event(base + 7, AWAY, CORNER, (101,), sec=600.0, player_id=AWAY_PLAYER, **kw),
    ]


def test_only_regular_shots_get_rows_and_goals_come_from_the_shot_itself():
    match = make_match(home_score=4, away_score=1)

    result = process_match(match, goal_scenario_events(), "Test League")

    assert [r.event_id for r in result.rows] == [1, 5]
    assert [r.is_goal for r in result.rows] == [1, 0]
    assert all(r.eligible for r in result.rows)
    assert result.kind_events == {"open_play": 2, "free_kick": 1, "penalty": 1}
    assert result.kind_goals == {"open_play": 1, "free_kick": 1, "penalty": 1}
    assert result.other_goal_events == 1  # the corner; the keeper's tag is not a goal event
    assert result.own_goal_events == 1
    assert (result.published_goals, result.rebuilt_goals) == (5, 5)
    assert result.mismatch is None


def test_keeper_goal_tag_never_labels_a_shot_and_scoreboard_never_picks_one():
    # Published 1-0 for the away side, but only the home keeper's event carries the Goal tag.
    match = make_match(home_score=0, away_score=1)
    events = [
        make_shot(1, AWAY, tags=(402, 1802), sec=10.0, player_id=AWAY_PLAYER),
        make_shot(2, AWAY, tags=(401, 1801), sec=20.0, player_id=AWAY_PLAYER),
        make_event(3, HOME, SAVE_ATTEMPT, (101, 1802), sec=22.0),
    ]
    match["teamsData"][str(AWAY)]["formation"]["lineup"][0]["goals"] = "1"

    result = process_match(match, events, "Test League")

    assert result.mismatch is not None
    assert result.mismatch["published"] == {HOME: 0, AWAY: 1}
    assert result.mismatch["rebuilt"] == {HOME: 0, AWAY: 0}
    assert result.mismatch["lineup_credited_scorers"] == [(str(AWAY), AWAY_PLAYER, "1")]
    assert len(result.mismatch["credited_scorer_shot_events"]) == 2
    assert [r.is_goal for r in result.rows] == [0, 0]  # labels untouched
    assert all(not r.eligible for r in result.rows)
    assert {r.primary_exclusion for r in result.rows} == {MATCH_SCORE_MISMATCH}


def test_other_matches_are_unaffected_by_a_mismatched_match():
    bad = make_match(5000, home_score=0, away_score=1)
    good = make_match(5001, home_score=1, away_score=0, dateutc="2017-10-01 12:00:00")
    events = [
        make_shot(1, HOME, tags=(402,), match_id=5000, sec=5.0),
        make_shot(2, HOME, tags=(101, 402), match_id=5001, sec=5.0),
    ]
    members = {"events_Test.json": events}

    result = build_table(FakeSource([bad, good], members))

    by_match = {r.match_id: r for r in result.rows}
    assert by_match[5000].eligible == 0 and by_match[5001].eligible == 1
    assert result.leagues[364].mismatched_matches == [5000]


def test_suspect_position_is_quarantined_not_corrected_and_boundary_is_exact():
    match = make_match()
    low = parse_shot(make_shot(1, positions=[{"x": 19, "y": 40}]), match)
    edge = parse_shot(make_shot(2, positions=[{"x": 20, "y": 40}]), match)

    row_low = make_row(low, match, "Test League", False)
    row_edge = make_row(edge, match, "Test League", False)

    assert (row_low.eligible, row_low.primary_exclusion) == (0, SUSPECT_POSITION)
    assert (row_low.x_pct, row_low.y_pct) == (19, 40)  # raw coordinates kept, nothing flipped
    assert row_low.x_m == pytest.approx(19 * 1.05)
    assert row_low.distance_m > 80  # still describes the recorded position
    assert row_edge.eligible == 1
    assert "OWN_HALF_POSITION" in row_edge.flags


def test_position_evidence_distinguishes_mirror_and_recorded_support():
    match = make_match()
    shot_low = make_shot(1, HOME, tags=(402,), sec=100.0, positions=[{"x": 6, "y": 57}])
    # Opponent event 1 s earlier at (6, 68) in its own frame: mirrors to (94, 32) in the shooter's
    # frame, far from the recorded (6, 57) but close to its mirror image (94, 43).
    opponent_mirror = make_event(2, AWAY, TOUCH, (), sec=99.0, positions=[{"x": 6, "y": 68}])
    result = process_match(match, [opponent_mirror, shot_low], "Test League")
    assert [s.verdict for s in result.suspects] == ["mirror_supported"]

    opponent_recorded = make_event(3, AWAY, TOUCH, (), sec=99.0, positions=[{"x": 94, "y": 43}])
    result = process_match(match, [opponent_recorded, shot_low], "Test League")
    assert [s.verdict for s in result.suspects] == ["recorded_supported"]

    keeper_only = make_event(4, AWAY, SAVE_ATTEMPT, (), sec=101.0, positions=[{"x": 94, "y": 43}])
    result = process_match(match, [shot_low, keeper_only], "Test League")
    assert [s.verdict for s in result.suspects] == ["no_evidence"]  # keeper events are ignored
    assert result.suspects[0].opponent_keeper_response is True


def test_several_problems_are_all_listed_in_precedence_order():
    match = make_match()
    shot = parse_shot(make_shot(1, period="E1", positions=[{"x": 5, "y": 50}]), match)

    row = make_row(shot, match, "Test League", True)

    assert (
        row.exclusion_reasons
        == f"{MATCH_SCORE_MISMATCH};{SUSPECT_POSITION};{INVALID_PERIOD_OR_TIME}"
    )
    assert row.primary_exclusion == MATCH_SCORE_MISMATCH
    assert row.eligible == 0


@pytest.mark.parametrize(
    ("event_kwargs", "expected_reason"),
    [
        ({"positions": []}, INVALID_POSITION),
        ({"positions": [{"x": 101, "y": 50}]}, INVALID_POSITION),
        ({"positions": [{"x": 50.5, "y": 50}]}, INVALID_POSITION),
        ({"period": "P"}, INVALID_PERIOD_OR_TIME),
        ({"sec": -1.0}, INVALID_PERIOD_OR_TIME),
    ],
)
def test_bad_positions_and_times_are_quarantined_with_rows_kept(event_kwargs, expected_reason):
    match = make_match()
    row = make_row(parse_shot(make_shot(1, **event_kwargs), match), match, "Test League", False)

    assert row.eligible == 0 and row.primary_exclusion == expected_reason
    if expected_reason == INVALID_POSITION:
        assert (row.x_pct, row.x_m, row.distance_m, row.angle_rad) == (None, None, None, None)


def test_unknown_label_is_quarantined_and_never_defaulted_to_a_miss():
    match = make_match()
    event = make_shot(1)
    event["tags"] = None

    row = make_row(parse_shot(event, match), match, "Test League", False)

    assert row.is_goal is None
    assert (row.eligible, row.primary_exclusion) == (0, LABEL_UNKNOWN)


def test_informational_flags_do_not_exclude():
    match = make_match()
    event = make_shot(1, player_id=0, tags=(201,))  # no body-part tag, unknown player

    row = make_row(parse_shot(event, match), match, "Test League", False)

    assert row.eligible == 1
    assert row.body_part_class == "unknown"
    assert set(row.flags.split(";")) == {"UNKNOWN_PLAYER", "BODY_PART_UNKNOWN"}
    assert row.player_id is None


def test_body_part_classes_and_geometry_columns_on_a_known_shot():
    match = make_match()
    right = make_row(parse_shot(make_shot(1, tags=(402,)), match), match, "L", False)
    left = make_row(parse_shot(make_shot(2, tags=(401,)), match), match, "L", False)
    head = make_row(parse_shot(make_shot(3, tags=(403,)), match), match, "L", False)
    assert [r.body_part_class for r in (right, left, head)] == [
        "right_foot",
        "left_foot",
        "head_or_body",
    ]

    centre = make_row(
        parse_shot(make_shot(4, positions=[{"x": 100, "y": 50}]), match), match, "L", False
    )
    assert (centre.x_m, centre.y_m) == (105.0, 34.0)
    assert centre.angle_deg == pytest.approx(180.0) and "ON_GOAL_LINE" in centre.flags
    assert 0 <= centre.angle_rad <= math.pi


def test_away_team_rows_use_the_event_team_not_row_order():
    match = make_match()
    events = [make_shot(1, AWAY, player_id=AWAY_PLAYER), make_shot(2, HOME, sec=20.0)]

    rows = process_match(match, events, "Test League").rows

    assert [(r.team_id, r.opponent_team_id, r.side) for r in rows] == [
        (AWAY, HOME, "away"),
        (HOME, AWAY, "home"),
    ]


def test_identity_failures_become_parse_errors_and_fail_the_integrity_check():
    match = make_match()
    events = [make_shot(1, HOME), make_shot(2, 999)]
    source = FakeSource([match], {"events_Test.json": events})

    result = build_table(source)

    assert len(result.rows) == 1 and len(result.parse_errors) == 1
    with pytest.raises(IntegrityError, match="identity validation"):
        check_integrity(result)


def test_duplicate_event_ids_fail_the_integrity_check():
    a, b = make_match(5000), make_match(5001, dateutc="2017-10-01 12:00:00")
    events = [
        make_shot(7, HOME, match_id=5000, tags=(101,)),
        make_shot(7, HOME, match_id=5001),  # same event id in another match
    ]
    a["teamsData"][str(HOME)]["score"] = 1
    b["teamsData"][str(HOME)]["score"] = 0

    result = build_table(FakeSource([a, b], {"events_Test.json": events}))

    with pytest.raises(IntegrityError, match="unique"):
        check_integrity(result)


def test_non_contiguous_or_unknown_matches_are_rejected():
    a, b = make_match(5000), make_match(5001)
    interleaved = [
        make_shot(1, match_id=5000),
        make_shot(2, match_id=5001),
        make_shot(3, match_id=5000),
    ]
    with pytest.raises(IntegrityError, match="not contiguous"):
        build_table(FakeSource([a, b], {"events_Test.json": interleaved}))
    with pytest.raises(IntegrityError, match="unknown match"):
        build_table(FakeSource([a], {"events_Test.json": [make_shot(1, match_id=9)]}))


def test_matches_without_events_fail_and_international_files_are_not_read():
    a, b = make_match(5000), make_match(5001)
    with pytest.raises(IntegrityError, match="no events"):
        build_table(FakeSource([a, b], {"events_Test.json": [make_shot(1, match_id=5000)]}))

    cup = make_match(6000, competition_id=102)
    cup_events = [make_shot(10, match_id=6000)]
    result = build_table(
        FakeSource(
            [a, cup],
            {"events_Test.json": [make_shot(1, match_id=5000)], "events_Cup.json": cup_events},
        )
    )
    assert [r.match_id for r in result.rows] == [5000]
    assert result.out_of_scope == {
        "events_Cup.json": {"competition": "Test Cup", "metadata_matches": 1}
    }


def fixture_source(match_order=(0, 1)):
    matches = [
        make_match(5000, home_score=1, away_score=0, dateutc="2017-09-30 16:30:00"),
        make_match(5001, home_score=0, away_score=1, dateutc="2017-08-12 15:00:00"),
    ]
    groups = [
        [
            make_shot(1, HOME, tags=(101, 402), sec=50.0, match_id=5000),
            make_shot(
                2, AWAY, tags=(403,), sec=20.0, match_id=5000, period="2H", player_id=AWAY_PLAYER
            ),
        ],
        [make_shot(3, AWAY, tags=(101, 401), sec=70.0, match_id=5001, player_id=AWAY_PLAYER)],
    ]
    events = list(itertools.chain.from_iterable(groups[i] for i in match_order))
    return FakeSource(matches, {"events_Test.json": events})


def test_output_is_deterministic_and_independent_of_event_file_order(tmp_path):
    first = build_table(fixture_source((0, 1)))
    second = build_table(fixture_source((1, 0)))
    write_csv(first.rows, tmp_path / "a.csv")
    write_csv(second.rows, tmp_path / "b.csv")

    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    raw = (tmp_path / "a.csv").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    rows = list(csv.DictReader((tmp_path / "a.csv").open(encoding="utf-8")))
    assert tuple(rows[0].keys()) == COLUMNS
    # chronological by match date first (5001 is earlier), then period, time and event id
    assert [r["event_id"] for r in rows] == ["3", "1", "2"]
    assert shot_table.sha256_file(tmp_path / "a.csv") == shot_table.sha256_file(tmp_path / "b.csv")


def test_every_column_has_one_role_and_the_feature_allowlist_is_strict():
    assert set(shot_table.COLUMN_ROLES) == set(COLUMNS)
    assert shot_table.FEATURE_ALLOWLIST == ("distance_m", "angle_rad")
    assert shot_table.CANDIDATE_FEATURES == ("body_part_class",)
    allowed = set(shot_table.FEATURE_ALLOWLIST) | set(shot_table.CANDIDATE_FEATURES)
    assert shot_table.TARGET == "is_goal" and shot_table.TARGET not in allowed
    identifiers = {c for c, role in shot_table.COLUMN_ROLES.items() if role == "identifier"}
    quality = {c for c, role in shot_table.COLUMN_ROLES.items() if role == "quality"}
    assert not allowed & (identifiers | quality)
    # No provider tag or post-shot field is written to the table at all.
    forbidden = ("tag", "accurate", "blocked", "zone", "end_", "xg", "save", "score", "result")
    assert not [c for c in COLUMNS if any(word in c for word in forbidden)]


def test_integrity_checks_pass_on_a_clean_build():
    result = build_table(fixture_source())
    lines = check_integrity(result)
    assert all(outcome == "pass" for _, outcome in lines)


def test_cli_end_to_end_writes_csv_manifest_and_report_reproducibly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(build_shots, "WyscoutSource", lambda *a, **k: fixture_source())
    out, report = tmp_path / "processed" / "shots.csv", tmp_path / "report.md"
    argv = ["--out", str(out), "--report", str(report), "--cache-dir", str(tmp_path / "c")]

    assert build_shots.main(argv) == 0
    first = out.read_bytes()
    manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["csv_sha256"] == shot_table.sha256_file(out)
    assert manifest["rows"] == 3 and manifest["columns"] == list(COLUMNS)
    text = report.read_text(encoding="utf-8")
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7."):
        assert heading in text
    assert "CC BY 4.0" in text and "10.6084/m9.figshare.c.4415000.v5" in text

    assert build_shots.main(argv) == 0
    assert out.read_bytes() == first
    capsys.readouterr()


def test_cli_offline_without_cache_exits_2_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "shots.csv"

    code = build_shots.main(
        [
            "--offline",
            "--cache-dir",
            str(tmp_path / "empty"),
            "--out",
            str(out),
            "--report",
            str(tmp_path / "r.md"),
        ]
    )

    assert code == 2
    assert not out.exists() and not (tmp_path / "r.md").exists()
    assert "--offline" in capsys.readouterr().err


def test_cli_integrity_failure_exits_1_and_writes_nothing(tmp_path, monkeypatch, capsys):
    source = FakeSource([make_match()], {"events_Test.json": [make_shot(1, 999)]})
    monkeypatch.setattr(build_shots, "WyscoutSource", lambda *a, **k: source)
    out = tmp_path / "shots.csv"

    code = build_shots.main(["--out", str(out), "--report", str(tmp_path / "r.md")])

    assert code == 1 and not out.exists()
    assert "integrity error" in capsys.readouterr().err


# --- statistics helpers used in the report -------------------------------------------------


def test_wilson_interval_and_rank_helpers_against_known_values():
    low, high = build_shots.wilson(5, 10)
    assert (low, high) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    assert build_shots.wilson(0, 50)[0] == pytest.approx(0.0, abs=1e-12)
    assert build_shots.average_ranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]
    assert build_shots.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert build_shots.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert build_shots.quantiles([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]
    # degenerate inputs must not crash the report
    assert math.isnan(build_shots.spearman([2, 2, 2], [1, 2, 3]))
    assert all(math.isnan(v) for v in build_shots.quantiles([]))
    assert build_shots.geometry_sensitivity([]) == []


# --- integration on the real cached archive (skipped on a clean checkout) -----------------

CACHED = (Path(DEFAULT_CACHE_DIR) / "events.zip").exists()


@pytest.mark.skipif(not CACHED, reason="Wyscout cache absent; run the build_shots CLI once")
def test_first_real_matches_produce_valid_rows():
    source = WyscoutSource(offline=True)
    matches = {m["wyId"]: m for m in source.matches()}
    competition = "English first division"
    checked = 0
    stream = source.iter_events("events_England.json")
    for match_id, group in itertools.islice(
        itertools.groupby(stream, key=lambda e: e["matchId"]), 3
    ):
        events = list(group)
        result = process_match(matches[match_id], events, competition)
        expected = sum(1 for e in events if shot_kind(e) == "open_play")
        assert len(result.rows) == expected > 0
        assert result.parse_errors == []
        assert result.mismatch is None  # these matches reconcile
        for row in result.rows:
            assert row.eligible == 1
            assert 0 <= row.angle_rad <= math.pi and row.distance_m > 0
            assert {row.team_id, row.opponent_team_id} == set(match_team_ids(matches[match_id]))
            checked += 1
    assert checked > 40
