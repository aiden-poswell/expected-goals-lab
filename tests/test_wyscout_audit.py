import itertools
from pathlib import Path

import pytest

from expected_goals_lab import wyscout_audit as audit
from expected_goals_lab.wyscout import DEFAULT_CACHE_DIR, WyscoutSource
from expected_goals_lab.wyscout_shots import match_team_ids, parse_shot, shot_kind
from tests.wyscout_helpers import (
    AWAY,
    AWAY_PLAYER,
    CORNER,
    HOME,
    HOME_PLAYER,
    PENALTY,
    SAVE_ATTEMPT,
    make_event,
    make_match,
    make_shot,
)


def matches_by_id(*matches):
    return {m["wyId"]: m for m in matches}


def test_scan_counts_identity_ordering_and_duplicates():
    match = make_match()
    events = [
        make_event(1, HOME, sec=5.0),
        make_event(2, AWAY, sec=4.0),  # earlier than the previous event in the same period
        make_event(2, HOME, sec=6.0),  # duplicate id
        make_event(4, 999, sec=7.0),  # team not in the match
        make_event(5, HOME, match_id=777),  # match not in metadata
    ]

    scan = audit.scan_events(events, matches_by_id(match), set())

    assert scan.n_events == 5
    assert scan.unknown_match == 1
    assert scan.unknown_team == 1
    assert scan.duplicate_ids == 1
    assert scan.order_violations == 1
    assert scan.match_ids == {5000: 4}


def test_scan_tallies_shots_goal_tags_and_coordinates():
    match = make_match()
    events = [
        make_shot(1, tags=(101, 402)),
        make_shot(2, tags=(401,), positions=[{"x": 4, "y": 50}, {"x": 100, "y": 100}]),
        make_shot(3, kind=PENALTY, tags=(402, 101), positions=[{"x": 89, "y": 50}]),
        make_shot(4, kind=PENALTY, tags=(402,), period="P", sec=1.0),
        make_shot(5, tags=(402,), player_id=0, positions=[{"x": 150, "y": 50}]),
        make_event(6, AWAY, SAVE_ATTEMPT, (101,), sec=11.0),
        make_event(7, AWAY, CORNER, (101,), sec=12.0),
    ]

    scan = audit.scan_events(events, matches_by_id(match), set())

    assert scan.shot_events[("open_play", False)] == 3
    assert scan.shot_events[("penalty", False)] == 1 and scan.shot_events[("penalty", True)] == 1
    assert scan.goal_tag_by_type[("Shot", "Shot")] == 1
    assert scan.goal_tag_by_type[("Save attempt", "Save attempt")] == 1
    assert scan.penalty_positions == {(89, 50): 1}
    assert scan.open_play_x_below_10 == {5000: 1}
    assert scan.shot_player_id_zero == 1 and scan.shot_position_out_of_range == 1
    assert scan.second_positions == {"(0,0)": 2, "(100,100)": 1, "absent": 2}


def test_reconciliation_agrees_and_reports_disagreement():
    match = make_match(home_score=1, away_score=0)
    ok_events = [
        make_shot(1, HOME, tags=(101, 402)),
        make_event(2, AWAY, SAVE_ATTEMPT, (101,)),  # must not give the away team a goal
    ]
    scan = audit.scan_events(ok_events, matches_by_id(match), set())

    assert audit.reconcile([scan], matches_by_id(match)) == (2, 2, [])

    scan = audit.scan_events([make_shot(1, HOME, tags=(402,))], matches_by_id(match), set())
    agree, total, failures = audit.reconcile([scan], matches_by_id(match))
    assert (agree, total) == (1, 2)
    assert len(failures) == 1 and "team 111: events give 0, record says 1" in failures[0]


def test_spread_is_even_and_deterministic():
    items = [{"n": i} for i in range(20)]
    assert [m["n"] for m in audit.spread(items, 5)] == [2, 6, 10, 14, 18]
    assert audit.spread(items, 50) == items


def test_select_sample_takes_domestic_by_date_and_international_shootouts():
    competitions = [
        {"wyId": 1, "type": "club", "name": "League"},
        {"wyId": 2, "type": "international", "name": "Cup"},
    ]
    league = [
        {"wyId": i, "competitionId": 1, "dateutc": f"2018-01-{20 - i:02d}", "duration": "Regular"}
        for i in range(1, 11)
    ]
    cup = [
        {"wyId": 100, "competitionId": 2, "dateutc": "2018-06-01", "duration": "Regular"},
        {"wyId": 101, "competitionId": 2, "dateutc": "2018-06-02", "duration": "Penalties"},
    ]

    forward = audit.select_sample(league + cup, competitions, 2)
    backward = audit.select_sample(list(reversed(league + cup)), competitions, 2)

    assert [m["wyId"] for m in forward] == [
        8,
        3,
        101,
    ]  # date order is ids 10..1; picks indices 2 and 7
    assert forward == backward


def test_audit_sample_validates_shooters_against_the_squad():
    match = make_match()
    events = [
        make_event(1, HOME, tags=(), player_id=HOME_PLAYER),
        make_event(2, AWAY, tags=(), player_id=AWAY_PLAYER),
        make_shot(3, HOME, player_id=HOME_PLAYER),
        make_shot(4, AWAY, player_id=AWAY_PLAYER),
        make_shot(5, AWAY, player_id=HOME_PLAYER),  # home player credited to the away team
        make_shot(6, 999),  # team outside the match
    ]
    scan = audit.scan_events(events, matches_by_id(match), {match["wyId"]})

    result = audit.audit_sample([scan], matches_by_id(match))

    assert [s.event_id for s in result.shots] == [3, 4, 5]
    assert len(result.errors) == 1 and "999" in result.errors[0]
    assert result.shooter_not_in_squad == 1
    assert result.team_ids_ok == 0  # team 999 appears among the match's events


def test_main_offline_without_cache_exits_2_and_writes_no_report(tmp_path, capsys):
    report = tmp_path / "report.md"

    code = audit.main(
        ["--offline", "--cache-dir", str(tmp_path / "empty"), "--report", report.as_posix()]
    )

    assert code == 2
    assert not report.exists()
    assert "--offline" in capsys.readouterr().err


# --- integration: real cached files, skipped on a clean checkout ---------------------------

CACHED = (Path(DEFAULT_CACHE_DIR) / "events.zip").exists() and (
    Path(DEFAULT_CACHE_DIR) / "matches.zip"
).exists()


@pytest.mark.skipif(not CACHED, reason="Wyscout cache absent; run the wyscout_audit CLI once")
def test_real_events_parse_against_real_match_metadata():
    source = WyscoutSource(offline=True)
    matches = matches_by_id(*source.matches())
    shots = 0
    for event in itertools.islice(source.iter_events("events_England.json"), 40_000):
        match = matches[event["matchId"]]
        assert str(event["teamId"]) in match["teamsData"]
        if shot_kind(event):
            shot = parse_shot(event, match)
            assert shot.team_id in match_team_ids(match)
            assert shot.opponent_team_id != shot.team_id
            assert shot.x is not None and shot.body_part is not None
            shots += 1
    assert shots > 100
