import pytest

from expected_goals_lab.wyscout_shots import (
    FREE_KICK,
    OPEN_PLAY,
    PENALTY,
    WyscoutShotError,
    event_tag_ids,
    goals_by_team,
    match_team_ids,
    parse_shot,
    published_score,
    shot_kind,
)
from tests.wyscout_helpers import (
    AWAY,
    CORNER,
    FREE_KICK_SHOT,
    HOME,
    HOME_PLAYER,
    SAVE_ATTEMPT,
    SHOT,
    TOUCH,
    make_event,
    make_match,
    make_shot,
)
from tests.wyscout_helpers import PENALTY as PENALTY_EVENT


def test_parse_open_play_shot_maps_fields_and_identity():
    shot = parse_shot(make_shot(7, tags=(101, 402, 1801, 201)), make_match())

    assert (shot.competition_id, shot.season_id, shot.match_id, shot.event_id) == (
        364,
        181150,
        5000,
        7,
    )
    assert (shot.team_id, shot.opponent_team_id, shot.player_id) == (HOME, AWAY, HOME_PLAYER)
    assert shot.kind == OPEN_PLAY and shot.period == "1H" and not shot.is_shootout
    assert (shot.x, shot.y) == (88, 41)
    assert shot.body_part == "right foot"
    assert shot.is_goal is True and shot.on_target is True and shot.is_blocked is False
    assert shot.malformed == ()


def test_event_taxonomy_distinguishes_shots_from_goal_tagged_non_shots():
    assert shot_kind(make_shot(kind=SHOT)) == OPEN_PLAY
    assert shot_kind(make_shot(kind=FREE_KICK_SHOT)) == FREE_KICK
    assert shot_kind(make_shot(kind=PENALTY_EVENT)) == PENALTY
    # Goal-tagged events that are not shots: a directly scored corner and a keeper's save attempt.
    assert shot_kind(make_event(event=CORNER, tags=(101,))) is None
    assert shot_kind(make_event(event=SAVE_ATTEMPT, tags=(101,))) is None
    assert shot_kind(make_event(event=(6, "Offside", "", ""))) is None  # '' sub-event id is real
    with pytest.raises(WyscoutShotError, match="not a shot"):
        parse_shot(make_event(event=CORNER, tags=(101,)), make_match())


def test_penalty_and_shootout_kick_are_flagged_not_dropped():
    in_play = parse_shot(make_shot(kind=PENALTY_EVENT, tags=(402, 101)), make_match())
    kick = parse_shot(make_shot(kind=PENALTY_EVENT, period="P"), make_match())

    assert in_play.kind == PENALTY and not in_play.is_shootout
    assert not in_play.is_non_penalty_attempt
    assert kick.is_shootout and not kick.is_non_penalty_attempt
    assert parse_shot(make_shot(kind=FREE_KICK_SHOT), make_match()).is_non_penalty_attempt


def test_team_identity_comes_from_the_event_not_row_order():
    match = make_match()
    shots = [
        parse_shot(make_shot(i, team_id=t), match) for i, t in [(1, AWAY), (2, AWAY), (3, HOME)]
    ]

    assert [s.team_id for s in shots] == [AWAY, AWAY, HOME]
    assert [s.opponent_team_id for s in shots] == [HOME, HOME, AWAY]
    assert match_team_ids(match) == (HOME, AWAY)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"team_id": 999}, "not in match"),
        ({"match_id": 9999}, "is not match 5000"),
    ],
)
def test_shot_from_another_match_or_team_is_rejected(overrides, message):
    with pytest.raises(WyscoutShotError, match=message):
        parse_shot(make_shot(**overrides), make_match())


def test_missing_or_non_integer_event_id_is_rejected():
    event = make_shot()
    del event["id"]
    with pytest.raises(WyscoutShotError, match="'id'"):
        parse_shot(event, make_match())
    with pytest.raises(WyscoutShotError, match="'id'"):
        parse_shot(make_shot(event_id="177959212"), make_match())


def test_absent_tags_are_unknown_not_false():
    event = make_shot()
    del event["tags"]

    shot = parse_shot(event, make_match())

    assert shot.tags is None and shot.malformed == ("tags",)
    assert shot.is_goal is None and shot.is_blocked is None and shot.on_target is None
    assert event_tag_ids({"tags": [{"id": 101}, "bad"]}) is None


def test_body_part_is_not_defaulted():
    none = parse_shot(make_shot(tags=(201,)), make_match())
    both = parse_shot(make_shot(tags=(401, 402)), make_match())
    head = parse_shot(make_shot(tags=(403,)), make_match())

    assert none.body_part is None and none.malformed == ()
    assert both.body_part is None and both.malformed == ("body_part",)
    assert head.body_part == "head/body"


def test_on_target_requires_exactly_one_accuracy_tag():
    on = parse_shot(make_shot(tags=(402, 1801)), make_match())
    off = parse_shot(make_shot(tags=(402, 1802)), make_match())
    neither = parse_shot(make_shot(tags=(402,)), make_match())
    both = parse_shot(make_shot(tags=(402, 1801, 1802)), make_match())

    assert (on.on_target, off.on_target, neither.on_target, both.on_target) == (
        True,
        False,
        None,
        None,
    )


def test_player_id_zero_is_unknown_not_a_player():
    shot = parse_shot(make_shot(player_id=0), make_match())
    assert shot.player_id is None and shot.malformed == ("player_id",)


@pytest.mark.parametrize(
    "positions",
    [
        [{"x": 120, "y": 50}],
        [{"x": 50.5, "y": 50}],
        [{"x": "88", "y": 41}],
        [{"y": 41}],
        [],
        None,
    ],
)
def test_unusable_start_position_is_none_and_flagged(positions):
    event = make_shot()
    event["positions"] = positions
    shot = parse_shot(event, make_match())
    assert (shot.x, shot.y) == (None, None)
    assert "position" in shot.malformed


def test_second_position_placeholder_is_ignored():
    shot = parse_shot(make_shot(positions=[{"y": 30, "x": 75}, {"y": 100, "x": 100}]), make_match())
    assert (shot.x, shot.y) == (75, 30)


def test_goals_by_team_ignores_keeper_events_counts_own_goals_and_shootout_excluded():
    match = make_match()
    events = [
        make_shot(1, HOME, tags=(101, 402)),  # home goal
        make_event(2, AWAY, SAVE_ATTEMPT, (101,)),  # same goal, logged on the away keeper
        make_event(3, AWAY, TOUCH, (102,)),  # away player's own goal: counts for home
        make_event(4, AWAY, CORNER, (101,)),  # corner scored directly: counts for away
        make_shot(5, HOME, kind=PENALTY_EVENT, tags=(101,), period="P"),  # shootout: excluded
    ]

    assert goals_by_team(events, match) == {HOME: 2, AWAY: 1}


def test_published_score_uses_score_or_score_et_by_duration():
    regular = make_match(home_score=2, away_score=1)
    knockout = make_match(duration="ExtraTime")
    knockout["teamsData"][str(HOME)]["scoreET"] = 3
    knockout["teamsData"][str(AWAY)]["scoreET"] = 3

    assert published_score(regular) == {HOME: 2, AWAY: 1}
    assert published_score(knockout) == {HOME: 3, AWAY: 3}  # `score` (1, 0) is not used here
