"""Builders for Wyscout-shaped test data.

Structure and field names mirror the real files audited in S01B (see
docs/WYSCOUT_DATA_AUDIT.md). IDs, names and values are synthetic test values.
"""

HOME, AWAY = 111, 222
HOME_PLAYER, AWAY_PLAYER, BENCH_PLAYER = 1001, 2001, 1002


def make_match(
    match_id: int = 5000,
    duration: str = "Regular",
    home_score=1,
    away_score=0,
    dateutc: str = "2017-09-30 16:30:00",
    competition_id: int = 364,
) -> dict:
    def side(team_id, side_name, score, players):
        return {
            "teamId": team_id,
            "side": side_name,
            "score": score,
            "scoreET": 0,
            "scoreP": 0,
            "scoreHT": 0,
            "hasFormation": 1,
            "formation": {
                "lineup": [{"playerId": p, "goals": "0", "ownGoals": "0"} for p in players[:1]],
                "bench": [{"playerId": p, "goals": "null", "ownGoals": "0"} for p in players[1:]],
                "substitutions": [],
            },
        }

    return {
        "wyId": match_id,
        "competitionId": competition_id,
        "seasonId": 181150,
        "dateutc": dateutc,
        "label": "Home Test - Away Test, 1 - 0",
        "status": "Played",
        "duration": duration,
        "teamsData": {
            str(HOME): side(HOME, "home", home_score, [HOME_PLAYER, BENCH_PLAYER]),
            str(AWAY): side(AWAY, "away", away_score, [AWAY_PLAYER]),
        },
    }


def make_event(
    event_id: int = 1,
    team_id: int = HOME,
    event: tuple[int, str, object, str] = (8, "Pass", 85, "Simple pass"),
    tags: tuple[int, ...] = (),
    period: str = "1H",
    sec: float = 10.0,
    match_id: int = 5000,
    player_id: int = HOME_PLAYER,
    positions: list | None = None,
) -> dict:
    event_id_, name, sub_id, sub_name = event
    return {
        "id": event_id,
        "eventId": event_id_,
        "eventName": name,
        "subEventId": sub_id,
        "subEventName": sub_name,
        "tags": [{"id": t} for t in tags],
        "playerId": player_id,
        "teamId": team_id,
        "matchId": match_id,
        "matchPeriod": period,
        "eventSec": sec,
        "positions": positions if positions is not None else [{"y": 41, "x": 88}, {"y": 0, "x": 0}],
    }


SHOT = (10, "Shot", 100, "Shot")
FREE_KICK_SHOT = (3, "Free Kick", 33, "Free kick shot")
PENALTY = (3, "Free Kick", 35, "Penalty")
CORNER = (3, "Free Kick", 30, "Corner")
SAVE_ATTEMPT = (9, "Save attempt", 91, "Save attempt")
TOUCH = (7, "Others on the ball", 72, "Touch")


def make_shot(event_id: int = 1, team_id: int = HOME, kind=SHOT, tags=(402,), **kwargs) -> dict:
    return make_event(event_id, team_id, kind, tags, **kwargs)
