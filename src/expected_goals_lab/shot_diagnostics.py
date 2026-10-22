"""Evidence gathering for the two integrity concerns found in S01B. Diagnostics only.

Nothing here changes a coordinate or a label. The results go into the audit report so a reviewer
can see why a shot or match was quarantined.

* ``position_evidence``: for a shot at a suspicious position, compare it with opponent events a
  few seconds either side. Each team's events are in its own attack-relative frame, so a point
  (x, y) for one team is (100 - x, 100 - y) for the other. If the shot is where the opponent's
  nearby events say the play is, "recorded" distances are small; if only the mirrored position
  matches, the shot's frame may be reversed. Goalkeeper events are ignored because their second
  position just repeats the shot's position.
* ``explain_mismatch``: for a match whose goal tags disagree with the published score, list the
  goal-tagged events, the lineup's own record of who scored, and what happened immediately before
  each goalkeeper event that carries the Goal tag.
"""

import math
from dataclasses import dataclass

from expected_goals_lab.wyscout_shots import (
    EVENT_SAVE_ATTEMPT,
    TAG_GOAL,
    TAG_OWN_GOAL,
    event_tag_ids,
    shot_kind,
)

EVIDENCE_WINDOW_SEC = 3.0
EVIDENCE_RADIUS = 20.0  # percentage-point distance counted as "agrees"
PLACEHOLDERS = {(0, 0), (100, 100)}  # sentinel positions used by non-located events


@dataclass(frozen=True)
class PositionEvidence:
    event_id: int
    match_id: int
    team_id: int
    period: str
    event_sec: float
    x_pct: int
    y_pct: int
    is_goal: bool
    verdict: str  # mirror_supported, recorded_supported, conflicting, no_evidence
    opponent_events: tuple  # (seconds from shot, event name, x, y, d_recorded, d_mirrored)
    # (event name, x, y) of the nearest same-team event within 10 s before / after the shot
    previous_same_team: tuple | None
    next_same_team: tuple | None
    opponent_keeper_response: bool  # an opponent Save attempt within 5 s after the shot
    other_team_shots_x: tuple  # x of the team's other regular shots in the period (up to 3 lowest)


def start_position(event: dict) -> tuple[int, int] | None:
    positions = event.get("positions")
    if not isinstance(positions, list) or not positions or not isinstance(positions[0], dict):
        return None
    x, y = positions[0].get("x"), positions[0].get("y")
    ok = all(isinstance(v, int) and not isinstance(v, bool) for v in (x, y))
    return (x, y) if ok else None


def position_evidence(events: list[dict], index: int) -> PositionEvidence:
    shot = events[index]
    x, y = start_position(shot)
    period, sec, team = shot["matchPeriod"], shot["eventSec"], shot["teamId"]

    opponent = []
    previous_same_team = None
    next_same_team = None
    keeper_response = False
    other_shots = []
    for i, other in enumerate(events):
        if i == index or other.get("matchPeriod") != period:
            continue
        dt = other["eventSec"] - sec
        if other["teamId"] == team:
            if shot_kind(other) == "open_play":
                position = start_position(other)
                if position:
                    other_shots.append(position[0])
            if -10 <= dt < 0 and i < index and start_position(other):
                previous_same_team = (other["eventName"], *start_position(other))
            if 0 <= dt <= 10 and i > index and start_position(other) and next_same_team is None:
                next_same_team = (other["eventName"], *start_position(other))
            continue
        if other.get("eventId") == EVENT_SAVE_ATTEMPT:
            keeper_response = keeper_response or 0 <= dt <= 5
            continue
        position = start_position(other)
        if abs(dt) <= EVIDENCE_WINDOW_SEC and position and position not in PLACEHOLDERS:
            ox, oy = position
            recorded = math.hypot(x - (100 - ox), y - (100 - oy))
            mirrored = math.hypot(x - ox, y - oy)
            opponent.append(
                (round(dt, 1), other["eventName"], ox, oy, round(recorded), round(mirrored))
            )

    near_recorded = any(row[4] <= EVIDENCE_RADIUS for row in opponent)
    near_mirrored = any(row[5] <= EVIDENCE_RADIUS for row in opponent)
    verdict = {
        (True, True): "conflicting",
        (True, False): "recorded_supported",
        (False, True): "mirror_supported",
        (False, False): "no_evidence",
    }[(near_recorded, near_mirrored)]
    tags = event_tag_ids(shot) or frozenset()
    return PositionEvidence(
        event_id=shot["id"],
        match_id=shot["matchId"],
        team_id=team,
        period=period,
        event_sec=sec,
        x_pct=x,
        y_pct=y,
        is_goal=TAG_GOAL in tags,
        verdict=verdict,
        opponent_events=tuple(opponent),
        previous_same_team=previous_same_team,
        next_same_team=next_same_team,
        opponent_keeper_response=keeper_response,
        other_team_shots_x=tuple(sorted(other_shots)[:3]),
    )


def _brief(event: dict) -> dict:
    positions = event.get("positions") or []
    return {
        "id": event.get("id"),
        "period": event.get("matchPeriod"),
        "sec": round(event.get("eventSec", 0), 1),
        "team": event.get("teamId"),
        "player": event.get("playerId"),
        "type": f"{event.get('eventName')} / {event.get('subEventName')}",
        "tags": sorted(t.get("id") for t in event.get("tags", []) if isinstance(t, dict)),
        "positions": [(p.get("x"), p.get("y")) for p in positions if isinstance(p, dict)],
    }


def explain_mismatch(match: dict, events: list[dict], rebuilt: dict, published: dict) -> dict:
    """Facts about a match whose rebuilt goals differ from the published score."""
    goal_events = [
        _brief(e) for e in events if (event_tag_ids(e) or frozenset()) & {TAG_GOAL, TAG_OWN_GOAL}
    ]
    credited = [
        (team_id, p["playerId"], p["goals"])
        for team_id, data in match["teamsData"].items()
        for p in data.get("formation", {}).get("lineup", [])
        + data.get("formation", {}).get("bench", [])
        if p.get("goals") not in ("0", "null", None)
    ]
    credited_ids = {player_id for _, player_id, _ in credited}
    before_keeper_goal = []
    last_by_credited = None
    for i, e in enumerate(events):
        tags = event_tag_ids(e) or frozenset()
        if e.get("eventId") == EVENT_SAVE_ATTEMPT and TAG_GOAL in tags:
            before_keeper_goal.append(
                {
                    "keeper_event": _brief(e),
                    "previous_events": [_brief(f) for f in events[max(0, i - 4) : i]],
                }
            )
            earlier = [f for f in events[:i] if f.get("playerId") in credited_ids]
            last_by_credited = _brief(earlier[-1]) if earlier else None
    return {
        "match_id": match["wyId"],
        "label": match["label"],
        "date_utc": match["dateutc"],
        "published": published,
        "rebuilt": rebuilt,
        "goal_tagged_events": goal_events,
        "lineup_credited_scorers": credited,
        "credited_scorer_shot_events": [
            _brief(e) for e in events if e.get("playerId") in credited_ids and shot_kind(e)
        ],
        "before_keeper_goal": before_keeper_goal,
        "credited_scorer_last_event_before_keeper_goal": last_by_credited,
    }
