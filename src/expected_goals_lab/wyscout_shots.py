"""Wyscout event taxonomy and shot parsing, from the real files audited in S01B.

Taxonomy (eventid2name.csv and tags2name.csv from the Figshare collection):

* Open-play shots:      eventId 10, subEventId 100 ("Shot").
* Free-kick shots:      eventId 3,  subEventId 33  ("Free kick shot").
* Penalties:            eventId 3,  subEventId 35  ("Penalty"). Shootout kicks are penalties
  with ``matchPeriod == "P"``.
* Corners (3/30), other free kicks and passes can carry the Goal tag but are not shots.
* Goal tag 101 also appears on the *goalkeeper's* "Save attempt" events (eventId 9), logged
  for the conceding team. Those are not goals scored and must be ignored when counting goals.
* Own goal tag 102 sits on an ordinary event (touch, clearance, pass) of the player who
  scored it, logged for his own team. The goal counts for the opponent.

An absent tag is not assumed to mean false for descriptive tags. Goal (101) is the exception,
and only because ``goals_by_team`` rebuilds the published score from it in 3,651 of 3,652
domestic team-matches (docs/WYSCOUT_DATA_AUDIT.md).
"""

from dataclasses import dataclass
from typing import Any

EVENT_SHOT, SUBEVENT_SHOT = 10, 100
EVENT_FREE_KICK, SUBEVENT_FREE_KICK_SHOT, SUBEVENT_PENALTY = 3, 33, 35
EVENT_SAVE_ATTEMPT = 9

TAG_GOAL, TAG_OWN_GOAL = 101, 102
TAG_LEFT_FOOT, TAG_RIGHT_FOOT, TAG_HEAD_OR_BODY = 401, 402, 403
TAG_BLOCKED, TAG_ACCURATE, TAG_NOT_ACCURATE = 2101, 1801, 1802
BODY_PART_TAGS = {
    TAG_LEFT_FOOT: "left foot",
    TAG_RIGHT_FOOT: "right foot",
    TAG_HEAD_OR_BODY: "head/body",
}
PERIODS = ("1H", "2H", "E1", "E2", "P")
SHOOTOUT_PERIOD = "P"

OPEN_PLAY, FREE_KICK, PENALTY = "open_play", "free_kick", "penalty"


class WyscoutShotError(ValueError):
    """The event cannot be tied to a shot, an event ID, the given match or one of its teams."""


@dataclass(frozen=True)
class WyscoutShot:
    competition_id: int
    season_id: int
    match_id: int
    event_id: int
    team_id: int
    opponent_team_id: int
    player_id: int | None  # None if absent, malformed, or 0 (not a valid Wyscout player ID)
    kind: str  # OPEN_PLAY, FREE_KICK or PENALTY
    period: str | None
    event_sec: float | None
    x: int | None  # 0-100, percentage of pitch length towards the opponent's goal
    y: int | None  # 0-100, percentage of pitch width
    body_part: str | None  # None if no body-part tag, or if tags disagree
    tags: frozenset[int] | None  # None if the tag list is absent or malformed
    malformed: tuple[str, ...] = ()

    @property
    def is_shootout(self) -> bool:
        return self.period == SHOOTOUT_PERIOD

    @property
    def is_goal(self) -> bool | None:
        """Goal tag present. None if the tags themselves are unusable."""
        return None if self.tags is None else TAG_GOAL in self.tags

    @property
    def is_blocked(self) -> bool | None:
        return None if self.tags is None else TAG_BLOCKED in self.tags

    @property
    def on_target(self) -> bool | None:
        """True/False from the accuracy tags 1801/1802; None if neither or both are present."""
        if self.tags is None or ((TAG_ACCURATE in self.tags) == (TAG_NOT_ACCURATE in self.tags)):
            return None
        return TAG_ACCURATE in self.tags

    @property
    def is_non_penalty_attempt(self) -> bool:
        """Open-play or free-kick shot taken in play (not a penalty, not a shootout kick)."""
        return self.kind != PENALTY and not self.is_shootout


def shot_kind(event: dict) -> str | None:
    """OPEN_PLAY / FREE_KICK / PENALTY for a shot-type event, None for anything else."""
    event_id, sub_id = event.get("eventId"), event.get("subEventId")
    if event_id == EVENT_SHOT and sub_id == SUBEVENT_SHOT:
        return OPEN_PLAY
    if event_id == EVENT_FREE_KICK and sub_id == SUBEVENT_FREE_KICK_SHOT:
        return FREE_KICK
    if event_id == EVENT_FREE_KICK and sub_id == SUBEVENT_PENALTY:
        return PENALTY
    return None


def match_team_ids(match: dict) -> tuple[int, int]:
    """(home_team_id, away_team_id) from a match record's ``teamsData``."""
    sides = {data["side"]: data["teamId"] for data in match["teamsData"].values()}
    return sides["home"], sides["away"]


def event_tag_ids(event: dict) -> frozenset[int] | None:
    """Tag IDs of an event; None if the tag list is absent or any entry is not ``{"id": int}``."""
    tags = event.get("tags")
    if not isinstance(tags, list):
        return None
    try:
        ids = frozenset(tag["id"] for tag in tags)
    except (TypeError, KeyError):
        return None
    return ids if all(type(i) is int for i in ids) else None


def parse_shot(event: dict, match: dict) -> WyscoutShot:
    """Normalise one shot-type event of ``match``. Raises ``WyscoutShotError`` on bad identity."""
    kind = shot_kind(event)
    if kind is None:
        raise WyscoutShotError(
            f"event {event.get('id')!r} is not a shot, free-kick shot or penalty"
        )
    event_id = event.get("id")
    if not _is_int(event_id):
        raise WyscoutShotError("shot event has no usable integer 'id'")
    if event.get("matchId") != match["wyId"]:
        raise WyscoutShotError(
            f"event {event_id}: matchId {event.get('matchId')!r} is not match {match['wyId']}"
        )
    home_id, away_id = match_team_ids(match)
    team_id = event.get("teamId")
    if team_id not in (home_id, away_id):
        raise WyscoutShotError(
            f"event {event_id}: teamId {team_id!r} is not in match {match['wyId']} "
            f"(home {home_id}, away {away_id})"
        )

    malformed: list[str] = []
    tags = event_tag_ids(event)
    if tags is None:
        malformed.append("tags")

    player_id = event.get("playerId")
    if not _is_int(player_id) or player_id == 0:
        malformed.append("player_id")
        player_id = None

    period = event.get("matchPeriod")
    if period not in PERIODS:
        malformed.append("period")
        period = None

    event_sec = event.get("eventSec")
    if isinstance(event_sec, bool) or not isinstance(event_sec, int | float):
        malformed.append("event_sec")
        event_sec = None

    x, y = _start_position(event.get("positions"))
    if x is None:
        malformed.append("position")

    body_parts = [BODY_PART_TAGS[t] for t in (tags or ()) if t in BODY_PART_TAGS]
    if len(body_parts) > 1:
        malformed.append("body_part")

    return WyscoutShot(
        competition_id=match["competitionId"],
        season_id=match["seasonId"],
        match_id=match["wyId"],
        event_id=event_id,
        team_id=team_id,
        opponent_team_id=away_id if team_id == home_id else home_id,
        player_id=player_id,
        kind=kind,
        period=period,
        event_sec=float(event_sec) if event_sec is not None else None,
        x=x,
        y=y,
        body_part=body_parts[0] if len(body_parts) == 1 else None,
        tags=tags,
        malformed=tuple(malformed),
    )


def goals_by_team(events: list[dict], match: dict) -> dict[int, int]:
    """Goals credited to each of the match's two teams in play (shootout excluded).

    Goal-tagged events count for their own team unless they are goalkeeper save attempts,
    which carry the tag for the conceding side. An own-goal-tagged event counts for the
    opponent. Only events with tag 101 or 102 are read, so callers may pre-filter to those.
    """
    home_id, away_id = match_team_ids(match)
    goals = {home_id: 0, away_id: 0}
    for event in events:
        tags = event_tag_ids(event)
        team_id = event.get("teamId")
        if tags is None or team_id not in goals or event.get("matchPeriod") == SHOOTOUT_PERIOD:
            continue
        if TAG_GOAL in tags and event.get("eventId") != EVENT_SAVE_ATTEMPT:
            goals[team_id] += 1
        if TAG_OWN_GOAL in tags:
            goals[away_id if team_id == home_id else home_id] += 1
    return goals


def published_score(match: dict) -> dict[int, int]:
    """Goals per team in play, from the match record.

    ``score`` for matches of regular duration; ``scoreET`` (includes extra time, excludes the
    shootout) otherwise. In domestic matches ``scoreET`` is 0 even when goals were scored.
    """
    field = "score" if match.get("duration") == "Regular" else "scoreET"
    return {data["teamId"]: data[field] for data in match["teamsData"].values()}


def _start_position(positions: Any) -> tuple[int | None, int | None]:
    """First entry of ``positions``. The second entry is a placeholder for shots (see audit)."""
    if not isinstance(positions, list) or not positions or not isinstance(positions[0], dict):
        return None, None
    x, y = positions[0].get("x"), positions[0].get("y")
    if _is_int(x) and _is_int(y) and 0 <= x <= 100 and 0 <= y <= 100:
        return x, y
    return None, None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
