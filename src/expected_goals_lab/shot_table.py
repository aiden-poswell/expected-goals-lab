"""Canonical table of regular Wyscout shots for the five domestic leagues.

One row per event with ``eventId`` 10 and ``subEventId`` 100 in the domestic league files.
Free-kick shots (3/33) and penalties (3/35) are counted but get no row. Every regular shot gets a
row, including those that cannot be used for modelling: those carry ``eligible = 0`` plus reason
codes, so nothing is dropped silently.

Exclusion reasons, in precedence order (``primary_exclusion`` is the first that applies):

1. ``MATCH_SCORE_MISMATCH``    goals rebuilt from tags differ from the published score of the match
2. ``INVALID_POSITION``        start position missing, malformed or outside 0-100
3. ``SUSPECT_POSITION``        x_pct below ``SUSPECT_X_PCT``: implausibly far from the attacked
                               goal; evidence in the audit report. Raw coordinates are kept
4. ``INVALID_PERIOD_OR_TIME``  period not 1H/2H or time missing or negative
5. ``LABEL_UNKNOWN``           the tag list is missing or malformed, so the outcome is unknown
6. ``GEOMETRY_ERROR``          distance/angle cannot be computed

Informational flags (do not exclude): ``UNKNOWN_PLAYER``, ``BODY_PART_UNKNOWN``,
``OWN_HALF_POSITION`` (x_pct < 50), ``ON_GOAL_LINE`` (x_pct = 100).

``is_goal`` is tag 101 on the shot event itself and nothing else. No post-shot tag is written to
the table.
"""

import csv
import hashlib
import itertools
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from pathlib import Path

from expected_goals_lab.geometry import REFERENCE_PITCH, Pitch, shot_geometry
from expected_goals_lab.shot_diagnostics import (
    PositionEvidence,
    explain_mismatch,
    position_evidence,
)
from expected_goals_lab.wyscout_shots import (
    EVENT_SAVE_ATTEMPT,
    OPEN_PLAY,
    TAG_GOAL,
    TAG_OWN_GOAL,
    WyscoutShot,
    WyscoutShotError,
    event_tag_ids,
    goals_by_team,
    match_team_ids,
    parse_shot,
    published_score,
    shot_kind,
)

MATCH_SCORE_MISMATCH = "MATCH_SCORE_MISMATCH"
INVALID_POSITION = "INVALID_POSITION"
SUSPECT_POSITION = "SUSPECT_POSITION"
INVALID_PERIOD_OR_TIME = "INVALID_PERIOD_OR_TIME"
LABEL_UNKNOWN = "LABEL_UNKNOWN"
GEOMETRY_ERROR = "GEOMETRY_ERROR"
EXCLUSION_PRECEDENCE = (
    MATCH_SCORE_MISMATCH,
    INVALID_POSITION,
    SUSPECT_POSITION,
    INVALID_PERIOD_OR_TIME,
    LABEL_UNKNOWN,
    GEOMETRY_ERROR,
)
SUSPECT_X_PCT = 20
REGULAR_PERIODS = ("1H", "2H")
PERIOD_RANK = {"1H": 0, "2H": 1, "E1": 2, "E2": 3, "P": 4}
BODY_PART_CLASS = {
    "left foot": "left_foot",
    "right foot": "right_foot",
    "head/body": "head_or_body",
}
SCHEMA_VERSION = 1


class IntegrityError(RuntimeError):
    """The extracted data violates an assumption the dataset relies on."""


@dataclass(frozen=True)
class ShotRow:
    event_id: int
    match_id: int
    competition_id: int
    competition: str
    season_id: int
    match_datetime_utc: str
    team_id: int
    opponent_team_id: int
    side: str
    player_id: int | None
    period: str | None
    event_sec: float | None
    x_pct: int | None
    y_pct: int | None
    x_m: float | None
    y_m: float | None
    distance_m: float | None
    angle_rad: float | None
    angle_deg: float | None
    body_part_class: str
    is_goal: int | None
    eligible: int
    exclusion_reasons: str
    primary_exclusion: str
    flags: str


COLUMNS = tuple(f.name for f in fields(ShotRow))

# Every column has exactly one role. Only ``feature_core`` and ``feature_candidate`` columns may
# be model inputs in S03; ``location_source`` columns are what the features are derived from.
# Identifiers are for grouping, splitting and auditing and must not be encoded as features.
COLUMN_ROLES = {
    "event_id": "identifier",
    "match_id": "identifier",
    "competition_id": "identifier",
    "competition": "identifier",
    "season_id": "identifier",
    "match_datetime_utc": "identifier",
    "team_id": "identifier",
    "opponent_team_id": "identifier",
    "player_id": "identifier",
    "side": "context_reserved",
    "period": "context_reserved",
    "event_sec": "context_reserved",
    "x_pct": "location_source",
    "y_pct": "location_source",
    "x_m": "location_source",
    "y_m": "location_source",
    "distance_m": "feature_core",
    "angle_rad": "feature_core",
    "angle_deg": "location_source",
    "body_part_class": "feature_candidate",
    "is_goal": "target",
    "eligible": "quality",
    "exclusion_reasons": "quality",
    "primary_exclusion": "quality",
    "flags": "quality",
}
FEATURE_ALLOWLIST = tuple(c for c in COLUMNS if COLUMN_ROLES[c] == "feature_core")
CANDIDATE_FEATURES = tuple(c for c in COLUMNS if COLUMN_ROLES[c] == "feature_candidate")
TARGET = "is_goal"


def make_row(
    shot: WyscoutShot,
    match: dict,
    competition: str,
    match_mismatch: bool,
    pitch: Pitch = REFERENCE_PITCH,
) -> ShotRow:
    reasons: set[str] = set()
    flags: list[str] = []
    geometry = None

    if match_mismatch:
        reasons.add(MATCH_SCORE_MISMATCH)
    if shot.x is None or shot.y is None:
        reasons.add(INVALID_POSITION)
    else:
        if shot.x < SUSPECT_X_PCT:
            reasons.add(SUSPECT_POSITION)
        if shot.x < 50:
            flags.append("OWN_HALF_POSITION")
        if shot.x == 100:
            flags.append("ON_GOAL_LINE")
        try:
            geometry = shot_geometry(shot.x, shot.y, pitch)
        except ValueError:
            reasons.add(GEOMETRY_ERROR)
    if shot.period not in REGULAR_PERIODS or shot.event_sec is None or shot.event_sec < 0:
        reasons.add(INVALID_PERIOD_OR_TIME)
    if shot.is_goal is None:
        reasons.add(LABEL_UNKNOWN)
    if shot.player_id is None:
        flags.append("UNKNOWN_PLAYER")
    if shot.body_part is None:
        flags.append("BODY_PART_UNKNOWN")

    ordered = [r for r in EXCLUSION_PRECEDENCE if r in reasons]
    home_id, _ = match_team_ids(match)
    return ShotRow(
        event_id=shot.event_id,
        match_id=shot.match_id,
        competition_id=shot.competition_id,
        competition=competition,
        season_id=shot.season_id,
        match_datetime_utc=match["dateutc"],
        team_id=shot.team_id,
        opponent_team_id=shot.opponent_team_id,
        side="home" if shot.team_id == home_id else "away",
        player_id=shot.player_id,
        period=shot.period,
        event_sec=shot.event_sec,
        x_pct=shot.x,
        y_pct=shot.y,
        x_m=geometry.x_m if geometry else None,
        y_m=geometry.y_m if geometry else None,
        distance_m=geometry.distance_m if geometry else None,
        angle_rad=geometry.angle_rad if geometry else None,
        angle_deg=geometry.angle_deg if geometry else None,
        body_part_class=BODY_PART_CLASS.get(shot.body_part, "unknown"),
        is_goal=None if shot.is_goal is None else int(shot.is_goal),
        eligible=int(not ordered),
        exclusion_reasons=";".join(ordered),
        primary_exclusion=ordered[0] if ordered else "",
        flags=";".join(flags),
    )


@dataclass
class MatchResult:
    match: dict
    rows: list[ShotRow] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    n_events: int = 0
    kind_events: Counter = field(default_factory=Counter)  # kind -> n, parsed or not
    kind_goals: Counter = field(default_factory=Counter)  # kind -> goal-tagged
    other_goal_events: int = 0  # goal-tagged attacking events that are not shot-type
    own_goal_events: int = 0
    published_goals: int = 0
    rebuilt_goals: int = 0
    mismatch: dict | None = None
    suspects: list[PositionEvidence] = field(default_factory=list)


def process_match(match: dict, events: list[dict], competition: str) -> MatchResult:
    result = MatchResult(match=match, n_events=len(events))
    tagged = []
    regular = []  # indices of regular-shot events
    for index, event in enumerate(events):
        kind = shot_kind(event)
        tags = event_tag_ids(event) or frozenset()
        goal = TAG_GOAL in tags
        if kind:
            result.kind_events[kind] += 1
            result.kind_goals[kind] += goal
            if kind == OPEN_PLAY:
                regular.append(index)
        elif goal and event.get("eventId") != EVENT_SAVE_ATTEMPT:
            result.other_goal_events += 1
        if goal or TAG_OWN_GOAL in tags:
            tagged.append(event)
        result.own_goal_events += TAG_OWN_GOAL in tags

    rebuilt = goals_by_team(tagged, match)
    published = published_score(match)
    result.rebuilt_goals, result.published_goals = sum(rebuilt.values()), sum(published.values())
    mismatched = rebuilt != published
    if mismatched:
        result.mismatch = explain_mismatch(match, events, rebuilt, published)

    for index in regular:
        event = events[index]
        try:
            shot = parse_shot(event, match)
        except WyscoutShotError as exc:
            result.parse_errors.append(str(exc))
            continue
        result.rows.append(make_row(shot, match, competition, mismatched))
        if shot.x is not None and shot.x < SUSPECT_X_PCT:
            result.suspects.append(position_evidence(events, index))
    return result


@dataclass
class LeagueTally:
    competition_id: int
    name: str
    metadata_matches: int
    matches: int = 0
    events: int = 0
    kind_events: Counter = field(default_factory=Counter)
    kind_goals: Counter = field(default_factory=Counter)
    other_goal_events: int = 0
    own_goal_events: int = 0
    published_goals: int = 0
    rebuilt_goals: int = 0
    mismatched_matches: list[int] = field(default_factory=list)


@dataclass
class BuildResult:
    rows: list[ShotRow]
    leagues: dict[int, LeagueTally]
    matches: dict[int, dict]
    parse_errors: list[str]
    mismatches: list[dict]
    suspects: list[PositionEvidence]
    out_of_scope: dict[str, dict]  # events member -> description, not scanned


def build_table(source) -> BuildResult:
    """Stream the domestic event files match by match and build every regular-shot row."""
    competitions = {c["wyId"]: c for c in source.competitions()}
    matches = {m["wyId"]: m for m in source.matches()}
    result = BuildResult([], {}, matches, [], [], [], {})

    for member in source.events_members():
        events = iter(source.iter_events(member))
        first = next(events, None)
        if first is None:
            continue
        first_match = matches.get(first.get("matchId"))
        if first_match is None:
            raise IntegrityError(f"{member}: first event refers to unknown match")
        competition = competitions[first_match["competitionId"]]
        in_competition = {
            m for m, rec in matches.items() if rec["competitionId"] == competition["wyId"]
        }
        if competition["type"] != "club":
            result.out_of_scope[member] = {
                "competition": competition["name"],
                "metadata_matches": len(in_competition),
            }
            continue

        tally = LeagueTally(competition["wyId"], competition["name"], len(in_competition))
        result.leagues[competition["wyId"]] = tally
        seen: set[int] = set()
        for match_id, group in itertools.groupby(
            itertools.chain([first], events), key=lambda e: e.get("matchId")
        ):
            match = matches.get(match_id)
            if match is None or match["competitionId"] != competition["wyId"]:
                raise IntegrityError(
                    f"{member}: events for match {match_id} not in {competition['name']}"
                )
            if match_id in seen:
                raise IntegrityError(f"{member}: events for match {match_id} are not contiguous")
            seen.add(match_id)
            outcome = process_match(match, list(group), competition["name"])
            _accumulate(result, tally, outcome)
        if seen != in_competition:
            raise IntegrityError(f"{member}: {len(in_competition - seen)} matches have no events")

    result.rows.sort(key=_row_order)
    return result


def _accumulate(result: BuildResult, tally: LeagueTally, outcome: MatchResult) -> None:
    result.rows.extend(outcome.rows)
    result.parse_errors.extend(outcome.parse_errors)
    result.suspects.extend(outcome.suspects)
    tally.matches += 1
    tally.events += outcome.n_events
    tally.kind_events += outcome.kind_events
    tally.kind_goals += outcome.kind_goals
    tally.other_goal_events += outcome.other_goal_events
    tally.own_goal_events += outcome.own_goal_events
    tally.published_goals += outcome.published_goals
    tally.rebuilt_goals += outcome.rebuilt_goals
    if outcome.mismatch:
        tally.mismatched_matches.append(outcome.match["wyId"])
        result.mismatches.append(outcome.mismatch)


def _row_order(row: ShotRow) -> tuple:
    return (
        row.match_datetime_utc,
        row.match_id,
        PERIOD_RANK.get(row.period, 9),
        row.event_sec if row.event_sec is not None else -1.0,
        row.event_id,
    )


def check_integrity(result: BuildResult) -> list[tuple[str, str]]:
    """Assertions on the finished table. Returns (check, outcome) lines; raises on failure."""
    rows, matches = result.rows, result.matches
    problems: list[str] = []
    lines: list[tuple[str, str]] = []

    def check(name: str, failures: int, detail: str = "") -> None:
        lines.append((name, "pass" if not failures else f"FAIL ({failures})"))
        if failures:
            problems.append(f"{name}: {failures} {detail}".strip())

    ids = Counter(r.event_id for r in rows)
    check("Event IDs unique", sum(1 for n in ids.values() if n > 1), "duplicated")
    check("Shot events failing identity validation", len(result.parse_errors), "events")
    unknown_match = [r for r in rows if r.match_id not in matches]
    check("Every row joins to a match record", len(unknown_match))
    bad_link = 0
    for r in rows:
        m = matches.get(r.match_id)
        if m is None:
            continue
        home_id, away_id = match_team_ids(m)
        if (
            r.team_id not in (home_id, away_id)
            or {r.team_id, r.opponent_team_id} != {home_id, away_id}
            or r.side != ("home" if r.team_id == home_id else "away")
            or r.competition_id != m["competitionId"]
            or r.season_id != m["seasonId"]
            or r.match_datetime_utc != m["dateutc"]
        ):
            bad_link += 1
    check("Team, side, league, season and date agree with the match record", bad_link)

    eligible = [r for r in rows if r.eligible]
    check(
        "Quarantined rows all carry a reason",
        sum(1 for r in rows if not r.eligible and not r.primary_exclusion),
    )
    check(
        "Eligible rows have a valid period, time, label and in-range integer position",
        sum(
            1
            for r in eligible
            if r.period not in REGULAR_PERIODS
            or r.event_sec is None
            or r.is_goal not in (0, 1)
            or not (isinstance(r.x_pct, int) and isinstance(r.y_pct, int))
            or not (SUSPECT_X_PCT <= r.x_pct <= 100 and 0 <= r.y_pct <= 100)
        ),
    )
    check(
        "Eligible rows have finite geometry, distance >= 0 and angle in [0, pi]",
        sum(
            1
            for r in eligible
            if not all(
                v is not None and math.isfinite(v)
                for v in (r.x_m, r.y_m, r.distance_m, r.angle_rad)
            )
            or r.distance_m < 0
            or not (0 <= r.angle_rad <= math.pi)
        ),
    )

    expected_regular = {cid: t.kind_events[OPEN_PLAY] for cid, t in result.leagues.items()}
    per_league = Counter(r.competition_id for r in rows)
    check(
        "Rows per league equal the regular-shot events seen in that league",
        sum(1 for cid, n in expected_regular.items() if per_league[cid] != n),
    )
    expected_goals = {cid: t.kind_goals[OPEN_PLAY] for cid, t in result.leagues.items()}
    row_goals = Counter(r.competition_id for r in rows if r.is_goal == 1)
    check(
        "Goal rows per league equal the goal-tagged regular-shot events seen",
        sum(1 for cid, n in expected_goals.items() if row_goals[cid] != n),
    )
    seasons: dict[int, set[int]] = {}
    for r in rows:
        seasons.setdefault(r.competition_id, set()).add(r.season_id)
    check(
        "Exactly one season ID per league",
        sum(1 for s in seasons.values() if len(s) != 1) + (len(seasons) != len(result.leagues)),
    )
    check(
        "Every metadata match of every domestic league has events",
        sum(1 for t in result.leagues.values() if t.matches != t.metadata_matches),
    )
    if problems:
        raise IntegrityError("; ".join(problems))
    return lines


def _fmt(value, spec: str | None = None) -> str:
    if value is None:
        return ""
    return format(value, spec) if spec else str(value)


_FLOAT_FORMATS = {
    "event_sec": ".3f",
    "x_m": ".6f",
    "y_m": ".6f",
    "distance_m": ".6f",
    "angle_rad": ".9f",
    "angle_deg": ".6f",
}


def write_csv(rows: Iterable[ShotRow], path: Path) -> None:
    """UTF-8, LF line endings, fixed column order and float formats: byte-stable output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(
                [_fmt(getattr(row, name), _FLOAT_FORMATS.get(name)) for name in COLUMNS]
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
