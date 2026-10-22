"""S01B Wyscout audit: source, coverage and shot taxonomy of the Pappalardo-Massucco dataset.

    uv run python -m expected_goals_lab.wyscout_audit

Downloads (once, checksum-verified, cached under data/cache/wyscout/) the pinned Figshare
files, then:

1. Describes the published corpus from match metadata only.
2. Scans every event file once and keeps *counts only* (event totals, shot-type events, goal
   tags, ID and ordering checks, score reconciliation). No shot table is built.
3. Audits individual shots only for a deterministic sample: ``--matches-per-competition``
   matches evenly spaced by date in each domestic league, plus every international match
   that went to a penalty shootout.
4. Prints the report and writes it to ``reports/wyscout_data_audit.md``.

With a warm cache nothing is downloaded. ``--offline`` fails instead of downloading.
"""

import argparse
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from expected_goals_lab import wyscout
from expected_goals_lab.common import DataAccessError, count_table, table
from expected_goals_lab.wyscout import WyscoutSource
from expected_goals_lab.wyscout_shots import (
    OPEN_PLAY,
    PENALTY,
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

DEFAULT_REPORT = Path("reports/wyscout_data_audit.md")
# Table 1 of Pappalardo et al. (2019), Scientific Data 6:236: events per competition file.
PUBLISHED_EVENTS = {
    "events_Spain.json": 628_659,
    "events_England.json": 643_150,
    "events_Italy.json": 647_372,
    "events_Germany.json": 519_407,
    "events_France.json": 632_807,
    "events_World_Cup.json": 101_759,
    "events_European_Championship.json": 78_140,
}


@dataclass
class Scan:
    """Counts from one pass over one competition's event file."""

    n_events: int = 0
    key_counts: Counter = field(default_factory=Counter)
    match_ids: Counter = field(default_factory=Counter)
    unknown_match: int = 0
    unknown_team: int = 0
    duplicate_ids: int = 0
    order_violations: int = 0  # eventSec going backwards within a (match, period)
    shot_events: Counter = field(default_factory=Counter)  # (kind, in_shootout) -> n
    shot_goals: Counter = field(default_factory=Counter)  # same key, Goal-tagged only
    goal_tag_by_type: Counter = field(default_factory=Counter)
    own_goal_tag_by_type: Counter = field(default_factory=Counter)
    penalty_positions: Counter = field(default_factory=Counter)  # non-shootout penalties
    open_play_x_bins: Counter = field(default_factory=Counter)  # (bin, goal-tagged) -> n
    second_positions: Counter = field(default_factory=Counter)  # shots: kind of 2nd position
    open_play_x_below_50: int = 0
    open_play_x_below_10: Counter = field(default_factory=Counter)  # matchId -> n
    open_play_x_below_10_periods: Counter = field(default_factory=Counter)  # matchPeriod -> n
    shot_player_id_zero: int = 0
    shot_position_out_of_range: int = 0
    tagged_events: dict[int, list[dict]] = field(default_factory=dict)  # events with tag 101/102
    sample_events: dict[int, list[dict]] = field(default_factory=dict)


def scan_events(events: Iterable[dict], matches: dict[int, dict], sample_ids: set[int]) -> Scan:
    scan = Scan()
    seen_ids: set = set()
    last_sec: dict[tuple, float] = {}
    for event in events:
        scan.n_events += 1
        scan.key_counts.update(event.keys())
        match = matches.get(event.get("matchId"))
        if match is None:
            scan.unknown_match += 1
            continue
        match_id = match["wyId"]
        scan.match_ids[match_id] += 1
        if str(event.get("teamId")) not in match["teamsData"]:
            scan.unknown_team += 1
        if event.get("id") in seen_ids:
            scan.duplicate_ids += 1
        seen_ids.add(event.get("id"))
        period_key = (match_id, event.get("matchPeriod"))
        if event.get("eventSec", 0) < last_sec.get(period_key, 0):
            scan.order_violations += 1
        last_sec[period_key] = event.get("eventSec", 0)
        if match_id in sample_ids:
            scan.sample_events.setdefault(match_id, []).append(event)

        tags = event_tag_ids(event) or frozenset()
        event_type = (event.get("eventName"), event.get("subEventName"))
        if TAG_GOAL in tags:
            scan.goal_tag_by_type[event_type] += 1
        if TAG_OWN_GOAL in tags:
            scan.own_goal_tag_by_type[event_type] += 1
        if tags & {TAG_GOAL, TAG_OWN_GOAL}:
            scan.tagged_events.setdefault(match_id, []).append(event)

        kind = shot_kind(event)
        if kind is None:
            continue
        shootout = event.get("matchPeriod") == "P"
        scan.shot_events[(kind, shootout)] += 1
        scan.shot_goals[(kind, shootout)] += TAG_GOAL in tags
        positions = event.get("positions") or [{}]
        first = positions[0] if isinstance(positions[0], dict) else {}
        if len(positions) > 1 and isinstance(positions[1], dict):
            second = (positions[1].get("x"), positions[1].get("y"))
            scan.second_positions[
                "(0,0)" if second == (0, 0) else "(100,100)" if second == (100, 100) else "other"
            ] += 1
        else:
            scan.second_positions["absent"] += 1
        x, y = first.get("x"), first.get("y")
        scan.shot_player_id_zero += event.get("playerId") == 0
        scan.shot_position_out_of_range += not (
            isinstance(x, int) and isinstance(y, int) and 0 <= x <= 100 and 0 <= y <= 100
        )
        if kind == PENALTY and not shootout and isinstance(x, int) and isinstance(y, int):
            scan.penalty_positions[(x, y)] += 1
        if kind == OPEN_PLAY and not shootout and isinstance(x, int):
            scan.open_play_x_bins[(min(x // 10 * 10, 90), TAG_GOAL in tags)] += 1
            scan.open_play_x_below_50 += x < 50
            if x < 10:
                scan.open_play_x_below_10[match_id] += 1
                scan.open_play_x_below_10_periods[event.get("matchPeriod")] += 1
    return scan


def spread(items: list[dict], k: int) -> list[dict]:
    """k items evenly spaced through the list, deterministic."""
    if k >= len(items):
        return items
    n = len(items)
    return [items[(2 * i + 1) * n // (2 * k)] for i in range(k)]


def select_sample(
    matches: list[dict], competitions: list[dict], per_competition: int
) -> list[dict]:
    """Per domestic league: evenly spaced by (dateutc, wyId). Plus all shootout matches."""
    chosen: list[dict] = []
    for competition in sorted(competitions, key=lambda c: c["wyId"]):
        in_competition = [m for m in matches if m["competitionId"] == competition["wyId"]]
        if competition["type"] == "club":
            ordered = sorted(in_competition, key=lambda m: (m["dateutc"], m["wyId"]))
            chosen += spread(ordered, per_competition)
        else:
            chosen += sorted(
                (m for m in in_competition if m["duration"] == "Penalties"),
                key=lambda m: (m["dateutc"], m["wyId"]),
            )
    return chosen


def squad_player_ids(match: dict, team_id: int) -> set[int]:
    formation = match["teamsData"][str(team_id)].get("formation", {})
    return {p["playerId"] for p in formation.get("lineup", []) + formation.get("bench", [])}


@dataclass
class SampleAudit:
    shots: list[WyscoutShot] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    matches: int = 0
    events: int = 0
    team_ids_ok: int = 0
    shooter_not_in_squad: int = 0


def audit_sample(scans: list[Scan], matches: dict[int, dict]) -> SampleAudit:
    result = SampleAudit()
    for scan in scans:
        for match_id, events in scan.sample_events.items():
            match = matches[match_id]
            result.matches += 1
            result.events += len(events)
            result.team_ids_ok += {e["teamId"] for e in events} == set(match_team_ids(match))
            for event in events:
                if shot_kind(event) is None:
                    continue
                try:
                    shot = parse_shot(event, match)
                except WyscoutShotError as exc:
                    result.errors.append(str(exc))
                    continue
                result.shots.append(shot)
                if shot.player_id is not None and shot.player_id not in squad_player_ids(
                    match, shot.team_id
                ):
                    result.shooter_not_in_squad += 1
    return result


def reconcile(scans: list[Scan], matches: dict[int, dict]) -> tuple[int, int, list[str]]:
    """(team-matches agreeing, team-matches checked, descriptions of disagreements)."""
    agree = total = 0
    failures: list[str] = []
    for scan in scans:
        for match_id in scan.match_ids:
            match = matches[match_id]
            rebuilt = goals_by_team(scan.tagged_events.get(match_id, []), match)
            for team_id, expected in published_score(match).items():
                total += 1
                if rebuilt[team_id] == expected:
                    agree += 1
                else:
                    failures.append(
                        f"match {match_id} ({match['label']}), team {team_id}: "
                        f"events give {rebuilt[team_id]}, record says {expected}"
                    )
    return agree, total, failures


def shot_columns(scan: Scan) -> list[str]:
    """'n (goals)' for open-play, free-kick, in-play penalty and shootout shot-type events."""

    def cell(keys: list[tuple[str, bool]]) -> str:
        return f"{sum(scan.shot_events[k] for k in keys)} ({sum(scan.shot_goals[k] for k in keys)})"

    return [
        cell([("open_play", False)]),
        cell([("free_kick", False)]),
        cell([("penalty", False)]),
        cell([(k, True) for k in ("open_play", "free_kick", "penalty")]),
    ]


def universal_keys(scans: Iterable[Scan]) -> set[str]:
    """Keys present on every event of every file."""
    return set.intersection(
        *({k for k, n in scan.key_counts.items() if n == scan.n_events} for scan in scans)
    )


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "n/a"


def render_report(
    source: WyscoutSource,
    competitions: list[dict],
    matches: dict[int, dict],
    scans: dict[str, Scan],
    sample: list[dict],
    audit: SampleAudit,
    args: argparse.Namespace,
) -> str:
    names = {c["wyId"]: c["name"] for c in competitions}
    out: list[str] = []
    add = out.extend

    add(
        [
            "# Wyscout data audit (S01B)",
            "",
            "Generated by `uv run python -m expected_goals_lab.wyscout_audit` on "
            f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC with `--matches-per-competition "
            f"{args.matches_per_competition}`. Sections 1-2 describe the published files from "
            "metadata. Section 3 is an archive-wide *count* scan (no shot-level extraction). "
            "Sections 4-6 are shot-level checks on the sample only.",
            "",
            "## 1. Source",
            "",
            f"- Collection: Pappalardo & Massucco, *Soccer match event dataset*, figshare, "
            f"doi:{wyscout.COLLECTION_DOI} ({wyscout.COLLECTION_URL}).",
            f"- Licence recorded for every file below: {wyscout.LICENCE} "
            f"({wyscout.LICENCE_URL}), as "
            "returned by the Figshare API on 2026-10-20. This program does not re-check it.",
            f"- Paper to cite: Pappalardo et al. (2019), Scientific Data 6:236, "
            f"doi:{wyscout.PAPER_DOI}.",
            "",
            table(
                [
                    "File",
                    "Figshare article (version)",
                    "File ID",
                    "Bytes",
                    "MD5 (verified on load)",
                ],
                [
                    [f.name, f"{f.article_id} (v{f.article_version})", f.file_id, f.size, f.md5]
                    for f in wyscout.ALL_FILES
                ],
            ),
            "",
        ]
    )

    rows = []
    for c in sorted(competitions, key=lambda c: (c["type"] != "club", c["name"])):
        ms = [m for m in matches.values() if m["competitionId"] == c["wyId"]]
        per_team: Counter = Counter()
        for m in ms:
            per_team.update(int(t) for t in m["teamsData"])
        dates = sorted(m["dateutc"] for m in ms)
        durations = Counter(m["duration"] for m in ms)
        rows.append(
            [
                c["name"],
                c["wyId"],
                c["type"],
                len(ms),
                len(per_team),
                f"{min(per_team.values())}-{max(per_team.values())}",
                f"{dates[0][:10]} to {dates[-1][:10]}",
                ", ".join(f"{k} {v}" for k, v in durations.items()),
            ]
        )
    add(
        [
            "## 2. Published corpus (from match metadata)",
            "",
            f"{len(matches)} matches in {len(competitions)} competitions; "
            f"{sum(1 for m in matches.values() if m['status'] != 'Played')} not `Played`; "
            f"seasons: {sorted({m['seasonId'] for m in matches.values()})}.",
            "",
            table(
                [
                    "Competition",
                    "wyId",
                    "Type",
                    "Matches",
                    "Teams",
                    "Matches per team",
                    "Dates (UTC)",
                    "Duration",
                ],
                rows,
            ),
            "",
            "## 3. Archive-wide count scan (every event file read once; counts only)",
            "",
        ]
    )
    scan_rows = []
    for member, scan in sorted(scans.items()):
        published = PUBLISHED_EVENTS.get(member)
        in_file = {matches[i]["competitionId"] for i in scan.match_ids}
        scan_rows.append(
            [
                member,
                "/".join(names[i] for i in sorted(in_file)),
                scan.n_events,
                published if published is not None else "n/a",
                "yes" if scan.n_events == published else "NO",
                len(scan.match_ids),
                scan.unknown_match,
                scan.unknown_team,
                scan.duplicate_ids,
                scan.order_violations,
            ]
        )
    add(
        [
            table(
                [
                    "Events file",
                    "Competition(s) in file",
                    "Events",
                    "Published (paper Table 1)",
                    "Equal",
                    "Matches with events",
                    "Unknown matchId",
                    "Unknown teamId",
                    "Duplicate event ids",
                    "eventSec going backwards",
                ],
                scan_rows,
            ),
            "",
            f"Matches in metadata with no events: "
            f"{len(matches) - sum(len(s.match_ids) for s in scans.values())}. Duplicate IDs are "
            "checked within each file only. Every event in every file has these keys: "
            + ", ".join(sorted(universal_keys(scans.values())))
            + ".",
            "",
            "Shot-type events (open-play `Shot` 10/100, `Free kick shot` 3/33, `Penalty` 3/35); "
            "shootout = `matchPeriod` `P`:",
            "",
        ]
    )
    kind_rows = []
    for member, scan in sorted(scans.items()):
        kind_rows.append([member, *shot_columns(scan)])
    total_scan = Scan()
    for scan in scans.values():
        total_scan.shot_events += scan.shot_events
        total_scan.shot_goals += scan.shot_goals
    kind_rows.append(["all files", *shot_columns(total_scan)])
    add(
        [
            table(
                [
                    "Events file",
                    "Open-play shots (goal-tagged)",
                    "Free-kick shots (goal-tagged)",
                    "Penalties in play (goal-tagged)",
                    "Shootout kicks (goal-tagged)",
                ],
                kind_rows,
            ),
            "",
        ]
    )

    goal_types = sum((s.goal_tag_by_type for s in scans.values()), Counter())
    og_types = sum((s.own_goal_tag_by_type for s in scans.values()), Counter())
    agree, total, failures = reconcile(list(scans.values()), matches)
    add(
        [
            "Goal tag (101) by event type, all files: "
            + ", ".join(f"{a} / {b}: {n}" for (a, b), n in goal_types.most_common())
            + ".",
            "",
            "Own-goal tag (102) by event type, all files: "
            + ", ".join(f"{a} / {b}: {n}" for (a, b), n in og_types.most_common())
            + ".",
            "",
            "Score reconciliation: goals rebuilt per team as Goal-tagged events other than "
            "goalkeeper `Save attempt`, outside the shootout, plus opponent's own-goal-tagged "
            "events, compared with `score` (regular-duration matches) or `scoreET` "
            "(extra-time and shootout matches): "
            f"**{agree} of {total} team-matches agree.**",
            "",
            *[f"- Disagreement: {f}" for f in failures],
            "" if failures else "",
        ]
    )

    shots = audit.shots
    n = len(shots)
    add(
        [
            "## 4. Sample",
            "",
            f"{len(sample)} matches selected: {args.matches_per_competition} per domestic league, "
            "evenly spaced by `(dateutc, wyId)` (index `(2i+1)*n // (2k)`), plus every "
            "international match decided by a penalty shootout. Sampled matches processed: "
            f"{audit.matches}; events: {audit.events}; shot-type events parsed: {n}.",
            "",
            table(
                ["Competition", "Matches", "Shot-type events", "Match IDs"],
                [
                    [
                        names[cid],
                        sum(1 for m in sample if m["competitionId"] == cid),
                        sum(1 for s in shots if s.competition_id == cid),
                        ", ".join(str(m["wyId"]) for m in sample if m["competitionId"] == cid),
                    ]
                    for cid in sorted({m["competitionId"] for m in sample})
                ],
            ),
            "",
        ]
    )
    if not n:
        add(["No shots were audited."])
        return "\n".join(out) + "\n"

    non_pen = [s for s in shots if s.is_non_penalty_attempt]
    add(
        [
            "## 5. Shots in the sample",
            "",
            count_table("Kind", Counter(s.kind for s in shots), n),
            "",
            count_table("Period", Counter(s.period for s in shots), n),
            "",
            f"Non-penalty attempts in play (open play + free kick, not shootout): {len(non_pen)}. "
            "Tag-derived properties below are over these.",
            "",
            count_table(
                "Goal tag (101) present", Counter(s.is_goal for s in non_pen), len(non_pen)
            ),
            "",
            count_table(
                "Blocked tag (2101) present", Counter(s.is_blocked for s in non_pen), len(non_pen)
            ),
            "",
            count_table(
                "On target (1801 accurate / 1802 not accurate; None = neither or both)",
                Counter(s.on_target for s in non_pen),
                len(non_pen),
            ),
            "",
            count_table(
                "Body part (None = no body-part tag, or conflicting tags)",
                Counter(s.body_part for s in non_pen),
                len(non_pen),
            ),
            "",
            "Missing or unusable parsed fields (all sampled shot-type events):",
            "",
            table(
                ["Field", "Missing or malformed", "Share"],
                [
                    [name, k, pct(k, n)]
                    for name, k in [
                        (
                            "player_id (absent, malformed or 0)",
                            sum(s.player_id is None for s in shots),
                        ),
                        ("period", sum(s.period is None for s in shots)),
                        ("event_sec", sum(s.event_sec is None for s in shots)),
                        ("start position x/y", sum(s.x is None for s in shots)),
                        ("tags", sum(s.tags is None for s in shots)),
                        (
                            "body_part (no tag or conflicting)",
                            sum(s.body_part is None for s in shots),
                        ),
                    ]
                ],
            ),
            "",
            "Identity and consistency checks on the sample:",
            "",
            table(
                ["Check", "Result"],
                [
                    [
                        "Shot events failing identity validation (matchId, teamId, event id)",
                        len(audit.errors),
                    ],
                    [
                        "Matches whose event teamIds equal the two teams in the match record",
                        f"{audit.team_ids_ok} of {audit.matches}",
                    ],
                    [
                        "Shooters not listed in their team's lineup or bench",
                        audit.shooter_not_in_squad,
                    ],
                    [
                        "Duplicate shot event ids",
                        n - len({(s.match_id, s.event_id) for s in shots}),
                    ],
                ],
            ),
            "",
            *[f"- {e}" for e in audit.errors[:5]],
            "",
            "## 6. Coordinates",
            "",
            "Wyscout API documentation (support.wyscout.com/matches-wyid-events): `x`, `y` are "
            'integer "percentage from the left corner of the attacking team". The paper adds that '
            "x is nearness to the opponent's goal. Checks on the data:",
            "",
        ]
    )
    low_x = sum((s.open_play_x_below_10 for s in scans.values()), Counter())
    low_x_periods = sum((s.open_play_x_below_10_periods for s in scans.values()), Counter())
    pen = sum((s.penalty_positions for s in scans.values()), Counter())
    pen_n = sum(pen.values())
    if pen_n:
        mean_x = sum(x * c for (x, _), c in pen.items()) / pen_n
        mean_y = sum(y * c for (_, y), c in pen.items()) / pen_n
        add(
            [
                f"- Penalties in play, all files ({pen_n}): mean start position x = {mean_x:.1f}, "
                f"y = {mean_y:.1f}; most common {pen.most_common(3)}. The penalty mark is 11 m "
                "from the "
                "goal line (Laws of the Game), so x near 89-90 implies a pitch length of about "
                f"{11 / (1 - mean_x / 100):.0f} m *if* the scale is linear and the mark is at "
                "11 m.",
            ]
        )
    bins = sum((s.open_play_x_bins for s in scans.values()), Counter())
    add(
        [
            "- Open-play shots (non-shootout), all files, by x band:",
            "",
            table(
                ["x band", "Shots", "Goal tag", "Share"],
                [
                    [
                        f"{b}-{b + 10}",
                        bins[(b, False)] + bins[(b, True)],
                        bins[(b, True)],
                        pct(bins[(b, True)], bins[(b, False)] + bins[(b, True)]),
                    ]
                    for b in range(0, 100, 10)
                ],
            ),
            "",
            f"- Open-play shots with x < 50: "
            f"{sum(s.open_play_x_below_50 for s in scans.values())}. "
            f"With x < 10: {sum(low_x.values())} shots in {len(low_x)} matches "
            f"({', '.join(f'{m} ({matches[m]["label"]})' for m in sorted(low_x))}). "
            f"By period: {dict(low_x_periods)}. Shots this far from the attacked goal that "
            "cluster in a few matches (with goals among them) are consistent with the "
            "attacking direction having been recorded the wrong way round for part of a "
            "match; that explanation is an inference, not something the data states.",
            f"- Shot-type events with playerId 0: "
            f"{sum(s.shot_player_id_zero for s in scans.values())}; "
            "with a start position outside 0-100 or not integer: "
            f"{sum(s.shot_position_out_of_range for s in scans.values())}.",
            "- Second entry of `positions` on shot-type events, all files: "
            + ", ".join(
                f"{k}: {v}"
                for k, v in sum(
                    (s.second_positions for s in scans.values()), Counter()
                ).most_common()
            )
            + ".",
            f"- Sampled shots with start position outside 0-100: "
            f"{sum(1 for s in shots if s.x is None)} (counted as malformed above).",
            "",
            "## 7. Limits of this report",
            "",
            f"Shot-level statements (section 5) cover {audit.matches} sampled matches; section 3 "
            "counts cover the whole archive but are tallies, not validated shot records. No model "
            "has been fitted and no xG metric is reported.",
        ]
    )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the Wyscout open event dataset.")
    parser.add_argument("--matches-per-competition", type=int, default=6)
    parser.add_argument("--cache-dir", type=Path, default=wyscout.DEFAULT_CACHE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--offline", action="store_true", help="never download; fail if uncached")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    source = WyscoutSource(args.cache_dir, offline=args.offline)
    try:
        competitions = source.competitions()
        matches = {m["wyId"]: m for m in source.matches()}
        sample = select_sample(list(matches.values()), competitions, args.matches_per_competition)
        sample_ids = {m["wyId"] for m in sample}
        scans: dict[str, Scan] = {}
        for member in source.events_members():
            print(f"Scanning {member}...", file=sys.stderr)
            scans[member] = scan_events(source.iter_events(member), matches, sample_ids)
    except DataAccessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    audit = audit_sample(list(scans.values()), matches)
    report = render_report(source, competitions, matches, scans, sample, audit, args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {args.report} (downloaded: {source.downloads or 'nothing'}).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
