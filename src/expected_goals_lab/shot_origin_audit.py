"""S08A: is the coordinate fed to the xG model a shot origin, and is it outcome-coded?

    uv run python -m expected_goals_lab.shot_origin_audit

Development-only diagnosis (docs/SHOT_ORIGIN_AUDIT_PROTOCOL.md). The source event archive is read
in two passes (the first collects the regular shots of development matches and the pass-end
links, the second every event of the case-study and goal-line matches). Each pass parses every
event of the five league files, one at a time, so events of held-out matches are loaded into
memory too; fields other than ``matchId`` are examined only for events of development
(train + validation) matches. The shot table comes through the guarded loaders that refuse the
test split. Nothing here changes a model, a cohort or a frozen artefact.

The result has two parts that are kept apart. The *mechanical screen* is computed and
reproducible. The *manual review* of the source documentation and event context is a dated human
judgement recorded in ``S08A_REVIEW``; it is never inferred from a hard-coded flag.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expected_goals_lab import portfolio_data as pdata
from expected_goals_lab import portfolio_figures as pf
from expected_goals_lab.common import DataAccessError, table
from expected_goals_lab.dataset import DEFAULT_CSV, OPEN_SPLITS, iter_csv_rows
from expected_goals_lab.evaluate_baselines import LEAGUE_NAMES
from expected_goals_lab.evaluate_nonlinear import (
    DEFAULT_PROTOCOL,
    DEFAULT_S03_REPORT,
    evaluate_nonlinear,
)
from expected_goals_lab.nonlinear import NONLINEAR_LOCATION_BODY_PART
from expected_goals_lab.splits import split_of_events
from expected_goals_lab.wyscout import WyscoutSource
from expected_goals_lab.wyscout_shots import (
    BODY_PART_TAGS,
    EVENT_SHOT,
    SUBEVENT_SHOT,
    TAG_GOAL,
    _start_position,
    event_tag_ids,
)

DEFAULT_REPORT = Path("reports/shot_origin_audit.md")
CASE_MATCH_ID = pf.RECORDED_MATCH_ID
LEAGUE_FILES = (
    "events_England.json",
    "events_France.json",
    "events_Germany.json",
    "events_Italy.json",
    "events_Spain.json",
)
GOAL_MOUTH_TAGS = range(1201, 1224)
BANDS = ("x = 100", "x = 99", "x = 98", "x = 95-97", "x = 90-94", "x < 90")
CENTRE_Y = (45, 55)  # the goal mouth spans about 5.4 percentage points either side of y = 50
PERIOD_ORDER = {"1H": 0, "2H": 1, "E1": 2, "E2": 3, "P": 4}
BODY_NAME = {401: "left_foot", 402: "right_foot", 403: "head_or_body"}
MARKER_XG = (0.05, 0.20, 0.50, 0.60)
ORIGINAL_XG_AREA = 2000.0  # S08's marker scale before S08A (points squared per unit xG)
PASS_MIN_NON_GOAL_SHARE = 0.30
PASS_MAX_SUBSET = 0.001
CONDITIONAL_MAX_SUBSET = 0.005


class AuditError(RuntimeError):
    """An audit precondition failed. Stop and report; do not repair."""


# --- pure helpers ------------------------------------------------------------------------------


def x_band(x: int) -> str:
    if x >= 100:
        return BANDS[0]
    if x == 99:
        return BANDS[1]
    if x == 98:
        return BANDS[2]
    if x >= 95:
        return BANDS[3]
    if x >= 90:
        return BANDS[4]
    return BANDS[5]


def diagnostic_flags(x: int, y: int) -> dict[str, bool]:
    """Separate diagnostic flags (never exclusion rules)."""
    return {
        "x>=95": x >= 95,
        "x>=98": x >= 98,
        "x=100": x == 100,
        "(100,50)": (x, y) == (100, 50),
    }


def centre_zone(x: int, y: int) -> str:
    """The rows of the second screening table: where an x >= 95 position sits."""
    near_centre = CENTRE_Y[0] <= y <= CENTRE_Y[1]
    if x == 100:
        if y == 50:
            return "(100, 50) exactly"
        return "x = 100, other y inside the goal mouth" if near_centre else "x = 100, y outside"
    if x >= 95:
        return "x = 95-99, y inside the goal mouth" if near_centre else "x = 95-99, y outside"
    return "x < 95"


def second_position_class(positions) -> str:
    """Classify the second ``positions`` entry of a shot: absent, placeholder or other."""
    if not isinstance(positions, list) or len(positions) < 2 or not isinstance(positions[1], dict):
        return "absent"
    pair = (positions[1].get("x"), positions[1].get("y"))
    if pair == (0, 0):
        return "(0, 0) placeholder"
    if pair == (100, 100):
        return "(100, 100) placeholder"
    return "other"


def raw_positions(event: dict) -> list[tuple]:
    positions = event.get("positions")
    if not isinstance(positions, list):
        return []
    return [(p.get("x"), p.get("y")) for p in positions if isinstance(p, dict)]


def is_regular_shot(event: dict) -> bool:
    return event.get("eventId") == EVENT_SHOT and event.get("subEventId") == SUBEVENT_SHOT


def goal_mouth_tags(event: dict) -> list[int]:
    tags = event_tag_ids(event) or frozenset()
    return sorted(t for t in tags if t in GOAL_MOUTH_TAGS)


def event_clock(event: dict) -> str:
    seconds = float(event["eventSec"])
    return f"{event['matchPeriod']} {int(seconds // 60):02d}:{int(seconds % 60):02d}"


def marker_diameter_m(xg: float, area_per_unit: float | None = None) -> float:
    """Diameter, in metres of pitch, of the Figure 3 marker for a given xG (current layout)."""
    area_per_unit = pf.XG_AREA_PER_UNIT if area_per_unit is None else area_per_unit
    fig = pf.plt.figure(figsize=pf.MATCH_FIG_SIZE)
    ax = fig.add_axes(pf.MATCH_AXES_RECT)
    pf.draw_pitch(ax)
    ax.set_ylim(*pf.MATCH_YLIM)
    fig.canvas.draw()
    origin = ax.transData.transform((0.0, 0.0))
    unit = ax.transData.transform((1.0, 0.0))
    points_per_metre = float(np.hypot(*(unit - origin))) * 72.0 / fig.dpi
    pf.plt.close(fig)
    diameter_points = 2.0 * np.sqrt(area_per_unit * xg / np.pi)
    return diameter_points / points_per_metre


def band_table(x: np.ndarray, goal: np.ndarray) -> list[list]:
    """Per band: shots, goals, non-goals, goal rate."""
    bands = np.array([x_band(int(v)) for v in x])
    rows = []
    for band in BANDS:
        mask = bands == band
        n, g = int(mask.sum()), int(goal[mask].sum())
        rows.append([band, n, g, n - g, f"{100 * g / n:.1f}%" if n else "n/a"])
    return rows


def zone_table(x: np.ndarray, y: np.ndarray, goal: np.ndarray) -> list[list]:
    zones = np.array([centre_zone(int(a), int(b)) for a, b in zip(x, y, strict=True)])
    order = [
        "(100, 50) exactly",
        "x = 100, other y inside the goal mouth",
        "x = 100, y outside",
        "x = 95-99, y inside the goal mouth",
        "x = 95-99, y outside",
        "x < 95",
    ]
    rows = []
    for zone in order:
        mask = zones == zone
        n, g = int(mask.sum()), int(goal[mask].sum())
        rows.append([zone, n, g, n - g, f"{100 * g / n:.1f}%" if n else "n/a"])
    return rows


def link_table(
    event_ids: np.ndarray, x: np.ndarray, goal: np.ndarray, links: dict[int, float]
) -> list[list]:
    """Does the origin sit where the team's previous pass ended? Share within 5 and 10 units."""
    groups = {
        "x = 100": x >= 100,
        "x = 95-99": (x >= 95) & (x < 100),
        "x < 95": x < 95,
    }
    rows = []
    for label, mask in groups.items():
        for outcome, flag in (("goals", 1), ("non-goals", 0)):
            sel = mask & (goal == flag)
            d = np.array([links[int(e)] for e in event_ids[sel] if int(e) in links])
            if not len(d):
                rows.append([label, outcome, int(sel.sum()), 0, "n/a", "n/a", "n/a"])
                continue
            rows.append(
                [
                    label,
                    outcome,
                    int(sel.sum()),
                    len(d),
                    f"{np.median(d):.1f}",
                    f"{100 * (d <= 5).mean():.0f}%",
                    f"{100 * (d <= 10).mean():.0f}%",
                ]
            )
    return rows


def link_reading(r) -> str:
    """The plain-language reading of the link table, with its own numbers."""
    cells = {(row[0], row[1]): row for row in r.link_table}
    goals, non_goals = cells[("x < 95", "goals")], cells[("x < 95", "non-goals")]
    near_g, near_n = cells[("x = 95-99", "goals")], cells[("x = 95-99", "non-goals")]
    return (
        "**Reading.** Where a preceding pass exists, the shot origin is very often exactly the "
        f"pass's end (median distance {goals[4]} for goals and {non_goals[4]} for non-goals at "
        f"x < 95; {near_g[4]} and {near_n[4]} at x = 95-99). The share within 5 units is "
        f"{goals[5]} (goals) against {non_goals[5]} (non-goals) at x < 95 and {near_g[5]} "
        f"against {near_n[5]} at x = 95-99: similar for goals and non-goals, which is what "
        "a start position predicts and not what an outcome-dependent position would. The "
        "x = 100 rows are too few to say anything either way. This is consistency evidence "
        "about what the first `positions` entry is; it is not proof against the provider's "
        "coding, and the Wyscout page does not say which entry is the start of a shot."
    )


def subset_of_concern(x: np.ndarray, goal: np.ndarray) -> dict:
    """Rows on the goal line (x = 100): their share of the development rows and their outcomes."""
    mask = x >= 100
    n = int(mask.sum())
    return {
        "rows": n,
        "share": n / len(x) if len(x) else 0.0,
        "goals": int(goal[mask].sum()),
        "non_goals": n - int(goal[mask].sum()),
    }


def mechanical_screen(
    mismatches: int, near_goal_non_goal_share: float, subset: dict
) -> tuple[str, list[str]]:
    """The reproducible part: PASS, CONDITIONAL PASS or FAIL by the criteria of protocol section 6.

    It checks what a computer can check: reconciliation with ``positions[0]``, the non-goal share
    of near-goal shots and the goal-line subset. It cannot tell an origin from a destination;
    that question belongs to the manual review, which is deliberately not an argument here.
    """
    reasons = []
    if mismatches:
        return "FAIL", [f"{mismatches} development shots differ from positions[0]"]
    if subset["share"] > CONDITIONAL_MAX_SUBSET:
        return "FAIL", [f"goal-line subset is {100 * subset['share']:.2f}% of development rows"]
    if near_goal_non_goal_share < PASS_MIN_NON_GOAL_SHARE:
        reasons.append(
            f"non-goals are {100 * near_goal_non_goal_share:.1f}% of x = 95-99 shots "
            f"(minimum {100 * PASS_MIN_NON_GOAL_SHARE:.0f}% for PASS)"
        )
    if subset["rows"] and subset["non_goals"] == 0:
        reasons.append(f"all {subset['rows']} goal-line records are goals")
    if subset["share"] > PASS_MAX_SUBSET:
        reasons.append(f"goal-line subset is {100 * subset['share']:.2f}% of development rows")
    return ("CONDITIONAL PASS", reasons) if reasons else ("PASS", ["all PASS criteria met"])


REVIEW_NO_EVIDENCE = "no destination evidence found"
REVIEW_EVIDENCE = "destination evidence found"
REVIEW_NOT_PERFORMED = "not performed"
REVIEW_CONCLUSIONS = (REVIEW_NO_EVIDENCE, REVIEW_EVIDENCE, REVIEW_NOT_PERFORMED)


@dataclass(frozen=True)
class ManualReview:
    """A dated human judgement about what the first ``positions`` entry is. Not computed."""

    conclusion: str
    reviewed_on: str
    basis: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.conclusion not in REVIEW_CONCLUSIONS:
            raise ValueError(f"unknown review conclusion {self.conclusion!r}")


S08A_REVIEW = ManualReview(
    conclusion=REVIEW_NO_EVIDENCE,
    reviewed_on="2026-10-22",
    basis=(
        "the Wyscout documentation describes positions as the place an event 'started/ended "
        "(only start if not a movement)', although it does not say which entry is the start "
        "of a shot",
        "the second entry of every development regular shot is a placeholder, (0, 0) or "
        "(100, 100), with the same goal rate for both",
        "where a same-team pass precedes the shot, the shot origin is usually exactly the "
        "end position of that pass, for goals and non-goals alike",
        "the two goals of the case-study match have plausible source-event sequences and agree "
        "with a public match report in team and minute",
        "an independent review of the S08A report accepted this as internal-consistency "
        "evidence, not as proof of physical ground truth",
    ),
)


def overall_finding(screen: str, review: ManualReview) -> tuple[str, str]:
    """Combine the mechanical screen with the manual review; return (finding, how it arose).

    A failed screen is a failed finding whatever the review says, and a review that found
    destination evidence fails it whatever the screen says. A missing review leaves the finding
    unresolved: the mechanical screen alone never produces a PASS.
    """
    if screen == "FAIL":
        return "FAIL", "the mechanical screen failed"
    if review.conclusion == REVIEW_EVIDENCE:
        return "FAIL", "the manual review found evidence that a location is a destination"
    if review.conclusion == REVIEW_NOT_PERFORMED:
        return "UNRESOLVED", "the mechanical screen cannot settle origin versus destination"
    return screen, "evidence-based review judgement; the mechanical screen alone is not enough"


# --- reading: development events only -----------------------------------------------------------


def development_events(events: Iterable[dict], development_matches: set[int]) -> Iterator[dict]:
    """Events of development matches. For any other event only ``matchId`` is read."""
    for event in events:
        if event["matchId"] in development_matches:
            yield event


PLACEHOLDERS = {(0, 0), (100, 100)}
PASS_EVENT = 8
LINK_WINDOW_S = 10.0


def _real_end(event: dict) -> tuple[int, int] | None:
    """End position of a pass, if the second entry is a real in-range position."""
    pos = raw_positions(event)
    if event.get("eventId") != PASS_EVENT or len(pos) < 2 or pos[1] in PLACEHOLDERS:
        return None
    a, b = pos[1]
    return (a, b) if isinstance(a, int) and isinstance(b, int) else None


def read_shots_and_context(
    source: WyscoutSource, development_matches: set[int], context_matches: set[int]
) -> tuple[dict[int, dict], dict[int, list[dict]], dict[int, float]]:
    """Regular shots of every development match, all events of the context matches, and for each
    shot the distance (percentage units) from its origin to the end of the same team's latest
    pass in the previous ``LINK_WINDOW_S`` seconds (where such a pass exists)."""
    shots: dict[int, dict] = {}
    context: dict[int, list[dict]] = {m: [] for m in context_matches}
    links: dict[int, float] = {}
    for member in LEAGUE_FILES:
        last: dict[tuple[int, int], tuple[str, float, tuple[int, int]]] = {}
        for event in development_events(source.iter_events(member), development_matches):
            key = (event["matchId"], event["teamId"])
            if is_regular_shot(event):
                shots[int(event["id"])] = event
                prev, origin = last.get(key), _start_position(event.get("positions"))
                if (
                    prev
                    and origin[0] is not None
                    and prev[0] == event["matchPeriod"]
                    and 0 <= event["eventSec"] - prev[1] <= LINK_WINDOW_S
                ):
                    links[int(event["id"])] = float(
                        np.hypot(origin[0] - prev[2][0], origin[1] - prev[2][1])
                    )
            end = _real_end(event)
            if end is not None:
                last[key] = (event["matchPeriod"], event["eventSec"], end)
            if event["matchId"] in context_matches:
                context[event["matchId"]].append(event)
    for events in context.values():
        events.sort(key=lambda e: (PERIOD_ORDER.get(e["matchPeriod"], 9), e["eventSec"]))
    return shots, context, links


def neighbours(events: list[dict], event_id: int, before: int = 3, after: int = 2) -> list[dict]:
    index = next(i for i, e in enumerate(events) if int(e["id"]) == event_id)
    return events[max(0, index - before) : index + after + 1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- the audit ----------------------------------------------------------------------------------


@dataclass
class AuditResult:
    n_development: int
    n_train: int
    n_validation: int
    reconciled: int
    mismatches: list[dict]
    unmatched: int
    goal_flag_mismatch: int
    body_part_mismatch: int
    second_position: list[list]
    link_table: list[list]
    goal_mouth_table: list[list]
    bands_overall: list[list]
    bands_by_league: dict[str, list[list]]
    zones: list[list]
    bands_by_body: dict[str, list[list]]
    x_decile_table: list[list]
    distance_table: list[list]
    goal_line: list[dict]
    subset: dict
    near_goal_non_goal_share: float
    case: list[dict]
    case_goal_context: dict[int, list[dict]]
    screen: str
    screen_reasons: list[str]
    review: ManualReview
    finding: str
    finding_basis: str
    marker_m: dict[float, float]  # at the original S08 scale
    marker_now: dict[float, float]  # at the current display scale
    s07_before: dict[str, str]
    s07_after: dict[str, str]
    provenance: dict


def s07_hashes(root: Path = Path(".")) -> dict[str, str]:
    files = {
        "report": root / "reports/final_test_evaluation.md",
        "summary": root / pdata.S07_SUMMARY,
        "log": root / pdata.S07_LOG,
    }
    return {name: sha256_file(path) for name, path in files.items()}


def run_audit(
    csv_path: Path = DEFAULT_CSV,
    manifest_path: Path | None = None,
    split_dir: Path = Path("data/processed"),
    validation_reference: Path = pf.VALIDATION_SUMMARY,
    s03_report: Path = DEFAULT_S03_REPORT,
    protocol: Path = DEFAULT_PROTOCOL,
    source: WyscoutSource | None = None,
    rows=iter_csv_rows,
    case_match_id: int = CASE_MATCH_ID,
    reference_scores: dict | None = None,
    s07_root: Path = Path("."),
    review: ManualReview = S08A_REVIEW,
) -> AuditResult:
    manifest_path = manifest_path or csv_path.with_suffix(".manifest.json")
    pdata.require_shot_table(csv_path)
    before = s07_hashes(s07_root)
    source = source or WyscoutSource(offline=True)
    if reference_scores is None:
        reference_scores = json.loads(validation_reference.read_text(encoding="utf-8"))[
            "validation_scores"
        ]
    ev = evaluate_nonlinear(csv_path, manifest_path, split_dir, 20, s03_report, protocol, rows)
    pf.check_reproduces_validation(ev, reference_scores)

    split_of = split_of_events(ev.identity, ev.plan)
    development_matches = {
        int(m)
        for e, m in zip(ev.identity.event_ids, ev.identity.match_ids, strict=True)
        if split_of[int(e)] in OPEN_SPLITS
    }
    parts = [ev.parts[s] for s in OPEN_SPLITS]
    cat = np.concatenate
    event_ids = cat([p.base.event_ids for p in parts])
    match_ids = cat([p.base.match_ids for p in parts])
    comp = cat([p.base.competition_ids for p in parts])
    x, y = cat([p.base.x_pct for p in parts]), cat([p.base.y_pct for p in parts])
    goal = cat([p.base.y for p in parts]).astype(int)
    body = cat([p.body_part for p in parts])
    distance = cat([p.base.features["distance_m"] for p in parts])
    if case_match_id not in development_matches:
        raise AuditError(f"match {case_match_id} is not a development match")

    # first pass over the archive: regular shots of development matches only
    shots, _, _ = read_shots_and_context(source, development_matches, set())

    # reconciliation with the source events
    mismatches, unmatched, goal_bad, body_bad = [], 0, 0, 0
    for eid, mid, px, py, g, bp in zip(event_ids, match_ids, x, y, goal, body, strict=True):
        raw = shots.get(int(eid))
        if raw is None or raw["matchId"] != int(mid):
            unmatched += 1
            continue
        origin = _start_position(raw.get("positions"))
        first = raw_positions(raw)[0] if raw_positions(raw) else (None, None)
        if origin != (int(px), int(py)) or first != (int(px), int(py)):
            mismatches.append({"event_id": int(eid), "csv": (int(px), int(py)), "raw": first})
        tags = event_tag_ids(raw) or frozenset()
        if (TAG_GOAL in tags) != bool(g):
            goal_bad += 1
        names = [BODY_NAME[t] for t in tags if t in BODY_PART_TAGS]
        if (names[0] if len(names) == 1 else "unknown") != str(bp):
            body_bad += 1

    # second position and goal-mouth tags, on every regular shot of the development matches
    second = Counter()
    zone_tags = Counter()
    for raw in shots.values():
        tags = event_tag_ids(raw) or frozenset()
        is_goal = TAG_GOAL in tags
        second[(second_position_class(raw.get("positions")), is_goal)] += 1
        zone_tags[(bool(goal_mouth_tags(raw)), is_goal)] += 1
    second_rows = []
    for cls in ("(0, 0) placeholder", "(100, 100) placeholder", "other", "absent"):
        g, ng = second[(cls, True)], second[(cls, False)]
        second_rows.append([cls, g + ng, g, ng, f"{100 * g / (g + ng):.1f}%" if g + ng else "n/a"])
    zone_rows = []
    for label, flag in (("has a goal-mouth tag (1201-1223)", True), ("no goal-mouth tag", False)):
        g, ng = zone_tags[(flag, True)], zone_tags[(flag, False)]
        zone_rows.append([label, g + ng, g, ng, f"{100 * g / (g + ng):.1f}%" if g + ng else "n/a"])

    # screening tables
    bands_by_league = {
        name: band_table(x[comp == cid], goal[comp == cid]) for cid, name in LEAGUE_NAMES.items()
    }
    bands_by_body = {
        cls: band_table(x[body == cls], goal[body == cls])
        for cls in ("right_foot", "left_foot", "head_or_body")
    }
    decile_rows = []
    for lo in range(0, 100, 10):
        mask = (x >= lo) & (x < (lo + 10 if lo < 90 else 101))
        n, g = int(mask.sum()), int(goal[mask].sum())
        label = f"x = {lo}-{lo + 9}" if lo < 90 else "x = 90-100"
        decile_rows.append([label, n, g, n - g, f"{100 * g / n:.1f}%" if n else "n/a"])
    distance_rows = []
    for label, mask in (("goals", goal == 1), ("non-goals", goal == 0)):
        d = distance[mask]
        q = np.percentile(d, [5, 25, 50, 75, 95])
        distance_rows.append([label, int(mask.sum()), *[f"{v:.1f}" for v in q]])
    near = (x >= 95) & (x <= 99)
    near_non_goal_share = float((goal[near] == 0).mean()) if near.any() else 1.0
    subset = subset_of_concern(x, goal)

    # goal-line records and the match under inspection need neighbouring events
    goal_line_ids = [int(e) for e, px in zip(event_ids, x, strict=True) if px >= 100]
    case_ids = [e for e, m in zip(event_ids, match_ids, strict=True) if int(m) == case_match_id]
    context_matches = {case_match_id} | {
        int(m) for e, m, px in zip(event_ids, match_ids, x, strict=True) if px >= 100
    }
    _, context, links = read_shots_and_context(source, development_matches, context_matches)

    link_rows = link_table(event_ids, x, goal, links)
    val = ev.parts["validation"]
    xg_of = dict(
        zip(
            val.base.event_ids.tolist(),
            ev.predictions["validation"][NONLINEAR_LOCATION_BODY_PART].tolist(),
            strict=True,
        )
    )
    angle_of = dict(
        zip(val.base.event_ids.tolist(), val.base.features["angle_rad"].tolist(), strict=True)
    )
    case = []
    for eid in case_ids:
        i = int(np.where(event_ids == eid)[0][0])
        raw = shots[int(eid)]
        case.append(
            {
                "event": raw,
                "clock": event_clock(raw),
                "team": int(raw["teamId"]),
                "player": int(raw["playerId"]),
                "goal": bool(goal[i]),
                "body": str(body[i]),
                "positions": raw_positions(raw),
                "origin": (int(x[i]), int(y[i])),
                "distance": float(distance[i]),
                "angle_deg": float(np.degrees(angle_of[int(eid)])),
                "xg": float(xg_of[int(eid)]),
                "flags": diagnostic_flags(int(x[i]), int(y[i])),
                "goal_mouth": goal_mouth_tags(raw),
                "sort": (PERIOD_ORDER[raw["matchPeriod"]], raw["eventSec"]),
            }
        )  # fmt: skip
    case.sort(key=lambda r: r["sort"])
    case_goal_context = {
        r["event"]["id"]: neighbours(context[case_match_id], int(r["event"]["id"]))
        for r in case
        if r["goal"]
    }

    goal_line = []
    for eid in goal_line_ids:
        i = int(np.where(event_ids == eid)[0][0])
        raw = shots[eid]
        mid = int(match_ids[i])
        goal_line.append(
            {
                "event": raw,
                "match_id": mid,
                "league": LEAGUE_NAMES[int(comp[i])],
                "split": split_of[int(eid)],
                "goal": bool(goal[i]),
                "body": str(body[i]),
                "origin": (int(x[i]), int(y[i])),
                "positions": raw_positions(raw),
                "goal_mouth": goal_mouth_tags(raw),
                "clock": event_clock(raw),
                "context": neighbours(context[mid], eid),
            }
        )

    screen, screen_reasons = mechanical_screen(
        len(mismatches) + unmatched + goal_bad + body_bad, near_non_goal_share, subset
    )
    finding, finding_basis = overall_finding(screen, review)
    after = s07_hashes(s07_root)
    if after != before:
        raise AuditError("an S07 artefact changed during the audit")
    return AuditResult(
        n_development=len(event_ids),
        n_train=len(ev.parts["train"]),
        n_validation=len(ev.parts["validation"]),
        reconciled=len(event_ids) - len(mismatches) - unmatched,
        mismatches=mismatches,
        unmatched=unmatched,
        goal_flag_mismatch=goal_bad,
        body_part_mismatch=body_bad,
        second_position=second_rows,
        link_table=link_rows,
        goal_mouth_table=zone_rows,
        bands_overall=band_table(x, goal),
        bands_by_league=bands_by_league,
        zones=zone_table(x, y, goal),
        bands_by_body=bands_by_body,
        x_decile_table=decile_rows,
        distance_table=distance_rows,
        goal_line=goal_line,
        subset=subset,
        near_goal_non_goal_share=near_non_goal_share,
        case=case,
        case_goal_context=case_goal_context,
        screen=screen,
        screen_reasons=screen_reasons,
        review=review,
        finding=finding,
        finding_basis=finding_basis,
        marker_m={v: marker_diameter_m(v, ORIGINAL_XG_AREA) for v in MARKER_XG},
        marker_now={v: marker_diameter_m(v) for v in MARKER_XG},
        s07_before=before,
        s07_after=after,
        provenance={
            "shot_table_sha256": ev.identity.csv_sha256,
            "split_assignment_sha256": ev.assignment_sha,
            "event_archive_files": list(LEAGUE_FILES),
        },
    )


# --- report ---------------------------------------------------------------------------------------


def fmt_positions(positions: list[tuple]) -> str:
    return " ; ".join(f"({a}, {b})" for a, b in positions)


def render_report(r: AuditResult) -> str:
    hdr = ["Group", "Shots", "Goals", "Non-goals", "Goal rate"]
    out: list[str] = []
    add = out.append
    add("# Shot-origin provenance audit (S08A)\n")
    add(
        "Development-only diagnosis of the coordinates fed to the xG model, written under "
        "`docs/SHOT_ORIGIN_AUDIT_PROTOCOL.md`. It changes no model, cohort, split or frozen "
        "artefact, and no held-out (test-window) feature, label, prediction or "
        "match outcome is inspected, aggregated or printed. The event archive is read in two "
        "passes (the first collects the regular shots of development matches and the pass-end "
        "links, the second every event of the case-study and goal-line matches); each pass "
        "parses every event of the five league files into memory one at a time, so events of "
        "held-out matches are loaded too, but fields other than `matchId` are examined only for "
        "events of development matches.\n"
    )
    add(f"## Finding: {r.finding} ({r.finding_basis})\n")
    add("### Mechanical screen (computed, reproducible)\n")
    add(
        f"**{r.screen}.** The screen checks reconciliation with `positions[0]`, the goal flag "
        "and body-part consistency, the non-goal share of x = 95-99 shots and the goal-line "
        "subset, against the criteria fixed in the protocol. It cannot tell an origin from a "
        "destination.\n"
    )
    for reason in r.screen_reasons:
        add(f"- {reason}")
    add("")
    add(f"### Manual review (a dated judgement, not computed): {r.review.conclusion}\n")
    add(f"Reviewed {r.review.reviewed_on}. Basis:\n")
    for item in r.review.basis:
        add(f"- {item}")
    add("")
    add(
        "- Caveats that stand whatever the finding: the Wyscout page does not state which "
        "`positions` entry is the start of a shot (the first-entry reading is supported by "
        "the documented 'start' semantics and by the evidence below); "
        f"{r.subset['rows']} development rows sit exactly on the goal line; the audit tests "
        "consistency, not physical truth; and it covers the development partition only."
    )
    add("")
    add(
        f"Development rows audited: {r.n_development:,} eligible regular shots "
        f"({r.n_train:,} train, {r.n_validation:,} validation). Reconciled with the source "
        f"event's `positions[0]`: {r.reconciled:,}; mismatches {len(r.mismatches)}; "
        f"not found in the archive {r.unmatched}; goal flag differs from tag 101 on the event: "
        f"{r.goal_flag_mismatch}; body-part class differs from tags 401-403: "
        f"{r.body_part_mismatch}.\n"
    )
    add("## 1. What the dataset calls a shot location\n")
    add(
        "- **Documentation.** Wyscout (support.wyscout.com/matches-wyid-events): `positions` is "
        '"Positions in which this event started/ended (only start if not a movement)"; x and y '
        'are "percentage from the left corner of the attacking team". The page says nothing more '
        "about shot events and does not tie the goal-mouth tags 1201-1223 to the coordinate "
        "fields. Which of two position entries is the start for a shot is therefore **not "
        "stated** in that page; the first-entry reading rests on the documented 'start' "
        "semantics, the schema audit of S01B and the reconciliation below. The Scientific Data "
        "paper was consulted in S01B and not re-fetched in S08A (publisher page redirects to "
        "authentication).\n"
        "- **Code.** `src/expected_goals_lab/wyscout_shots.py` `parse_shot` line 161 -> "
        "`_start_position` (lines 219-226): first entry of `positions` only, integers in 0-100. "
        "`shot_table.make_row` copies it to `x_pct`/`y_pct` and derives distance and angle with "
        "`geometry.shot_geometry`. No tag is read to obtain a position.\n"
    )
    add("**Second `positions` entry on every regular shot of the development matches** "
        "(goal flag from tag 101 on the shot):\n")  # fmt: skip
    add(table(hdr, r.second_position) + "\n")
    add("**Goal-mouth tags (post-shot information, never read by the parser):**\n")
    add(table(hdr, r.goal_mouth_table) + "\n")
    add(
        "**Does the origin sit where the team's previous pass ended?** For development shots "
        "preceded within 10 s, in the same period, by a pass of the same team that has a real "
        "end position, the distance (percentage units; 1 unit is about 1 m along and 0.7 m "
        "across the pitch) between the shot origin and that pass's end. Short distances are what "
        "an origin (where the shooter received the ball) predicts; a destination would not "
        "follow the pass.\n"
    )
    add(
        table(
            [
                "x band",
                "Outcome",
                "Shots",
                "With a preceding pass",
                "Median dist",
                "Within 5",
                "Within 10",
            ],
            r.link_table,
        )
        + "\n"
    )
    add(link_reading(r) + "\n")
    add("## 2. Match 2516947 (Borussia Dortmund v Augsburg, 2018-02-26), event by event\n")
    add(
        "Sorted by period and time. All rows come from the shot table with the source event's "
        "full `positions` array beside them; xG is from the model fitted on the original "
        "training split only. Flags are diagnostic only.\n"
    )
    rows = []
    for k, c in enumerate(r.case, start=1):
        f = c["flags"]
        flag = ",".join(n for n, v in f.items() if v) or "-"
        rows.append(
            [
                k, c["clock"], c["event"]["id"], c["team"], c["player"],
                "GOAL" if c["goal"] else "no goal", c["body"], fmt_positions(c["positions"]),
                f"({c['origin'][0]}, {c['origin'][1]})", f"{c['distance']:.2f}",
                f"{c['angle_deg']:.1f}", f"{c['xg']:.3f}", flag,
            ]
        )  # fmt: skip
    add(
        table(
            [
                "#",
                "Clock",
                "Event id",
                "Team",
                "Player",
                "Outcome",
                "Body",
                "positions (all)",
                "Origin x,y %",
                "Dist m",
                "Angle deg",
                "xG",
                "Flags",
            ],
            rows,
        )  # fmt: skip
        + "\n"
    )
    add(
        "Each printed origin equals the first entry of the source event's `positions` (checked "
        "for every row above and for all development shots in the reconciliation line).\n"
    )
    for c in (c for c in r.case if c["goal"]):
        add(f"### Goal, event {c['event']['id']} ({c['clock']}, team {c['team']})\n")
        add(
            f"Own tags: {sorted(event_tag_ids(c['event']) or [])}; goal-mouth tags "
            f"{c['goal_mouth'] or 'none'}. Origin ({c['origin'][0]}, {c['origin'][1]}) %, "
            f"{c['distance']:.2f} m from the goal centre, angle {c['angle_deg']:.1f} degrees, "
            f"xG {c['xg']:.3f}. Neighbouring events in the match (three before, two after):\n"
        )
        ctx = [
            [e["id"], event_clock(e), e["eventName"], e["subEventName"], e["teamId"],
             fmt_positions(raw_positions(e)),
             "<- the shot" if e["id"] == c["event"]["id"] else ""]
            for e in r.case_goal_context[c["event"]["id"]]
        ]  # fmt: skip
        add(table(["Event id", "Clock", "Event", "Sub-event", "Team", "positions", ""], ctx) + "\n")
    add(EXTERNAL_NOTE + "\n")
    add("### Why the markers look so large\n")
    add(
        "The chart draws marker area proportional to xG. On the chart's pitch scale the "
        "diameter of a marker, in metres of pitch, was, at the S08 scale "
        f"({ORIGINAL_XG_AREA:.0f} points squared per unit xG): "
        + ", ".join(f"xG {v:.2f}: {m:.1f} m" for v, m in r.marker_m.items())
        + ". A marker for a goal at xG of about 0.6 was therefore wider than the 5.5 m six-yard "
        "box is deep, so the circle covered far more pitch than the distance of the recorded "
        "origin suggests. That is a display effect, separate from whether the coordinate is "
        "right. At the current scale "
        f"({pf.XG_AREA_PER_UNIT:.0f} points squared per unit): "
        + ", ".join(f"xG {v:.2f}: {m:.1f} m" for v, m in r.marker_now.items())
        + ". Both goals of this match have an estimated chance of "
        + ", ".join(f"{c['xg']:.3f}" for c in r.case if c["goal"])
        + " from a recorded distance of "
        + ", ".join(f"{c['distance']:.2f} m" for c in r.case if c["goal"])
        + " and an angle of "
        + ", ".join(f"{c['angle_deg']:.0f} degrees" for c in r.case if c["goal"])
        + ": a very high probability is what a model built on distance and angle gives to a "
        "shot about 4 m from goal that sees most of the goal mouth (compare the goal rate "
        "of shots recorded at x = 95-99 inside the goal mouth in table 3.3), so the large "
        "circles reflect the recorded closeness, not something questionable in the "
        "coordinates.\n"
    )
    add("## 3. Development-only outcome screen\n")
    add("### 3.1 Origin x by band, all leagues\n")
    add(table(["x band (%)", "Shots", "Goals", "Non-goals", "Goal rate"], r.bands_overall) + "\n")
    add("### 3.2 The same, by league\n")
    for name, rows_ in r.bands_by_league.items():
        add(f"**{name}**\n")
        add(table(["x band (%)", "Shots", "Goals", "Non-goals", "Goal rate"], rows_) + "\n")
    add("### 3.3 Goal-line and centre-of-goal positions\n")
    add(table(["Position group", "Shots", "Goals", "Non-goals", "Goal rate"], r.zones) + "\n")
    add("### 3.4 By body-part class\n")
    for cls, rows_ in r.bands_by_body.items():
        add(f"**{cls}**\n")
        add(table(["x band (%)", "Shots", "Goals", "Non-goals", "Goal rate"], rows_) + "\n")
    add("### 3.5 Where goals and non-goals are taken\n")
    add(table(["x band (%)", "Shots", "Goals", "Non-goals", "Goal rate"], r.x_decile_table) + "\n")
    add("Distance to the goal centre (m), percentiles:\n")
    add(table(["Group", "Shots", "5%", "25%", "50%", "75%", "95%"], r.distance_table) + "\n")
    add("### 3.6 Goal-line records (x = 100) in the development partition\n")
    s = r.subset
    add(
        f"{s['rows']} records, {100 * s['share']:.3f}% of development rows: {s['goals']} goals and "
        f"{s['non_goals']} non-goals. The S02 audit counted seven goal-line shots in the whole "
        "cohort; the development partition holds the records listed here, and nothing is "
        "examined or stated about the rest. A high x is not by itself taken as an error: "
        "several of these follow a pass that ended at the same position, i.e. the shooter "
        "received the ball on the goal line.\n"
    )
    if r.goal_line:
        gl = [
            [g["event"]["id"], g["match_id"], g["league"], g["split"], g["clock"],
             "GOAL" if g["goal"] else "no goal", g["body"], f"({g['origin'][0]}, {g['origin'][1]})",
             fmt_positions(g["positions"]), g["goal_mouth"] or "-"]
            for g in r.goal_line
        ]  # fmt: skip
        add(
            table(
                [
                    "Event id",
                    "Match",
                    "League",
                    "Split",
                    "Clock",
                    "Outcome",
                    "Body",
                    "Origin %",
                    "positions (all)",
                    "Goal-mouth tags",
                ],
                gl,
            )  # fmt: skip
            + "\n"
        )
        for g in r.goal_line:
            add(f"Neighbouring events of event {g['event']['id']} (qualitative context only):\n")
            ctx = [
                [e["id"], event_clock(e), e["eventName"], e["teamId"],
                 fmt_positions(raw_positions(e)),
                 "<- the shot" if e["id"] == g["event"]["id"] else ""]
                for e in g["context"]
            ]  # fmt: skip
            add(table(["Event id", "Clock", "Event", "Team", "positions", ""], ctx) + "\n")
    add("## 4. Technical review of the S08 claims and display\n")
    add(review_text(r) + "\n")
    add("## 4b. Limits of this audit\n")
    add(LIMITS + "\n")
    add("## 5. Frozen artefacts\n")
    add(
        table(
            ["Artefact", "SHA-256 before the audit", "SHA-256 after"],
            [[k, f"`{r.s07_before[k]}`", f"`{r.s07_after[k]}`"] for k in r.s07_before],
        )
        + "\n"
    )
    add(
        "Shot table SHA-256 "
        f"`{r.provenance['shot_table_sha256']}`; split assignment SHA-256 "
        f"`{r.provenance['split_assignment_sha256']}`. `REAL_EXECUTION_ENABLED` is unchanged. "
        "Data: Pappalardo & Massucco (2019), Wyscout open event data, CC BY 4.0, "
        "doi:10.6084/m9.figshare.c.4415000.v5; filtered, quality-screened and augmented for "
        "this project.\n"
    )
    return "\n".join(out)


EXTERNAL_NOTE = (
    "### External corroboration note (outside the data; it overrides nothing)\n\n"
    "A public match report (Sporting Life, "
    "https://www.sportinglife.com/football/live/38908/borussia-dortmund-vs-fc-augsburg, "
    "fetched 2026-10-22) gives Borussia Dortmund 1-1 Augsburg on 26 February 2018, with Marco "
    "Reus (Dortmund) scoring in the 16th minute and Kevin Danso (Augsburg) in the 73rd. The "
    "dataset's goal events are Dortmund (team 2447) at 1H 15:08 and Augsburg (team 2481) at 2H "
    "27:05, which is the 16th minute and the 73rd (45 + 27). Team and minute agree. The "
    "player ids of the two goals were not matched to names (no players file was used), and the "
    "report gives no positions, so it neither confirms nor refutes the coordinates in the "
    "table. The data show the Augsburg goal coming two seconds after a saved headed shot at "
    "2H 27:03, which followed a corner at 2H 27:01.\n"
)

LIMITS = (
    "- The audit checks internal consistency and agreement with the provider's documented "
    "'start' semantics. It cannot prove that the provider's coordinates are physically "
    "correct, and no video was consulted.\n"
    "- Positions are integer percentages, about 1 m along and 0.7 m across the pitch; very "
    "close shots are sensitive to that quantisation.\n"
    "- Quarantined rows (`x_pct` below 20, a score-mismatch match) are outside the "
    "cohort and were not re-examined.\n"
    "- Everything here concerns the development partition. Nothing is inferred about the "
    "held-out window, and the S07 results stay locked and are not re-run or re-interpreted."
)


def review_text(r: AuditResult) -> str:
    """Findings of the technical review of S08 (Audit D)."""
    return (
        "1. **Reading paths.** `dataset.read_identity`, `dataset._parse_partitions`, "
        "`candidate_data._join_body_parts` and `portfolio_data.read_team_columns` all traverse "
        "every line of the shot table, as S03-S05 already did. Held-out rows are touched only "
        "for identity columns (event id, match id, league, date, eligibility); "
        "`load_partitions` and `load_body_parts` refuse the test split, and "
        "`read_team_columns` reads team and side only for requested validation events. The S08 "
        'wording "no test-window row is read" was therefore too broad and now says: no '
        "held-out feature, outcome or prediction is inspected, scored or used to fit a model or "
        "build a figure.\n\n"
        "2. **What Figure 1 compares.** The held-out evaluation compared the location + "
        "body-part recipe with the distance + angle recipe. They differ in their location terms "
        "as well as the body-part tag, so it is not a body-part ablation. The smooth-location "
        "model without the tag was a validation-stage comparison only and was not part of the "
        "held-out result. Wording in the story and on Figure 1 was adjusted; no causal claim "
        "is made.\n\n"
        "3. **Match map direction.** Figure 3 draws the home team attacking right and the away "
        "team attacking left in both halves. That is a display convention, not the real "
        "direction of play (teams switch ends at half-time); the figure now says so.\n\n"
        "4. **Markers and footnotes.** The markers were large: with the original scale "
        f"({ORIGINAL_XG_AREA:.0f} points squared per unit xG) a goal at xG 0.6 spanned "
        f"{r.marker_m[0.60]:.1f} m of pitch at the original scale, more than the six-yard box "
        "is deep. Given the audit result, only the display was changed: the scale is now "
        f"{pf.XG_AREA_PER_UNIT:.0f} points squared per unit, a small dot marks each recorded "
        "position, and the footers were shortened. The plotted probabilities, team totals, "
        "inputs and selection rule are identical. Figure 2 is unchanged (common colour "
        "scale and support mask).\n\n"
        "5. **Figure 1.** The frozen held-out metrics are untouched; only the headline "
        "and footer wording changed. A tidy chart is not evidence that its data are right; "
        "that rests on this audit."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Development-only shot-origin audit (S08A).")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        result = run_audit()
    except (AuditError, pdata.FigureInputError, FileNotFoundError, DataAccessError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result), encoding="utf-8")
    print(f"finding: {result.finding} ({result.finding_basis})")
    print(f"mechanical screen: {result.screen}; manual review: {result.review.conclusion}")
    for reason in result.screen_reasons:
        print(f"  {reason}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
