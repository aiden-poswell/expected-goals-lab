"""S02: build the canonical regular-shot table for the five Wyscout domestic leagues.

    uv run python -m expected_goals_lab.build_shots

Reads the checksum-pinned Figshare archives through ``wyscout.WyscoutSource`` (cached under
data/cache/wyscout/), streams the five domestic event files match by match, and writes:

* data/processed/wyscout_regular_shots.csv       one row per regular shot (git-ignored)
* data/processed/wyscout_regular_shots.manifest.json
* reports/shot_dataset_audit.md                  compact aggregated audit (committed)

The CSV is byte-for-byte reproducible for a given archive and code version. Nothing is trained,
split or fitted. Field definitions: docs/SHOT_DATASET_SPEC.md.
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from expected_goals_lab import wyscout
from expected_goals_lab.common import DataAccessError, table
from expected_goals_lab.geometry import ALTERNATIVE_PITCHES, REFERENCE_PITCH, shot_geometry
from expected_goals_lab.shot_table import (
    CANDIDATE_FEATURES,
    COLUMN_ROLES,
    COLUMNS,
    EXCLUSION_PRECEDENCE,
    FEATURE_ALLOWLIST,
    SCHEMA_VERSION,
    SUSPECT_POSITION,
    SUSPECT_X_PCT,
    TARGET,
    BuildResult,
    IntegrityError,
    ShotRow,
    build_table,
    check_integrity,
    sha256_file,
    write_csv,
)
from expected_goals_lab.wyscout import WyscoutSource
from expected_goals_lab.wyscout_shots import FREE_KICK, OPEN_PLAY, PENALTY

DEFAULT_OUT = Path("data/processed/wyscout_regular_shots.csv")
DEFAULT_REPORT = Path("reports/shot_dataset_audit.md")
SUSPECT_THRESHOLDS = (10, 20, 30, 50)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return centre - half, centre + half


def rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    low, high = wilson(k, n)
    return f"{100 * k / n:.2f}% [{100 * low:.2f}, {100 * high:.2f}]"


def quantiles(values: list[float], points=(0, 0.25, 0.5, 0.75, 1.0)) -> list[float]:
    if not values:
        return [math.nan] * len(points)
    ordered = sorted(values)
    out = []
    for q in points:
        position = q * (len(ordered) - 1)
        low, high = math.floor(position), math.ceil(position)
        out.append(ordered[low] + (ordered[high] - ordered[low]) * (position - low))
    return out


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = average_ranks(a), average_ranks(b)
    mean_a, mean_b = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb, strict=True))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    if var_a == 0 or var_b == 0:
        return math.nan  # a constant sequence has no rank correlation
    return cov / math.sqrt(var_a * var_b)


def geometry_sensitivity(rows: list[ShotRow]) -> list[list]:
    """Recompute distance and angle for the eligible rows under each alternative pitch."""
    if not rows:
        return []
    reference = {
        "distance": [r.distance_m for r in rows],
        "angle": [r.angle_rad for r in rows],
    }
    output = []
    for name, pitch in ALTERNATIVE_PITCHES.items():
        alt = [shot_geometry(r.x_pct, r.y_pct, pitch) for r in rows]
        for label, ref, new in (
            ("distance_m", reference["distance"], [g.distance_m for g in alt]),
            (
                "angle_deg",
                [math.degrees(a) for a in reference["angle"]],
                [g.angle_deg for g in alt],
            ),
        ):
            diffs = [abs(a - b) for a, b in zip(ref, new, strict=True)]
            n = len(ref)
            ref_pct = [(r - 0.5) / n for r in average_ranks(ref)]
            new_pct = [(r - 0.5) / n for r in average_ranks(new)]
            shift = [abs(a - b) for a, b in zip(ref_pct, new_pct, strict=True)]
            output.append(
                [
                    name,
                    label,
                    f"{sum(diffs) / n:.3f}",
                    f"{max(diffs):.3f}",
                    f"{spearman(ref, new):.5f}",
                    f"{max(shift):.4f}",
                    f"{100 * sum(1 for s in shift if s > 0.05) / n:.2f}%",
                ]
            )
    return output


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process in MB, if the platform offers it."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(Counters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            ok = psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            )
            return counters.PeakWorkingSetSize / 2**20 if ok else None
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 1024
    except Exception:  # noqa: BLE001 - diagnostics must never break the build
        return None


def render_report(
    result: BuildResult,
    integrity: list[tuple[str, str]],
    csv_sha256: str,
    args: argparse.Namespace,
) -> str:
    rows = result.rows
    eligible = [r for r in rows if r.eligible]
    leagues = sorted(result.leagues.values(), key=lambda t: t.name)
    out: list[str] = []
    add = out.extend

    def count(rs, cid, **conds):
        return sum(
            1
            for r in rs
            if r.competition_id == cid and all(getattr(r, k) == v for k, v in conds.items())
        )

    add(
        [
            "# Shot dataset audit (S02)",
            "",
            "Generated by `uv run python -m expected_goals_lab.build_shots` on "
            f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Every figure is computed in that run from "
            "the "
            "pinned Wyscout archives. Aggregates only: the shot table itself is not committed.",
            "",
            "Source: Pappalardo & Massucco (2019), *Soccer match event dataset*, figshare, "
            f"doi:{wyscout.COLLECTION_DOI}; Pappalardo et al. (2019), Scientific Data 6:236, "
            f"doi:{wyscout.PAPER_DOI}; data collected by Wyscout; {wyscout.LICENCE} "
            f"({wyscout.LICENCE_URL}). The data were processed and transformed for this project.",
            "",
            f"Output: `{args.out.as_posix()}`, {len(rows)} rows, {len(COLUMNS)} columns, "
            f"SHA-256 `{csv_sha256}`. Pinned inputs: "
            + ", ".join(f"{f.name} (md5 `{f.md5}`)" for f in (wyscout.EVENTS, wyscout.MATCHES))
            + ".",
            "",
            "## 1. From events to model-eligible rows, by league",
            "",
            "Only the five domestic league files are read for rows. Out of scope and not scanned: "
            + "; ".join(
                f"{member} ({info['competition']}, {info['metadata_matches']} matches in metadata)"
                for member, info in sorted(result.out_of_scope.items())
            )
            + ".",
            "",
            "Event counts (goal-tagged in brackets; tag 101 on the shot-type event itself):",
            "",
        ]
    )
    tally_rows = []
    for t in leagues:
        tally_rows.append(
            [
                t.name,
                t.matches,
                t.events,
                f"{t.kind_events[OPEN_PLAY]} ({t.kind_goals[OPEN_PLAY]})",
                f"{t.kind_events[FREE_KICK]} ({t.kind_goals[FREE_KICK]})",
                f"{t.kind_events[PENALTY]} ({t.kind_goals[PENALTY]})",
            ]
        )
    totals = Counter()
    for t in leagues:
        totals.update({"m": t.matches, "e": t.events})
        for k in (OPEN_PLAY, FREE_KICK, PENALTY):
            totals[k] += t.kind_events[k]
            totals[k + "g"] += t.kind_goals[k]
    tally_rows.append(
        [
            "**Five leagues**",
            totals["m"],
            totals["e"],
            f"{totals[OPEN_PLAY]} ({totals[OPEN_PLAY + 'g']})",
            f"{totals[FREE_KICK]} ({totals[FREE_KICK + 'g']})",
            f"{totals[PENALTY]} ({totals[PENALTY + 'g']})",
        ]
    )
    add(
        [
            table(
                [
                    "League",
                    "Matches",
                    "All events",
                    "Regular shots (10/100)",
                    "Free-kick shots (3/33)",
                    "Penalties (3/35)",
                ],
                tally_rows,
            ),
            "",
            "Rows in the shot table. *Quarantined* rows are kept with reason codes and `eligible "
            "= 0`:",
            "",
        ]
    )
    row_table = []
    for t in leagues:
        cid = t.competition_id
        n, q, e = count(rows, cid), count(rows, cid, eligible=0), count(rows, cid, eligible=1)
        g = count(rows, cid, eligible=1, is_goal=1)
        row_table.append([t.name, n, q, e, g, e - g, rate(g, e)])
    n_all, q_all = len(rows), sum(1 for r in rows if not r.eligible)
    g_all = sum(1 for r in eligible if r.is_goal == 1)
    row_table.append(
        [
            "**Five leagues**",
            n_all,
            q_all,
            len(eligible),
            g_all,
            len(eligible) - g_all,
            rate(g_all, len(eligible)),
        ]
    )
    add(
        [
            table(
                [
                    "League",
                    "Regular-shot rows",
                    "Quarantined",
                    "Eligible",
                    "Eligible goals",
                    "Eligible non-goals",
                    "Goal rate [95% Wilson]",
                ],
                row_table,
            ),
            "",
            "Goal rates describe the eligible cohort only; they are descriptive, not a model "
            "result.",
            "",
            "## 2. Quarantine reasons",
            "",
        ]
    )
    reason_rows = []
    for reason in EXCLUSION_PRECEDENCE:
        any_r = [r for r in rows if reason in r.exclusion_reasons.split(";")]
        prim = [r for r in rows if r.primary_exclusion == reason]
        reason_rows.append([reason, len(any_r), len(prim), sum(1 for r in any_r if r.is_goal == 1)])
    add(
        [
            "A row can have several reasons; `primary` is the first in this precedence order "
            "(top to bottom).",
            "",
            table(
                ["Reason", "Rows with reason", "Rows where primary", "Goal-tagged among them"],
                reason_rows,
            ),
            "",
            "Informational flags (do not exclude): "
            + ", ".join(
                f"{flag} {sum(1 for r in rows if flag in r.flags.split(';'))}"
                for flag in (
                    "UNKNOWN_PLAYER",
                    "BODY_PART_UNKNOWN",
                    "OWN_HALF_POSITION",
                    "ON_GOAL_LINE",
                )
            )
            + ". Body-part classes among eligible rows: "
            + ", ".join(
                f"{k} {v}" for k, v in Counter(r.body_part_class for r in eligible).most_common()
            )
            + ".",
            "",
            "## 3. Goal bridge: published scores to tagged events",
            "",
            "Per league, published goals (sum of both teams' scores in play) against what the tags "
            "explain. Own goals (tag 102) and penalty, free-kick and corner goals are separate "
            "categories and are not treated as regular-shot misses.",
            "",
        ]
    )
    bridge = []
    for t in leagues:
        explained = (
            t.kind_goals[OPEN_PLAY]
            + t.kind_goals[FREE_KICK]
            + t.kind_goals[PENALTY]
            + t.other_goal_events
            + t.own_goal_events
        )
        bridge.append(
            [
                t.name,
                t.published_goals,
                t.kind_goals[OPEN_PLAY],
                t.kind_goals[FREE_KICK],
                t.kind_goals[PENALTY],
                t.other_goal_events,
                t.own_goal_events,
                t.published_goals - explained,
            ]
        )
    cols = list(zip(*[b[1:] for b in bridge], strict=True))
    bridge.append(["**Five leagues**", *[sum(c) for c in cols]])
    add(
        [
            table(
                [
                    "League",
                    "Published goals",
                    "Regular shots",
                    "Free-kick shots",
                    "Penalties",
                    "Other attacking events",
                    "Own goals (102)",
                    "Unexplained",
                ],
                bridge,
            ),
            "",
            "Matches whose rebuilt goals differ from the published score: "
            + (", ".join(str(m) for t in leagues for m in t.mismatched_matches) or "none")
            + ".",
            "",
        ]
    )
    for detail in result.mismatches:
        mid = detail["match_id"]
        affected = [r for r in rows if r.match_id == mid]
        add(
            [
                f"### 3.1 Match {mid}: {detail['label']} ({detail['date_utc']} UTC)",
                "",
                f"- Published goals by team: {detail['published']}. Goals rebuilt from tags: "
                f"{detail['rebuilt']}.",
                f"- Lineup records credit these players with a goal (team, player, `goals`): "
                f"{detail['lineup_credited_scorers']}. That field is a separate provider record; "
                "this "
                "project does not use it as a label.",
                "- Goal-tagged or own-goal-tagged events in the match:",
                "",
                table(
                    ["Event id", "Period", "Sec", "Team", "Player", "Type", "Tags", "Positions"],
                    [
                        [
                            e["id"],
                            e["period"],
                            e["sec"],
                            e["team"],
                            e["player"],
                            e["type"],
                            e["tags"],
                            e["positions"],
                        ]
                        for e in detail["goal_tagged_events"]
                    ],
                ),
                "",
                "- Shot-type events (regular, free-kick or penalty) by the credited scorer(s):",
                "",
                table(
                    ["Event id", "Period", "Sec", "Team", "Player", "Type", "Tags", "Positions"],
                    [
                        [
                            e["id"],
                            e["period"],
                            e["sec"],
                            e["team"],
                            e["player"],
                            e["type"],
                            e["tags"],
                            e["positions"],
                        ]
                        for e in detail["credited_scorer_shot_events"]
                    ],
                ),
                "",
            ]
        )
        for item in detail["before_keeper_goal"]:
            add(
                [
                    f"- Events immediately before keeper event {item['keeper_event']['id']}:",
                    "",
                    table(
                        [
                            "Event id",
                            "Period",
                            "Sec",
                            "Team",
                            "Player",
                            "Type",
                            "Tags",
                            "Positions",
                        ],
                        [
                            [
                                e["id"],
                                e["period"],
                                e["sec"],
                                e["team"],
                                e["player"],
                                e["type"],
                                e["tags"],
                                e["positions"],
                            ]
                            for e in item["previous_events"]
                        ],
                    ),
                    "",
                ]
            )
        last = detail["credited_scorer_last_event_before_keeper_goal"]
        add(
            [
                f"- Last event by a credited scorer before the keeper's goal event: "
                f"{last['type'] if last else 'none'}"
                + (
                    f" (event {last['id']}, {last['period']} {last['sec']} s, tags {last['tags']}, "
                    f"positions {last['positions']})"
                    if last
                    else ""
                )
                + ".",
                f"- **Treatment:** every regular-shot row of the match ({len(affected)} rows, "
                f"{sum(1 for r in affected if r.is_goal == 1)} goal-tagged) is quarantined with "
                "`MATCH_SCORE_MISMATCH`. No label was edited. The events show the only goal tag "
                "on a "
                "goalkeeper event and none on any attacking event; they do not prove which "
                "provider "
                "event, if any, was the scoring shot, so the scoreboard is not used to pick one.",
                "",
            ]
        )

    add(
        [
            f"## 4. Suspect positions (x_pct < {SUSPECT_X_PCT})",
            "",
            "Regular shots recorded below this x are quarantined with `SUSPECT_POSITION`. The raw "
            "coordinates are kept and nothing is flipped. For each, opponent events within 3 s are "
            "compared with the recorded position and its mirror `(100-x, 100-y)` (radius 20 "
            "points; "
            "goalkeeper events excluded). Verdicts: `mirror_supported` (only the mirror agrees), "
            "`recorded_supported`, `conflicting`, `no_evidence`. These are diagnostics, not "
            "corrections.",
            "",
        ]
    )
    suspects = sorted(result.suspects, key=lambda s: (s.x_pct, s.match_id, s.event_sec))
    names = {t.competition_id: t.name for t in leagues}
    match_league = {m: names.get(rec["competitionId"]) for m, rec in result.matches.items()}
    add(
        [
            table(
                [
                    "Match",
                    "League",
                    "Team",
                    "Period",
                    "Sec",
                    "x,y",
                    "Goal-tagged",
                    "Verdict",
                    "Keeper save after",
                    "Previous same-team event",
                    "Next same-team event",
                    "Team's other shots, lowest x",
                ],
                [
                    [
                        s.match_id,
                        match_league[s.match_id],
                        s.team_id,
                        s.period,
                        round(s.event_sec),
                        f"{s.x_pct},{s.y_pct}",
                        "yes" if s.is_goal else "no",
                        s.verdict,
                        "yes" if s.opponent_keeper_response else "no",
                        s.previous_same_team or "-",
                        s.next_same_team or "-",
                        list(s.other_team_shots_x),
                    ]
                    for s in suspects
                ],
            ),
            "",
            "Verdict counts: "
            + ", ".join(f"{k} {v}" for k, v in Counter(s.verdict for s in suspects).most_common())
            + ". "
            f"Below x = 10: {sum(1 for s in suspects if s.x_pct < 10)} shots in "
            f"{len({s.match_id for s in suspects if s.x_pct < 10})} matches, of which "
            f"{sum(1 for s in suspects if s.x_pct < 10 and s.period == '2H')} in the second half.",
            "",
            "Cohort sensitivity to the cutoff (rows that are otherwise eligible; raw x_pct):",
            "",
        ]
    )
    base = [
        r
        for r in rows
        if set(r.exclusion_reasons.split(";")) <= {"", SUSPECT_POSITION} and r.x_pct is not None
    ]
    sens = []
    for cut in SUSPECT_THRESHOLDS:
        kept = [r for r in base if r.x_pct >= cut]
        removed = [r for r in base if r.x_pct < cut]
        kept_goals = sum(1 for r in kept if r.is_goal == 1)
        sens.append(
            [
                f"x_pct >= {cut}" + (" (used)" if cut == SUSPECT_X_PCT else ""),
                len(kept),
                len(removed),
                sum(1 for r in removed if r.is_goal == 1),
                rate(kept_goals, len(kept)),
            ]
        )
    add(
        [
            table(
                [
                    "Eligibility cutoff",
                    "Eligible rows",
                    "Removed by cutoff",
                    "Goals removed",
                    "Goal rate of kept rows [95% Wilson]",
                ],
                sens,
            ),
            "",
            "## 5. Geometry",
            "",
            f"Reference pitch {REFERENCE_PITCH.length_m:g} x {REFERENCE_PITCH.width_m:g} m, goal "
            f"{REFERENCE_PITCH.goal_width_m} m; `x_m = 105 * x_pct / 100`, `y_m = 68 * y_pct / "
            "100`, "
            "goal centre (105, 34). Actual stadium dimensions are not in the data, so distances "
            "and "
            "angles below are approximations under that assumption (eligible rows).",
            "",
        ]
    )
    dist = [r.distance_m for r in eligible]
    ang = [r.angle_deg for r in eligible]
    add(
        [
            table(
                ["Quantity", "min", "25%", "median", "75%", "max"],
                [
                    ["distance_m", *[f"{v:.2f}" for v in quantiles(dist)]],
                    ["angle_deg", *[f"{v:.2f}" for v in quantiles(ang)]],
                ],
            ),
            "",
            "Sensitivity to the assumed pitch (same 7.32 m goal; eligible rows; differences are "
            "alternative minus reference, in absolute value). *Max percentile shift* is the "
            "largest "
            "change in a shot's rank position (0-1) among all eligible shots; *Share >0.05* is the "
            "fraction of shots whose percentile moves by more than five points.",
            "",
            table(
                [
                    "Pitch",
                    "Quantity",
                    "Mean abs diff",
                    "Max abs diff",
                    "Spearman rho vs reference",
                    "Max percentile shift",
                    "Share > 0.05",
                ],
                geometry_sensitivity(eligible),
            ),
            "",
            "## 6. Integrity assertions on the full table",
            "",
            table(["Check", "Result"], integrity),
            "",
            "## 7. Unresolved issues",
            "",
            "- Sequence evidence for the suspect positions is mixed; the cause (a frame reversal "
            "over "
            "part of a sequence, wrong tagging, or something else) is not established, and shots "
            "with "
            "a wrong frame at ordinary-looking positions cannot be detected from coordinates "
            "alone.",
            "- One match has a goal that is tagged only on the opponent goalkeeper's event; it is "
            "quarantined, not repaired.",
            "- Regular shots are not guaranteed to be open play: they may follow set pieces.",
            "- Pitch size, goal position and the x = 100 convention rest on documentation and the "
            "penalty-spot check; per-stadium dimensions are unknown.",
            "- Provider tagging is human and unaudited; one season, five leagues, no provider xG.",
            "- Nothing here has been split, fitted or evaluated.",
        ]
    )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the canonical Wyscout regular-shot table.")
    parser.add_argument("--cache-dir", type=Path, default=wyscout.DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--offline", action="store_true", help="never download; fail if uncached")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    started = time.perf_counter()

    source = WyscoutSource(args.cache_dir, offline=args.offline)
    try:
        result = build_table(source)
        integrity = check_integrity(result)
    except DataAccessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except IntegrityError as exc:
        print(f"integrity error: {exc}", file=sys.stderr)
        return 1

    write_csv(result.rows, args.out)
    digest = sha256_file(args.out)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "columns": list(COLUMNS),
        "rows": len(result.rows),
        "eligible_rows": sum(r.eligible for r in result.rows),
        "csv_sha256": digest,
        "column_roles": COLUMN_ROLES,
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "candidate_features": list(CANDIDATE_FEATURES),
        "target": TARGET,
        "pitch": {
            "length_m": REFERENCE_PITCH.length_m,
            "width_m": REFERENCE_PITCH.width_m,
            "goal_width_m": REFERENCE_PITCH.goal_width_m,
        },
        "suspect_x_pct": SUSPECT_X_PCT,
        "exclusion_precedence": list(EXCLUSION_PRECEDENCE),
        "sources": [
            {"name": f.name, "figshare_file_id": f.file_id, "bytes": f.size, "md5": f.md5}
            for f in (wyscout.EVENTS, wyscout.MATCHES, wyscout.COMPETITIONS)
        ],
        "collection_doi": wyscout.COLLECTION_DOI,
        "licence": wyscout.LICENCE,
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = render_report(result, integrity, digest, args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    elapsed = time.perf_counter() - started
    memory = peak_memory_mb()
    print(report)
    print(
        f"Wrote {args.out} ({len(result.rows)} rows, sha256 {digest}) and {args.report}. "
        f"Downloaded: {source.downloads or 'nothing'}. Runtime {elapsed:.0f} s"
        + (f", peak memory {memory:.0f} MB." if memory else "."),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
