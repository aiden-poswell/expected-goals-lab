"""S03: chronological split and validation-only comparison of three xG baselines.

    uv run python -m expected_goals_lab.evaluate_baselines

Input: the S02 shot table and manifest (data/processed/). Output: the split assignment and its
manifest (data/processed/, git-ignored) and reports/baseline_validation.md.

The test window is not modelled or scored: this module never asks for its row-level labels or
features, and the report holds no statistic computed from them. (Its aggregate goal count is
nevertheless inferable from earlier cohort audit totals; see docs/BASELINE_PROTOCOL.md.)
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from expected_goals_lab.baselines import (
    CONSTANT,
    DISTANCE_ONLY,
    DISTANCE_PLUS_ANGLE,
    LOGISTIC_PARAMS,
    FittedModel,
    fit_baselines,
    with_pitch,
)
from expected_goals_lab.common import table
from expected_goals_lab.dataset import (
    DEFAULT_CSV,
    OPEN_SPLITS,
    ContractError,
    Identity,
    Partition,
    RowReader,
    iter_csv_rows,
    load_partitions,
    read_identity,
)
from expected_goals_lab.geometry import ALTERNATIVE_PITCHES
from expected_goals_lab.scoring import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    Interval,
    Scores,
    brier_terms,
    calibration_bins,
    cluster_bootstrap_differences,
    log_loss_terms,
    score,
)
from expected_goals_lab.splits import (
    ASSIGNMENT_NAME,
    SPLIT_MANIFEST_NAME,
    SplitPlan,
    assign_date_splits,
    split_manifest,
    split_of_events,
    summarise,
    verify_split,
    write_assignment,
)

DEFAULT_REPORT = Path("reports/baseline_validation.md")
LEAGUE_NAMES = {
    364: "English first division",
    412: "French first division",
    426: "German first division",
    524: "Italian first division",
    795: "Spanish first division",
}
MIN_LEAGUE_SHOTS, MIN_LEAGUE_GOALS = 1000, 100
MODELS = (CONSTANT, DISTANCE_ONLY, DISTANCE_PLUS_ANGLE)
PAIRS = [
    (DISTANCE_PLUS_ANGLE, DISTANCE_ONLY),
    (DISTANCE_PLUS_ANGLE, CONSTANT),
    (DISTANCE_ONLY, CONSTANT),
]
METRIC_TERMS = {"log_loss": log_loss_terms, "brier": brier_terms}


@dataclass
class Evaluation:
    identity: Identity
    plan: SplitPlan
    split_summary: dict
    assignment_sha: str
    partitions: dict[str, Partition]
    models: dict[str, FittedModel]
    scores: dict[str, dict[str, Scores]]
    intervals: dict[str, dict[tuple, Interval]]
    by_league: dict[int, dict]
    calibration: list[dict]
    sensitivity: dict[str, dict[str, Scores]]
    goal_line: dict[str, dict]
    drift: dict[str, dict]
    resamples: int


def _predictions(models: dict[str, FittedModel], part: Partition) -> dict[str, np.ndarray]:
    return {name: models[name].predict_proba(part) for name in MODELS}


def _intervals(part: Partition, preds: dict[str, np.ndarray], resamples: int) -> dict:
    return {
        metric: cluster_bootstrap_differences(
            part.match_ids,
            {name: fn(part.y, preds[name]) for name in MODELS},
            PAIRS,
            resamples,
        )
        for metric, fn in METRIC_TERMS.items()
    }


def _subset(part: Partition, mask: np.ndarray) -> Partition:
    return replace(
        part,
        event_ids=part.event_ids[mask],
        match_ids=part.match_ids[mask],
        competition_ids=part.competition_ids[mask],
        dates=part.dates[mask],
        features={k: v[mask] for k, v in part.features.items()},
        x_pct=part.x_pct[mask],
        y_pct=part.y_pct[mask],
        y=part.y[mask],
    )


def evaluate(
    csv_path: Path,
    manifest_path: Path,
    out_dir: Path,
    resamples: int = BOOTSTRAP_RESAMPLES,
    rows: RowReader = iter_csv_rows,
) -> Evaluation:
    identity = read_identity(csv_path, manifest_path, rows)
    plan = assign_date_splits(identity.dates.tolist())
    verify_split(identity, plan)
    assignment_sha = write_assignment(out_dir / ASSIGNMENT_NAME, identity, plan)
    (out_dir / SPLIT_MANIFEST_NAME).write_text(
        json.dumps(split_manifest(identity, plan, assignment_sha), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = summarise(identity, plan)

    parts = load_partitions(csv_path, split_of_events(identity, plan), OPEN_SPLITS, rows)
    train, validation = parts["train"], parts["validation"]
    models = fit_baselines(train)
    preds = {split: _predictions(models, parts[split]) for split in OPEN_SPLITS}
    scores = {
        split: {name: score(parts[split].y, preds[split][name]) for name in MODELS}
        for split in OPEN_SPLITS
    }
    intervals = _intervals(validation, preds["validation"], resamples)

    by_league = {}
    for cid in sorted(set(validation.competition_ids.tolist())):
        mask = validation.competition_ids == cid
        n, goals = int(mask.sum()), int(validation.y[mask].sum())
        entry = {
            "n": n,
            "goals": goals,
            "shown": n >= MIN_LEAGUE_SHOTS and goals >= MIN_LEAGUE_GOALS,
        }
        if entry["shown"]:
            sub = _subset(validation, mask)
            sub_preds = {name: preds["validation"][name][mask] for name in MODELS}
            entry["scores"] = {name: score(sub.y, sub_preds[name]) for name in MODELS}
            entry["intervals"] = _intervals(sub, sub_preds, resamples)
        by_league[int(cid)] = entry

    sensitivity = {}
    for pitch_name, pitch in ALTERNATIVE_PITCHES.items():
        alt_train, alt_val = with_pitch(train, pitch), with_pitch(validation, pitch)
        alt_models = fit_baselines(alt_train)
        sensitivity[pitch_name] = {
            name: score(alt_val.y, alt_models[name].predict_proba(alt_val)) for name in MODELS
        }

    goal_line = {}
    for split in OPEN_SPLITS:
        part = parts[split]
        mask = part.x_pct == 100
        goal_line[split] = {
            "n": int(mask.sum()),
            "goals": int(part.y[mask].sum()),
            "angles_deg": sorted(
                round(float(np.degrees(a)), 1) for a in part.features["angle_rad"][mask]
            ),
            "angle_range_deg": (
                round(float(np.degrees(part.features["angle_rad"].min())), 2),
                round(float(np.degrees(part.features["angle_rad"].max())), 2),
            ),
        }

    drift = {}
    for split in OPEN_SPLITS:
        part = parts[split]
        drift[split] = {
            "rows": len(part),
            "matches": int(len(set(part.match_ids.tolist()))),
            "dates": int(len(set(part.dates.tolist()))),
            "goals": int(part.y.sum()),
            "rate": float(part.y.mean()),
            "distance_mean": float(part.features["distance_m"].mean()),
            "distance_median": float(np.median(part.features["distance_m"])),
            "angle_mean": float(np.degrees(part.features["angle_rad"]).mean()),
            "angle_median": float(np.median(np.degrees(part.features["angle_rad"]))),
        }

    return Evaluation(
        identity=identity,
        plan=plan,
        split_summary=summary,
        assignment_sha=assignment_sha,
        partitions=parts,
        models=models,
        scores=scores,
        intervals=intervals,
        by_league=by_league,
        calibration=calibration_bins(validation.y, preds["validation"][DISTANCE_PLUS_ANGLE]),
        sensitivity=sensitivity,
        goal_line=goal_line,
        drift=drift,
        resamples=resamples,
    )


def _fmt_interval(iv: Interval, digits: int) -> str:
    return f"{iv.estimate:+.{digits}f} [{iv.low:+.{digits}f}, {iv.high:+.{digits}f}]"


def _reading(iv: Interval, better: str, worse: str) -> str:
    """Plain-language reading of a difference a - b where lower is better."""
    if iv.high < 0:
        return f"{better}; the interval is entirely below zero"
    if iv.low > 0:
        return f"{worse}; the interval is entirely above zero"
    return "no clear difference on these validation matches; the interval includes zero"


def _angle_summary(ll: dict, br: dict) -> str:
    """One honest sentence on whether adding angle to distance is distinguishable from nothing."""
    ll_iv, br_iv = (
        ll[(DISTANCE_PLUS_ANGLE, DISTANCE_ONLY)],
        br[(DISTANCE_PLUS_ANGLE, DISTANCE_ONLY)],
    )
    both_gain = ll[(DISTANCE_PLUS_ANGLE, CONSTANT)].estimate
    share = ll[(DISTANCE_ONLY, CONSTANT)].estimate / both_gain if both_gain else float("nan")
    gain = (
        f" Distance alone gives {100 * share:.0f}% of the log-loss gain over the constant baseline."
    )
    ll_clear, br_clear = ll_iv.high < 0, br_iv.high < 0
    if ll_clear and br_clear:
        return (
            "Summary on angle: both scoring rules favour adding angle, and both intervals exclude "
            "zero on these validation matches." + gain
        )
    if ll_clear != br_clear:
        which = "Brier score" if br_clear else "log loss"
        return (
            "Summary on angle: the two scoring rules disagree about whether adding angle is "
            f"distinguishable from no change; only the {which} interval excludes zero, so treat "
            "the evidence for angle as weak." + gain
        )
    return (
        "Summary on angle: neither interval excludes zero, so these validation matches do not "
        "separate the two-feature model from distance only." + gain
    )


def _score_rows(scores: dict[str, Scores]) -> list[list]:
    return [
        [name, s.n, s.goals, f"{s.log_loss:.4f}", f"{s.brier:.5f}", f"{s.auc:.3f}"]
        for name, s in scores.items()
    ]


SCORE_HEADERS = ["Model", "Shots", "Goals", "Log loss", "Brier", "ROC AUC"]


def render_report(ev: Evaluation, csv_path: Path) -> str:
    train_p = ev.partitions["train"]
    out: list[str] = []
    add = out.extend
    val = ev.scores["validation"]
    ll, br = ev.intervals["log_loss"], ev.intervals["brier"]
    add(
        [
            "# Baseline validation (S03)",
            "",
            f"Generated by `uv run python -m expected_goals_lab.evaluate_baselines` on "
            f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC. **Validation only.** The test window has "
            "not been modelled or scored, and its row-level labels/features have not been "
            "accessed by S03; however, its aggregate goal count is indirectly inferable from "
            "earlier cohort audit totals. No test-row statistic, prediction or metric appears "
            "here. Protocol: `docs/BASELINE_PROTOCOL.md`.",
            "",
            f"Input: `{csv_path.as_posix()}`, SHA-256 `{ev.identity.csv_sha256}`, "
            f"{ev.identity.n_rows} rows of which {ev.identity.n_eligible} eligible (quarantined "
            "rows never enter). Data: Pappalardo & Massucco (2019), Wyscout open event data, "
            "CC BY 4.0, doi:10.6084/m9.figshare.c.4415000.v5; processed for this project.",
            "",
            "## 1. Findings supported by the validation data",
            "",
        ]
    )
    k = DISTANCE_PLUS_ANGLE, DISTANCE_ONLY
    add(
        [
            f"On the {val[CONSTANT].n} validation shots ({val[CONSTANT].goals} goals, "
            f"{ev.split_summary['validation']['matches']} matches, "
            f"{ev.plan.boundaries['validation']['first_date']} to "
            f"{ev.plan.boundaries['validation']['last_date']}), after fitting on earlier matches "
            "only:",
            "",
            f"- Constant training rate: log loss {val[CONSTANT].log_loss:.4f}, Brier "
            f"{val[CONSTANT].brier:.5f}.",
            f"- Distance only: log loss {val[DISTANCE_ONLY].log_loss:.4f}, Brier "
            f"{val[DISTANCE_ONLY].brier:.5f}, AUC {val[DISTANCE_ONLY].auc:.3f}.",
            f"- Distance plus angle: log loss {val[DISTANCE_PLUS_ANGLE].log_loss:.4f}, Brier "
            f"{val[DISTANCE_PLUS_ANGLE].brier:.5f}, AUC {val[DISTANCE_PLUS_ANGLE].auc:.3f}.",
            "- Distance only versus constant, log loss: "
            f"{_fmt_interval(ll[(DISTANCE_ONLY, CONSTANT)], 4)}: "
            + _reading(
                ll[(DISTANCE_ONLY, CONSTANT)], "distance-only was better", "distance-only was worse"
            )
            + ".",
            "- Distance plus angle versus constant, log loss: "
            f"{_fmt_interval(ll[(DISTANCE_PLUS_ANGLE, CONSTANT)], 4)}: "
            + _reading(
                ll[(DISTANCE_PLUS_ANGLE, CONSTANT)],
                "distance-plus-angle was better",
                "distance-plus-angle was worse",
            )
            + ".",
            f"- Distance plus angle versus distance only, log loss: {_fmt_interval(ll[k], 4)}: "
            + _reading(ll[k], "adding angle was better", "adding angle was worse")
            + f"; Brier {_fmt_interval(br[k], 5)}: "
            + _reading(br[k], "adding angle was better", "adding angle was worse")
            + ".",
            "",
            _angle_summary(ll, br),
            "",
            "Intervals are 95% percentile intervals of a match-cluster bootstrap "
            f"({ev.resamples} resamples, seed {BOOTSTRAP_SEED}). They describe sensitivity to "
            "which "
            "validation matches were drawn. They are not guarantees for future matches or seasons, "
            "and there is one season of data. Nothing here concerns the test split.",
            "",
            "## 2. Splits",
            "",
            "Chronological by UTC match date; all matches on a date share a split; 70% / 85% "
            f"cumulative cut points over the {len(ev.plan.date_split)} distinct dates, rounded "
            "half "
            "up. Counts only for the sealed test split.",
            "",
        ]
    )
    summary = ev.split_summary
    rows = []
    for split in ("train", "validation", "test"):
        s = summary[split]
        rows.append([split, s["first_date"], s["last_date"], s["dates"], s["matches"], s["rows"]])
    add(
        [
            table(["Split", "First date", "Last date", "Dates", "Matches", "Eligible shots"], rows),
            "",
        ]
    )
    league_rows = []
    for cid, name in LEAGUE_NAMES.items():
        league_rows.append(
            [name]
            + [
                f"{summary[s]['rows_by_competition'].get(cid, 0)} shots / "
                f"{summary[s]['matches_by_competition'].get(cid, 0)} matches"
                for s in ("train", "validation", "test")
            ]
        )
    add([table(["League", "Train", "Validation", "Test"], league_rows), ""])
    ok = all(
        [
            summary["train"]["last_date"] < summary["validation"]["first_date"],
            summary["validation"]["last_date"] < summary["test"]["first_date"],
        ]
    )
    add(
        [
            "Checks passed: every eligible event and match is in exactly one split; no date is in "
            f"two splits; train < validation < test in time ({'yes' if ok else 'NO'}). Split "
            f"assignment file SHA-256 `{ev.assignment_sha}`.",
            "",
            "Train and validation, with goals (no test-window goal count is stated here; see the "
            "caveat at the top):",
            "",
        ]
    )
    drift_rows = []
    for split in OPEN_SPLITS:
        d = ev.drift[split]
        drift_rows.append(
            [
                split,
                d["rows"],
                d["matches"],
                d["goals"],
                f"{100 * d['rate']:.2f}%",
                f"{d['distance_mean']:.2f} / {d['distance_median']:.2f}",
                f"{d['angle_mean']:.2f} / {d['angle_median']:.2f}",
                f"{d['rows'] / d['matches']:.2f}",
            ]
        )
    add(
        [
            table(
                [
                    "Split",
                    "Shots",
                    "Matches",
                    "Goals",
                    "Goal rate",
                    "Distance m, mean / median",
                    "Angle deg, mean / median",
                    "Shots per match",
                ],
                drift_rows,
            ),
            "",
            "## 3. Validation scores",
            "",
            "Lower log loss and Brier are better. AUC is secondary. Models were fitted on the "
            "train "
            "split only.",
            "",
            table(SCORE_HEADERS, _score_rows(val)),
            "",
            "Train, in-sample (a check for overfitting, not a performance claim):",
            "",
            table(SCORE_HEADERS, _score_rows(ev.scores["train"])),
            "",
            "### 3.1 Differences with match-cluster bootstrap intervals (validation)",
            "",
        ]
    )
    diff_rows = []
    for a, b in PAIRS:
        diff_rows.append([f"{a} - {b}", _fmt_interval(ll[(a, b)], 4), _fmt_interval(br[(a, b)], 5)])
    add(
        [table(["Difference (first minus second)", "Log loss [95%]", "Brier [95%]"], diff_rows), ""]
    )

    add(["### 3.2 By league (validation)", ""])
    league_rows = []
    for cid, entry in ev.by_league.items():
        name = LEAGUE_NAMES.get(cid, str(cid))
        if not entry["shown"]:
            league_rows.append(
                [
                    name,
                    entry["n"],
                    entry["goals"],
                    f"not shown: below {MIN_LEAGUE_SHOTS} shots or {MIN_LEAGUE_GOALS} goals",
                    "",
                    "",
                    "",
                ]
            )
            continue
        s = entry["scores"]
        iv = entry["intervals"]["log_loss"][k]
        league_rows.append(
            [
                name,
                entry["n"],
                entry["goals"],
                f"{s[CONSTANT].log_loss:.4f} / {s[DISTANCE_ONLY].log_loss:.4f} / "
                f"{s[DISTANCE_PLUS_ANGLE].log_loss:.4f}",
                f"{s[CONSTANT].brier:.5f} / {s[DISTANCE_ONLY].brier:.5f} / "
                f"{s[DISTANCE_PLUS_ANGLE].brier:.5f}",
                f"{s[DISTANCE_PLUS_ANGLE].auc:.3f}",
                _fmt_interval(iv, 4),
            ]
        )
    add(
        [
            table(
                [
                    "League",
                    "Shots",
                    "Goals",
                    "Log loss: constant / distance / distance+angle",
                    "Brier: constant / distance / distance+angle",
                    "AUC (distance+angle)",
                    "Log loss, +angle minus distance only [95%]",
                ],
                league_rows,
            ),
            "",
            "League samples are small; the intervals are wide and differences between leagues "
            "should not be over-read.",
            "",
            "### 3.3 Calibration bins (validation, distance plus angle)",
            "",
            "Ten equal-frequency bins on the predicted probability. Bin counts are small enough "
            "that observed rates are noisy; this is a description, not a demonstration of "
            "calibration.",
            "",
            table(
                ["Bin", "Shots", "Mean prediction", "Observed goal rate", "Goals"],
                [
                    [
                        i + 1,
                        b["n"],
                        f"{b['mean_prediction']:.4f}",
                        f"{b['observed_rate']:.4f}",
                        b["goals"],
                    ]
                    for i, b in enumerate(ev.calibration)
                ],
            ),
            "",
            "## 4. Fitted models (train)",
            "",
            f"`StandardScaler` then `LogisticRegression({LOGISTIC_PARAMS})`, default L2 penalty, "
            "no "
            f"search. The constant baseline is the training goal rate "
            f"{ev.models[CONSTANT].estimator.rate:.5f}, exactly "
            f"{int(train_p.y.sum())}/{len(train_p)}.",
            "",
        ]
    )
    coef_rows = []
    for name in (DISTANCE_ONLY, DISTANCE_PLUS_ANGLE):
        c = ev.models[name].coefficients()
        for feature in c["features"]:
            coef_rows.append(
                [
                    name,
                    feature,
                    f"{c['standardised'][feature]:+.4f}",
                    f"{c['per_unit'][feature]:+.5f}",
                ]
            )
        coef_rows.append(
            [
                name,
                "intercept",
                f"{c['intercept_standardised']:+.4f}",
                f"{c['intercept_per_unit']:+.4f}",
            ]
        )
    d_only = ev.models[DISTANCE_ONLY].coefficients()["standardised"]["distance_m"]
    d_both = ev.models[DISTANCE_PLUS_ANGLE].coefficients()["standardised"]
    add(
        [
            table(
                [
                    "Model",
                    "Term",
                    "Coefficient per standard deviation",
                    "Coefficient per unit (m or rad)",
                ],
                coef_rows,
            ),
            "",
            "Direction check: farther shots should score less often (negative distance "
            "coefficient) and wider angles should score more often (positive angle coefficient). "
            f"Observed: distance {'negative' if d_only < 0 else 'POSITIVE'} in the distance-only "
            f"model; distance {'negative' if d_both['distance_m'] < 0 else 'POSITIVE'} and angle "
            f"{'positive' if d_both['angle_rad'] > 0 else 'NEGATIVE'} in the two-feature model. "
            "Distance and angle are strongly related (both come from position), so the split of "
            "credit between them in the two-feature model should not be read as separate effects.",
            "",
            "## 5. Sensitivity to the assumed pitch size (validation)",
            "",
            "Pre-specified: the three models refitted on train and scored on validation with "
            "distance and angle recomputed from the raw percentages on another pitch (goal 7.32 "
            "m). "
            "Changes are against the 105 x 68 m reference above.",
            "",
        ]
    )
    sens_rows = []
    for pitch_name, scores in ev.sensitivity.items():
        for name in MODELS:
            s, r = scores[name], val[name]
            sens_rows.append(
                [
                    pitch_name,
                    name,
                    f"{s.log_loss:.4f}",
                    f"{s.log_loss - r.log_loss:+.5f}",
                    f"{s.brier:.5f}",
                    f"{s.brier - r.brier:+.6f}",
                    f"{s.auc:.3f}",
                ]
            )
    add(
        [
            table(["Pitch", "Model", "Log loss", "Change", "Brier", "Change", "AUC"], sens_rows),
            "",
            "## 6. Goal-line shots (`x_pct = 100`)",
            "",
            "S02 flags these as a degenerate-angle edge case (angle 180 or 0 degrees). They stay "
            "in "
            "the locked cohort and enter the models as ordinary rows; the cohort was not changed "
            "on "
            "the basis of any result. Counts in the open splits (the test split was not examined):",
            "",
            table(
                [
                    "Split",
                    "Shots at x = 100",
                    "Goals",
                    "Angles (deg)",
                    "Angle range of the whole split (deg)",
                ],
                [
                    [s, g["n"], g["goals"], g["angles_deg"], g["angle_range_deg"]]
                    for s, g in ev.goal_line.items()
                ],
            ),
            "",
            "Recommendation for a later slice: a sensitivity analysis that excludes or clips these "
            "rows, decided before looking at test results.",
            "",
            "## 7. Limits and departures",
            "",
            "- One season (2017/18), five leagues. Validation is a single contiguous window of 26 "
            "match dates; it supports a statement about those matches, not about other seasons.",
            "- Teams and players appear in train and validation, as they would in a within-season "
            "forecast.",
            "- Distances and angles assume a 105 x 68 m pitch; positions are integer percentages.",
            "- Labels come from provider tagging (tag 101 on the shot event). "
            f"{ev.identity.n_rows - ev.identity.n_eligible} rows quarantined by S02 never enter.",
            "- Only distance and angle were used. No body part, context, time or identifiers.",
            "- Departures from `docs/BASELINE_PROTOCOL.md`: none to the models, split, metrics or "
            "thresholds. A summary sentence on angle was added to this report after the first "
            "validation run (wording only; see the protocol, section 9).",
        ]
    )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validation-only comparison of xG baselines.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="split files (default: CSV folder)"
    )
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    manifest_path = args.manifest or args.csv.with_suffix(".manifest.json")
    out_dir = args.out_dir or args.csv.parent
    started = time.perf_counter()
    try:
        evaluation = evaluate(args.csv, manifest_path, out_dir, args.resamples)
    except FileNotFoundError as exc:
        print(
            f"error: {exc} not found. Build the shot table first: "
            "`uv run python -m expected_goals_lab.build_shots`.",
            file=sys.stderr,
        )
        return 2
    except (ContractError, ValueError) as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 1
    report = render_report(evaluation, args.csv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {args.report}. Runtime {time.perf_counter() - started:.0f} s.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
