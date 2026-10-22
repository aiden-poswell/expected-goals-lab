"""S04: validation-only comparison of two nonlinear candidates with the S03 reference model.

    uv run python -m expected_goals_lab.evaluate_nonlinear

Compares, on the validation split only (docs/NONLINEAR_PROTOCOL.md):

* reference            S03 ``distance_plus_angle`` (unchanged code)
* nonlinear_location   spline bases of distance and angle
* nonlinear_location_body_part   the same plus the Wyscout body-part class

The test window is not modelled or scored and no row-level test value is read. (Its aggregate goal
count is inferable from earlier cohort audit totals; this module neither states nor computes it.)
No candidate is selected for final testing.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn

from expected_goals_lab.baselines import DISTANCE_PLUS_ANGLE, LOGISTIC_PARAMS, fit_baselines
from expected_goals_lab.candidate_data import (
    BODY_PART_VOCABULARY,
    CandidatePartition,
    load_body_parts,
)
from expected_goals_lab.common import table
from expected_goals_lab.dataset import (
    DEFAULT_CSV,
    OPEN_SPLITS,
    ContractError,
    Identity,
    RowReader,
    iter_csv_rows,
    load_partitions,
    read_identity,
    sha256_bytes_of,
)
from expected_goals_lab.nonlinear import (
    NONLINEAR_LOCATION,
    NONLINEAR_LOCATION_BODY_PART,
    SPLINE_PARAMS,
    check_probabilities,
    fit_candidates,
)
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
    SPLIT_MANIFEST_NAME,
    SplitPlan,
    assign_date_splits,
    split_of_events,
    summarise,
    verify_split,
    write_assignment,
)

DEFAULT_REPORT = Path("reports/nonlinear_validation.md")
DEFAULT_PROTOCOL = Path("docs/NONLINEAR_PROTOCOL.md")
DEFAULT_S03_REPORT = Path("reports/baseline_validation.md")
SUMMARY_NAME = "nonlinear_validation_summary.json"
REFERENCE = DISTANCE_PLUS_ANGLE
A, B = NONLINEAR_LOCATION, NONLINEAR_LOCATION_BODY_PART
MODELS = (REFERENCE, A, B)
PAIRS = [(A, REFERENCE), (B, REFERENCE), (B, A)]
METRIC_TERMS = {"log_loss": log_loss_terms, "brier": brier_terms}
FRAGILE_FRACTION = 0.25
# Published S02/S03 identity of the cohort, enforced whenever the input table has this SHA-256.
PUBLISHED_CSV_SHA = "58eb62c0f786406c7df1e457a981784ab10d0514ee5a3ec73c2119c83f0af96e"
PUBLISHED_ASSIGNMENT_SHA = "2f995fc0337436619f794d8c64c7f8397bf81d8e28deca9f2add8369c881da15"
PUBLISHED_SPLIT_ROWS = {"train": 28263, "validation": 5509, "test": 6653}


@dataclass
class NonlinearEvaluation:
    identity: Identity
    plan: SplitPlan
    split_summary: dict
    assignment_sha: str
    parts: dict[str, CandidatePartition]
    scores: dict[str, dict[str, Scores]]
    intervals: dict[str, dict[tuple, Interval]]
    sensitivity: dict
    calibration: dict[str, list[dict]]
    body_part_table: list[list]
    body_part_terms: list[list]
    checks: list[tuple[str, str]]
    resamples: int
    provenance: dict
    # Open-split predictions and the fitted models, exposed for S05's validation diagnostics.
    predictions: dict[str, dict[str, np.ndarray]]
    reference_model: object
    candidates: dict


# --- identity and reproduction checks ---------------------------------------------------------


def check_split_identity(
    identity, summary, assignment_sha, split_manifest_path
) -> list[tuple[str, str]]:
    checks = []
    if split_manifest_path.exists():
        try:
            recorded = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(
                f"S03 split manifest {split_manifest_path} is unreadable: {exc}"
            ) from exc
        if recorded.get("assignment_csv_sha256") != assignment_sha:
            raise ContractError("the split assignment differs from the one S03 recorded")
        if recorded.get("input_csv_sha256") != identity.csv_sha256:
            raise ContractError("the S03 split manifest was made from a different shot table")
        for split in ("train", "validation", "test"):
            if recorded["splits"][split]["rows"] != summary[split]["rows"]:
                raise ContractError(f"{split} row count differs from S03's recorded split")
        checks.append(
            ("Split assignment equals the one S03 recorded (SHA-256 and row counts)", "pass")
        )
    else:
        checks.append(
            (
                "S03 split manifest not found; split identity checked by recomputation only",
                "not checked",
            )
        )
    if identity.csv_sha256 == PUBLISHED_CSV_SHA:
        if assignment_sha != PUBLISHED_ASSIGNMENT_SHA:
            raise ContractError("the split assignment SHA-256 differs from the published S03 value")
        got = {s: summary[s]["rows"] for s in PUBLISHED_SPLIT_ROWS}
        if got != PUBLISHED_SPLIT_ROWS:
            raise ContractError(
                f"split sizes {got} differ from the published {PUBLISHED_SPLIT_ROWS}"
            )
        checks.append(("Assignment SHA-256 and split sizes equal the published S03 values", "pass"))
    else:
        checks.append(
            (
                "Input table is not the published S02 table; published S03 values not enforced",
                "not checked",
            )
        )
    return checks


def parse_s03_reference(report_path: Path) -> tuple[int, int, str, str, str] | None:
    """(shots, goals, log loss, Brier, AUC) as printed for ``distance_plus_angle`` in S03's "
    "report."""
    if not report_path.exists():
        return None
    section = report_path.read_text(encoding="utf-8").split("## 3. Validation scores", 1)
    if len(section) < 2:
        return None
    pattern = rf"^\| {REFERENCE} \| (\d+) \| (\d+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|"
    match = re.search(pattern, section[1], flags=re.MULTILINE)
    if not match:
        return None
    n, goals, ll, brier, auc = match.groups()
    return int(n), int(goals), ll, brier, auc


def check_reference_reproduces_s03(reference: Scores, report_path: Path) -> tuple[str, str]:
    printed = parse_s03_reference(report_path)
    if printed is None:
        return (
            "S03 report not found or unparsable; reference reproduction not checked",
            "not checked",
        )
    n, goals, ll, brier, auc = printed
    got = (
        reference.n,
        reference.goals,
        f"{reference.log_loss:.4f}",
        f"{reference.brier:.5f}",
        f"{reference.auc:.3f}",
    )
    if got != (n, goals, ll, brier, auc):
        raise ContractError(
            f"regenerated S03 reference {got} differs from the S03 report {printed}"
        )
    return (
        "Regenerated S03 reference equals the committed S03 report (to its printed precision)",
        "pass",
    )


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=20)

    try:
        head = run("rev-parse", "HEAD")
        dirty = run("status", "--porcelain")
        if head.returncode != 0:
            return {"commit": "unavailable", "uncommitted_changes": None}
        return {"commit": head.stdout.strip(), "uncommitted_changes": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unavailable", "uncommitted_changes": None}


# --- evaluation --------------------------------------------------------------------------------


def _terms(y, preds: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    return {metric: {m: fn(y, preds[m]) for m in MODELS} for metric, fn in METRIC_TERMS.items()}


def _intervals(match_ids, y, preds, resamples) -> dict[str, dict[tuple, Interval]]:
    return {
        metric: cluster_bootstrap_differences(match_ids, terms, PAIRS, resamples)
        for metric, terms in _terms(y, preds).items()
    }


def evaluate_nonlinear(
    csv_path: Path,
    manifest_path: Path,
    out_dir: Path,
    resamples: int = BOOTSTRAP_RESAMPLES,
    s03_report: Path = DEFAULT_S03_REPORT,
    protocol: Path = DEFAULT_PROTOCOL,
    rows: RowReader = iter_csv_rows,
) -> NonlinearEvaluation:
    identity = read_identity(csv_path, manifest_path, rows)
    plan = assign_date_splits(identity.dates.tolist())
    verify_split(identity, plan)
    summary = summarise(identity, plan)
    with tempfile.TemporaryDirectory() as scratch:  # do not overwrite S03's own outputs
        assignment_sha = write_assignment(Path(scratch) / "assignment.csv", identity, plan)
    checks = check_split_identity(identity, summary, assignment_sha, out_dir / SPLIT_MANIFEST_NAME)

    split_of = split_of_events(identity, plan)
    base = load_partitions(csv_path, split_of, OPEN_SPLITS, rows)
    parts = load_body_parts(csv_path, base, split_of, OPEN_SPLITS, rows)
    train, validation = parts["train"], parts["validation"]
    for name, part in parts.items():
        if len(set(part.base.y.tolist())) < 2:
            raise ValueError(f"the {name} cohort has a single outcome class")

    reference_model = fit_baselines(train.base)[REFERENCE]
    candidates = fit_candidates(train)
    preds = {}
    for split, part in parts.items():
        preds[split] = {
            REFERENCE: check_probabilities(reference_model.predict_proba(part.base), REFERENCE),
            A: candidates[A].predict_proba(part),
            B: candidates[B].predict_proba(part),
        }
    scores = {
        split: {m: score(parts[split].base.y, preds[split][m]) for m in MODELS}
        for split in OPEN_SPLITS
    }
    checks.append(check_reference_reproduces_s03(scores["validation"][REFERENCE], s03_report))

    y, match_ids = validation.base.y, validation.base.match_ids
    intervals = _intervals(match_ids, y, preds["validation"], resamples)

    keep = validation.base.x_pct != 100  # the goal-line records: dropped from scoring only
    kept_preds = {m: preds["validation"][m][keep] for m in MODELS}
    sensitivity = {
        "removed": int((~keep).sum()),
        "removed_goals": int(y[~keep].sum()),
        "scores": {m: score(y[keep], kept_preds[m]) for m in MODELS},
        "intervals": _intervals(match_ids[keep], y[keep], kept_preds, resamples),
    }

    calibration = {m: calibration_bins(y, preds["validation"][m]) for m in (A, B)}
    pipeline_b = candidates[B].pipeline
    categories = pipeline_b.named_steps["features"].named_transformers_["body_part"].categories_[0]
    weights = pipeline_b.named_steps["logit"].coef_[0][-len(categories) :]  # one-hot block is last
    body_terms = [[str(c), f"{w:+.4f}"] for c, w in zip(categories, weights, strict=True)]
    body_rows = []
    for cls in BODY_PART_VOCABULARY:
        cells = [cls]
        for split in OPEN_SPLITS:
            mask = parts[split].body_part == cls
            n, goals = int(mask.sum()), int(parts[split].base.y[mask].sum())
            cells += [n, f"{100 * goals / n:.2f}%" if n else "n/a"]
        body_rows.append(cells)

    provenance = {
        **git_state(),
        "input_csv_sha256": identity.csv_sha256,
        "assignment_csv_sha256": assignment_sha,
        "protocol_sha256": sha256_bytes_of(protocol) if protocol.exists() else "unavailable",
        "scikit_learn": sklearn.__version__,
    }
    return NonlinearEvaluation(
        identity=identity,
        plan=plan,
        split_summary=summary,
        assignment_sha=assignment_sha,
        parts=parts,
        scores=scores,
        intervals=intervals,
        sensitivity=sensitivity,
        calibration=calibration,
        body_part_table=body_rows,
        body_part_terms=body_terms,
        checks=checks,
        resamples=resamples,
        provenance=provenance,
        predictions=preds,
        reference_model=reference_model,
        candidates=candidates,
    )


# --- report ------------------------------------------------------------------------------------


def _iv(iv: Interval, digits: int) -> str:
    return f"{iv.estimate:+.{digits}f} [{iv.low:+.{digits}f}, {iv.high:+.{digits}f}]"


def is_fragile(iv: Interval) -> bool:
    """Excludes zero, but its nearest edge is within a quarter of the estimate's magnitude."""
    if not (iv.high < 0 or iv.low > 0):
        return False
    nearest = min(abs(iv.low), abs(iv.high))
    return nearest < FRAGILE_FRACTION * abs(iv.estimate)


def reading(ll: Interval, br: Interval, comparator: str) -> str:
    """The pre-stated reading rule of docs/NONLINEAR_PROTOCOL.md, section 6."""
    if ll.estimate < 0 and ll.high < 0 and br.estimate < 0:
        text = (
            f"improvement over {comparator}: log-loss interval below zero and "
            "Brier estimate below zero"
        )
        return text + (
            "; the interval is fragile (nearest edge close to zero)" if is_fragile(ll) else ""
        )
    if ll.estimate > 0 and ll.low > 0 and br.estimate > 0:
        return (
            f"worse than {comparator}: log-loss interval above zero and Brier estimate above zero"
        )
    reasons = []
    if (ll.estimate < 0) != (br.estimate < 0):
        reasons.append("log loss and Brier point estimates disagree in direction")
    if ll.low <= 0 <= ll.high:
        reasons.append("the log-loss interval includes zero")
    return f"no clear improvement over {comparator}: " + ("; ".join(reasons) or "criteria not met")


def verdict(ll: Interval, br: Interval) -> str:
    """The category of ``reading``: improvement, worse or no clear improvement."""
    return reading(ll, br, "x").split(" over x")[0].split(" than x")[0].split(":")[0]


def sensitivity_flags(ev) -> list[str]:
    """Sentences for every comparison whose reading differs between main and sensitivity."""
    flags = []
    for a, b in PAIRS:
        main = verdict(ev.intervals["log_loss"][(a, b)], ev.intervals["brier"][(a, b)])
        reduced = verdict(
            ev.sensitivity["intervals"]["log_loss"][(a, b)],
            ev.sensitivity["intervals"]["brier"][(a, b)],
        )
        if main != reduced:
            flags.append(
                f'- **{a} vs {b}**: the reading is "{main}" on all validation shots but '
                f'"{reduced}" once the {ev.sensitivity["removed"]} goal-line shots are left out '
                "of scoring. The conclusion for this comparison depends on those few records."
            )
    return flags


_SCORE_HEADERS = ["Model", "Shots", "Goals", "Log loss", "Brier", "ROC AUC"]


def _score_rows(scores: dict[str, Scores]) -> list[list]:
    return [
        [m, s.n, s.goals, f"{s.log_loss:.4f}", f"{s.brier:.5f}", f"{s.auc:.3f}"]
        for m, s in scores.items()
    ]


def _diff_table(intervals: dict) -> str:
    rows = []
    for a, b in PAIRS:
        ll, br = intervals["log_loss"][(a, b)], intervals["brier"][(a, b)]
        rows.append([f"{a} - {b}", _iv(ll, 4), _iv(br, 5), reading(ll, br, b)])
    return table(
        ["Difference (first minus second)", "Log loss [95%]", "Brier [95%]", "Reading"], rows
    )


def render_report(ev: NonlinearEvaluation, csv_path: Path) -> str:
    val, train = ev.parts["validation"], ev.parts["train"]
    vs = ev.scores["validation"]
    ll, br = ev.intervals["log_loss"], ev.intervals["brier"]
    n_matches = len(set(val.base.match_ids.tolist()))
    n_dates = len(set(val.base.dates.tolist()))
    p = ev.provenance
    out: list[str] = []
    add = out.extend
    add(
        [
            "# Nonlinear location and body-part validation (S04)",
            "",
            f"Generated by `uv run python -m expected_goals_lab.evaluate_nonlinear` on "
            f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC. **Validation only; no model is selected for "
            "final testing.** The test window has not been modelled or scored, and its row-level "
            "labels/features have not been accessed by S04 (nor by S03); however, its aggregate "
            "goal count is indirectly inferable from earlier cohort audit totals. No test-row "
            "statistic appears here. Protocol: `docs/NONLINEAR_PROTOCOL.md`.",
            "",
            "## Provenance",
            "",
            f"- Input table `{csv_path.as_posix()}`: SHA-256 `{p['input_csv_sha256']}`; "
            f"{ev.identity.n_rows} rows, {ev.identity.n_eligible} eligible.",
            f"- Split assignment SHA-256 `{p['assignment_csv_sha256']}`. Protocol file SHA-256 "
            f"`{p['protocol_sha256']}`.",
            f"- Git commit `{p['commit']}`"
            + (" (working tree has uncommitted changes)" if p["uncommitted_changes"] else "")
            + f"; scikit-learn {p['scikit_learn']}.",
            "- Data: Pappalardo & Massucco (2019), Wyscout open event data, CC BY 4.0, "
            "doi:10.6084/m9.figshare.c.4415000.v5; processed for this project.",
            "",
            "Checks run before any score was produced:",
            "",
            table(["Check", "Result"], [list(c) for c in ev.checks]),
            "",
            "## 1. What the validation data support",
            "",
            f"Validation: {vs[REFERENCE].n} shots, {vs[REFERENCE].goals} goals, {n_matches} "
            "matches "
            f"on {n_dates} dates ({ev.plan.boundaries['validation']['first_date']} to "
            f"{ev.plan.boundaries['validation']['last_date']}). Fitted on {len(train)} earlier "
            "shots. One season and one contiguous window.",
            "",
        ]
    )
    for a, b in PAIRS:
        add(
            [
                f"- **{a} vs {b}**: {reading(ll[(a, b)], br[(a, b)], b)}. Log loss "
                f"{_iv(ll[(a, b)], 4)}, Brier {_iv(br[(a, b)], 5)}."
            ]
        )
    add(
        [
            "",
            "Reading rule (fixed in the protocol before fitting): improvement requires a negative "
            "log-loss estimate whose 95% interval excludes zero and a negative Brier estimate; "
            "otherwise there is no clear improvement. Intervals are exploratory and unadjusted "
            "(six are reported), from a match-cluster bootstrap "
            f"({ev.resamples} resamples, seed {BOOTSTRAP_SEED}); an interval that excludes zero "
            f"but whose nearest edge is within {int(100 * FRAGILE_FRACTION)}% of the estimate's "
            "magnitude is called fragile. Nothing is called statistically significant, and none "
            "of this establishes behaviour in other seasons.",
            "",
            "## 2. Samples",
            "",
            'Body-part class (Wyscout tag-derived; `head_or_body` comes from tag 403, "head/body", '
            "and is not a confirmed header). Train and validation only:",
            "",
            table(
                [
                    "Class",
                    "Train shots",
                    "Train goal rate",
                    "Validation shots",
                    "Validation goal rate",
                ],
                ev.body_part_table,
            ),
            "",
            "## 3. Candidates (fixed; no search)",
            "",
            "- **Reference**: S03 `distance_plus_angle`, StandardScaler + "
            f"LogisticRegression({LOGISTIC_PARAMS}).",
            f"- **{A}**: for `distance_m` and `angle_rad` separately, "
            f"SplineTransformer({SPLINE_PARAMS}) "
            f"then StandardScaler; concatenated; LogisticRegression({LOGISTIC_PARAMS}).",
            f'- **{B}**: the same spline blocks plus OneHotEncoder(handle_unknown="ignore") of '
            "`body_part_class` (categories from train, not scaled); same LogisticRegression.",
            "",
            "## 4. Scores",
            "",
            "Validation (lower log loss and Brier are better; AUC is secondary):",
            "",
            table(_SCORE_HEADERS, _score_rows(vs)),
            "",
            "Train, in-sample, an overfitting diagnostic only:",
            "",
            table(_SCORE_HEADERS, _score_rows(ev.scores["train"])),
            "",
            "### 4.1 Paired differences (validation, match-cluster bootstrap)",
            "",
            _diff_table(ev.intervals),
            "",
            "### 4.2 Fitted body-part terms (train)",
            "",
            "Coefficients of the one-hot terms in `" + B + "` on the log-odds scale, after the "
            "spline terms for location. They are fixed-penalty (L2) estimates from the training "
            "shots, shown only to give the direction of what the model learned. A larger term "
            "raises predicted odds relative to the other classes at the same location. They say "
            "nothing about heading as a skill, because `head_or_body` is a coarse tag.",
            "",
            table(["Class", "Coefficient"], ev.body_part_terms),
            "",
            "## 5. Sensitivity: goal-line records excluded from scoring only",
            "",
            f"The {ev.sensitivity['removed']} validation shots at `x_pct = 100` "
            f"({ev.sensitivity['removed_goals']} goals) are dropped from the scoring set. Models "
            "are "
            "not refitted and stay trained on the full fixed train cohort; the main scores above "
            "are "
            "unchanged. This is not a retrained alternative cohort, and no goal-line record of the "
            "test window was read.",
            "",
            table(_SCORE_HEADERS, _score_rows(ev.sensitivity["scores"])),
            "",
            _diff_table(ev.sensitivity["intervals"]),
            "",
            *(
                sensitivity_flags(ev)
                or ["No reading changes when the goal-line shots are left out."]
            ),
            "",
            "## 6. Calibration bins (validation, no calibration fitted)",
            "",
            "Ten equal-frequency bins of the predicted probability. A description only: bin counts "
            "are small and observed rates are noisy.",
            "",
        ]
    )
    for model in (A, B):
        add(
            [
                f"**{model}**",
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
                        for i, b in enumerate(ev.calibration[model])
                    ],
                ),
                "",
            ]
        )
    add(
        [
            "## 7. Limits",
            "",
            "- One season (2017/18) and one contiguous validation window; no claim about other "
            "seasons, leagues or club-grade accuracy.",
            "- `body_part_class` is a coarse provider class. `head_or_body` mixes headers with "
            "other "
            "body parts, so any difference associated with it is a statement about that tag, not "
            "about headers.",
            "- Splines add flexibility (six columns per feature) with a fixed penalty; no tuning "
            "was "
            "done and none is implied to be optimal.",
            "- Distances and angles assume a 105 x 68 m pitch.",
            "- Departures from `docs/NONLINEAR_PROTOCOL.md`: none to the data, split, candidates, "
            "metrics, comparisons, reading rule, bootstrap or sensitivity. Two reporting additions "
            "were made after the first run: the sentence flagging readings that change in the "
            "goal-line sensitivity, and the table of fitted body-part terms (protocol, section "
            "12).",
        ]
    )
    return "\n".join(out) + "\n"


def summary_json(ev: NonlinearEvaluation) -> dict:
    """Aggregate numbers only, for the git-ignored run record."""
    return {
        "provenance": ev.provenance,
        "checks": [list(c) for c in ev.checks],
        "validation_scores": {m: vars(s) for m, s in ev.scores["validation"].items()},
        "train_scores": {m: vars(s) for m, s in ev.scores["train"].items()},
        "intervals": {
            metric: {f"{a} - {b}": vars(iv) for (a, b), iv in pairs.items()}
            for metric, pairs in ev.intervals.items()
        },
        "resamples": ev.resamples,
        "seed": BOOTSTRAP_SEED,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validation-only S04 comparison.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="S03 outputs and run record folder"
    )
    parser.add_argument("--s03-report", type=Path, default=DEFAULT_S03_REPORT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    manifest_path = args.manifest or args.csv.with_suffix(".manifest.json")
    out_dir = args.out_dir or args.csv.parent
    started = time.perf_counter()
    try:
        evaluation = evaluate_nonlinear(
            args.csv, manifest_path, out_dir, args.resamples, args.s03_report, args.protocol
        )
    except FileNotFoundError as exc:
        print(
            f"error: {exc} not found. Build the shot table first: "
            "`uv run python -m expected_goals_lab.build_shots`.",
            file=sys.stderr,
        )
        return 2
    except (ContractError, ValueError, RuntimeError, Warning) as exc:
        # Contract breaks, sealed-split requests (RuntimeError), invalid inputs and promoted
        # convergence warnings all stop the run without writing a report.
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    report = render_report(evaluation, args.csv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SUMMARY_NAME).write_text(
        json.dumps(summary_json(evaluation), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report)
    print(f"Wrote {args.report}. Runtime {time.perf_counter() - started:.0f} s.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
