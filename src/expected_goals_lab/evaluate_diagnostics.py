"""S05: validation-only diagnostics, figures and report for the selected model.

    uv run python -m expected_goals_lab.evaluate_diagnostics

The model decision is fixed in docs/MODEL_SELECTION_DECISION.md and the scope in
docs/DIAGNOSTICS_PROTOCOL.md. This module reuses S04's guarded evaluation (so the only data path is
the S03/S04 loaders and the unchanged candidate factories), requires S04's published validation
numbers to be reproduced first, and then computes aggregate-only diagnostics on validation rows.
The test window is not read, scored or described. Nothing is tuned or selected here.
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from expected_goals_lab import figures
from expected_goals_lab.baselines import DISTANCE_PLUS_ANGLE
from expected_goals_lab.common import table
from expected_goals_lab.dataset import (
    DEFAULT_CSV,
    ContractError,
    RowReader,
    iter_csv_rows,
    sha256_bytes_of,
)
from expected_goals_lab.diagnostics import (
    CLOSE_RANGE_M,
    NEIGHBOURHOOD_DEG,
    NEIGHBOURHOOD_M,
    SPARSE_BELOW,
    adjusted_by_class,
    bins_covering_diagonal,
    bins_missing_diagonal,
    calibration_rows,
    illustrative_predictions,
    league_rows,
    league_summary,
    overall_tally,
    raw_body_part_rows,
)
from expected_goals_lab.evaluate_baselines import LEAGUE_NAMES
from expected_goals_lab.evaluate_nonlinear import (
    DEFAULT_PROTOCOL as S04_PROTOCOL,
)
from expected_goals_lab.evaluate_nonlinear import (
    DEFAULT_S03_REPORT,
    NonlinearEvaluation,
    evaluate_nonlinear,
)
from expected_goals_lab.nonlinear import NONLINEAR_LOCATION_BODY_PART
from expected_goals_lab.scoring import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED

REFERENCE, SELECTED = DISTANCE_PLUS_ANGLE, NONLINEAR_LOCATION_BODY_PART
DEFAULT_REPORT = Path("reports/model_diagnostics.md")
DEFAULT_FIGURES = Path("figures")
DEFAULT_S04_REPORT = Path("reports/nonlinear_validation.md")
DEFAULT_PROTOCOL = Path("docs/DIAGNOSTICS_PROTOCOL.md")
SUMMARY_NAME = "diagnostics_summary.json"
FIGURE_STEMS = ("validation_model_comparison", "validation_calibration", "body_part_and_location")


@dataclass
class DiagnosticsResult:
    s04: NonlinearEvaluation
    leagues: list[dict]
    league_counts: dict
    calibration: dict[str, list[dict]]
    tally: dict[str, dict]
    raw_validation: list[dict]
    raw_train: list[dict]
    adjusted: list[dict]
    fitted_terms: dict[str, float]
    illustrations: list[dict]
    checks: list[tuple[str, str]]
    provenance: dict
    first: dict | None = None  # wording for the first screen, derived from the numbers


def parse_s04_validation_scores(report_path: Path) -> dict[str, tuple]:
    """Validation score rows as printed in the committed S04 report: model -> (n, goals, ll, "
    "brier, auc)."""
    if not report_path.exists():
        raise ContractError(f"{report_path} not found: S04 numbers cannot be reproduced")
    text = report_path.read_text(encoding="utf-8")
    if "## 4. Scores" not in text:
        raise ContractError(f"{report_path} has no scores section")
    section = text.split("## 4. Scores", 1)[1].split("Train, in-sample", 1)[0]
    rows = {}
    for match in re.finditer(
        r"^\| (\w+) \| (\d+) \| (\d+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|",
        section,
        re.MULTILINE,
    ):
        name, n, goals, ll, brier, auc = match.groups()
        rows[name] = (int(n), int(goals), ll, brier, auc)
    return rows


def check_reproduces_s04(s04: NonlinearEvaluation, report_path: Path) -> tuple[str, str]:
    printed = parse_s04_validation_scores(report_path)
    for model, scores in s04.scores["validation"].items():
        if model not in printed:
            raise ContractError(f"model {model} is missing from the S04 report")
        got = (
            scores.n,
            scores.goals,
            f"{scores.log_loss:.4f}",
            f"{scores.brier:.5f}",
            f"{scores.auc:.3f}",
        )
        if got != printed[model]:
            raise ContractError(
                f"regenerated {model} {got} differs from the S04 report {printed[model]}"
            )
    return (
        "Regenerated S04 validation numbers equal the committed S04 report (to printed precision)",
        "pass",
    )


def evaluate_diagnostics(
    csv_path: Path,
    manifest_path: Path,
    out_dir: Path,
    resamples: int = BOOTSTRAP_RESAMPLES,
    s03_report: Path = DEFAULT_S03_REPORT,
    s04_report: Path = DEFAULT_S04_REPORT,
    s04_protocol: Path = S04_PROTOCOL,
    protocol: Path = DEFAULT_PROTOCOL,
    rows: RowReader = iter_csv_rows,
) -> DiagnosticsResult:
    s04 = evaluate_nonlinear(
        csv_path, manifest_path, out_dir, resamples, s03_report, s04_protocol, rows
    )
    checks = list(s04.checks)
    checks.append(check_reproduces_s04(s04, s04_report))

    val, train = s04.parts["validation"], s04.parts["train"]
    y = val.base.y
    p_ref, p_sel = s04.predictions["validation"][REFERENCE], s04.predictions["validation"][SELECTED]

    leagues = league_rows(val.base, p_ref, p_sel, LEAGUE_NAMES, resamples)
    checks.append(("League shots and goals add up to the validation totals", "pass"))
    calibration = {"reference": calibration_rows(y, p_ref), "selected": calibration_rows(y, p_sel)}
    checks.append(("Calibration bin shots and goals add up to the validation totals", "pass"))
    tally = {"reference": overall_tally(y, p_ref), "selected": overall_tally(y, p_sel)}

    pipeline = s04.candidates[SELECTED].pipeline
    categories = pipeline.named_steps["features"].named_transformers_["body_part"].categories_[0]
    weights = pipeline.named_steps["logit"].coef_[0][-len(categories) :]
    fitted_terms = {str(c): float(w) for c, w in zip(categories, weights, strict=True)}
    illustrations = illustrative_predictions(pipeline, train)
    for loc in illustrations:
        distance, angle = loc["distance_m"], np.radians(loc["angle_deg"])
        lo_d, hi_d = (
            train.base.features["distance_m"].min(),
            train.base.features["distance_m"].max(),
        )
        lo_a, hi_a = train.base.features["angle_rad"].min(), train.base.features["angle_rad"].max()
        if not (lo_d <= distance <= hi_d and lo_a <= angle <= hi_a):
            raise ContractError(
                f"illustrative position {loc['location']!r} is outside the training range"
            )
    checks.append(("Illustrative positions lie inside the training feature range", "pass"))

    provenance = {**s04.provenance, "diagnostics_protocol_sha256": _sha(protocol)}
    result = DiagnosticsResult(
        s04=s04,
        leagues=leagues,
        league_counts=league_summary(leagues),
        calibration=calibration,
        tally=tally,
        raw_validation=raw_body_part_rows(val.body_part, y, val.base.features["distance_m"]),
        raw_train=raw_body_part_rows(
            train.body_part, train.base.y, train.base.features["distance_m"]
        ),
        adjusted=adjusted_by_class(val.body_part, y, p_ref, p_sel),
        fitted_terms=fitted_terms,
        illustrations=illustrations,
        checks=checks,
        provenance=provenance,
    )
    result.first = first_screen(result)
    return result


def _sha(path: Path) -> str:
    return sha256_bytes_of(path) if path.exists() else "unavailable"


# --- wording derived from the numbers ----------------------------------------------------------


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _num(x: float | None, digits: int) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _signed(x: float | None, digits: int) -> str:
    return "n/a" if x is None else f"{x:+.{digits}f}".replace("+-", "-")


def body_part_story(result: DiagnosticsResult) -> dict:
    """Facts about raw versus location-adjusted body-part patterns, stated only if the data show "
    "them."""
    raw = {r["class"]: r for r in result.raw_validation}
    train_raw = {r["class"]: r for r in result.raw_train}
    feet = ("right_foot", "left_foot")
    head = "head_or_body"
    if any(
        raw[c]["conversion"] is None or train_raw[c]["conversion"] is None for c in (*feet, head)
    ):
        return {
            "conflict": False,
            "text": "A raw versus adjusted comparison is not possible: a class has no shots.",
        }
    feet_n = sum(raw[c]["shots"] for c in feet)
    feet_goals = sum(raw[c]["goals"] for c in feet)
    feet_rate = feet_goals / feet_n
    higher_raw = train_raw[head]["conversion"] > max(train_raw[c]["conversion"] for c in feet) and (
        raw[head]["conversion"] > feet_rate
    )
    lower_adjusted = result.fitted_terms[head] < min(result.fitted_terms[c] for c in feet)
    closer = raw[head]["median_distance"] < min(raw[c]["median_distance"] for c in feet)
    conflict = higher_raw and lower_adjusted
    if conflict and closer:
        feet_medians = [raw[c]["median_distance"] for c in feet]
        feet_close = sum(raw[c]["close_shots"] for c in feet) / feet_n
        text = (
            "Head/body shots convert more often than foot shots in both windows "
            f"(validation {_pct(raw[head]['conversion'])} versus {_pct(feet_rate)}), yet the "
            f"fitted head/body term ({result.fitted_terms[head]:+.2f}) is the lowest of the three. "
            "The raw gap is largely a matter of location: head/body shots are taken closer to goal "
            f"(validation median {raw[head]['median_distance']:.1f} m versus "
            f"{min(feet_medians):.1f}–{max(feet_medians):.1f} m for the feet; "
            f"{_pct(raw[head]['close_share'], 0)} within {CLOSE_RANGE_M:.0f} m versus "
            f"{_pct(feet_close, 0)}). Once the model knows where the shot was taken, it gives a "
            "head/body shot lower odds than a foot shot from the same spot. A plausible football "
            "reading, which these data cannot test, is that close-range head/body shots often "
            "follow crosses with defenders and the goalkeeper nearby, while close-range foot "
            "shots include cut-backs and through balls that leave more time."
        )
    else:
        text = (
            "The raw and the location-adjusted ordering of the classes do not conflict in a way "
            "this table can illustrate; see the numbers above."
        )
    return {"conflict": conflict, "closer": closer, "text": text}


def _missing_bins_text(rows: list[dict]) -> str:
    missing = bins_missing_diagonal(rows)
    if not missing:
        return ""
    parts = ", ".join(f"bin {i} (by {gap:.4f})" for i, gap in missing)
    return (
        f" The bins that miss are {parts}; the gap is the distance in probability between the "
        "bin's mean prediction and the nearest edge of its interval."
    )


def adjusted_story(result: DiagnosticsResult) -> str:
    """Where the location-only benchmark and the observed rate disagree for head/body shots."""
    head = {r["class"]: r for r in result.adjusted}["head_or_body"]
    if not head["shots"] or head["observed"] is None:
        return ""
    gap = head["mean_reference"] - head["observed"]
    direction = "over-predicts" if gap > 0 else "under-predicts"
    return (
        f"The location-only benchmark {direction} head/body shots: it averages "
        f"{_pct(head['mean_reference'])} against {_pct(head['observed'])} observed "
        f"({head['shots']} shots); the selected model averages {_pct(head['mean_selected'])}. "
        "That mis-estimate, in a class concentrated close to goal, is where the tag adds "
        "information."
    )


def first_screen(result: DiagnosticsResult) -> dict:
    s04 = result.s04
    v = s04.scores["validation"]
    ref, sel = v[REFERENCE], v[SELECTED]
    ll = s04.intervals["log_loss"][(SELECTED, REFERENCE)]
    counts = result.league_counts
    cover = {k: bins_covering_diagonal(result.calibration[k]) for k in ("reference", "selected")}
    both_better = sel.log_loss < ref.log_loss and sel.brier < ref.brier
    s1 = (
        f"On the {sel.n:,} validation shots, adding Wyscout's body-part tag to flexible distance "
        "and "
        f"angle curves gave {'better' if both_better else 'mixed'} goal probabilities than "
        "distance and "
        f"angle alone: log loss {sel.log_loss:.4f} versus {ref.log_loss:.4f} and Brier score "
        f"{sel.brier:.5f} versus {ref.brier:.5f} (paired log-loss difference "
        f"{_signed(ll.estimate, 4)}, 95% interval [{_signed(ll.low, 4)}, {_signed(ll.high, 4)}])."
    )
    s2 = (
        f"The selected model has the lower log loss in {counts['lower_log_loss']} of "
        f"{counts['leagues']} leagues and the lower Brier score in {counts['lower_brier']} of "
        f"{counts['leagues']}, and its predicted rate falls inside the observed-rate interval in "
        f"{cover['selected']} of 10 probability bins (benchmark: {cover['reference']}); these are "
        "descriptive counts, not tests."
    )
    story = body_part_story(result)
    s3 = (
        "The body-part term is an association in tagged data, not the effect of heading: "
        "head/body shots convert more often overall, largely because they are taken closer "
        "to goal, but from the "
        "same spot the model gives them lower odds."
        if story["conflict"] and story["closer"]
        else "The body-part term is an association in tagged data, not a measured effect of how "
        "a shot is struck."
    )
    return {"s1": s1, "s2": s2, "s3": s3, "cover": cover, "story": story}


# --- figures ------------------------------------------------------------------------------------


def figure_inputs(result: DiagnosticsResult) -> dict:
    s04 = result.s04
    v = s04.scores["validation"]
    ref, sel = v[REFERENCE], v[SELECTED]
    window = (
        f"{s04.plan.boundaries['validation']['first_date']} to "
        f"{s04.plan.boundaries['validation']['last_date']}"
    )
    caption = (
        f"Wyscout open event data, five domestic leagues, 2017/18. Validation only: {sel.n:,} "
        "shots, "
        f"{sel.goals} goals, matches played {window}. The later test window is not used."
    )
    ll, br = (
        s04.intervals["log_loss"][(SELECTED, REFERENCE)],
        s04.intervals["brier"][(SELECTED, REFERENCE)],
    )
    better = sel.log_loss < ref.log_loss and sel.brier < ref.brier
    first = result.first
    return {
        "comparison": {
            "headline": (
                "Adding the body-part tag improved validation log loss and Brier score"
                if better
                else "Selected model versus benchmark on validation: mixed result"
            ),
            "subtitle": "Selected (spline location + body-part tag) versus benchmark (distance "
            f"+ angle), {sel.n:,} validation shots",
            "reference": {"log_loss": ref.log_loss, "brier": ref.brier, "auc": ref.auc},
            "selected": {"log_loss": sel.log_loss, "brier": sel.brier, "auc": sel.auc},
            "diff_log_loss": (ll.low, ll.high),
            "diff_brier": (br.low, br.high),
            "caption": caption,
        },
        "calibration": {
            "headline": "Predicted and observed goal rates by probability bin, validation matches",
            "subtitle": (
                f"Selected model: {first['cover']['selected']} of 10 bins have the bin's mean "
                "prediction "
                f"inside the observed-rate interval; benchmark: {first['cover']['reference']} of 10"
            ),
            "calibration": result.calibration,
            "tally": result.tally,
            "caption": caption,
        },
        "body_part": {
            "headline": (
                "Head/body shots score more often overall, but the model rates them lower from "
                "the same spot"
                if first["story"]["conflict"] and first["story"]["closer"]
                else "Body-part tag: observed goal rates and location-adjusted predictions"
            ),
            "subtitle": "Wyscout tag classes: right foot, left foot, head/body (tag 403, not a "
            "confirmed header)",
            "raw": result.raw_validation,
            "overall_rate": sel.goals / sel.n,
            "illustrations": result.illustrations,
            "caption": caption,
        },
    }


def make_figures(result: DiagnosticsResult, folder: Path) -> list[Path]:
    inputs = figure_inputs(result)
    paths = []
    paths += figures.plot_model_comparison(folder / FIGURE_STEMS[0], inputs["comparison"])
    paths += figures.plot_calibration(folder / FIGURE_STEMS[1], inputs["calibration"])
    paths += figures.plot_body_part(folder / FIGURE_STEMS[2], inputs["body_part"])
    return paths


# --- report -------------------------------------------------------------------------------------


def _iv(interval, digits: int = 4) -> str:
    if interval is None:
        return "n/a"
    return f"[{interval[0]:+.{digits}f}, {interval[1]:+.{digits}f}]"


def render_report(
    result: DiagnosticsResult, csv_path: Path, report_path: Path, figures_dir: Path
) -> str:
    s04 = result.s04
    v = s04.scores["validation"]
    ref, sel = v[REFERENCE], v[SELECTED]
    first = result.first
    n_matches = len(set(s04.parts["validation"].base.match_ids.tolist()))
    rel = Path(os.path.relpath(figures_dir, report_path.parent)).as_posix()
    ll = s04.intervals["log_loss"][(SELECTED, REFERENCE)]
    br = s04.intervals["brier"][(SELECTED, REFERENCE)]
    p = result.provenance
    out: list[str] = []
    add = out.extend
    add(
        [
            "# Does the body-part tag help predict goals? Validation diagnostics (S05)",
            "",
            "**Question.** When a shot is taken from a given spot, does knowing whether Wyscout "
            "tagged it "
            "as a right foot, left foot or head/body shot make the goal probability more "
            "accurate than "
            "distance and angle alone, and does that hold in every league?",
            "",
            f"**Finding.** {first['s1']} {first['s2']} {first['s3']}",
            "",
            table(
                [
                    "Validation, 2018-02-26 to 2018-04-13",
                    "Benchmark: distance + angle",
                    "Selected: spline location + body-part tag",
                ],
                [
                    [
                        "Shots / goals / matches",
                        f"{ref.n:,} / {ref.goals} / {n_matches}",
                        f"{sel.n:,} / {sel.goals} / {n_matches}",
                    ],
                    ["Log loss (lower is better)", f"{ref.log_loss:.4f}", f"{sel.log_loss:.4f}"],
                    ["Brier score (lower is better)", f"{ref.brier:.5f}", f"{sel.brier:.5f}"],
                    ["ROC AUC", f"{ref.auc:.3f}", f"{sel.auc:.3f}"],
                    [
                        "Selected minus benchmark, log loss",
                        "",
                        f"{_signed(ll.estimate, 4)} {_iv((ll.low, ll.high))}",
                    ],
                    [
                        "Selected minus benchmark, Brier",
                        "",
                        f"{_signed(br.estimate, 5)} {_iv((br.low, br.high), 5)}",
                    ],
                ],
            ),
            "",
            "**Limitation.** This is one season (2017/18), one contiguous validation window and "
            "provider-tagged data; the interval comes from a bootstrap of validation matches that "
            "ignores selection among candidates and dependence between matches of one team, so it "
            "supports a statement about these matches and not about other seasons. The later test "
            "window has not been scored.",
            "",
            f"Figures: [model comparison]({rel}/{FIGURE_STEMS[0]}.png) · "
            f"[calibration]({rel}/{FIGURE_STEMS[1]}.png) · "
            f"[body part and location]({rel}/{FIGURE_STEMS[2]}.png) (SVG versions alongside).",
            "",
            f"![Model comparison]({rel}/{FIGURE_STEMS[0]}.png)",
            "",
            "---",
            "",
            "## Method and diagnostics",
            "",
            f"Generated by `uv run python -m expected_goals_lab.evaluate_diagnostics` on "
            f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Validation only; descriptive and exploratory "
            "except for the reproduction checks (`docs/DIAGNOSTICS_PROTOCOL.md`). The model "
            "decision (`docs/MODEL_SELECTION_DECISION.md`) is provisional and was fixed from S04 "
            "results before these breakdowns were computed. The test window has not been "
            "modelled or "
            "scored, and its row-level labels/features have not been accessed; however, its "
            "aggregate "
            "goal count is indirectly inferable from earlier cohort audit totals.",
            "",
            f"- Input table `{csv_path.as_posix()}` SHA-256 `{p['input_csv_sha256']}`; split "
            "assignment "
            f"SHA-256 `{p['assignment_csv_sha256']}`; diagnostics protocol SHA-256 "
            f"`{p['diagnostics_protocol_sha256']}`.",
            f"- Git commit `{p['commit']}`"
            + (" (working tree has uncommitted changes)" if p["uncommitted_changes"] else "")
            + f"; scikit-learn {p['scikit_learn']}.",
            "- Data: Pappalardo & Massucco (2019), Wyscout open event data, CC BY 4.0, "
            "doi:10.6084/m9.figshare.c.4415000.v5; collected by Wyscout; processed for this "
            "project.",
            "",
            table(["Check run before any diagnostic", "Result"], [list(c) for c in result.checks]),
            "",
            "### A. By league (validation, descriptive)",
            "",
            "Differences are selected minus benchmark, so negative means the selected model scored "
            "better. The bracketed intervals use the same match-cluster bootstrap as S04 and only "
            "indicate sampling noise: no threshold or significance label is applied, and the five "
            "leagues are not combined into a test. AUC is shown only with at least "
            "10 goals and 10 non-goals.",
            "",
        ]
    )
    rows = []
    for r in result.leagues:
        rows.append(
            [
                r["name"],
                r["shots"],
                r["goals"],
                r["matches"],
                _pct(r["conversion"], 2),
                _num(r["reference"] and r["reference"]["log_loss"], 4),
                _num(r["selected"] and r["selected"]["log_loss"], 4),
                f"{_signed(r['diff_log_loss'], 4)} {_iv(r['interval_log_loss'])}"
                if r["diff_log_loss"] is not None
                else "n/a",
                _num(r["reference"] and r["reference"]["brier"], 5),
                _num(r["selected"] and r["selected"]["brier"], 5),
                f"{_signed(r['diff_brier'], 5)}" if r["diff_brier"] is not None else "n/a",
                _num(r["reference"] and r["reference"]["auc"], 3),
                _num(r["selected"] and r["selected"]["auc"], 3),
            ]
        )
    add(
        [
            table(
                [
                    "League",
                    "Shots",
                    "Goals",
                    "Matches",
                    "Conversion",
                    "LL benchmark",
                    "LL selected",
                    "LL diff [interval]",
                    "Brier benchmark",
                    "Brier selected",
                    "Brier diff",
                    "AUC benchmark",
                    "AUC selected",
                ],
                rows,
            ),
            "",
            f"Selected model has the lower log loss in {result.league_counts['lower_log_loss']} of "
            f"{result.league_counts['leagues']} leagues and the lower Brier score in "
            f"{result.league_counts['lower_brier']} of {result.league_counts['leagues']}. League "
            "names come "
            "from the Wyscout competition mapping verified in S01B.",
            "",
            "### B. Calibration (validation, no calibration fitted)",
            "",
            "Ten equal-frequency bins per model. The Wilson 95% interval treats shots as "
            "independent "
            "and therefore understates uncertainty from shots within a match.",
            "",
        ]
    )
    for key, title in (
        ("reference", "Benchmark: distance + angle"),
        ("selected", "Selected: spline location + body-part tag"),
    ):
        t = result.tally[key]
        add(
            [
                f"**{title}.** Sum of predictions {t['expected_goals']:.1f} against {t['goals']} "
                "observed goals "
                f"(mean prediction {_pct(t['mean_prediction'], 2)}, conversion "
                f"{_pct(t['conversion'], 2)}); "
                f"{first['cover'][key]} of 10 bins have the mean prediction inside the "
                "observed-rate interval." + _missing_bins_text(result.calibration[key]),
                "",
                table(
                    ["Bin", "Shots", "Goals", "Mean prediction", "Observed rate", "95% interval"],
                    [
                        [
                            i + 1,
                            b["shots"],
                            b["goals"],
                            f"{b['mean_prediction']:.4f}",
                            f"{b['observed']:.4f}",
                            f"[{b['low']:.4f}, {b['high']:.4f}]",
                        ]
                        for i, b in enumerate(result.calibration[key])
                    ],
                ),
                "",
            ]
        )
    add(
        [
            "### C. Body-part tag class: association, not a causal effect",
            "",
            'Wyscout `head_or_body` is tag 403 ("head/body"): it mixes headers with other '
            "body-part "
            f"shots and is **not a confirmed header**. A class with fewer than {SPARSE_BELOW} "
            "shots in a "
            "group is marked sparse.",
            "",
            "**C1. Raw, as observed.**",
            "",
        ]
    )
    for window, data in (("Train", result.raw_train), ("Validation", result.raw_validation)):
        add(
            [
                f"*{window}*",
                "",
                table(
                    [
                        "Class",
                        "Shots",
                        "Goals",
                        "Goal rate",
                        "95% interval",
                        "Median distance (m)",
                        f"Within {CLOSE_RANGE_M:.0f} m",
                    ],
                    [
                        [
                            r["label"],
                            r["shots"] if not r["sparse"] else f"{r['shots']} (sparse)",
                            r["goals"],
                            _pct(r["conversion"], 2),
                            "n/a"
                            if r["low"] is None
                            else f"[{_pct(r['low'], 2)}, {_pct(r['high'], 2)}]",
                            _num(r["median_distance"], 1),
                            f"{r['close_shots']} ({_pct(r['close_share'], 0)})"
                            if r["shots"]
                            else "n/a",
                        ]
                        for r in data
                    ],
                ),
                "",
            ]
        )
    add(
        [
            "**C2. Validation shots: observed against each model's mean prediction.** The "
            "benchmark "
            "uses location only; the selected model also uses the tag.",
            "",
            table(
                [
                    "Class",
                    "Shots",
                    "Observed rate",
                    "Benchmark mean prediction",
                    "Selected mean prediction",
                ],
                [
                    [
                        r["label"],
                        r["shots"],
                        _pct(r["observed"], 2),
                        _pct(r["mean_reference"], 2),
                        _pct(r["mean_selected"], 2),
                    ]
                    for r in result.adjusted
                ],
            ),
            "",
            adjusted_story(result),
            "",
            "**C3. Fitted body-part terms (train).** Log-odds scale, L2-penalised, after the "
            "spline "
            "terms for location; only differences between classes mean anything.",
            "",
            table(
                ["Class", "Coefficient"],
                [[c, f"{w:+.4f}"] for c, w in sorted(result.fitted_terms.items())],
            ),
            "",
            "**C4. Why a higher raw rate can coexist with a lower adjusted term.** "
            + first["story"]["text"],
            "",
            "**C5. Fixed-location illustrations (model predictions, not observations).** The "
            "selected "
            "model's predicted probability for each class at four fixed positions chosen before "
            "any "
            "result was seen. 'Nearby' counts training shots of that class within "
            f"{NEIGHBOURHOOD_M} m of the "
            f"position's distance and {NEIGHBOURHOOD_DEG:.0f} degrees of its angle; fewer than "
            f"{SPARSE_BELOW} "
            "means the prediction there is extrapolated. These are not a measured effect of "
            "changing "
            "how a shot is struck: shots are not assigned a body part at random.",
            "",
            table(
                ["Position", "Distance (m)", "Right foot", "Left foot", "Head/body (tag 403)"],
                [
                    [
                        f"{loc['location']} (x {loc['x_pct']}, y {loc['y_pct']})",
                        f"{loc['distance_m']:.1f}",
                        *[
                            f"{_pct(c['probability'], 1)}; nearby {c['training_shots_nearby']}"
                            + (" (sparse)" if c["sparse"] else "")
                            for c in loc["cells"]
                        ],
                    ]
                    for loc in result.illustrations
                ],
            ),
            "",
            "### D. Robustness and limits",
            "",
            "- **Goal-line records.** The seven `x_pct = 100` records and the S02 quarantine "
            "policy are "
            "unchanged in the main analysis. S04's own sensitivity left the "
            f"{s04.sensitivity['removed']} "
            f"validation goal-line shots ({s04.sensitivity['removed_goals']} goals) out of "
            "scoring only, without refitting: "
            f"the selected model still scored better than the benchmark on log loss "
            f"({s04.sensitivity['scores'][SELECTED].log_loss:.4f} versus "
            f"{s04.sensitivity['scores'][REFERENCE].log_loss:.4f}) "
            f"and Brier score ({s04.sensitivity['scores'][SELECTED].brier:.5f} versus "
            f"{s04.sensitivity['scores'][REFERENCE].brier:.5f}). It is repeated from S04, not "
            "re-optimised.",
            "- **Coordinates and tags.** Positions are integer percentages with an assumed 105 x "
            "68 m "
            "pitch; labels and body-part tags come from human provider tagging; `head_or_body` "
            "is coarse.",
            "- **Design.** One season, five leagues, one chronological cut. The same teams and "
            "players "
            "appear in the train, validation and test windows.",
            "- **Uncertainty.** Match-level resampling does not capture dependence between "
            "matches of "
            "the same team or league, nor the selection of this model among candidates. League "
            "rows "
            "are small, their intervals wide, and no inferential threshold is applied.",
            "- **Not done.** No calibration fitting, no new feature or model, no change to the "
            "cohort "
            "or the decision, and no scoring of the held-out test window.",
            "",
            "### Post hoc additions and departures",
            "",
            "Four reporting and presentation additions were made after the first run had been "
            "seen; none "
            "changes the data, models, split, calculations, thresholds or the fixed decision: "
            "(1) naming the calibration bins whose interval misses the mean prediction, "
            "(2) the sentence on the benchmark's over-prediction of head/body shots, "
            "(3) hedged wording of the football explanation and a softened first-screen sentence, "
            "(4) figure re-layout after inspection and clamped Wilson intervals. See "
            "`docs/DIAGNOSTICS_PROTOCOL.md`, section 10.",
        ]
    )
    return "\n".join(out) + "\n"


def summary_json(result: DiagnosticsResult) -> dict:
    """Aggregates only, for the git-ignored run record."""
    return {
        "provenance": result.provenance,
        "checks": [list(c) for c in result.checks],
        "leagues": result.leagues,
        "calibration": result.calibration,
        "tally": result.tally,
        "raw_validation": result.raw_validation,
        "adjusted": result.adjusted,
        "fitted_terms": result.fitted_terms,
        "illustrations": result.illustrations,
        "resamples": result.s04.resamples,
        "seed": BOOTSTRAP_SEED,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validation-only S05 diagnostics.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--s03-report", type=Path, default=DEFAULT_S03_REPORT)
    parser.add_argument("--s04-report", type=Path, default=DEFAULT_S04_REPORT)
    parser.add_argument("--s04-protocol", type=Path, default=S04_PROTOCOL)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    manifest_path = args.manifest or args.csv.with_suffix(".manifest.json")
    out_dir = args.out_dir or args.csv.parent
    started = time.perf_counter()
    try:
        result = evaluate_diagnostics(
            args.csv,
            manifest_path,
            out_dir,
            args.resamples,
            args.s03_report,
            args.s04_report,
            args.s04_protocol,
            args.protocol,
        )
    except FileNotFoundError as exc:
        print(
            f"error: {exc} not found. Build the shot table first: "
            "`uv run python -m expected_goals_lab.build_shots`.",
            file=sys.stderr,
        )
        return 2
    except (ContractError, ValueError, RuntimeError, Warning) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    paths = make_figures(result, args.figures_dir)
    report = render_report(result, args.csv, args.report, args.figures_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SUMMARY_NAME).write_text(
        json.dumps(summary_json(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report)
    print(
        f"Wrote {args.report} and {len(paths)} figure files in {args.figures_dir}. "
        f"Runtime {time.perf_counter() - started:.0f} s.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
