# Methodology

A short technical account of what was done, in the order it happened, with pointers to the full protocols and reports. Everything here describes one season of one provider's data; none of it is a claim about other seasons, other leagues or live use.

## 1. Question and data

**Question.** How much of a shot's chance of becoming a goal is explained by where it is taken, and does the provider's body-part tag add predictive value?

**Data.** Wyscout open event data (Pappalardo & Massucco, 2019; CC BY 4.0; see [DATA_LICENSE_AND_ATTRIBUTION.md](DATA_LICENSE_AND_ATTRIBUTION.md)): the five domestic league files for 2017/18, 1,826 matches. Files are pinned by Figshare file ID, size and MD5 and verified on every load. The data contain no provider xG, no freeze frames and no shot end location, so there is no external benchmark and no defender or goalkeeper information.

**Cohort.** Regular shots only (`eventId` 10, `subEventId` 100). Penalties and direct free-kick shots are counted but not modelled; own goals are not shots. "Regular" is a provider label, not a claim of open play: such a shot can follow a corner or a free kick. 40,461 regular shots exist; 36 are quarantined (18 in one match whose events do not reproduce the published score, 18 recorded implausibly far from the goal they were aimed at), leaving **40,425 eligible rows**. Quarantined rows keep their raw values and are never repaired, flipped or silently re-admitted. [SHOT_DATASET_SPEC.md](SHOT_DATASET_SPEC.md) lists every rule and known provider issue.

**Label.** `is_goal` is the provider's goal tag on the shot event itself. Goal tags on goalkeeper save events, post-shot tags (accuracy, blocked, goal-mouth zone), the final score and the lineup goal counts never create or change a label.

## 2. Features

- **Distance** to the goal centre and **angle** subtended by the posts, from the shot's start position. Wyscout positions are integer percentages from the attacking team's point of view (x = 100 is the goal attacked). Metres use an *assumed* 105 m x 68 m pitch and a 7.32 m goal; the provider publishes no pitch size, and alternative sizes were probed as a sensitivity check ([GEOMETRY_VALIDATION.md](GEOMETRY_VALIDATION.md)). A position is about 1 m along and 0.7 m across the pitch, so very close shots are coarse.
- **Body part class**: left foot, right foot, or `head_or_body`. The last is the provider's tag 403 ("head/body"); it is **not** a confirmed header.
- Nothing from after the shot, no second position entry, no provider tag other than the body-part class and no identifier is a model input. The allowlist is enforced in code.

## 3. Chronological partitions

Matches are split by UTC match date, not at random: the earliest 70% of the 178 distinct match dates are **training** (1,270 matches, 28,263 shots, 4 Aug 2017 to 25 Feb 2018), the next 15% **validation** (257 matches, 5,509 shots, 26 Feb to 13 Apr 2018) and the last 15% **test** (298 matches, 6,653 shots, 14 Apr to 20 May 2018). Whole dates, hence whole matches, fall in one partition. The assignment is deterministic (integer arithmetic) and its SHA-256 is recorded. Transforms (scaling, spline knots, one-hot categories) are fitted inside the model pipeline on the rows passed to `fit`, so they see training rows only.

## 4. Models

Both are logistic regressions with the library's default L2 penalty (`C = 1.0`, `lbfgs`, `max_iter = 1000`), fixed in advance and not tuned; a convergence warning is an error.

- **Benchmark `distance_plus_angle`:** standardised distance and angle.
- **Candidate `nonlinear_location_body_part`:** cubic spline bases of distance and of angle separately (5 quantile knots, constant extrapolation), standardised, plus a one-hot of the body-part class.
- A constant rate and a distance-only model were also fitted as references.

The two recipes differ in their location terms (smooth curves rather than straight-line terms) *and* in the body-part tag. A validation-stage variant with the smooth terms but no body-part tag was inconclusive; it was not part of the held-out evaluation. The held-out comparison is therefore a comparison of two recipes, not an ablation of body part, and the body-part term is an association in provider-tagged data, not a causal effect of how a shot is struck.

## 5. Validation (exploratory)

Metrics are log loss (primary), Brier score and ROC AUC (secondary); classification accuracy is not used. Uncertainty comes from a paired bootstrap that resamples whole matches (1,000 resamples, seed 20180513). On the validation window the candidate scored log loss 0.2845 against 0.2930 and Brier 0.08197 against 0.08468. This stage was exploratory: models were compared and one was chosen, so its intervals do not account for that choice. Diagnostics by league, probability bin and body-part class ([model_diagnostics.md](../reports/model_diagnostics.md)) did not change the choice, which was frozen in [MODEL_SELECTION_DECISION.md](MODEL_SELECTION_DECISION.md).

## 6. The one-time, score-held-out evaluation

The protocol ([FINAL_TEST_PROTOCOL.md](FINAL_TEST_PROTOCOL.md)) was fixed and hash-pinned before any test row was scored: refit the two recipes once on training plus validation (33,772 shots), score the untouched test window once, report a predetermined set of aggregates, and do not change anything afterwards. It was run once; the log records one start and one completion. The result is in [reports/final_test_evaluation.md](../reports/final_test_evaluation.md) and `results/final_test_summary.json`.

The window was held out from *scoring*, not from all outcome information: its aggregate goal count was inferable from published cohort totals before the evaluation. It is therefore described as score-held-out, not outcome-blind.

## 7. Results and their limits

On 6,653 test shots (298 matches) the candidate had log loss 0.2905 against 0.2998 and Brier 0.08447 against 0.08752 for the benchmark (AUC 0.787 against 0.767). The paired match bootstrap gives a log-loss difference of -0.0093 (95% interval -0.0120 to -0.0066) and a Brier difference of -0.00305 (-0.00393 to -0.00215), which meets the pre-stated reading rule for an improvement. These intervals resample matches and so ignore dependence between matches of one team or league and the selection of the candidate; they are exploratory. Both models expect slightly fewer goals than were scored (about 715 and 712 against 738). No probability calibration was fitted; calibration tables are in the report and show no single direction of miscalibration across bins, with shot-level intervals that understate uncertainty.

Limits that matter: one season; the same teams and players in every partition; provider tags and coordinates; no goalkeeper, defender or pressure context; regular shots only; one chronological cut. Nothing here supports claims about player finishing ability, goalkeeper performance, scouting value, other seasons or causal effects.

## 8. Checking the shot locations

Because the model's main input is a coordinate, a development-only audit checked what that coordinate is ([SHOT_ORIGIN_AUDIT_PROTOCOL.md](SHOT_ORIGIN_AUDIT_PROTOCOL.md), [reports/shot_origin_audit.md](../reports/shot_origin_audit.md)). All 33,772 training and validation coordinates equal the first `positions` entry of the source event; the second entry is a placeholder on every one of those shots; shot origins usually coincide with the end of the team's preceding pass; and goal rates rise smoothly towards the goal with many non-goals at every distance. Six records sit exactly on the goal line. The result is an evidence-based review judgement on top of a passing mechanical screen: it is internal-consistency evidence for the development data, not proof of physical ground truth, and it says nothing about the held-out window.

## 9. Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md): what can be rebuilt from the public files, which tests need the download, and why the published held-out result is a record to inspect rather than a number the supplied code will regenerate without a deliberate new lock.
