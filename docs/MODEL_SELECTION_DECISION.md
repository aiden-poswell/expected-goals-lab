# Model selection decision (provisional)

Dated 2026-10-21. Written before any S05 diagnostic breakdown was computed or inspected. Based only on the S04 validation results already published in `reports/nonlinear_validation.md` (S04 commit `fbd96a926816982db2719c15815db4b1908c71dc`).

## Decision

| Role | Model | Definition |
|---|---|---|
| **Primary candidate** for one eventual final-test evaluation | `nonlinear_location_body_part` | cubic spline bases of `distance_m` and `angle_rad` (separately, 5 quantile knots, standardised) plus a one-hot of the three Wyscout body-part tag classes, then `LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)`, fitted on the training split only |
| **Predeclared benchmark** shown alongside it | S03 `distance_plus_angle` | `StandardScaler` + the same logistic regression on `distance_m` and `angle_rad` |
| Diagnostic only | `nonlinear_location` | kept in the S04 reports; **not eligible** to replace the primary on the basis of S05 exploratory analysis |

## Basis

From the S04 validation window only (5,509 shots, 587 goals, 257 matches, 26 match dates, 2018-02-26 to 2018-04-13):

- Log loss 0.2845 (primary candidate) versus 0.2930 (benchmark); Brier score 0.08197 versus 0.08468.
- Paired match-cluster bootstrap, log-loss difference (primary minus benchmark) -0.0085, 95% interval [-0.0115, -0.0057]; the Brier estimate points the same way.
- The spline location model alone (`nonlinear_location`) was inconclusive against the benchmark (log loss -0.0008, interval [-0.0018, +0.0002]), and its reading changed with the two goal-line shots excluded from scoring. The gain is therefore attributed to the body-part class and not to the splines.

## Status and limits of this decision

- **Provisional.** It is a choice among validation results, not a result from the future test window. The bootstrap is exploratory: it does not capture the uncertainty of having chosen among candidates, dependence between matches of the same team or league, or performance in another season.
- **No further tuning.** No model, feature, hyperparameter, cohort or exclusion rule is to be changed before the single final-test evaluation. The final test evaluation has not been performed, and S05 does not perform it.
- **The test window is not outcome-blind at the aggregate level.** It has not been modelled or scored and its row-level labels and features have not been accessed, but its aggregate goal count is inferable from previously published cohort totals.
- **Integrity errors stop the work.** If an S05 diagnostic uncovers an integrity error, the flaw is documented and the work stops for review. The model or cohort is not changed silently.
- **Weaker diagnostics do not change the decision.** If a league, bin or subgroup looks weaker for the primary candidate, that is reported faithfully and the fixed decision stands. Any change needs a separately documented protocol departure and review, not a model search.

## Association is not a causal effect

The body-part term in the primary candidate describes how goal rates differ between shots tagged `right_foot`, `left_foot` and `head_or_body` **after** accounting for where on the pitch the shot was taken. That is a statistical association in one season of provider-tagged data. It is not a measured effect of changing how a shot is struck: shots are not assigned a body part at random. Players, situations and defensive set-ups that lead to a header differ from those that lead to a foot shot in ways the model cannot see (for example crosses delivered under pressure, a defender close by, or a goalkeeper off the line). `head_or_body` is Wyscout tag 403 ("head/body"), which mixes headers with other body-part shots. It is not a confirmed header and nothing here estimates what heading does to a goal probability.
