# S04 protocol: nonlinear location and body-part classes

Written and frozen before any S04 candidate was fitted. Section 12 lists departures; at the time of writing there are none.

## 1. Question

After the established distance + angle logistic regression, do (a) nonlinear relationships with location and (b) the limited Wyscout body-part classes improve held-out, within-season goal probabilities on **validation** matches?

This is a within-season comparison for five leagues on one contiguous validation window (26 match dates, 2018-02-26 to 2018-04-13). It does not speak to other seasons, other leagues or club-grade accuracy. **No model is selected for final testing in S04.**

## 2. Status of the test window

The test window has not been modelled or scored, and its row-level labels/features have not been accessed by S03; however, its aggregate goal count is indirectly inferable from earlier cohort audit totals. S04 keeps the same posture: it reads no row-level label, feature, location or body-part value of a test row, scores nothing there, and neither states nor computes a test goal count. It is not entirely outcome-blind at the aggregate level, and no S04 text claims otherwise.

## 3. Data and split identity

- Input: the S02 shot table, SHA-256 `58eb62c0f786406c7df1e457a981784ab10d0514ee5a3ec73c2119c83f0af96e`, with its manifest. Same eligibility rules (36 quarantined rows never enter).
- Split: exactly S03's date-level chronological assignment (train 2017-08-04 to 2018-02-25, validation 2018-02-26 to 2018-04-13, test 2018-04-14 to 2018-05-20). Published counts of eligible shots: train 28,263, validation 5,509, test 6,653. S04 recomputes the assignment and requires its SHA-256 (`2f995fc0337436619f794d8c64c7f8397bf81d8e28deca9f2add8369c881da15`) and the three counts to match S03 whenever the input table has the published SHA; any difference is a hard error and means regeneration found an integrity issue to explain, not to work around.
- Outcome: `is_goal`, the provider's tag 101 on the shot event. No score reconstruction and no goalkeeper-derived label.

## 4. Reference model

S03's `distance_plus_angle`: `Pipeline(StandardScaler, LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000))` on `distance_m` and `angle_rad`, fitted by the unchanged S03 code. S04 regenerates its validation predictions and requires its log loss, Brier score and AUC to equal the figures in the committed `reports/baseline_validation.md` to the precision printed there. The model is not relabelled or retuned.

## 5. Candidates (fixed; no search)

All components below are fitted on **train only** and applied to validation by `transform` / `predict`.

**A. `nonlinear_location`.** For each of `distance_m` and `angle_rad` separately:
`SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="constant", include_bias=False)`, followed by `StandardScaler`. The two blocks (6 columns each) are concatenated and passed to `LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)`. Knots are the training quantiles; values outside the training range take the boundary basis values (constant extrapolation). The scaler is applied after the spline so each basis column has unit variance in the training data, which keeps the fixed L2 penalty comparable across columns.

**B. `nonlinear_location_body_part`.** Exactly A's two spline blocks, plus a one-hot encoding of `body_part_class` (`OneHotEncoder(handle_unknown="ignore", sparse_output=False)`, categories learned from train, not scaled), then the same `LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)`. The full one-hot (no dropped level) is used; the L2 penalty handles the collinearity with the intercept.

`body_part_class` is a Wyscout **tag-derived class**: `right_foot` and `left_foot` come from tags 401/402, and `head_or_body` from tag 403, which means "head/body". It is not a confirmed header and is not an independently observed biomechanical category. Nothing in S04 interprets it as one.

Unseen-category policy: the data layer fails closed on any value outside `{right_foot, left_foot, head_or_body}` (including `unknown`) in train or validation, and the model fit fails if train lacks any of the three. `handle_unknown="ignore"` is therefore only a last line of defence and should never be exercised.

Hard errors: convergence warnings; non-finite or out-of-[0, 1] predicted probabilities; a train or validation cohort with a single outcome class; a missing or duplicated event ID in the guarded join; an unexplained category; any mismatch in split identity.

If the specified design produces a real numerical problem it is reported before any other architecture is tried. No variants are tried until one wins.

## 6. Metrics and comparisons

- Primary: validation log loss. Also Brier score and ROC AUC. All three models' scores are shown with the shot count, goal count and match count. Train scores are shown only as an overfitting diagnostic.
- Comparisons: A minus reference, B minus reference, B minus A.
- Uncertainty: the S03 paired match-cluster bootstrap on validation matches, 1,000 resamples, seed 20180513, 95% percentile intervals of the difference in log loss and in Brier score; the same resamples serve every comparison.
- These intervals are **exploratory and unadjusted**: six intervals are reported without correction for multiplicity, and one season and one validation window are available. An interval that excludes zero is described as *fragile* when its edge nearest to zero is closer to zero than a quarter of the magnitude of the point estimate (fixed here, before any candidate is fitted).

**Pre-stated reading rule.** A candidate is described as *improving* on its comparator only if the log-loss point estimate is below zero, the 95% interval excludes zero, and the Brier point estimate is also below zero. If log loss and Brier disagree, or the log-loss interval includes zero, the report says there is no clear improvement. Nothing is called statistically significant.

## 7. Sensitivity: goal-line records

The seven records with `x_pct = 100` (S02 flag) have degenerate angles. The sensitivity re-scores the **validation** predictions with those rows excluded **from scoring only**. Models are not refitted and remain trained on the full fixed train cohort. The main scores are unchanged. The report gives the number of shots (and goals) removed and repeats the paired intervals on the reduced scoring set. This is not a retrained alternative cohort. No goal-line record of the test window is read.

## 8. Calibration

For each of A and B, ten equal-frequency bins of the validation predictions, with shot count, mean predicted probability and observed goal rate. No calibration is fitted. The bins are a description, not evidence of calibration. No shot-level outcomes or identities are published.

## 9. Data-access contract

S04 needs one column beyond S03: `body_part_class`. A dedicated guarded loader reads it **only for eligible train and validation rows**, joined to the S03 partitions by `event_id`. It validates that every partition event is found exactly once, that no extra event is returned, and that the vocabulary is the expected one. Asking it for the test split raises. S03's feature allowlist (`distance_m`, `angle_rad`) is unchanged and its baselines still reject `body_part_class`. Model code receives arrays from the guarded layer, never raw CSV rows.

## 10. Reproducibility and provenance

The report records the input SHA-256, the split assignment SHA-256, this protocol's SHA-256, the Git commit (and whether the working tree had uncommitted changes) and the candidate definitions. Everything runs offline from the S02 table.

## 11. Not done in S04

No test-window evaluation, no model selection for final testing, no hyperparameter search, no new data or cohorts, no calibration fitting, no thresholds changed, no tracking or pressure features, no plots.

## 12. Departures

Two reporting additions were made after the first validation run. Neither changes the data, split, candidates, metrics, comparisons, reading rule, bootstrap seed or resample count, or the sensitivity definition:

1. The report now states, for any comparison whose reading differs between the main analysis and the goal-line sensitivity, that its conclusion depends on those few records. (In the first run, `nonlinear_location` versus the reference read "no clear improvement" on all validation shots but "improvement, fragile" with the 2 goal-line shots left out of scoring.) The main reading is unchanged.
2. A table of the fitted one-hot body-part terms of the second candidate (train-fitted) was added to give the direction of what the model learned.

Any further departure is listed here and in `reports/nonlinear_validation.md`.

## 13. Limits

One season; validation is a single contiguous window; distance and angle assume a 105 x 68 m pitch; labels and body-part tags depend on provider tagging; teams and players recur across windows.
