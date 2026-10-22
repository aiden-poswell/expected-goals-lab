# S05 protocol: validation diagnostics

Written before any S05 breakdown was computed. Section 10 lists post hoc additions and departures; at the time of writing there are none. The model decision is fixed in `docs/MODEL_SELECTION_DECISION.md`.

## 1. Scope

Descriptive, validation-only diagnostics of two already-fitted models, to check whether the S04 improvement is coherent across leagues and probability ranges and to say what the body-part term does and does not mean. No model, feature, hyperparameter, cohort or exclusion rule changes. The test window is not modelled, scored or read. This is not a release and not a deployable-xG claim.

## 2. Models

- **Selected** (provisional primary): `nonlinear_location_body_part`.
- **Benchmark**: S03 `distance_plus_angle`.
- `nonlinear_location` is not part of S05 except where S04's own results are quoted.

Both are refitted by the unchanged S03/S04 code on the training split. Before any diagnostic is computed, S05 requires the validation log loss, Brier score, AUC, shot count and goal count of both models to equal the figures printed in the committed `reports/nonlinear_validation.md` (log loss to 4 decimals, Brier to 5, AUC to 3), and requires all S04 identity checks to pass (input SHA-256, split assignment SHA-256 and sizes, S03 reproduction). Any difference is a hard error.

## 3. Allowed data

Only what S04 already reads: eligible train and validation rows, with `distance_m`, `angle_rad`, `x_pct`, `y_pct`, `is_goal`, `body_part_class`, league and match IDs, through the guarded loaders. No test-window row is read, scored or described. Outputs hold aggregates only: no event ID, match ID, shot-level prediction or shot-level outcome is written to any report, figure or committed file.

## 4. Diagnostic A: leagues

For each of the five domestic leagues (names from the verified `evaluate_baselines.LEAGUE_NAMES` mapping, Wyscout competition IDs 364, 412, 426, 524, 795): validation shots, goals, matches, conversion; log loss and Brier score of both models; the difference (selected minus benchmark) in each; the number of leagues in which the selected model has the lower log loss and the lower Brier score.

- Intervals for league differences use the S03 paired match-cluster bootstrap (1,000 resamples, seed 20180513). They are shown as an indication of sampling noise only. **No threshold, reading rule or significance label is applied to them**, and the five leagues are not pooled into a test.
- AUC is shown only if the league has at least 10 goals and at least 10 non-goals in validation; otherwise `n/a`. A league with no shots or no goals gets `n/a` for any quantity that is undefined; nothing is divided by zero.
- An unknown league ID in the validation rows is a hard error.

## 5. Diagnostic B: calibration

Ten equal-frequency bins of each model's validation predictions (the S03/S04 binning rule). For each bin: shots, goals, mean prediction, observed rate and a 95% Wilson interval for the observed rate. The Wilson interval treats shots as independent, so it understates uncertainty from shots within a match; it is an indication only. Overall tally: sum of predictions (expected goals) against observed goals, and mean prediction against conversion. Bin shots and goals must sum to the validation totals (hard check).

**No calibration is fitted**, and no calibrator, parameter or threshold is chosen on these data. The tally describes this validation window only and says nothing about another season.

## 6. Diagnostic C: body-part tag class

Wyscout `head_or_body` is tag 403 ("head/body"), not a confirmed header; `right_foot` and `left_foot` come from tags 402 and 401.

1. **Raw:** per class, in train and in validation: shots, goals, conversion with 95% Wilson interval, median shot distance, and the share of shots within 12 m of the goal centre ("close range", fixed here). A class with fewer than 30 shots in a group is marked *sparse*.
2. **Location-adjusted, observed shots:** per class in validation, observed conversion beside the mean predicted probability of the benchmark (location only) and of the selected model.
3. **Fitted terms:** the selected model's train-fitted one-hot coefficients (log-odds scale, L2-penalised, relative to one another).
4. **Fixed-location illustrations (model predictions, not observations):** the selected model's predicted probability for each class at four fixed pitch positions, fixed here before any result was seen. Positions are in attack-relative percentages, converted with the S02 geometry (105 x 68 m): central at the penalty-spot area (x 90, y 50; 10.5 m); central at the edge of the box (x 84, y 50; 16.8 m); wide inside the box (x 90, y 35); central long range (x 75, y 50; 26.3 m). For each position the number of **training** shots of each class within 1.5 m of its distance and 5 degrees of its angle is reported; a class with fewer than 30 such shots there is marked sparse and its prediction is flagged as extrapolated. The positions are an illustration, not a prediction target, and not a measured effect of changing body part.

No statement is made that changing how a shot is struck changes its probability by any amount.

## 7. Diagnostic D: robustness and limits

Reported as stated, not re-optimised: the seven goal-line records and the quarantine policy are kept exactly as in S02; S04's goal-line sensitivity (leaving the validation goal-line shots out of scoring only, with no refit) is summarised from its own results. Limits listed: Wyscout integer-percentage coordinates and the assumed 105 x 68 m pitch; human tagging and the coarse body-part tag; one season; teams and players recur across the train, validation and test windows; a single chronological cut; match-level resampling that ignores dependence between matches of the same team, and selection among candidates that the interval does not reflect.

## 8. Confirmatory and exploratory

- **Confirmatory (pass/fail):** reproduction of the S04 validation numbers and identity checks; calibration bin totals; league denominators equal validation totals.
- **Exploratory and descriptive:** everything else. In particular league differences, bin-wise agreement, body-part contrasts and the illustrative predictions generate no pass/fail criterion. A weaker result in a league, bin or class is reported and the fixed decision stands.

## 9. Figures

Three static figures (PNG and SVG) in `figures/`: model comparison, calibration, and body-part with location. Each states the dataset, season, that it is validation only, and the CC BY 4.0 attribution. Colours: blue for the selected model, grey for the benchmark, orange for observed statistics. Value axes that do not start at zero say so on the figure.

## 10. Post hoc additions and departures

Four additions were made after the first S05 run had been seen. None changes the data, models, split, calculations, thresholds or the fixed decision:

1. The calibration section now names the bins whose interval misses the mean prediction and by how much (in the first run the selected model had 8 of 10 bins inside against 10 of 10 for the benchmark).
2. The body-part section now states where the location-only benchmark and the observed rate disagree for head/body shots (the benchmark averaged 20.7% against 11.7% observed in the first run).
3. The football explanation of raw versus adjusted body-part patterns was reworded to a hedged "plausible reading that these data cannot test", and the first-screen sentence on body part was softened from "because" to "largely because".
4. Figures were re-laid-out after inspecting the images (overlapping labels, a clipped title, an unformatted subtitle), and Wilson intervals are clamped to bracket the observed rate after a zero-goal bin broke error-bar drawing in synthetic tests.

Any further departure is listed here and in `reports/model_diagnostics.md`.
