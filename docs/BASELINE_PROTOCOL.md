# S03 protocol: chronological splits and xG baselines

Written and fixed before any baseline was fitted. The split boundaries below were produced by `uv run python -m expected_goals_lab.splits`, which reads no label and no feature. Anything decided after seeing a validation result is listed in section 9.

## 1. Question

How much better do shot distance and goal-mouth angle predict goals than a constant, training-set goal rate, on **later matches of the same 2017/18 season**?

This is a within-season forecasting question for five leagues. It says nothing about other seasons, other leagues or club-grade accuracy.

## 2. Input

The S02 shot table `data/processed/wyscout_regular_shots.csv` and its manifest, never the raw provider events. On every run:

- the CSV's SHA-256 must equal the manifest's;
- the manifest must carry `schema_version` 1 and the exact column list, column roles, feature allowlist, candidate features and target that S02 defines;
- the CSV header must equal that column list; `event_id` must be unique; `eligible` must be 0 or 1; the row and eligible-row counts must equal the manifest's; eligible rows must have no exclusion reason and quarantined rows must have one;
- on the open splits, `is_goal` must be 0 or 1 and `distance_m`, `angle_rad` finite and in range.

Any failure stops the run (exit code 1; exit code 2 if the files are absent). Quarantined rows are discarded when the identity pass reads them and never reach a model.

## 3. Model inputs

Only `distance_m` and `angle_rad` (S02's `FEATURE_ALLOWLIST`). Asking for any other column, including `body_part_class`, identifiers, flags, the target, position columns, period and time, raises `FeatureNotAllowedError`. Identifiers are used only to assign splits and to group the bootstrap.

## 4. Split

- Unit: UTC calendar dates of matches that have at least one eligible shot. S02's `match_datetime_utc` is the provider's `dateutc`.
- 178 distinct dates, sorted. Cut points by integer arithmetic: train gets the first `(70 x 178 + 50) // 100 = 125`; validation gets up to `(85 x 178 + 50) // 100 = 151`, i.e. 26 more; test gets the last 27.
- Every match and shot takes the split of its date. Dates are distinct, so there are no ties; no date, match or shot straddles a boundary.
- No shuffling and no shot-level random split.

| Split | Dates | First | Last | Eligible shots | Matches |
|---|---|---|---|---|---|
| train | 125 | 2017-08-04 | 2018-02-25 | 28,263 | 1,270 |
| validation | 26 | 2018-02-26 | 2018-04-13 | 5,509 | 257 |
| test | 27 | 2018-04-14 | 2018-05-20 | 6,653 | 298 |

Counts by league are in `reports/baseline_validation.md`. The shares of shots (69.9%, 13.6%, 16.5%) differ from the 70/15/15 shares of dates because matchdays differ in size and the end of a season is more crowded. The boundaries were not adjusted.

Teams and players appear in all three windows, as they would if a model were refitted during a season and used for the remaining matches. The test window is the end of the season, when incentives (title, relegation, nothing to play for) may differ from earlier months; validation and test therefore need not resemble each other.

## 5. The sealed test split

**Precise status (corrected after S03 review).** The test window has not been modelled or scored, and its row-level labels/features have not been accessed by S03; however, its aggregate goal count is indirectly inferable from earlier cohort audit totals. S02's whole-cohort report publishes the eligible goal total, and S03 reports train and validation goal totals, so a reader can subtract. The test window is therefore **not entirely outcome-blind**. The mechanism below prevents row-level access, scoring and any test-derived statistic from entering S03; it does not hide that aggregate. No explicit test goal count is stated or computed anywhere in S03 or later slices.

In S03 the test split's row-level labels, features, locations and predictions, and any statistic derived from them, are not read, computed, shown or logged. Mechanism:

1. `dataset.read_identity` reads only `event_id`, `match_id`, `competition_id`, `match_datetime_utc`, `eligible`, `exclusion_reasons`.
2. `dataset.load_partitions` parses features and labels only for rows whose event is in a requested open split, and raises `SealedSplitError` if `test` is requested. For other rows no further column is read.
3. The test split is reported only through counts of rows, matches and dates by league, and its date range.
4. The test suite replaces the CSV reader with a spy that raises if a feature, location or target column of a test-date row is read, then runs the whole evaluation.

The SHA-256 check reads every byte of the file but interprets none of it. A later, separately approved release decision may unseal the test split once every candidate model and metric is fixed.

## 6. Candidate models, fixed in advance

All are fitted on **train** only and scored on **validation** only. No hyperparameter is searched.

| Name | Prediction |
|---|---|
| `constant_train_rate` | the fraction of goals among eligible train shots, for every shot |
| `distance_only` | logistic regression on `distance_m` |
| `distance_plus_angle` | logistic regression on `distance_m` and `angle_rad` |

Logistic models: `Pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000))` from scikit-learn, with its default L2 penalty. `C = 1.0` is the library default, chosen as a stated convention, not tuned; with about 28,000 training shots and two standardised inputs the penalty is mild. The scaler is fitted inside the pipeline on train rows only. `lbfgs` is deterministic, so there is no model seed.

## 7. Metrics and uncertainty (validation only)

- Primary: log loss and Brier score (lower is better). Predictions are clipped to `[1e-15, 1 - 1e-15]` for the logarithm. Secondary: ROC AUC. Not accuracy.
- Sample size and goal count are always shown. Train in-sample scores are also shown as a check for overfitting.
- **Match-cluster bootstrap** of validation matches: 1,000 resamples, seed 20180513, each resample draws the validation matches with replacement and keeps every shot of a drawn match. Reported: the point estimate and the 2.5th and 97.5th percentiles of the difference in log loss and in Brier score for (distance + angle) minus distance-only, (distance + angle) minus constant, and distance-only minus constant. These intervals are descriptive: they reflect which validation matches were drawn, not a guarantee for other matches or seasons.
- By league: the same metrics only for leagues with at least 1,000 validation shots and at least 100 validation goals; smaller ones are not shown. Per-league intervals use the same method on that league's matches.
- Calibration: validation predictions of `distance_plus_angle` in 10 equal-frequency bins, with counts, mean prediction and observed goal rate. It is a description, not evidence of calibration.

## 8. Pre-specified sensitivity and flagged edge case

- **Pitch size.** The three models are refitted on train and scored on validation with `distance_m` and `angle_rad` recomputed from the S02 raw percent coordinates for a 100 x 64 m and a 110 x 70 m pitch (goal 7.32 m). The constant baseline does not change. Reference stays 105 x 68. The test split is not used.
- **Goal-line shots.** S02 flags seven shots with `x_pct = 100` (angle 180 or 0 degrees, a degenerate value). They remain in the locked cohort and enter the models as ordinary rows. S03 counts them in train and validation, shows their angle values, and does not alter the cohort. A sensitivity analysis that excludes or clips them is recommended for a later slice and is not run here.

## 9. Departures and decisions made after seeing results

After the first validation run, one sentence was added to the report generator (the summary on whether adding angle is distinguishable from no change, written from the intervals) because log loss and Brier disagreed. It changes the report's wording only: no model, split, metric, resample count, seed or threshold changed. Any other departure is listed in `reports/baseline_validation.md`.

## 10. Limits

One season. Validation (26 dates, late February to mid-April) is a single contiguous window. Distance and angle are approximations of physical values under an assumed pitch. Labels depend on provider tagging.
