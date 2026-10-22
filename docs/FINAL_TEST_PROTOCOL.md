# Final held-out test protocol

Written in slice S06, **before** any model has been scored on the final test window. This document fixes what the one final evaluation does, what it reports and when it must stop. It is a pre-registration for a single exploratory-but-disciplined evaluation, not a claim of confirmatory inference.

**Status: FINAL TEST NOT YET OPENED.** Nothing in S06 evaluates, reads row-level data from, or reveals a score from the test window. The execution (slice S07) needs its own review and an explicit instruction.

Wording that must stay accurate: *the test window has not been modelled or scored, and its row-level labels and features have not been accessed by S03 to S06; however, its aggregate goal count is indirectly inferable from earlier cohort audit totals.* This protocol never states or computes that count.

## 1. Research question and primary comparison

**Question.** On shots from a later, untouched period of the same season, does an expected-goals model that uses smooth location terms and the tagged body-part class predict goals better than a plain distance-and-angle logistic model, judged by proper scoring rules?

| Role | Model | Definition |
|---|---|---|
| **Primary candidate** | `nonlinear_location_body_part` | per-feature cubic `SplineTransformer` (5 quantile knots, constant extrapolation) on `distance_m` and `angle_rad`, standardised, plus a one-hot of the three body-part classes, then `LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)` |
| **Benchmark** | `distance_plus_angle` | `StandardScaler` and the same logistic regression on `distance_m` and `angle_rad` |
| Reference only | `constant_development_rate` | the goal fraction of the development labels for every shot. **Predeclared as included**; it is not part of the comparison |

- **Primary metric: log loss** (lower is better). **Co-primary supporting metric: Brier score** (lower is better). **Secondary: ROC AUC** (higher is better; reported as not available if the test window had one outcome class only).
- **Direction.** The primary candidate is "better" when its log loss is lower than the benchmark's. Differences are reported as primary minus benchmark, so negative favours the primary.
- **Reading rule**, identical to the validation-stage rule: an improvement requires a log-loss difference below zero with a 95% interval entirely below zero and a Brier difference below zero; the reverse pattern is "worse"; anything else is "no clear improvement". The rule decides the wording only. The intervals are exploratory (section 6).
- **The test window is not a tuning venue.** No model, feature, hyperparameter, threshold, bin edge, cohort rule or date is chosen, changed or revisited because of a test result. The model choice was frozen in `docs/MODEL_SELECTION_DECISION.md` before S05 diagnostics were computed; S05 diagnostics did not reselect it.

## 2. Cohort, data identity and label

| Item | Frozen value |
|---|---|
| Source | Wyscout open event data, Pappalardo & Massucco (2019), figshare, [doi:10.6084/m9.figshare.c.4415000.v5](https://doi.org/10.6084/m9.figshare.c.4415000.v5), CC BY 4.0 (paper: Scientific Data 6:236, [doi:10.1038/s41597-019-0247-7](https://doi.org/10.1038/s41597-019-0247-7)); data collected by Wyscout; filtered, quality-screened and augmented for this project |
| Scope | five domestic leagues, season 2017/18 (Wyscout competition ids 364 England, 412 France, 426 Germany, 524 Italy, 795 Spain), 1,826 matches |
| Shot table | `data/processed/wyscout_regular_shots.csv`, SHA-256 `58eb62c0f786406c7df1e457a981784ab10d0514ee5a3ec73c2119c83f0af96e`, 40,461 rows of which 40,425 eligible |
| S02 manifest | `data/processed/wyscout_regular_shots.manifest.json`, file SHA-256 `19ad674e454836a00001ef04bd34fd6c5312fd10645e662b4c3382907a4c4251` |
| Split assignment | `data/processed/wyscout_split_assignment.csv`, SHA-256 `2f995fc0337436619f794d8c64c7f8397bf81d8e28deca9f2add8369c881da15` |
| Split manifest | `data/processed/wyscout_split_manifest.json`, file SHA-256 `51a9d64a6d93eb060eee9ec495f98688ed39eeb91bd3356fed897ff933bc4aed` |

Hashes were taken from file bytes and from the published split manifest. No test row was parsed to obtain them.

- **Label.** `is_goal` is Wyscout tag 101 on the shot event itself. Goal tags on goalkeeper "Save attempt" events and own-goal tags (102) on other events never create or change a label.
- **Cohort.** Regular shots only (`eventId` 10, `subEventId` 100). Penalties and free-kick shots are counted but excluded. A row is eligible unless quarantined for one of: `MATCH_SCORE_MISMATCH`, `INVALID_POSITION`, `SUSPECT_POSITION` (`x_pct` below 20), `INVALID_PERIOD_OR_TIME`, `LABEL_UNKNOWN`, `GEOMETRY_ERROR`. Quarantine is frozen; no row is re-admitted and no coordinate is repaired or mirrored.
- **Windows (UTC match date; chronological; whole dates, hence whole matches).**

| Split | First date | Last date | Shots | Matches | Dates |
|---|---|---|---|---|---|
| train | 2017-08-04 | 2018-02-25 | 28,263 | 1,270 | 125 |
| validation | 2018-02-26 | 2018-04-13 | 5,509 | 257 | 26 |
| test | 2018-04-14 | 2018-05-20 | 6,653 | 298 | 27 |

  Cut points are 70% and 85% of the 178 distinct dates, rounded half up in integer arithmetic. The partition sizes are the published ones in the split manifest.
- **Pitch and coordinates.** Wyscout percentages are mapped to a 105 m by 68 m pitch with the attacking goal at x = 100% and a 7.32 m goal; distance is to the goal centre and the angle is the angle subtended by the posts. The pitch size is an assumption (the provider does not publish one); it was probed in S03 and not changed.
- **Known issues, frozen as they are.**
  - **Chelsea – Manchester City (matchId 2499781):** the only Goal tag in the match is on the goalkeeper's save event; the scoring shot is not tagged. All 18 regular-shot rows of the match are quarantined and no label was changed.
  - **Goal-line shots:** seven shots in the cohort are recorded at `x_pct` = 100; they are kept with a degenerate angle (0 or π). Their split membership is not recorded in this document.
  - **Tag 403** is "head/body", not a confirmed header. The body-part term describes association within one season of provider tagging, not a causal effect of how a shot is struck.
  - Label and tag noise from provider tagging, undetected coordinate errors, and shots whose frame is wrong at an ordinary-looking location cannot be ruled out.

## 3. Final fitting policy

1. The **development set** is the original training split plus the original validation split. It has no match and no calendar date in common with the test split (checked from identity columns before any fit).
2. Both fixed recipes (primary and benchmark) are **refitted once, from scratch, on the development set**: the standard scaler's mean and scale, the spline knots, the one-hot categories and the logistic weights are all estimated from development data only. The optional constant reference uses the development goal fraction.
3. Only after the fitted pipelines have been verified against the locked recipes (section 9) is the test partition read and **scored without further fitting**, on one scoring occasion. "Once" refers to that occasion, not to a single pass over the shot table: the source file is walked more than once in one evaluation (identity columns, then the development loaders, then the authorised test loaders). The earlier passes read identity columns and development rows only; tests prove that no held-out feature or label is accessed before both development models are fitted and verified.
4. **Why this differs from the validation-stage models.** The models compared in S03 to S05 were fitted on the training split alone so that the validation window was held out. The final evaluation spends the validation window as extra fitting data, because its purpose is to measure the best-supported version of the already-chosen recipes on unseen later data. The final scores are therefore not comparable to the validation scores as like-for-like: more training data, a later fitting window and a different evaluation window all change at once. No claim is made that a difference from the validation scores is due to any single cause.
5. No hyperparameter, knot count, regularisation strength, feature or threshold changes in the refit. A fitted model that fails to converge, produces non-finite or out-of-range probabilities, or has a body-part class missing from the development data stops the evaluation (section 7).

## 4. Outcome-access sequence

1. **S06 (this slice):** synthetic fixtures and code review only. The metadata-only preflight (`python -m expected_goals_lab.final_evaluation`) verifies hashes, counts, recipes, dependencies and the implementation inventory without opening a row of the test window, and without parsing the shot table.
2. **S07:** the only slice that may open the test rows, after an explicit instruction and a deliberate `--execute --confirm-final-test` invocation. The test rows are scored once, in one execution, after the development models are fitted and verified (see section 3 on file passes).
3. **No repeat comparisons for selection.** A second look at test results to choose among models, thresholds or settings is forbidden.
4. **Completion, failures and retries.** The execution log (`reports/final_test_execution_log.jsonl`) has three outcomes: `started`, `failed` and `completed`. `completed` is appended only after both aggregate files (the Markdown report and the JSON summary) have been written to temporary files and moved into place; if rendering or writing fails, nothing is marked completed and any output written by that attempt is removed. These situations each stop a new execution until an operator has diagnosed the cause and supplies a retry reason: a `failed` attempt, an attempt that `started` and recorded no outcome (a crash or interruption), and output files that exist without a `completed` entry. The reason is recorded together with the previous outcome. Restarting the command never retries silently, and a corrupt log stops everything. A retry reason is not permission to change any model, feature, cohort, threshold or setting; it must follow a fix to a demonstrated execution fault, not a poor result. Once an execution has completed, further runs are refused. The log holds outcomes, hashes of the two output files and exception class names only, never data.
5. **Reruns of unchanged code are not new evidence**; they would reproduce the same numbers.

## 5. Predetermined report

The report (`reports/final_test_evaluation.md`) contains these fields, and no others are added after the numbers are seen:

- number of test shots, goals and matches, and the test window dates;
- for each model: log loss, Brier score and ROC AUC;
- primary minus benchmark differences in log loss and Brier score with **paired match-cluster bootstrap 95% percentile intervals** (whole matches resampled with replacement; the same resamples for both models; 1,000 resamples; seed 20180513; exploratory), and the reading-rule wording that follows from them. These bootstrap settings are frozen: the runner takes them from the lock, offers no option to change them, and refuses any different value before a held-out row is read;
- expected goals (sum of predicted probabilities) against observed goals, for each model;
- a ten-bin equal-frequency calibration table for each of the two compared models, with Wilson intervals that are labelled as treating shots as independent;
- results per league, **descriptive only**: shots, goals, matches, log loss and Brier score of each model and their differences; no test or threshold is attached to a league, and per-league bootstrap intervals are neither computed nor reported (the only intervals in the evaluation are the overall paired match-cluster ones above), and AUC is shown only with at least 10 goals and 10 non-goals.

No shot-level-independent intervals are used for the primary comparison. Only the preselected models are evaluated. The report contains aggregates only, never a shot, a prediction row, a match identifier or a player.

## 6. Honesty requirements

The report and any write-up must state, in plain terms:

- one season, five leagues, one chronology; no claim about other seasons or leagues;
- the same teams (and players) appear in development and test windows;
- labels and body-part tags come from provider tagging, and tag 403 is not a confirmed header;
- the test window's aggregate goal count was indirectly inferable beforehand, so it was held out from scoring, not from all outcome information;
- model selection was exploratory and used the validation window;
- the match bootstrap does not capture dependence between matches of the same team or league, or the uncertainty of having chosen the primary among candidates, so its intervals are exploratory;
- **if the primary candidate does worse than the benchmark, that is published as the result and the choice is not changed.**

## 7. Stop policy (one-way)

The evaluation stops, writes no results and requests review if any of the following happens. It never "repairs and continues":

- a data hash, manifest hash, split-manifest hash, split assignment, row count, window boundary or per-split count differs from section 2;
- the shot table, manifests or assignment were modified (a hash differs from the lock);
- labels or features are invalid (non-binary labels, non-finite or out-of-range features, an unknown body-part class, a quarantine flag inconsistent with the manifest);
- development and test overlap in event, match or calendar date, or the splits are not strictly ordered in time;
- a dependency version, the lockfile, `pyproject.toml`, the protocol file or any file of the frozen inventory differs from the lock, or a fitted pipeline differs from its locked recipe;
- a model fails to converge or emits non-finite or out-of-range probabilities.

After a stop, neither the data, the cohort, the dates nor the models are altered to make the evaluation pass; the cause is reviewed first.

## 8. Git and data safety

- No raw archives, processed shot tables, row-level predictions or per-shot outputs are committed or published. The only artefacts are the aggregate report, an aggregate summary and the execution log (outcomes only).
- Published aggregates carry the CC BY 4.0 attribution with the DOI given in section 2, state that the data were filtered and augmented, and do not imply endorsement by the dataset authors or Wyscout.
- No StatsBomb data or outputs are used.
- Nothing is pushed or merged by the evaluation.

## 9. Frozen implementation inventory

The preflight compares SHA-256 hashes of the files below (text normalised to LF line endings, so a CRLF checkout hashes identically) with the values in `src/expected_goals_lab/final_lock.py`.

| File | SHA-256 |
|---|---|
| `pyproject.toml` | `ff1798cd2ec6fa75f5e23e40e59fb41d68c2ad516f22fb7ab0107044386a8dbf` |
| `uv.lock` | `c1dcff92b478985080586d56c0fe57012a2d0d6e7e0cdf19f6b6bf83e8037ccb` |
| `docs/MODEL_SELECTION_DECISION.md` | `4260cdff6694e4e8d820e51661206b0efc28410cffeee896e1a5481f529cd3cf` |
| `src/expected_goals_lab/geometry.py` | `7d9afb9a3506f6eb071da3239aafde243d19dd01f318c6746fa8db675991b79e` |
| `src/expected_goals_lab/wyscout.py` | `bcc8fe3359e5fd302e2171952d015eaa1ca74f49a8f02c8f250975c53ecf6333` |
| `src/expected_goals_lab/wyscout_shots.py` | `919e038376d8ef454f2699ecc7f997eb6682117f8b1d4cad639ef5a7a9d4a9a1` |
| `src/expected_goals_lab/shot_table.py` | `b2a3befb24801aa4221ad438a44e00ed5535c06615a4b8020e475bde51a441c5` |
| `src/expected_goals_lab/shot_diagnostics.py` | `bd213a1c0377e8f065a93c281441d023a8c74a5e3e193035f7748265957c2ab4` |
| `src/expected_goals_lab/build_shots.py` | `3a6b3b221ad2a893b57b32a5372429a4c2c3db4aad71d4ea2b5e7387db90bd4b` |
| `src/expected_goals_lab/dataset.py` | `e2846b531c9195c2d7f8b4cffee133738885d3538fe728b716c9e1bb156c6be1` |
| `src/expected_goals_lab/splits.py` | `c334d311e6d9fc74c02c3b2dae47a46f78d07f2993bf1946eb49e845b314dad5` |
| `src/expected_goals_lab/baselines.py` | `bba08ac4a8a80490ff85baa657fe13300b4ce0cc61281b8887f8bc040cb8df78` |
| `src/expected_goals_lab/scoring.py` | `fe2a90ffc0e2644f420de616fb41c688f58ea0af1bb9a4d73173c01f6494ada3` |
| `src/expected_goals_lab/candidate_data.py` | `bd02f0ea9d0f0051cb16c904119f9090645e6b73865ebe9099018d17a13618a3` |
| `src/expected_goals_lab/nonlinear.py` | `77a85df5b886f48d88d9aaa05ac686356a3672c9962d82aafa5fd4b75b49fc94` |
| `src/expected_goals_lab/evaluate_baselines.py` | `219f9ce897bd33446f3e375aeef96c2b717d54d96b206b6ddbbc0a2ae2a0cc1e` |
| `src/expected_goals_lab/evaluate_nonlinear.py` | `b32913dce7452746621c0f2979a88eb26b9e4995c3a2057c497dce9c3eceead6` |
| `src/expected_goals_lab/diagnostics.py` | `e54caf1d506fee134b1e8356b143a4f0e1d7f8caeeb68a797e640b5897788a3e` |
| `src/expected_goals_lab/audit.py` | `8b497dd8bfca5e70f88b63c512c25f964e7796350291f293936c74971f262c2a` |

Other locked values: model decision file `docs/MODEL_SELECTION_DECISION.md` above; the primary recipe SHA-256 `d0c467b8f1de0550be8512c62463186c3031a7ea77b320109f09e614372b6291` and the benchmark recipe SHA-256 `e2365f7a42f291ef5cbca22e95e7f89d41a59154941819091d6c71d6424b2fd1` are hashes of a canonical description (class names and parameters of every pipeline stage, and the column each stage reads); bootstrap seed 20180513 with 1000 resamples; packages as pinned in `uv.lock`: joblib 1.6.0, numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1, threadpoolctl 3.7.0 (Python 3.12).

**Entry points.** `uv run python -m expected_goals_lab.final_evaluation` (or `--dry-run`) is metadata-only. `--execute --confirm-final-test` runs the evaluation; both switches are required and it is switched off for the real data (`REAL_EXECUTION_ENABLED`) until S07.

**What needs renewed review.** Any change to a file in the inventory, to `final_evaluation.py` or `final_lock.py`, to the lockfile or the pinned package versions, to a locked hash or count, to this protocol, or to the model decision file. The only change expected between S06 and S07 is switching on `REAL_EXECUTION_ENABLED`, which the S07 instruction must authorise explicitly, quoting the reviewed commit. `final_evaluation.py` and `final_lock.py` are not in the inventory (they would have to contain their own hash); the S07 instruction pins the reviewed commit instead. The protocol's own hash is held in `final_lock.py`.
