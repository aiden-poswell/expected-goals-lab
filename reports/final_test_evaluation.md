# Final held-out test evaluation

**These results come from the single pre-registered final evaluation (`docs/FINAL_TEST_PROTOCOL.md`, SHA-256 `5e4cc0b2971261fa5c807c777804a1dae00394747453eb5dcb4dfd126b28fdae`). They are not a basis for choosing, tuning or changing any model, feature, cohort or date.** Rerunning unchanged code is a reproducibility check, not new independent evidence.

The test window was held out from scoring, not from all outcome information: its aggregate goal count was already inferable from published cohort totals. Model selection was exploratory and used the validation window. This is one season, five leagues, one chronological cut.

Generated 2026-10-21 11:56 UTC. Data: Pappalardo & Massucco (2019), *Soccer match event dataset*, figshare, CC BY 4.0, doi:10.6084/m9.figshare.c.4415000.v5; Pappalardo et al. (2019), Scientific Data 6:236, doi:10.1038/s41597-019-0247-7; data collected by Wyscout; processed for this project.

## Identity

| Item | Value |
|---|---|
| Shot table | `data/processed/wyscout_regular_shots.csv` SHA-256 `58eb62c0f786406c7df1e457a981784ab10d0514ee5a3ec73c2119c83f0af96e` |
| Split assignment SHA-256 | `2f995fc0337436619f794d8c64c7f8397bf81d8e28deca9f2add8369c881da15` |
| Lockfile SHA-256 | `c1dcff92b478985080586d56c0fe57012a2d0d6e7e0cdf19f6b6bf83e8037ccb` |
| Git commit | `bbf38a858691b7000e2074516fb51ae48ecfb3f7` |
| scikit-learn | 1.9.1 |
| Bootstrap | 1000 match-level resamples, seed 20180513 |

Development (original train + validation, used only to fit): 33,772 shots, 3,530 goals, 1,527 matches on 151 dates. Test: 6,653 shots, 738 goals, 298 matches on 27 dates (2018-04-14 to 2018-05-20).

## Result

Primary comparison (`nonlinear_location_body_part` versus `distance_plus_angle`): improvement over distance_plus_angle: log-loss interval below zero and Brier estimate below zero.

| Model | Shots | Goals | Log loss | Brier | ROC AUC | Expected goals |
|---|---|---|---|---|---|---|
| nonlinear_location_body_part | 6653 | 738 | 0.2905 | 0.08447 | 0.787 | 714.6 |
| distance_plus_angle | 6653 | 738 | 0.2998 | 0.08752 | 0.767 | 711.9 |
| constant_development_rate (reference only) | 6653 | 738 | 0.3487 | 0.09866 | 0.500 | 695.4 |

Lower log loss and Brier score are better; higher AUC is better. Log loss is the primary metric, Brier the co-primary supporting score, AUC secondary.

Paired match-cluster bootstrap, primary minus benchmark (exploratory 95% percentile intervals; resampling whole matches):

| Metric | Difference | 95% interval |
|---|---|---|
| Log loss | -0.0093 | [-0.0120, -0.0066] |
| Brier | -0.00305 | [-0.00393, -0.00215] |

## Expected versus observed goals

| Model | Expected goals | Observed goals | Observed / expected |
|---|---|---|---|
| nonlinear_location_body_part | 714.6 | 738 | 1.033 |
| distance_plus_angle | 711.9 | 738 | 1.037 |
| constant_development_rate | 695.4 | 738 | 1.061 |

## Calibration (ten equal-frequency bins; no calibration fitted)

The Wilson intervals treat shots as independent and so understate uncertainty.

**nonlinear_location_body_part**

| Bin | Shots | Goals | Mean prediction | Observed rate | 95% interval |
|---|---|---|---|---|---|
| 1 | 666 | 11 | 0.0171 | 0.0165 | [0.0092, 0.0293] |
| 2 | 666 | 14 | 0.0233 | 0.0210 | [0.0126, 0.0350] |
| 3 | 666 | 21 | 0.0318 | 0.0315 | [0.0207, 0.0477] |
| 4 | 665 | 30 | 0.0447 | 0.0451 | [0.0318, 0.0637] |
| 5 | 665 | 35 | 0.0630 | 0.0526 | [0.0381, 0.0723] |
| 6 | 665 | 64 | 0.0823 | 0.0962 | [0.0761, 0.1210] |
| 7 | 665 | 51 | 0.1054 | 0.0767 | [0.0588, 0.0994] |
| 8 | 665 | 110 | 0.1395 | 0.1654 | [0.1391, 0.1956] |
| 9 | 665 | 135 | 0.1898 | 0.2030 | [0.1742, 0.2352] |
| 10 | 665 | 267 | 0.3776 | 0.4015 | [0.3649, 0.4392] |

**distance_plus_angle**

| Bin | Shots | Goals | Mean prediction | Observed rate | 95% interval |
|---|---|---|---|---|---|
| 1 | 666 | 8 | 0.0168 | 0.0120 | [0.0061, 0.0235] |
| 2 | 666 | 14 | 0.0272 | 0.0210 | [0.0126, 0.0350] |
| 3 | 666 | 19 | 0.0372 | 0.0285 | [0.0183, 0.0441] |
| 4 | 665 | 30 | 0.0509 | 0.0451 | [0.0318, 0.0637] |
| 5 | 665 | 51 | 0.0679 | 0.0767 | [0.0588, 0.0994] |
| 6 | 665 | 61 | 0.0843 | 0.0917 | [0.0721, 0.1161] |
| 7 | 665 | 92 | 0.1073 | 0.1383 | [0.1142, 0.1667] |
| 8 | 665 | 99 | 0.1384 | 0.1489 | [0.1238, 0.1779] |
| 9 | 665 | 126 | 0.1869 | 0.1895 | [0.1615, 0.2210] |
| 10 | 665 | 238 | 0.3533 | 0.3579 | [0.3224, 0.3951] |

## By league (descriptive only)

Differences are primary minus benchmark (negative favours the primary). Per-league bootstrap intervals are neither computed nor reported, and no threshold or significance statement is attached; denominators are shown. AUC appears only with at least 10 goals and 10 non-goals.

| League | Shots | Goals | Matches | LL benchmark | LL primary | LL diff | Brier benchmark | Brier primary | Brier diff | AUC benchmark | AUC primary |
|---|---|---|---|---|---|---|---|---|---|---|---|
| English first division | 1201 | 127 | 56 | 0.2900 | 0.2810 | -0.0090 | 0.08368 | 0.08118 | -0.00250 | 0.768 | 0.790 |
| French first division | 1376 | 153 | 59 | 0.3119 | 0.2996 | -0.0123 | 0.09156 | 0.08774 | -0.00382 | 0.746 | 0.773 |
| German first division | 980 | 133 | 44 | 0.3428 | 0.3313 | -0.0115 | 0.10395 | 0.09949 | -0.00445 | 0.773 | 0.788 |
| Italian first division | 1562 | 159 | 70 | 0.2717 | 0.2674 | -0.0043 | 0.07820 | 0.07685 | -0.00135 | 0.795 | 0.804 |
| Spanish first division | 1534 | 166 | 69 | 0.2978 | 0.2871 | -0.0106 | 0.08590 | 0.08226 | -0.00364 | 0.752 | 0.776 |

## Limits

- One season (2017/18), five leagues, one chronological cut; no claim about other seasons or leagues.
- The same teams and players appear in development and test.
- Labels and the body-part tag come from human provider tagging; Wyscout tag 403 is "head/body", not a confirmed header.
- The match bootstrap ignores dependence between matches of one team or league and the selection of the primary model among candidates.
- Known data issues are frozen in the protocol: one match with a score discrepancy is quarantined, and shots recorded very far from the attacked goal (x below 20% of pitch length) are quarantined as suspect.
- If the primary model did worse than the benchmark, that is the result; the choice is not revisited.

## Execution record

1 execution-log entries existed when this report was rendered. The `completed` entry is appended only after this report and its JSON summary are both in place; retries carry a recorded reason in `reports/final_test_execution_log.jsonl`.
