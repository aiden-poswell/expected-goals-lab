# Shot-origin provenance audit (S08A)

Development-only diagnosis of the coordinates fed to the xG model, written under `docs/SHOT_ORIGIN_AUDIT_PROTOCOL.md`. It changes no model, cohort, split or frozen artefact, and no held-out (test-window) feature, label, prediction or match outcome is inspected, aggregated or printed. The event archive is read in two passes (the first collects the regular shots of development matches and the pass-end links, the second every event of the case-study and goal-line matches); each pass parses every event of the five league files into memory one at a time, so events of held-out matches are loaded too, but fields other than `matchId` are examined only for events of development matches.

## Finding: PASS (evidence-based review judgement; the mechanical screen alone is not enough)

### Mechanical screen (computed, reproducible)

**PASS.** The screen checks reconciliation with `positions[0]`, the goal flag and body-part consistency, the non-goal share of x = 95-99 shots and the goal-line subset, against the criteria fixed in the protocol. It cannot tell an origin from a destination.

- all PASS criteria met

### Manual review (a dated judgement, not computed): no destination evidence found

Reviewed 2026-10-22. Basis:

- the Wyscout documentation describes positions as the place an event 'started/ended (only start if not a movement)', although it does not say which entry is the start of a shot
- the second entry of every development regular shot is a placeholder, (0, 0) or (100, 100), with the same goal rate for both
- where a same-team pass precedes the shot, the shot origin is usually exactly the end position of that pass, for goals and non-goals alike
- the two goals of the case-study match have plausible source-event sequences and agree with a public match report in team and minute
- an independent review of the S08A report accepted this as internal-consistency evidence, not as proof of physical ground truth

- Caveats that stand whatever the finding: the Wyscout page does not state which `positions` entry is the start of a shot (the first-entry reading is supported by the documented 'start' semantics and by the evidence below); 6 development rows sit exactly on the goal line; the audit tests consistency, not physical truth; and it covers the development partition only.

Development rows audited: 33,772 eligible regular shots (28,263 train, 5,509 validation). Reconciled with the source event's `positions[0]`: 33,772; mismatches 0; not found in the archive 0; goal flag differs from tag 101 on the event: 0; body-part class differs from tags 401-403: 0.

## 1. What the dataset calls a shot location

- **Documentation.** Wyscout (support.wyscout.com/matches-wyid-events): `positions` is "Positions in which this event started/ended (only start if not a movement)"; x and y are "percentage from the left corner of the attacking team". The page says nothing more about shot events and does not tie the goal-mouth tags 1201-1223 to the coordinate fields. Which of two position entries is the start for a shot is therefore **not stated** in that page; the first-entry reading rests on the documented 'start' semantics, the schema audit of S01B and the reconciliation below. The Scientific Data paper was consulted in S01B and not re-fetched in S08A (publisher page redirects to authentication).
- **Code.** `src/expected_goals_lab/wyscout_shots.py` `parse_shot` line 161 -> `_start_position` (lines 219-226): first entry of `positions` only, integers in 0-100. `shot_table.make_row` copies it to `x_pct`/`y_pct` and derives distance and angle with `geometry.shot_geometry`. No tag is read to obtain a position.

**Second `positions` entry on every regular shot of the development matches** (goal flag from tag 101 on the shot):

| Group | Shots | Goals | Non-goals | Goal rate |
|---|---|---|---|---|
| (0, 0) placeholder | 18772 | 1985 | 16787 | 10.6% |
| (100, 100) placeholder | 15017 | 1548 | 13469 | 10.3% |
| other | 0 | 0 | 0 | n/a |
| absent | 0 | 0 | 0 | n/a |

**Goal-mouth tags (post-shot information, never read by the parser):**

| Group | Shots | Goals | Non-goals | Goal rate |
|---|---|---|---|---|
| has a goal-mouth tag (1201-1223) | 25817 | 3532 | 22285 | 13.7% |
| no goal-mouth tag | 7972 | 1 | 7971 | 0.0% |

**Does the origin sit where the team's previous pass ended?** For development shots preceded within 10 s, in the same period, by a pass of the same team that has a real end position, the distance (percentage units; 1 unit is about 1 m along and 0.7 m across the pitch) between the shot origin and that pass's end. Short distances are what an origin (where the shooter received the ball) predicts; a destination would not follow the pass.

| x band | Outcome | Shots | With a preceding pass | Median dist | Within 5 | Within 10 |
|---|---|---|---|---|---|---|
| x = 100 | goals | 4 | 4 | 8.1 | 50% | 50% |
| x = 100 | non-goals | 2 | 2 | 13.4 | 0% | 50% |
| x = 95-99 | goals | 801 | 602 | 0.0 | 71% | 80% |
| x = 95-99 | non-goals | 1856 | 1268 | 0.0 | 66% | 78% |
| x < 95 | goals | 2725 | 2290 | 0.0 | 67% | 77% |
| x < 95 | non-goals | 28384 | 22817 | 0.0 | 62% | 73% |

**Reading.** Where a preceding pass exists, the shot origin is very often exactly the pass's end (median distance 0.0 for goals and 0.0 for non-goals at x < 95; 0.0 and 0.0 at x = 95-99). The share within 5 units is 67% (goals) against 62% (non-goals) at x < 95 and 71% against 66% at x = 95-99: similar for goals and non-goals, which is what a start position predicts and not what an outcome-dependent position would. The x = 100 rows are too few to say anything either way. This is consistency evidence about what the first `positions` entry is; it is not proof against the provider's coding, and the Wyscout page does not say which entry is the start of a shot.

## 2. Match 2516947 (Borussia Dortmund v Augsburg, 2018-02-26), event by event

Sorted by period and time. All rows come from the shot table with the source event's full `positions` array beside them; xG is from the model fitted on the original training split only. Flags are diagnostic only.

| # | Clock | Event id | Team | Player | Outcome | Body | positions (all) | Origin x,y % | Dist m | Angle deg | xG | Flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1H 00:41 | 231046361 | 2447 | 14789 | no goal | right_foot | (87, 36) ; (0, 0) | (87, 36) | 16.64 | 20.8 | 0.092 | - |
| 2 | 1H 06:47 | 231047310 | 2481 | 65180 | no goal | left_foot | (89, 60) ; (100, 100) | (89, 60) | 13.40 | 27.0 | 0.162 | - |
| 3 | 1H 13:06 | 231047433 | 2481 | 65180 | no goal | head_or_body | (94, 66) ; (100, 100) | (94, 66) | 12.57 | 17.7 | 0.048 | - |
| 4 | 1H 15:08 | 231046534 | 2447 | 15024 | GOAL | right_foot | (97, 54) ; (0, 0) | (97, 54) | 4.16 | 80.3 | 0.661 | x>=95 |
| 5 | 1H 24:32 | 231047620 | 2481 | 224965 | no goal | right_foot | (91, 76) ; (100, 100) | (91, 76) | 20.05 | 10.1 | 0.034 | - |
| 6 | 1H 31:19 | 231047697 | 2481 | 16432 | no goal | right_foot | (76, 31) ; (100, 100) | (76, 31) | 28.32 | 13.2 | 0.021 | - |
| 7 | 1H 35:11 | 231046840 | 2447 | 15024 | no goal | right_foot | (73, 37) ; (0, 0) | (73, 37) | 29.70 | 13.4 | 0.020 | - |
| 8 | 1H 38:58 | 231046874 | 2447 | 14789 | no goal | right_foot | (83, 51) ; (0, 0) | (83, 51) | 17.86 | 23.1 | 0.089 | - |
| 9 | 2H 09:58 | 231047971 | 2481 | 14997 | no goal | head_or_body | (91, 31) ; (100, 100) | (91, 31) | 16.01 | 15.9 | 0.030 | - |
| 10 | 2H 12:22 | 231048008 | 2481 | 14997 | no goal | right_foot | (92, 69) ; (100, 100) | (92, 69) | 15.41 | 15.3 | 0.073 | - |
| 11 | 2H 19:22 | 231048066 | 2481 | 65075 | no goal | head_or_body | (89, 46) ; (100, 100) | (89, 46) | 11.87 | 33.6 | 0.088 | - |
| 12 | 2H 21:08 | 231047139 | 2447 | 272363 | no goal | right_foot | (76, 76) ; (0, 0) | (76, 76) | 30.78 | 11.2 | 0.018 | - |
| 13 | 2H 23:39 | 231047170 | 2447 | 14789 | no goal | right_foot | (90, 69) ; (0, 0) | (90, 69) | 16.65 | 16.2 | 0.069 | - |
| 14 | 2H 25:06 | 231048134 | 2481 | 16432 | no goal | right_foot | (74, 39) ; (100, 100) | (74, 39) | 28.31 | 14.2 | 0.023 | - |
| 15 | 2H 27:03 | 231048161 | 2481 | 366217 | no goal | head_or_body | (90, 48) ; (100, 100) | (90, 48) | 10.59 | 37.9 | 0.108 | - |
| 16 | 2H 27:05 | 231048162 | 2481 | 366217 | GOAL | right_foot | (97, 46) ; (100, 100) | (97, 46) | 4.16 | 80.3 | 0.661 | x>=95 |
| 17 | 2H 29:55 | 231047255 | 2447 | 272363 | no goal | right_foot | (75, 37) ; (0, 0) | (75, 37) | 27.70 | 14.3 | 0.023 | - |
| 18 | 2H 37:57 | 231047446 | 2447 | 38093 | no goal | left_foot | (88, 44) ; (0, 0) | (88, 44) | 13.24 | 29.7 | 0.173 | - |
| 19 | 2H 40:45 | 231047499 | 2447 | 272363 | no goal | right_foot | (87, 46) ; (0, 0) | (87, 46) | 13.92 | 29.0 | 0.151 | - |
| 20 | 2H 48:23 | 231048325 | 2481 | 14900 | no goal | left_foot | (87, 64) ; (100, 100) | (87, 64) | 16.64 | 20.8 | 0.098 | - |

Each printed origin equals the first entry of the source event's `positions` (checked for every row above and for all development shots in the reconciliation line).

### Goal, event 231046534 (1H 15:08, team 2447)

Own tags: [101, 201, 402, 1209, 1801]; goal-mouth tags [1209]. Origin (97, 54) %, 4.16 m from the goal centre, angle 80.3 degrees, xG 0.661. Neighbouring events in the match (three before, two after):

| Event id | Clock | Event | Sub-event | Team | positions |  |
|---|---|---|---|---|---|---|
| 231046531 | 1H 15:02 | Others on the ball | Acceleration | 2447 | (64, 42) ; (90, 76) |  |
| 231046533 | 1H 15:06 | Pass | Cross | 2447 | (90, 76) ; (97, 54) |  |
| 231047473 | 1H 15:07 | Others on the ball | Touch | 2481 | (3, 50) ; (3, 46) |  |
| 231046534 | 1H 15:08 | Shot | Shot | 2447 | (97, 54) ; (0, 0) | <- the shot |
| 231047474 | 1H 15:09 | Save attempt | Reflexes | 2481 | (100, 100) ; (10, 24) |  |
| 231047477 | 1H 16:21 | Pass | Simple pass | 2481 | (28, 74) ; (40, 71) |  |

### Goal, event 231048162 (2H 27:05, team 2481)

Own tags: [101, 201, 402, 1203, 1801]; goal-mouth tags [1203]. Origin (97, 46) %, 4.16 m from the goal centre, angle 80.3 degrees, xG 0.661. Neighbouring events in the match (three before, two after):

| Event id | Clock | Event | Sub-event | Team | positions |  |
|---|---|---|---|---|---|---|
| 231048160 | 2H 27:01 | Free Kick | Corner | 2481 | (100, 100) ; (90, 48) |  |
| 231048161 | 2H 27:03 | Shot | Shot | 2481 | (90, 48) ; (100, 100) |  |
| 231047226 | 2H 27:04 | Save attempt | Reflexes | 2447 | (0, 0) ; (10, 52) |  |
| 231048162 | 2H 27:05 | Shot | Shot | 2481 | (97, 46) ; (100, 100) | <- the shot |
| 231047228 | 2H 27:05 | Save attempt | Reflexes | 2447 | (0, 0) ; (0, 0) |  |
| 231047232 | 2H 28:14 | Pass | Simple pass | 2447 | (42, 62) ; (35, 75) |  |

### External corroboration note (outside the data; it overrides nothing)

A public match report (Sporting Life, https://www.sportinglife.com/football/live/38908/borussia-dortmund-vs-fc-augsburg, fetched 2026-10-22) gives Borussia Dortmund 1-1 Augsburg on 26 February 2018, with Marco Reus (Dortmund) scoring in the 16th minute and Kevin Danso (Augsburg) in the 73rd. The dataset's goal events are Dortmund (team 2447) at 1H 15:08 and Augsburg (team 2481) at 2H 27:05, which is the 16th minute and the 73rd (45 + 27). Team and minute agree. The player ids of the two goals were not matched to names (no players file was used), and the report gives no positions, so it neither confirms nor refutes the coordinates in the table. The data show the Augsburg goal coming two seconds after a saved headed shot at 2H 27:03, which followed a corner at 2H 27:01.


### Why the markers look so large

The chart draws marker area proportional to xG. On the chart's pitch scale the diameter of a marker, in metres of pitch, was, at the S08 scale (2000 points squared per unit xG): xG 0.05: 2.4 m, xG 0.20: 4.8 m, xG 0.50: 7.6 m, xG 0.60: 8.3 m. A marker for a goal at xG of about 0.6 was therefore wider than the 5.5 m six-yard box is deep, so the circle covered far more pitch than the distance of the recorded origin suggests. That is a display effect, separate from whether the coordinate is right. At the current scale (800 points squared per unit): xG 0.05: 1.5 m, xG 0.20: 3.0 m, xG 0.50: 4.8 m, xG 0.60: 5.3 m. Both goals of this match have an estimated chance of 0.661, 0.661 from a recorded distance of 4.16 m, 4.16 m and an angle of 80 degrees, 80 degrees: a very high probability is what a model built on distance and angle gives to a shot about 4 m from goal that sees most of the goal mouth (compare the goal rate of shots recorded at x = 95-99 inside the goal mouth in table 3.3), so the large circles reflect the recorded closeness, not something questionable in the coordinates.

## 3. Development-only outcome screen

### 3.1 Origin x by band, all leagues

| x band (%) | Shots | Goals | Non-goals | Goal rate |
|---|---|---|---|---|
| x = 100 | 6 | 4 | 2 | 66.7% |
| x = 99 | 69 | 32 | 37 | 46.4% |
| x = 98 | 276 | 108 | 168 | 39.1% |
| x = 95-97 | 2312 | 661 | 1651 | 28.6% |
| x = 90-94 | 8790 | 1412 | 7378 | 16.1% |
| x < 90 | 22319 | 1313 | 21006 | 5.9% |

### 3.2 The same, by league

**English first division**

| x band (%) | Shots | Goals | Non-goals | Goal rate |
|---|---|---|---|---|
| x = 100 | 1 | 1 | 0 | 100.0% |
| x = 99 | 9 | 5 | 4 | 55.6% |
| x = 98 | 48 | 23 | 25 | 47.9% |
| x = 95-97 | 546 | 153 | 393 | 28.0% |
| x = 90-94 | 1858 | 323 | 1535 | 17.4% |
| x < 90 | 4770 | 282 | 4488 | 5.9% |

**French first division**

| x band (%) | Shots | Goals | Non-goals | Goal rate |
|---|---|---|---|---|
| x = 100 | 1 | 0 | 1 | 0.0% |
| x = 99 | 15 | 9 | 6 | 60.0% |
| x = 98 | 57 | 21 | 36 | 36.8% |
| x = 95-97 | 420 | 115 | 305 | 27.4% |
| x = 90-94 | 1758 | 275 | 1483 | 15.6% |
| x < 90 | 4696 | 300 | 4396 | 6.4% |

**German first division**

| x band (%) | Shots | Goals | Non-goals | Goal rate |
|---|---|---|---|---|
| x = 100 | 2 | 1 | 1 | 50.0% |
| x = 99 | 11 | 5 | 6 | 45.5% |
| x = 98 | 54 | 21 | 33 | 38.9% |
| x = 95-97 | 374 | 120 | 254 | 32.1% |
| x = 90-94 | 1595 | 240 | 1355 | 15.0% |
| x < 90 | 3872 | 225 | 3647 | 5.8% |

**Italian first division**

| x band (%) | Shots | Goals | Non-goals | Goal rate |
|---|---|---|---|---|
| x = 100 | 1 | 1 | 0 | 100.0% |
| x = 99 | 16 | 6 | 10 | 37.5% |
| x = 98 | 68 | 24 | 44 | 35.3% |
| x = 95-97 | 526 | 130 | 396 | 24.7% |
| x = 90-94 | 1770 | 284 | 1486 | 16.0% |
| x < 90 | 4862 | 249 | 4613 | 5.1% |

**Spanish first division**

| x band (%) | Shots | Goals | Non-goals | Goal rate |
|---|---|---|---|---|
| x = 100 | 1 | 1 | 0 | 100.0% |
| x = 99 | 18 | 7 | 11 | 38.9% |
| x = 98 | 49 | 19 | 30 | 38.8% |
| x = 95-97 | 446 | 143 | 303 | 32.1% |
| x = 90-94 | 1809 | 290 | 1519 | 16.0% |
| x < 90 | 4119 | 257 | 3862 | 6.2% |

### 3.3 Goal-line and centre-of-goal positions

| Position group | Shots | Goals | Non-goals | Goal rate |
|---|---|---|---|---|
| (100, 50) exactly | 0 | 0 | 0 | n/a |
| x = 100, other y inside the goal mouth | 3 | 3 | 0 | 100.0% |
| x = 100, y outside | 3 | 1 | 2 | 33.3% |
| x = 95-99, y inside the goal mouth | 956 | 489 | 467 | 51.2% |
| x = 95-99, y outside | 1701 | 312 | 1389 | 18.3% |
| x < 95 | 31109 | 2725 | 28384 | 8.8% |

### 3.4 By body-part class

**right_foot**

| x band (%) | Shots | Goals | Non-goals | Goal rate |
|---|---|---|---|---|
| x = 100 | 4 | 3 | 1 | 75.0% |
| x = 99 | 31 | 14 | 17 | 45.2% |
| x = 98 | 130 | 53 | 77 | 40.8% |
| x = 95-97 | 913 | 261 | 652 | 28.6% |
| x = 90-94 | 3327 | 615 | 2712 | 18.5% |
| x < 90 | 12838 | 758 | 12080 | 5.9% |

**left_foot**

| x band (%) | Shots | Goals | Non-goals | Goal rate |
|---|---|---|---|---|
| x = 100 | 2 | 1 | 1 | 50.0% |
| x = 99 | 24 | 13 | 11 | 54.2% |
| x = 98 | 73 | 28 | 45 | 38.4% |
| x = 95-97 | 595 | 200 | 395 | 33.6% |
| x = 90-94 | 2234 | 391 | 1843 | 17.5% |
| x < 90 | 8171 | 503 | 7668 | 6.2% |

**head_or_body**

| x band (%) | Shots | Goals | Non-goals | Goal rate |
|---|---|---|---|---|
| x = 100 | 0 | 0 | 0 | n/a |
| x = 99 | 14 | 5 | 9 | 35.7% |
| x = 98 | 73 | 27 | 46 | 37.0% |
| x = 95-97 | 804 | 200 | 604 | 24.9% |
| x = 90-94 | 3229 | 406 | 2823 | 12.6% |
| x < 90 | 1310 | 52 | 1258 | 4.0% |

### 3.5 Where goals and non-goals are taken

| x band (%) | Shots | Goals | Non-goals | Goal rate |
|---|---|---|---|---|
| x = 0-9 | 0 | 0 | 0 | n/a |
| x = 10-19 | 0 | 0 | 0 | n/a |
| x = 20-29 | 1 | 0 | 1 | 0.0% |
| x = 30-39 | 8 | 0 | 8 | 0.0% |
| x = 40-49 | 45 | 1 | 44 | 2.2% |
| x = 50-59 | 62 | 3 | 59 | 4.8% |
| x = 60-69 | 691 | 14 | 677 | 2.0% |
| x = 70-79 | 8699 | 211 | 8488 | 2.4% |
| x = 80-89 | 12813 | 1084 | 11729 | 8.5% |
| x = 90-100 | 11453 | 2217 | 9236 | 19.4% |

Distance to the goal centre (m), percentiles:

| Group | Shots | 5% | 25% | 50% | 75% | 95% |
|---|---|---|---|---|---|---|
| goals | 3530 | 4.0 | 7.5 | 11.5 | 15.4 | 25.0 |
| non-goals | 30242 | 7.8 | 12.9 | 18.0 | 25.5 | 32.4 |

### 3.6 Goal-line records (x = 100) in the development partition

6 records, 0.018% of development rows: 4 goals and 2 non-goals. The S02 audit counted seven goal-line shots in the whole cohort; the development partition holds the records listed here, and nothing is examined or stated about the rest. A high x is not by itself taken as an error: several of these follow a pass that ended at the same position, i.e. the shooter received the ball on the goal line.

| Event id | Match | League | Split | Clock | Outcome | Body | Origin % | positions (all) | Goal-mouth tags |
|---|---|---|---|---|---|---|---|---|---|
| 190069551 | 2499774 | English first division | train | 2H 13:36 | GOAL | left_foot | (100, 47) | (100, 47) ; (0, 0) | [1203] |
| 196248194 | 2516802 | German first division | train | 1H 05:40 | no goal | left_foot | (100, 62) | (100, 62) ; (0, 0) | [1213] |
| 221759018 | 2500890 | French first division | train | 2H 08:05 | no goal | right_foot | (100, 68) | (100, 68) ; (100, 100) | [1216] |
| 230367516 | 2516954 | German first division | train | 1H 12:11 | GOAL | right_foot | (100, 52) | (100, 52) ; (0, 0) | [1201] |
| 231561790 | 2565801 | Spanish first division | validation | 1H 39:28 | GOAL | right_foot | (100, 49) | (100, 49) ; (0, 0) | [1201] |
| 238857904 | 2576251 | Italian first division | validation | 2H 03:01 | GOAL | right_foot | (100, 57) | (100, 57) ; (0, 0) | [1201] |

Neighbouring events of event 190069551 (qualitative context only):

| Event id | Clock | Event | Team | positions |  |
|---|---|---|---|---|---|
| 190068820 | 2H 13:28 | Pass | 1625 | (64, 58) ; (74, 38) |  |
| 190068821 | 2H 13:32 | Pass | 1625 | (74, 38) ; (96, 77) |  |
| 190068823 | 2H 13:35 | Pass | 1625 | (96, 77) ; (100, 47) |  |
| 190069551 | 2H 13:36 | Shot | 1625 | (100, 47) ; (0, 0) | <- the shot |
| 190069552 | 2H 13:39 | Save attempt | 1628 | (100, 100) ; (0, 53) |  |
| 190069028 | 2H 14:32 | Pass | 1628 | (38, 31) ; (43, 12) |  |

Neighbouring events of event 196248194 (qualitative context only):

| Event id | Clock | Event | Team | positions |  |
|---|---|---|---|---|---|
| 196250101 | 1H 05:37 | Duel | 2453 | (2, 32) ; (2, 36) |  |
| 196248190 | 1H 05:38 | Shot | 2444 | (98, 64) ; (0, 0) |  |
| 196248135 | 1H 05:40 | Save attempt | 2453 | (100, 100) ; (2, 36) |  |
| 196248194 | 1H 05:40 | Shot | 2444 | (100, 62) ; (0, 0) | <- the shot |
| 196248138 | 1H 05:47 | Interruption | 2453 | (1, 33) ; (100, 100) |  |
| 196248140 | 1H 06:13 | Free Kick | 2453 | (100, 100) ; (57, 9) |  |

Neighbouring events of event 221759018 (qualitative context only):

| Event id | Clock | Event | Team | positions |  |
|---|---|---|---|---|---|
| 221759161 | 2H 08:00 | Others on the ball | 3776 | (13, 45) ; (18, 54) |  |
| 221759017 | 2H 08:01 | Shot | 3774 | (82, 46) ; (100, 100) |  |
| 221759163 | 2H 08:03 | Save attempt | 3776 | (0, 0) ; (15, 46) |  |
| 221759018 | 2H 08:05 | Shot | 3774 | (100, 68) ; (100, 100) | <- the shot |
| 221759165 | 2H 08:21 | Interruption | 3776 | (0, 41) ; (0, 0) |  |
| 221759166 | 2H 08:38 | Free Kick | 3776 | (0, 0) ; (34, 55) |  |

Neighbouring events of event 230367516 (qualitative context only):

| Event id | Clock | Event | Team | positions |  |
|---|---|---|---|---|---|
| 230367519 | 1H 12:08 | Shot | 2445 | (96, 70) ; (0, 0) |  |
| 230367397 | 1H 12:09 | Save attempt | 2462 | (100, 100) ; (4, 30) |  |
| 230367398 | 1H 12:10 | Others on the ball | 2462 | (1, 50) ; (0, 48) |  |
| 230367516 | 1H 12:11 | Shot | 2445 | (100, 52) ; (0, 0) | <- the shot |
| 230367402 | 1H 12:13 | Save attempt | 2462 | (100, 100) ; (100, 100) |  |
| 230367527 | 1H 13:15 | Duel | 2445 | (59, 10) ; (55, 29) |  |

Neighbouring events of event 231561790 (qualitative context only):

| Event id | Clock | Event | Team | positions |  |
|---|---|---|---|---|---|
| 231561754 | 1H 39:18 | Free Kick | 698 | (73, 100) ; (73, 86) |  |
| 231561756 | 1H 39:23 | Pass | 698 | (73, 86) ; (94, 64) |  |
| 231561785 | 1H 39:26 | Shot | 698 | (94, 64) ; (0, 0) |  |
| 231561790 | 1H 39:28 | Shot | 698 | (100, 49) ; (0, 0) | <- the shot |
| 231562007 | 1H 39:30 | Save attempt | 677 | (100, 100) ; (6, 36) |  |
| 231561791 | 1H 39:30 | Duel | 698 | (99, 49) ; (66, 85) |  |

Neighbouring events of event 238857904 (qualitative context only):

| Event id | Clock | Event | Team | positions |  |
|---|---|---|---|---|---|
| 238857897 | 2H 02:56 | Others on the ball | 3161 | (86, 7) ; (89, 10) |  |
| 238857899 | 2H 02:57 | Others on the ball | 3161 | (89, 10) ; (96, 27) |  |
| 238857902 | 2H 03:00 | Pass | 3161 | (96, 27) ; (100, 57) |  |
| 238857904 | 2H 03:01 | Shot | 3161 | (100, 57) ; (0, 0) | <- the shot |
| 238858347 | 2H 03:03 | Save attempt | 3194 | (100, 100) ; (0, 43) |  |
| 238858371 | 2H 04:01 | Pass | 3194 | (49, 49) ; (39, 41) |  |

## 4. Technical review of the S08 claims and display

1. **Reading paths.** `dataset.read_identity`, `dataset._parse_partitions`, `candidate_data._join_body_parts` and `portfolio_data.read_team_columns` all traverse every line of the shot table, as S03-S05 already did. Held-out rows are touched only for identity columns (event id, match id, league, date, eligibility); `load_partitions` and `load_body_parts` refuse the test split, and `read_team_columns` reads team and side only for requested validation events. The S08 wording "no test-window row is read" was therefore too broad and now says: no held-out feature, outcome or prediction is inspected, scored or used to fit a model or build a figure.

2. **What Figure 1 compares.** The held-out evaluation compared the location + body-part recipe with the distance + angle recipe. They differ in their location terms as well as the body-part tag, so it is not a body-part ablation. The smooth-location model without the tag was a validation-stage comparison only and was not part of the held-out result. Wording in the story and on Figure 1 was adjusted; no causal claim is made.

3. **Match map direction.** Figure 3 draws the home team attacking right and the away team attacking left in both halves. That is a display convention, not the real direction of play (teams switch ends at half-time); the figure now says so.

4. **Markers and footnotes.** The markers were large: with the original scale (2000 points squared per unit xG) a goal at xG 0.6 spanned 8.3 m of pitch at the original scale, more than the six-yard box is deep. Given the audit result, only the display was changed: the scale is now 800 points squared per unit, a small dot marks each recorded position, and the footers were shortened. The plotted probabilities, team totals, inputs and selection rule are identical. Figure 2 is unchanged (common colour scale and support mask).

5. **Figure 1.** The frozen held-out metrics are untouched; only the headline and footer wording changed. A tidy chart is not evidence that its data are right; that rests on this audit.

## 4b. Limits of this audit

- The audit checks internal consistency and agreement with the provider's documented 'start' semantics. It cannot prove that the provider's coordinates are physically correct, and no video was consulted.
- Positions are integer percentages, about 1 m along and 0.7 m across the pitch; very close shots are sensitive to that quantisation.
- Quarantined rows (`x_pct` below 20, a score-mismatch match) are outside the cohort and were not re-examined.
- Everything here concerns the development partition. Nothing is inferred about the held-out window, and the S07 results stay locked and are not re-run or re-interpreted.

## 5. Frozen artefacts

| Artefact | SHA-256 before the audit | SHA-256 after |
|---|---|---|
| report | `40e65a82ed0452ba3fab104f31cd0be7e1b0f970b4093f01b2a16b819697ff77` | `40e65a82ed0452ba3fab104f31cd0be7e1b0f970b4093f01b2a16b819697ff77` |
| summary | `97a28e6929a97ff4c699b4d4d538d4b6765c989bdfa66379a1150835b4765edf` | `97a28e6929a97ff4c699b4d4d538d4b6765c989bdfa66379a1150835b4765edf` |
| log | `69019e59ac4c8917478c3aaae1f9817414ceddc0928d7ef3cd20077f29d41761` | `69019e59ac4c8917478c3aaae1f9817414ceddc0928d7ef3cd20077f29d41761` |

Shot table SHA-256 `58eb62c0f786406c7df1e457a981784ab10d0514ee5a3ec73c2119c83f0af96e`; split assignment SHA-256 `2f995fc0337436619f794d8c64c7f8397bf81d8e28deca9f2add8369c881da15`. `REAL_EXECUTION_ENABLED` is unchanged. Data: Pappalardo & Massucco (2019), Wyscout open event data, CC BY 4.0, doi:10.6084/m9.figshare.c.4415000.v5; filtered, quality-screened and augmented for this project.
