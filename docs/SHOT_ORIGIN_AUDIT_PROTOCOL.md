# S08A protocol: shot-origin provenance audit

Written before any S08A audit number was computed. S08A is a diagnosis, not a modelling slice: it checks that the coordinates fed to the xG model are shot **origins** and are not outcome-coded. It does not change, re-run or re-score anything frozen.

## 1. Why

The S08 match map (Wyscout match 2516947, Borussia Dortmund v Augsburg, 2018-02-26) draws two very large scored-shot markers at the goal mouths. Both may be real close-range finishes, but a systematic link between unusual coordinates and goal tags would compromise the research whatever the chronological split. The S07 held-out evaluation took place once and stays locked: no rerun, no re-scoring, no selective filtering, no new "independent" claim.

## 2. Invariants

1. The real final evaluation is not executed, `REAL_EXECUTION_ENABLED` stays False, and the S07 report, JSON summary and execution log are not written. Their SHA-256 hashes are recorded before and after (reading the frozen hashes is allowed).
2. No change to S01-S07 features, recipes, coefficients, split assignment, the S02 dataset, the cohort or any protocol lock.
3. **Development only.** Only train and validation events are audited. The development set is derived from the frozen split (recomputed from identity columns and equal to the recorded assignment hash). For test-window rows nothing is inspected, logged, aggregated, printed or used: no coordinates, goal labels, body parts, predictions or match-specific outcomes.
4. No shot origin is invented, edited, flipped or removed. Diagnostic flags are not exclusion rules.
5. Raw archives and shot-level exports stay out of version control and out of the review patch. The committed report is match-level (one validation match) plus aggregates.

**What "development only" means precisely.** The Wyscout event archive is one JSON file per league; reading it parses every event, held-out ones included, into memory. That is an I/O pass over the file, not an examination of held-out features or outcomes. The audit code selects events by `matchId` against the development set and touches no other field of an event from a match outside it; a test enforces this with a trap that raises if any other field of such an event is read. The shot table is read through the existing guarded loaders (`load_partitions`, `load_body_parts`), which refuse the test split.

## 3. Audit A: what the dataset calls a shot location

- **Source documentation.** Wyscout events documentation (<https://support.wyscout.com/matches-wyid-events>, fetched in S08A): `positions` is "Positions in which this event started/ended (only start if not a movement)"; `x`/`y` are "X/Y position in percentage from the left corner of the attacking team". The page does not describe shot events beyond the label "Shot - Shot" (event 10, sub-event 100) and does not define the goal-mouth tags 1201-1223 in terms of the coordinate fields. The Scientific Data paper (doi:10.1038/s41597-019-0247-7) was consulted in S01B/S02 (`docs/GEOMETRY_VALIDATION.md`) for the attacker-relative frame; it was not re-fetched in S08A (the publisher page redirects to an authentication step). Taxonomy files `eventid2name.csv` and `tags2name.csv` are pinned Figshare files read through `WyscoutSource`.
- **Local code.** The shot origin is chosen in `src/expected_goals_lab/wyscout_shots.py`: `parse_shot` line 161 calls `_start_position(event.get("positions"))` (lines 219-226), which reads only `positions[0]` and returns `None` unless it is a dict with integer `x` and `y` in 0-100. `shot_table.make_row` takes `x_pct`/`y_pct` from that result and computes distance and angle through `geometry.shot_geometry`. No tag is read to obtain a position.
- **Checks (development matches).** (a) For every eligible development regular shot, the CSV's `x_pct`/`y_pct` must equal `positions[0]` of the source event (any mismatch is a FAIL). (b) The second entry of `positions` is classified (absent, (0,0), (100,100), other) and cross-tabulated with the goal flag, to see whether it is a placeholder or something carrying outcome information. (c) Goal-mouth tags (1201-1223) are tabulated against the origin to confirm they are post-shot information that the parser never reads. (d) Synthetic tests show the parser ignores a different second entry and any goal-zone tag. Anything the documentation does not settle is labelled uncertain.

## 4. Audit B: the S08 example match

For match 2516947, every shot row of the shot table, sorted by period and time: event id, team, player, period and mm:ss, goal flag, body-part tag, all elements of the source `positions`, the selected origin, distance, angle, the training-only model's xG, eligibility. Flags at `x >= 95`, `x >= 98`, `x == 100` and `(x, y) == (100, 50)` are separate columns. For the goals, the shot's own goal-mouth tags and the neighbouring events (three before, two after) give qualitative context; the preceding event is not treated as ground truth for the shooter. The marker-area formula (`XG_AREA_PER_UNIT` points squared per unit xG) is converted to metres on the pitch of the chart. An external corroboration note, if any, comes from a public match report, is labelled as such and never overrides a source coordinate.

## 5. Audit C: development-only outcome-leakage screen

On train + validation eligible regular shots, with denominators and goal and non-goal counts side by side (no p-values):

1. `x_pct` bins, mutually exclusive: `= 100`, `99`, `98`, `95-97`, `90-94`, `< 90`; overall and by league.
2. A second table isolating `(100, 50)`, other `x = 100` positions and other positions near the centre of the goal mouth (`45 <= y <= 55`; the goal mouth is about 5.4 percentage points either side of 50), against everything else with `x >= 95`.
3. The same bins by body-part class.
4. Distance (m) summaries for goals and non-goals, and counts by 10-unit `x` bins.
5. The goal-line records (`x = 100`) in the development partition, listed with the neighbouring events of their match.

Nothing here is a search for features.

## 6. Verdict criteria (fixed before any result)

The subset of concern is the set of development rows recorded on the goal line (`x = 100`, degenerate angle), together with any other band that turns out to be (almost) all goals.

- **PASS:** origins reconcile with `positions[0]` for every development shot; no sign that the second entry or any tag is used; the bands at `x >= 95` hold a substantial share of non-goals (at least 30% non-goals in the 95-99 range taken together); the subset of concern is at most 0.1% of development rows **and** contains non-goals as well as goals.
- **CONDITIONAL PASS:** origins reconcile and no destination is used, but the subset of concern is larger than 0.1% (up to 0.5% of development rows) or, however small, consists only of goals. The effect on the model is bounded and documented, the subset is named, and any claim must carry the caveat.
- **FAIL:** a mismatch with `positions[0]`, evidence that the recorded location is a destination, or a broad pattern (more than 0.5% of rows) in which location follows outcome. A FAIL stops the programme: earlier S07 results become historical results of a compromised pipeline, and a corrected model would be a new experiment on newly defined independent evaluation data.

## 7. Outputs

`reports/shot_origin_audit.md` (match 2516947 event by event, the development-only summary, the verdict), `src/expected_goals_lab/shot_origin_audit.py` (pure helpers and a CLI, `uv run python -m expected_goals_lab.shot_origin_audit`), tests, and minimal corrections to S08 wording grounded in the audit. The S08 figure is changed only if the audit passes and only in display; probabilities, totals, inputs and the selection rule stay identical.

## 8. Post-review clarification (S09, not predeclared)

Sections 1-7 above are unchanged and were written before any audit result. After the S08A report an independent review accepted the evidence and found two presentation defects, corrected in S09 without altering any count, table or criterion:

1. **Verdict wording.** The S08A code passed a literal `False` as "destination evidence" into the verdict function, so an automatic PASS looked as if the computer had checked origin versus destination. It had not: that question needs judgement. The report now separates a **mechanical screen** (computed: reconciliation with `positions[0]`, goal-flag and body-part consistency, the non-goal share of x = 95-99 shots, the goal-line subset, against the criteria of section 6) from a **manual review** (a dated human judgement recorded in `S08A_REVIEW`, with its basis listed). The overall finding is called a *finding*, not a verdict: it is PASS only as an evidence-based review judgement on top of a passing screen. A failed screen or a review that finds destination evidence fails the finding, and a missing review leaves it UNRESOLVED. The criteria themselves, and every raw count in the report, are the same as before; the regenerated report differs from the S08A one only in this header text.
2. **Archive passes.** The S08A text said the event archive is read "in one pass". The code makes two (the first collects regular shots of development matches and pass-end links, the second collects every event of the case-study and goal-line matches). Each parses every event of the five league files into memory one at a time, so events of held-out matches are loaded as well; fields other than `matchId` are examined only for events of development matches. The wording now says so.

The origin evidence remains internal-consistency evidence for the development partition, not proof of physical ground truth.
