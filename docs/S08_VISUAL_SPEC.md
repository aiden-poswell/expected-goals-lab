# S08 visual specification: football-facing figures and one match case study

Written before any S08 figure was drawn and before the case-study match was chosen. The modelling protocol is frozen and the one-time held-out evaluation (S07) is finished; S08 explains and presents, it does not select, tune or re-score anything.

## 1. Audience and question

**Audience:** a football-literate reader (analyst, scout, coach, fan) who has not read the code. **Football question:** how much of a shot's quality is explained by where it is taken, and does the provider's body-part tag add anything on top, on shots the model had never seen?

## 2. What each figure shows, and from what

| # | Figure | Source | New computation |
|---|---|---|---|
| 1 | Held-out model evidence | the frozen S07 JSON summary only | none: values are read and formatted; relative differences are arithmetic on those values |
| 2 | Shot quality across the pitch | the model fitted on the **original training split only** (S04/S05 selected recipe, unchanged) | probabilities on a regular grid of pitch positions, for each body-part class; no data row is scored |
| 3 | One real match | the same training-only model, applied to **one validation-window match** | the model's probabilities for that match's regular shots |

Provenance for all three: Pappalardo & Massucco (2019), Wyscout open event data, CC BY 4.0, doi:10.6084/m9.figshare.c.4415000.v5, filtered, quality-screened and augmented for this project; paper Scientific Data 6:236, doi:10.1038/s41597-019-0247-7.

## 3. Hard boundaries

- No held-out (test-window) feature, outcome or prediction is inspected, scored or used to fit a model or build a figure. As in S03-S05 the shot table is traversed in full to find rows, and identity columns of held-out rows (event id, match id, league, date) are read for the split; nothing else of them is. No final-test shot is scored, plotted or summarised by S08. (Corrected in S08A: the first wording, "no test-window row is read", overstated what a full traversal of the shot table implies.) The S07 numbers are read from `results/final_test_summary.json` (the private research repository wrote this file to `data/processed/final_test_summary.json`; this public package publishes the identical, byte-for-byte artefact at `results/final_test_summary.json` instead), whose SHA-256 must equal the `summary_sha256` in the completed entry of the S07 execution log, and whose protocol hash must equal the locked one; otherwise the figure refuses to build.
- The final evaluation is not run and the real-execution gate stays closed. S07 outputs are never written, moved or deleted.
- Figures 2 and 3 use `evaluate_nonlinear`, the unchanged S04 training-only pipeline, through the guarded loaders that refuse the test split. Its validation scores must equal those recorded in `data/processed/nonlinear_validation_summary.json` (to 1e-9) or the build stops.
- Nothing is tuned, chosen or filtered because of how it looks.

## 4. Plotting cohort

- **Figure 1:** the 6,653 regular shots of the S07 test window, as aggregates from the frozen summary.
- **Figure 2:** no shots; model estimates on a grid. A grid cell is drawn only where the training split holds at least 25 shots of that body-part class within ±3 m in each direction; elsewhere it is left blank, so nothing is extrapolated into positions the model has not seen for that class. Domain: the attacking 45 m of the pitch. Wyscout coordinates are attacker-relative, so every shot is aimed at the goal at x = 100% (105 m) and the figure shows one half-pitch attacking to the right.
- **Figure 3:** the regular shots of one validation-window match. Regular shots exclude penalties, direct free-kick shots and own goals, so team totals need not equal the match score.

## 5. Case-study match: selection rule (fixed before any score, goal or plot was looked at)

Among matches in the **original validation window** (2018-02-26 to 2018-04-13), take those in which both teams have at least one eligible regular shot and the two teams together have at least 10. From these choose the **earliest by match date, ties broken by the lowest match ID**. The rule uses only match date, match ID, team identity and the count of eligible shots. It uses no outcome, no goal, no model output and no plot. The selection code reads no target or prediction column.

**Recorded result of the rule** (selection step run before the match's goals, scores or model outputs were looked at; it printed only the items below): Wyscout match **2516947**, 2018-02-26, Borussia Dortmund (home, team 2447, 9 eligible regular shots) v Augsburg (away, team 2481, 11 eligible regular shots), 20 eligible regular shots in all. It is the earliest qualifying validation match; it was not chosen for how its shots look. The build stops if the rule ever returns a different match.

## 6. Football mapping checks

- Wyscout positions are percentages from the acting team's own point of view with x = 100% the goal attacked; metres use the project's assumed 105 m x 68 m pitch and 7.32 m goal. Pitch drawing uses the laws-of-the-game proportions (penalty area 40.32 m x 16.5 m, six-yard box 18.32 m x 5.5 m, penalty spot 11 m, centre circle and penalty arc radius 9.15 m).
- Figure 3 shows both teams on one full pitch: the home team attacks the right-hand goal as recorded; the away team's positions are rotated 180 degrees (x -> 100 - x, y -> 100 - y) so that it attacks the left-hand goal. Home and away come from the match metadata, and each shot row's team and side must agree with it.
- Team names come from the provider match label; the build fails if the label, the per-team side data and the shot rows disagree. No name, position or number is invented.
- `head_or_body` (tag 403) is labelled "Head / other body", never "header".

## 7. Layout and accessibility

Static PNG (200 dpi) and SVG, one message per figure, headline states the finding, a footer carries the data source, licence, the frozen/validation status and a method note. Colour pair blue/orange, chosen to be distinguishable with common colour-vision deficiencies and in greyscale by lightness and by marker fill (filled = goal). Text is at least 8 pt at the figure's native size, which keeps it readable when the figure is scaled to a phone width. No club badges, kits or trademarks. Axes that do not start at zero say so on the figure.

## 8. What the figures must not claim

No live-play accuracy, no player or club evaluation, no scouting validation and no causal effect of how a shot is struck. One season, one provider, shared teams across windows.
