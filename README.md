# Expected Goals Lab

**How much of a shot's chance of becoming a goal is explained by where it is taken, and does the provider's body-part tag add predictive value?** A one-season, five-league study on public Wyscout event data, with chronological partitions, one score-held-out evaluation and an audit of the shot coordinates.

![Estimated chance of a goal across the attacking end for right-foot, left-foot and head / other-body shots](figures/portfolio/fig2_shot_quality_map.png)

*Estimated chance of a goal for a regular shot from each spot, attacking the goal on the right, from a model fitted on training matches only. Blank areas have too few training shots of that kind to estimate (details in the figure note).*

## What was found

On 6,653 regular shots from 298 matches played 14 April to 20 May 2018 (five leagues), a model using smooth location terms plus the body-part tag scored better than a plain distance-and-angle model:

- **Log loss** 0.2905 against 0.2998 (lower is better); difference −0.0093, 95% interval −0.0120 to −0.0066.
- **Brier score** 0.08447 against 0.08752; difference −0.00305, interval −0.00393 to −0.00215.
- ROC AUC 0.787 against 0.767.

The two models were fitted once on earlier matches (training plus validation) and scored once on these later ones. The comparison is between two *recipes* that differ in their location terms as well as in the body-part tag, so it does not isolate the effect of body part, and the intervals (a bootstrap over matches) are exploratory. Full numbers: [reports/final_test_evaluation.md](reports/final_test_evaluation.md). A one-page narrative: [reports/football_research_story.md](reports/football_research_story.md).

**Limits, in short.** One historical season and one chronological split; the same teams and players appear in every partition; shots and body parts are provider tags (the "head / other body" tag is not a confirmed header); only regular shots are modelled (no penalties, direct free kicks or own goals); the model sees no goalkeeper, defender or pressure information. The held-out window was held out from *scoring*, not from all outcome information: its aggregate goal count could be inferred beforehand, so it is called score-held-out, not outcome-blind. Nothing here is evidence about other seasons, player finishing, goalkeepers, scouting value or causal effects.

## Why it matters

Raw conversion rates ignore the chance a shot offered: a header from the penalty spot and a shot from 25 metres are not the same opportunity. Distance and angle capture most of the geometry; the body-part tag adds information about how the shot was taken, but as a *statistical association* in tagged data, not a measure of what changing technique would do. xG describes the quality of the shots taken, not finishing skill, defending or tactics.

## Three figures

**1. Held-out evidence.** The score comparison above, with goals scored against goals expected as context.

<img src="figures/portfolio/fig1_heldout_evidence.png" alt="Dot plots of held-out log loss and Brier score for three models, and a bar chart of goals scored versus expected" width="480">

**2. Shot-quality surfaces** (the image at the top). Estimated chance of a goal by pitch position for each body-part tag, one colour scale, blank where training data are thin.

**3. One historical match.** Borussia Dortmund v Augsburg, 26 February 2018, a validation-window match chosen by a rule fixed beforehand (the earliest qualifying validation match). Marker area is the model's estimated chance; each team's regular-shot xG is shown beside the provider's score. Regular-shot xG excludes penalties, direct free kicks and own goals, so it need not match the score. Attack directions are a display convention.

<img src="figures/portfolio/fig3_match_shot_map.png" alt="Shot map of one match with marker area proportional to estimated chance of a goal" width="640">

## What the model sees, and what it does not

**Sees:** distance to the goal centre, the angle the goal mouth subtends, and a three-way body-part tag (left foot, right foot, head / other body). **Does not see:** defenders, goalkeeper position, pressure, assist type, match state, the shooter or team, shot technique or end location, or anything after the shot. Positions are integer percentages of an *assumed* 105 m x 68 m pitch. 36 of 40,461 regular shots are quarantined for documented data issues (one match whose events do not reproduce its score, and shots recorded implausibly far from the goal they targeted); they are never repaired or silently re-admitted ([docs/SHOT_DATASET_SPEC.md](docs/SHOT_DATASET_SPEC.md)).

## Data, licence and citation

Data: Pappalardo, L. & Massucco, E. (2019), *Soccer match event dataset*, figshare, [doi:10.6084/m9.figshare.c.4415000.v5](https://doi.org/10.6084/m9.figshare.c.4415000.v5), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); paper: Pappalardo et al. (2019), *A public data set of spatio-temporal match events in soccer competitions*, Scientific Data 6:236, [doi:10.1038/s41597-019-0247-7](https://doi.org/10.1038/s41597-019-0247-7). The underlying event data were collected by Wyscout. The data were filtered to regular shots, quality-screened and augmented with derived geometry for this project; the figures and tables here are adaptations. The dataset authors and Wyscout have not endorsed this work.

**Raw event archives and the complete processed shot table are not redistributed here**: the code downloads the public Figshare files and checks them against pinned checksums, and the full per-shot table stays local and git-ignored. What *is* published is aggregate results, derived figures, one historical match's shot visualisation, and a few event-level excerpts quoted in the audit reports to document specific data issues (see [docs/DATA_LICENSE_AND_ATTRIBUTION.md](docs/DATA_LICENSE_AND_ATTRIBUTION.md) for the exact distinction). The original code is MIT-licensed ([LICENSE](LICENSE)); that licence does not cover the dataset, which keeps its own terms. To cite this work, see [CITATION.cff](CITATION.cff).

## Method and evaluation design

[docs/METHODOLOGY.md](docs/METHODOLOGY.md) is the short technical account: cohort and label, features, chronological partitions, training-only transforms, the two models, exploratory validation, the one-time score-held-out evaluation, bootstrap limits and the shot-origin audit. The protocols were written before the numbers they govern: [baseline](docs/BASELINE_PROTOCOL.md), [nonlinear candidates](docs/NONLINEAR_PROTOCOL.md), [diagnostics](docs/DIAGNOSTICS_PROTOCOL.md), [model selection](docs/MODEL_SELECTION_DECISION.md), [final evaluation](docs/FINAL_TEST_PROTOCOL.md), and the [shot-origin audit](docs/SHOT_ORIGIN_AUDIT_PROTOCOL.md) with its [report](reports/shot_origin_audit.md).

## Run it

Python 3.12 and [uv](https://docs.astral.sh/uv/). In PowerShell:

```powershell
uv sync
uv run pytest -q                                          # runs offline; skips the tests below that need the download
uv run python -m expected_goals_lab.build_shots           # downloads ~78 MB of public data once
uv run python -m expected_goals_lab.evaluate_nonlinear    # validation-stage comparison
uv run python -m expected_goals_lab.portfolio_figures     # rebuilds the three figures
```

The validation-stage pipeline and the three portfolio figures are fully reproducible this way. The one-time held-out evaluation is not meant to be run again: its scorer intentionally refuses to execute on this public snapshot (the snapshot differs from the exact research revision that was evaluated, and its result is already published), so running it again would only replay a known outcome, not test anything new. [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) explains which tests need the download and the exact reasons the scorer refuses to run here.

## Layout

`src/expected_goals_lab/` code; `tests/` tests; `docs/` protocols and methodology; `reports/` aggregate reports (including the held-out evaluation record); `results/` the held-out summary; `figures/` published figures. Raw data, caches and the locally built shot table are git-ignored.
