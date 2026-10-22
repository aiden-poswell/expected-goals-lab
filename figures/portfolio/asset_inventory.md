# Portfolio figure inventory

Built with `uv run python -m expected_goals_lab.portfolio_figures`. Aggregates and one validation match only; no test-window shot-level data, no raw or derived shot tables.

## Provenance

- Model: `nonlinear_location_body_part` (recipe SHA-256 `d0c467b8f1de0550be8512c62463186c3031a7ea77b320109f09e614372b6291`), fitted on the original training split only (28,263 shots). Figures 2 and 3 only.
- Shot table SHA-256 `58eb62c0f786406c7df1e457a981784ab10d0514ee5a3ec73c2119c83f0af96e`; split assignment SHA-256 `2f995fc0337436619f794d8c64c7f8397bf81d8e28deca9f2add8369c881da15`.
- Figure 1 reads the frozen held-out summary, SHA-256 `97a28e6929a97ff4c699b4d4d538d4b6765c989bdfa66379a1150835b4765edf` (protocol `5e4cc0b2971261fa5c807c777804a1dae00394747453eb5dcb4dfd126b28fdae`); nothing is recomputed.
- Figure 3 case study: Wyscout match 2516947 (Borussia Dortmund v Augsburg), chosen by docs/S08_VISUAL_SPEC.md section 5. Regular-shot totals by team id: 2447: 9 shots, 1 goal(s), xG 1.30; 2481: 11 shots, 1 goal(s), xG 1.35.

## Files

| File | Bytes | Pixels | SHA-256 |
|---|---|---|---|
| `fig1_heldout_evidence.png` | 277,528 | 1840 x 2200 | `05709cbccfc113c3ac18f81f988fa9c43368d3d7a733cdcc60d6533eed4e36c4` |
| `fig1_heldout_evidence.svg` | 202,017 |  | `5112eb07fad0c9350b12c6b75b59dba9a9000afd596207cdebf349c689319c1a` |
| `fig2_shot_quality_map.png` | 283,583 | 2520 x 1680 | `d33f987f957e7cefa153ed5b41a36ee0c4bfdaae08900a1b11e912e7c8e686e5` |
| `fig2_shot_quality_map.svg` | 185,712 |  | `54a6b88edfbd3516e096021cfe7061004c40edd66ad825325d2aa2734dcaec3c` |
| `fig3_match_shot_map.png` | 262,460 | 2200 x 1800 | `6f0a6a8c26cf7fe4975113d90e9f82d1e390a1bd2a1aa6dbad96668502bbbda3` |
| `fig3_match_shot_map.svg` | 182,795 |  | `065811c4f2fad533fcf5baf105583b85cae84555f8d6914090e42c88560b98cb` |

Data: Pappalardo & Massucco (2019), Wyscout open event data, CC BY 4.0, doi:10.6084/m9.figshare.c.4415000.v5; filtered, quality-screened and augmented for this project.
