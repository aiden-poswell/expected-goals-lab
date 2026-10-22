# Final test preflight sign-off (S06)

## FINAL TEST NOT YET OPENED

The real held-out test window has not been opened, parsed or scored. The protocol it will be evaluated under is [FINAL_TEST_PROTOCOL.md](FINAL_TEST_PROTOCOL.md). This page lists what S06 has verified, what it deliberately did not do, and what risk remains. It is a checklist for a reviewer, not a result.

## S06A hardening (reviewed separately from S06)

S06 (commit `683f1bb1ff7587e641f6637fe5a778c66cc90491`) was reviewed and three implementation inconsistencies were found. S06A corrects them on its own branch, and its patch is reviewed on its own before any real evaluation. **The real-data gate remains CLOSED (`REAL_EXECUTION_ENABLED = False`) and no final-test command has been executed.**

1. **Bootstrap settings are frozen.** The command line has no `--resamples` option (and flag abbreviations are off). The runner takes the 1,000 resamples and seed 20180513 from the lock; against the real lock any different programmatic value is refused before the gate, the preflight, any log entry or any row read. Synthetic locks may still use fewer resamples for fast tests.
2. **Completion is trustworthy.** Report and JSON summary are rendered, written to temporary files, moved into place, and only then is `completed` appended (with the two file hashes). Any failure leaves no `completed` entry and removes outputs from that attempt. A failed attempt, an attempt that started with no outcome, output files without a completed entry, and a corrupt log each stop execution until an operator supplies a recorded retry reason; a reason is not permission to change any model. A completed evaluation is never rerun or overwritten.
3. **League results are descriptive.** A compact final-only function gives counts, log loss, Brier, their differences and the predeclared AUC rule. No per-league bootstrap is computed, and none appears in the report or JSON. The S05 validation diagnostics are untouched.

The protocol was also clarified: "once" means one scoring occasion, not one pass over the shot table. The protocol text changed, so only its locked hash changed (`bddf6498...` to `5e4cc0b2...`). The data, split, inventory and recipe hashes are unchanged, and none of the 19 inventoried files was modified.

## Checks completed in S06

| Check | How | Outcome |
|---|---|---|
| Frozen data identity (shot table, S02 manifest, split assignment, split manifest) | byte-level SHA-256 and published JSON only; no shot row parsed | matches the locked values |
| Split counts and date windows | published split manifest versus the lock | match; windows strictly ordered |
| Protocol, model decision, lockfile, `pyproject.toml` and the 19-file implementation inventory | LF-normalised SHA-256 | match the lock |
| Installed package versions | compared with `uv.lock` and the lock | match |
| Model recipes | canonical description of both pipelines, hashed | match the lock; fitting does not change a recipe hash |
| Dry run opens the shot table and the assignment only as bytes | test with a spy on `Path.open` and with `csv` parsing disabled | passes on synthetic data and on the real local files |
| Dry run survives a shot table that no parser could read | test (garbage bytes with a consistent lock) | passes |
| `--execute` without both flags, `--dry-run` combined with `--execute`, and the closed real-data gate | tests | exit 2, no row reader called, nothing written |
| One changed byte in any of nine locked files, or one changed lock value | tests | fails closed, no row reader called, no results, no log |
| S06A: render, temp-file, move-into-place and completion-record failures; interrupted attempt; stale outputs; corrupt log | tests (`test_final_hardening.py`) | never marked completed; no silent retry; no overwrite |
| Recomputed split differs from the locked split | test | stops after identity columns only, before any feature or label is loaded |
| Access order | test with a trap on sensitive columns of held-out-date rows | no sensitive column is read until both development models are fitted and verified |
| Models use development data only | test inverting every held-out label; test refitting independently on train + validation | expected goals and fitted models are unchanged by held-out labels; scores equal the independent refit |
| Hand-calculated log loss, Brier, ROC AUC (ties, single class), expected goals | tests | match |
| Resampling is by match | test with two matches of three and one shot | only match-level weightings occur |
| Sparse and single-class cases | tests | empty leagues and absent AUC render as "n/a"; one match gives no interval and no reading |
| Report neither hard-codes counts nor a conclusion | tests with changed counts and with a losing primary | the wording follows the numbers; attribution and DOI present; no identifiers |
| One-shot log and retries | tests | completed run cannot repeat; a retry needs a recorded reason; the log holds outcomes only |

## Checks blocked or not done on purpose

- Execution on the real data (`--execute --confirm-final-test`): switched off by `REAL_EXECUTION_ENABLED` and never invoked in S06.
- Any score, label, feature, prediction or goal count from the test window.
- Confirming that the real shot table parses and fits end to end under the final procedure: that happens only in S07, on the real rows.
- A clean-checkout run: the real-data preflight test, and tests from earlier slices that need the local shot table or cache, are skipped when those files are absent.

## Remaining risks

1. **Untested on real rows until S07.** The procedure is verified on synthetic data. A data-shaped fault (a rare body-part class missing from development, non-convergence, a league with no test shots) would surface only at execution; the protocol says to stop and review, not to repair.
2. **The loaders were modified.** `dataset.py` and `candidate_data.py` gained a token-gated route to the sealed split; `fit_baselines` and `fit_candidates` gained a `fit_split` keyword. The public loaders still refuse the test split and the old behaviour is unchanged (existing tests pass). A reviewer should read those diffs first.
3. **The lock modules are outside the hash inventory.** `final_evaluation.py` and `final_lock.py` cannot contain their own hash. S07 must pin the reviewed commit; the only intended change before S07 is switching on `REAL_EXECUTION_ENABLED`.
4. **Environment drift.** The lock pins package versions and the dependency lockfile. A different platform's floating-point behaviour could move scores in the last digits; that is not a protocol failure.
5. **Statistical limits stated in the protocol:** one season, repeated teams across windows, provider tagging noise, tag 403 is not a confirmed header, the aggregate test goal count was already inferable, and the match bootstrap omits team/league dependence and selection uncertainty.
6. **Development refit changes the comparison base.** Final scores come from models fitted on train + validation; they are not directly comparable with the validation-stage scores.

Nothing was committed or pushed in S06.
