# Reproducibility

What can be rebuilt from the public files, what needs the download, and what the published held-out result is and is not.

## Environment

Python 3.12 and [uv](https://docs.astral.sh/uv/). Windows PowerShell shown; the commands are the same elsewhere.

```powershell
uv sync                      # installs the pinned dependencies from uv.lock
uv run pytest -q             # offline; mostly synthetic fixtures, one test against real published data (see below)
uv run ruff check .
```

## Build the shot table from the public data

```powershell
uv run python -m expected_goals_lab.build_shots
```

Downloads the five pinned Figshare files (about 78 MB; `events.zip` is most of it) into `data/cache/wyscout/`, checks each against its pinned size and MD5, and writes `data/processed/wyscout_regular_shots.csv` and its manifest. Both directories are git-ignored; nothing from the dataset is committed. Later runs use the cache and make no network requests. The data are CC BY 4.0; see [DATA_LICENSE_AND_ATTRIBUTION.md](DATA_LICENSE_AND_ATTRIBUTION.md). If you run a command that needs the table before building it, it stops and prints this instruction instead of producing anything.

The table's SHA-256 should be `58eb62c0f786406c7df1e457a981784ab10d0514ee5a3ec73c2119c83f0af96e`, which the later steps check.

## Validation-stage steps (these use training and validation matches only)

```powershell
uv run python -m expected_goals_lab.splits                  # split assignment only
uv run python -m expected_goals_lab.evaluate_baselines      # three baselines, validation window
uv run python -m expected_goals_lab.evaluate_nonlinear      # two candidates, validation window
uv run python -m expected_goals_lab.evaluate_diagnostics    # diagnostics and validation figures
uv run python -m expected_goals_lab.shot_origin_audit       # development-only origin audit
uv run python -m expected_goals_lab.portfolio_figures       # the three portfolio figures
```

Each of these recomputes numbers that are also printed in the committed reports, checks that they match, and stops on a difference. None reads a test-window feature or outcome (the shot table is still traversed for identity columns). Floating-point results can differ in the last digits across platforms and library versions; the checks use the precision printed in the reports. `portfolio_figures` reads the published held-out summary in `results/` and does not recompute it.

## Tests

`uv run pytest -q` needs no network. Most tests use synthetic fixtures in place of the shot table, so they need no data either. One test is the exception: `test_real_s07_summary_is_read_without_being_touched` runs offline too, but it reads real data — the committed, already-published `results/final_test_summary.json` ships with this repository, so that one test checks it directly (hashing it before and after, to prove it is only read) without building anything or downloading anything. The remaining tests that need a built shot table, its cache, or a full local pipeline run are skipped when those are absent, and run once you have built the table or filled the cache:

| Test module | Needs |
|---|---|
| `test_real_s07_summary_is_read_without_being_touched` (in `test_portfolio_figures.py`) | nothing beyond this repository's own committed `results/final_test_summary.json` |
| `test_evaluate_nonlinear`, `test_evaluate_diagnostics` (real-table reproduction checks) | `data/processed/wyscout_regular_shots.csv` |
| `test_final_evaluation` (real metadata-only preflight) | the shot table, manifests and split assignment |
| `test_shot_table`, `test_wyscout_audit` (real-archive checks) | the cached Figshare files |

Synthetic tests check code behaviour (leakage traps, split logic, scoring arithmetic, figure inputs); they are not evidence about football.

## The held-out evaluation is a published record, not a regenerable number

The held-out evaluation was run once, on 21 October 2026, under a protocol pinned by hashes ([FINAL_TEST_PROTOCOL.md](FINAL_TEST_PROTOCOL.md)). Its aggregate outputs are published unchanged: [`reports/final_test_evaluation.md`](../reports/final_test_evaluation.md), `results/final_test_summary.json` and `reports/final_test_execution_log.jsonl` (outcomes and file hashes only). The figure and audit code verify the summary's SHA-256 against the log before using it.

Publishing the code does not make the result a new independent test: the test window's scores are now known, the models were chosen before they were scored, and any run of the same code on the same data is a **replay**, not new evidence.

**Why the supplied runner will not execute a replay as it stands.** The runner (`final_evaluation.py`) refuses to run unless the files of its pinned implementation inventory match the hashes in `final_lock.py`, and those hashes belong to the research repository's revision that was evaluated. This public snapshot is not byte-identical to it: 13 of the 19 inventory entries are identical (including `uv.lock`, the model-selection decision and the geometry, splitting, scoring and model modules), but `pyproject.toml` (project metadata), and four modules whose only change is provider-neutral imports (`wyscout.py`, `build_shots.py`, `evaluate_baselines.py`, `evaluate_nonlinear.py`), differ, and one entry (`audit.py`, an earlier audit tool that was not carried over) is absent. The execution gate is also closed (`REAL_EXECUTION_ENABLED = False`) and the log records a completed evaluation, so a second run is refused in any case.

A replay would therefore need a deliberate new lock for this snapshot and a separate decision to open the gate, and it should be labelled a replay. The metadata-only preflight (`uv run python -m expected_goals_lab.final_evaluation`) runs offline, reads no shot row, and lists exactly which checks differ. The data-identity checks (shot table, manifests, split assignment, protocol hash, lockfile, package versions, model recipes) are the part you can verify for yourself.

The runner's `--execute --confirm-final-test` mode is refused while the gate is closed and is deliberately not part of the normal workflow.
