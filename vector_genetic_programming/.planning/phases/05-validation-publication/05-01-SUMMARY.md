---
phase: "05-validation-publication"
plan: "01"
subsystem: "analysis"
tags: ["walk-forward", "dsr", "validation", "multi-seed"]
dependency_graph:
  requires: ["vgp.data.splitter.WalkForwardSplitter", "vgp.evolution.loop.run_evolution", "vgp.backtest.runner.evaluate"]
  provides: ["vgp.analysis.runner.WalkForwardRunner", "vgp.analysis.runner.WindowSpec", "vgp.analysis.runner.generate_windows", "vgp.analysis.dsr.compute_dsr", "vgp.analysis.dsr.aggregate_seeds", "vgp.analysis.dsr.save_results_csv"]
  affects: ["Phase 5 plan 02 (plots.py)", "Phase 5 plan 03 (community release)"]
tech_stack:
  added: ["scipy.stats.norm", "scipy.stats.skew", "scipy.stats.kurtosis", "dateutil.relativedelta"]
  patterns: ["Walk-forward window generation", "Structural OOS enforcement via local variable scoping", "Bailey & Lopez de Prado DSR formula", "Helper function extraction for test patchability"]
key_files:
  created:
    - "vgp/analysis/runner.py"
    - "vgp/analysis/dsr.py"
    - "tests/test_validation.py"
  modified:
    - "vgp/analysis/__init__.py"
decisions:
  - "Extracted _get_is_returns() as separate module-level helper so tests can patch IS returns extraction without requiring actual GP tree execution or vectorbt"
  - "Imported compute_dsr at module level (not deferred) so tests can patch vgp.analysis.runner.compute_dsr"
  - "Used T=600 (2024-01-01 to ~2025-08-22) for synthetic test data — large enough for train_end=2024-12-31 split"
  - "compute_dsr returns 0.0 for flat returns (std==0), insufficient observations (<2), or non-positive sr_var — all indicate no evidence of skill"
metrics:
  duration: "~12 min"
  completed: "2026-06-13"
  tasks_completed: 2
  files_changed: 4
---

# Phase 5 Plan 01: Walk-Forward Runner + DSR Module Summary

Walk-forward validation harness with structural OOS enforcement, multi-seed aggregation, Deflated Sharpe Ratio computation, and CSV persistence — provides the statistical backbone for Phase 5 validation.

## What Was Built

### `vgp/analysis/runner.py`

- `WindowSpec` dataclass: captures date boundaries for one rolling window (`window_id`, `train_end`, `val_start`, `val_end`, `test_start`, `test_end`)
- `generate_windows(total_start, total_end, train_months=12, val_months=2, oos_months=3, step_months=3)`: produces 4 non-overlapping walk-forward windows over the 2024-01-01 to 2026-04-01 data range using `dateutil.relativedelta` for calendar-accurate month arithmetic
- `WalkForwardRunner`: orchestrates multi-seed evolution per window, enforcing structural OOS isolation (test_fm is a local variable, passed to `evaluate()` exactly once, never to `run_evolution()`)
- `_get_is_returns(individual, train_fm, train_eval_cfg)`: separated helper for re-running IS backtest to extract per-period returns for DSR — patchable in tests without vectorbt

### `vgp/analysis/dsr.py`

- `compute_dsr(returns, sr_hat, n_trials, periods_per_year=252)`: implements Bailey & Lopez de Prado (2014) Proposition 3 using `scipy.stats.norm.ppf/cdf`, `scipy.stats.skew`, and `scipy.stats.kurtosis(fisher=True)` for excess kurtosis. Guards: flat returns (std=0) → 0.0; T<2 → 0.0; non-positive sr_var → 0.0
- `aggregate_seeds(seed_results)`: computes `median_oos_sharpe`, `iqr_oos_sharpe`, `median_dsr`, `n_seeds_positive_oos` across seeds for one window
- `save_results_csv(results, path)`: writes experiment results to CSV using `pd.DataFrame.to_csv()` (no mlflow per CLAUDE.md constraint)

### `vgp/analysis/__init__.py`

Updated to export all six public names: `WalkForwardRunner`, `WindowSpec`, `generate_windows`, `compute_dsr`, `aggregate_seeds`, `save_results_csv`.

### `tests/test_validation.py`

8 tests covering VAL-01 through VAL-04:

| Test | Requirement | Method |
|------|-------------|--------|
| `test_generate_windows_count` | VAL-01 | Asserts `len(windows) == 4` |
| `test_generate_windows_non_overlapping` | VAL-01 | Asserts `test_start[i+1] > test_end[i]` for all pairs |
| `test_generate_windows_dates` | VAL-01 | Asserts first window: `train_end='2024-12-31'`, `test_start='2025-03-01'`, `test_end='2025-05-31'` |
| `test_runner_oos_not_passed_to_evolution` | VAL-02 | Patches `run_evolution`; asserts `call.args[1].shape[0] < full_T` |
| `test_runner_iterates_seeds` | VAL-03 | `seeds=[0,1,2]` → 3 result dicts with keys `{window_id, seed, train_end, test_start, test_end, is_sharpe, oos_sharpe, dsr, n_nodes_best}` |
| `test_compute_dsr_returns_float_in_range` | VAL-04 | `compute_dsr(normal_returns, sr_hat=1.0, n_trials=10)` in [0.0, 1.0] |
| `test_compute_dsr_flat_returns_zero` | VAL-04 | `compute_dsr(zeros(100), sr_hat=0.0, n_trials=1) == 0.0` |
| `test_aggregate_seeds_positive_count` | VAL-04 | Two seeds [1.0, -0.5] → `n_seeds_positive_oos=1`, `median_oos_sharpe=0.25` |

## Deviations from Plan

### Auto-added: `_get_is_returns()` helper (Rule 2 — missing critical functionality)

**Found during:** Task 2 test design

**Issue:** The plan's `run_window()` implementation called `vbt.Portfolio.from_signals()` and `TreeEvaluator.execute()` directly in the method body for DSR computation. This prevented VAL-03 from being testable without actual GP tree execution and a real vectorbt run — mock individuals cannot be executed by `TreeEvaluator`.

**Fix:** Extracted the IS backtest re-run into a separate module-level function `_get_is_returns(individual, train_fm, train_eval_cfg)`. Tests patch `vgp.analysis.runner._get_is_returns` to return synthetic returns. The production code path is unchanged.

**Files modified:** `vgp/analysis/runner.py`

### Auto-added: Module-level `compute_dsr` import (Rule 2 — test patchability)

**Found during:** Task 2 test design

**Issue:** The plan's action code imported `compute_dsr` via a deferred import inside `run_window()` (`from vgp.analysis.dsr import compute_dsr`). This made `compute_dsr` unpatchable at `vgp.analysis.runner.compute_dsr` — the patch target would not exist at module level.

**Fix:** Moved `from vgp.analysis.dsr import compute_dsr` to module-level import in `runner.py`. This follows standard Python patch conventions and has no architectural downside.

**Files modified:** `vgp/analysis/runner.py`

## Known Stubs

None — all exported functions have complete implementations. `_get_is_returns()` is fully implemented with the vectorbt call; the test patches it for speed, not because it's stubbed.

## Threat Flags

No new network endpoints, auth paths, file access patterns, or schema changes were introduced. `save_results_csv()` writes to researcher-supplied paths (no path traversal surface — research-only context per T-05-03 in plan threat model).

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| `vgp/analysis/runner.py` exists | FOUND |
| `vgp/analysis/dsr.py` exists | FOUND |
| `vgp/analysis/__init__.py` exists | FOUND |
| `tests/test_validation.py` exists | FOUND |
| `.planning/phases/05-validation-publication/05-01-SUMMARY.md` exists | FOUND |
| Commit `976d58f` (Task 1 — implementation) | FOUND |
| Commit `ba105ce` (Task 2 — tests) | FOUND |
| `pytest tests/test_validation.py` — 8 passed, 0 failed | PASSED |
