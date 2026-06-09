---
phase: 03-gp-core-evaluation
plan: "03"
subsystem: backtest
tags: [vectorbt, numpy, evaluate, fitness, nsga2, transaction-costs, backtest-runner]

# Dependency graph
requires:
  - phase: 03-gp-core-evaluation
    plan: "01"
    provides: "build_pset() factory, Vector/Scalar type tokens, creator.Individual at module level"
  - phase: 03-gp-core-evaluation
    plan: "02"
    provides: "TreeEvaluator.execute() returning float32 [T] signal in {-1, 0, +1}"

provides:
  - "EvalConfig dataclass with fee_bps=10.0, min_trades=50, freq='1D', init_cash=10_000.0"
  - "evaluate(individual, feature_matrix, config) -> (sharpe, total_return, -tree_size) without deap import"
  - "BacktestRunner stateful wrapper for Phase 4 multiprocessing use"
  - "7 contract tests covering EVAL-01 through EVAL-04 (import boundary, tuple structure, costs, worst-fitness)"
  - "Phase 3 complete: single GP tree can be compiled, executed, and evaluated end-to-end"

affects: [04-evolution-engine, 05-validation-publication]

# Tech tracking
tech-stack:
  added: [vectorbt==1.0.0]
  patterns:
    - "Deferred GP imports inside evaluate() — no module-level deap in backtest layer (D-15)"
    - "fee_per_side = (fee_bps / 2.0) / 10_000.0 — round-trip bps to per-side decimal"
    - "upon_opposite_entry='close' with size_type='percent' — handles long-to-short reversals"
    - "sign_changes filter: np.sum(np.abs(np.diff(signals, axis=0)) > 0) across all assets"
    - "worst_fitness = (-np.inf, -np.inf, float(-tree_size)) — rankable by NSGA-II, not NaN"

key-files:
  created:
    - vgp/backtest/runner.py
    - tests/test_evaluate.py
  modified:
    - vgp/backtest/__init__.py

key-decisions:
  - "Deferred GP imports inside evaluate() body preserves D-15 (no deap in backtest) while keeping the function callable"
  - "fee_bps is round-trip; fee_per_side = (fee_bps / 2.0) / 10_000.0 — matches CLAUDE.md #3 (costs inside evaluate)"
  - "upon_opposite_entry='close' prevents double-position accumulation when signal reverses with size_type='percent'"
  - "worst_fitness uses float(-tree_size) not int, ensuring all three elements are Python float for NSGA-II"

patterns-established:
  - "Pattern: evaluate() accepts individual as opaque object — len(individual) gives tree_size without importing deap"
  - "Pattern: BacktestRunner.run() is the Phase 4 DEAP toolbox registration target"

requirements-completed: [EVAL-01, EVAL-02, EVAL-03, EVAL-04]

# Metrics
duration: 5min
completed: 2026-06-09
---

# Phase 3 Plan 03: BacktestRunner and evaluate() Summary

**vectorbt evaluate() with fee_per_side transaction costs, 50-trade sign-change filter, and deap-free import boundary — completing the Phase 3 GP-to-fitness pipeline**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-06-09T16:50:24Z
- **Completed:** 2026-06-09T16:54:58Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Created `vgp/backtest/runner.py` with `EvalConfig` dataclass, `evaluate()` function (no module-level deap import, transaction costs inside, worst-fitness for < 50 trades, NaN guard), and `BacktestRunner` class for Phase 4
- Updated `vgp/backtest/__init__.py` to export `evaluate`, `EvalConfig`, `BacktestRunner` while preserving the architecture invariant docstring
- Created `tests/test_evaluate.py` with 7 tests covering all four EVAL requirements: import boundary audit (D-15), 3-tuple structure, -tree_size convention, non-NaN Sharpe, fee effect comparison, worst-fitness path, and -inf rankability
- Phase 3 complete: 21/21 tests pass across GP-08, GP-05–07, and EVAL-01–04; end-to-end smoke confirms a single GP tree produces a real (Sharpe, total_return, -tree_size) tuple

## End-to-End Smoke Test

```
# Seed 55: T=400, F=12, A=3
result = evaluate(ind, fm, cfg)
# Output: (-0.6249507084964314, -0.05479372344127462, -6.0)
```

## Task Commits

1. **Task 1: Create vgp/backtest/runner.py** — `3e75be9` (feat)
2. **Task 2: Create tests/test_evaluate.py** — `31d38eb` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

| File | Change | Description |
|------|--------|-------------|
| `vgp/backtest/runner.py` | created | EvalConfig, evaluate(), BacktestRunner — vectorbt integration |
| `vgp/backtest/__init__.py` | modified | Re-exports: evaluate, EvalConfig, BacktestRunner |
| `tests/test_evaluate.py` | created | 7 tests — EVAL-01 through EVAL-04 |

## Test Results

```
tests/test_gp_primitives.py: 6/6 PASSED
tests/test_tree_evaluator.py: 8/8 PASSED
tests/test_evaluate.py: 7/7 PASSED
Total: 21 passed in 3.62s
```

## Import Boundary Verification

```python
import sys
before = set(sys.modules.keys())
from vgp.backtest import evaluate, EvalConfig
after = set(sys.modules.keys())
# [m for m in after-before if 'deap' in m] == []
# "boundary OK"
```

## Decisions Made

1. **Deferred GP imports inside evaluate() body:** Using `from vgp.gp.tree_evaluator import TreeEvaluator` and `from vgp.gp.gp_types import build_pset` inside the function body (not at module top-level) maintains the D-15 architecture invariant while keeping evaluate() callable. This passes the `test_backtest_runner_does_not_import_deap_eval01` sys.modules inspection test.

2. **fee_per_side = (fee_bps / 2.0) / 10_000.0:** The EvalConfig stores round-trip basis points (10.0 = 10 bps). The evaluate() function divides by 2 for per-side then by 10_000 for decimal (0.0005). This matches CLAUDE.md constraint #3 (costs inside evaluate, not post-hoc).

3. **upon_opposite_entry='close' with size_type='percent':** Without this, vectorbt accumulates double-positions when a long signal is immediately followed by a short signal (or vice versa) in a percent-sized portfolio. The 'close' disposition closes the existing position first. This is Pitfall 2 from RESEARCH.md.

4. **worst_fitness uses float(-tree_size):** All three elements of the worst-fitness tuple are Python float (not int) to ensure NSGA-II domination comparison works uniformly. -inf is produced by float arithmetic — no special casting needed.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

- **vectorbt dependency on pandas 2.3.3:** Installing vectorbt==1.0.0 downgraded pandas from 3.x to 2.3.3 (vectorbt's dependency constraint). This produces a pip resolver warning. The tests pass because test_evaluate.py uses pandas DatetimeIndex patterns compatible with both versions, and the project's pyproject.toml pin (pandas>=3.0.0) is the correct long-term requirement. This is a pre-existing environment tension documented in RESEARCH.md (vectorbt 1.0.0 vs pandas 3.0 compatibility). Not blocking for Phase 3 test correctness — Phase 4 may need to address this tension.

## Known Stubs

None. evaluate() is fully wired to TreeEvaluator and vectorbt.

## Threat Surface Scan

No new network endpoints, auth paths, or trust boundary changes introduced.

Threat mitigations from Plan threat register:
- T-03-07 (Transaction costs tampered): fees applied inside evaluate() via fee_per_side — confirmed by test_transaction_costs_applied_inside_eval02
- T-03-08 (DoS from degenerate individual): worst_fitness returned (not exception) — confirmed by test_below_50_trades_returns_worst_fitness_eval03
- T-03-10 (deap import boundary): test_backtest_runner_does_not_import_deap_eval01 enforces at test time; runner.py has no module-level deap import (grep verified)

## Next Phase Readiness

Phase 4 (Evolution Engine) can now:
- Import `from vgp.backtest import evaluate, EvalConfig, BacktestRunner`
- Register `runner.run` as the DEAP toolbox evaluation function
- Pass `EvalConfig` with real `close_prices` DataFrame from Phase 2 data pipeline
- Run NSGA-II evolution loop with confidence that fitness tuples are valid for domination sorting
- Note: vectorbt pandas version tension should be resolved in Phase 4 setup

---
*Phase: 03-gp-core-evaluation*
*Completed: 2026-06-09*
