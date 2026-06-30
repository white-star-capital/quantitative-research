---
phase: 03-gp-core-evaluation
plan: "01"
subsystem: gp
tags: [deap, numpy, typed-gp, primitives, nsga2, rolling-windows, stride-tricks]

# Dependency graph
requires:
  - phase: 02-data-pipeline
    provides: "FEATURE_NAMES list (12 features), float32 [T×F×A] array dtype convention, .to_numpy() idiom"

provides:
  - "Vector and Scalar type tokens with Scalar as subclass of Vector (DEAP type-chain safety)"
  - "11 module-level GP primitive functions: 5 arithmetic + 6 rolling aggregations"
  - "build_pset() factory returning PrimitiveSetTyped with 12 named inputs (FEATURE_NAMES)"
  - "creator.FitnessMulti (weights=(1,1,1)) and creator.Individual at module level"
  - "vgp.gp package re-exports: Vector, Scalar, build_pset, TreeEvaluator (stub until 03-02)"

affects: [03-02-tree-evaluator, 03-03-backtest-runner, 04-evolution-engine]

# Tech tracking
tech-stack:
  added: [deap==1.4.4, numpy.lib.stride_tricks.sliding_window_view]
  patterns:
    - "Module-level creator.create() with hasattr guard for interactive session safety"
    - "Scalar IS-A Vector subclass to prevent DEAP typed tree dead-ends"
    - "Module-level ephemeral constant generator function (not lambda) for pickling"
    - "sliding_window_view for O(1) memory rolling windows — no per-bar Python loops"
    - "np.where + np.errstate for protected division (no NaN/Inf propagation)"

key-files:
  created:
    - vgp/gp/primitives.py
    - vgp/gp/gp_types.py
    - .gitignore
  modified:
    - vgp/gp/__init__.py

key-decisions:
  - "D-09 implemented: creator.create() at module level with hasattr guard prevents TypeError on repeated import"
  - "Scalar defined as subclass of Vector (class Scalar(Vector)) — resolves DEAP type-chain dead-end for rolling→arithmetic trees"
  - "Ephemeral constant uses _rand_scalar_int() module-level function instead of lambda — required for multiprocessing.Pool pickling"
  - "rolling_std pads first (window-1) positions with zeros (not input values) — std of one element is undefined; zeros is a neutral pad"
  - "rolling_mean/max/min pads with input values — preserves signal continuity for warm-up period"

patterns-established:
  - "Pattern: All GP primitives are module-level functions, accept float32 ndarray, return float32 ndarray"
  - "Pattern: Ephemeral constant generators must be module-level named functions, not lambdas"
  - "Pattern: Type tokens (Vector, Scalar) are empty classes, never instantiated — DEAP uses issubclass() checks"

requirements-completed: [GP-01, GP-02, GP-03]

# Metrics
duration: 4min
completed: 2026-06-09
---

# Phase 3 Plan 01: GP Types and Primitives Summary

**DEAP typed GP foundation: Vector/Scalar type token hierarchy, 11 vectorized primitive functions, and build_pset() factory registering 12 named FEATURE_NAMES inputs with parsimony-pressure NSGA-II fitness**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-06-09T16:31:00Z
- **Completed:** 2026-06-09T16:35:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Created `vgp/gp/primitives.py` with Vector/Scalar type tokens (Scalar as subclass of Vector) and all 11 module-level primitive functions — 5 arithmetic + 6 rolling aggregations using `sliding_window_view`
- Created `vgp/gp/gp_types.py` with `creator.FitnessMulti`/`creator.Individual` at module level (D-09), `build_pset()` factory returning PrimitiveSetTyped with 12 FEATURE_NAMES inputs and 11 registered primitives
- Updated `vgp/gp/__init__.py` to re-export `Vector`, `Scalar`, `build_pset`, `TreeEvaluator` (with graceful fallback for TreeEvaluator until Plan 03-02)
- Set up `.venv` with deap==1.4.4 and numpy==2.2.6 (RESEARCH.md environment issue resolved)
- Added `.gitignore` to exclude `.venv/`, `__pycache__/`, `*.egg-info/` and other generated artifacts

## Task Commits

Each task was committed atomically:

1. **Task 1: Create vgp/gp/gp_types.py — creator definitions and build_pset() factory** - `b6b2448` (feat)
2. **Task 2: Create vgp/gp/primitives.py — Vector/Scalar types and all 11 primitive functions** - `e1bdf54` (feat)

**Plan metadata:** (docs commit below)

## Files Created/Modified

- `vgp/gp/primitives.py` — Vector/Scalar type tokens, 5 arithmetic + 6 rolling primitive functions (all module-level, float32, no pandas)
- `vgp/gp/gp_types.py` — creator.FitnessMulti/Individual at module level, build_pset() factory with 12 FEATURE_NAMES inputs and 11 registered primitives
- `vgp/gp/__init__.py` — Updated with re-exports: Vector, Scalar, build_pset, TreeEvaluator (try/except fallback)
- `.gitignore` — Added to exclude .venv/, __pycache__/, *.egg-info/ and other generated artifacts

## Decisions Made

1. **Scalar IS-A Vector subclass:** `class Scalar(Vector)` ensures DEAP's `issubclass()` type check allows rolling aggregation outputs to satisfy Vector input slots in arithmetic primitives. Without this, trees using rolling ops as arithmetic inputs would be type-invalid. (Resolves Assumption A1 from RESEARCH.md)

2. **Module-level `_rand_scalar_int()` instead of lambda:** DEAP warns that lambda-based ephemeral constants cannot be pickled by `multiprocessing.Pool`. Replacing the lambda with a module-level function eliminates the `RuntimeWarning` and ensures Phase 4 parallel evaluation works correctly.

3. **rolling_std pads with zeros:** Standard deviation of a single element is undefined (mathematically 0 with ddof=0). Padding with zeros is semantically neutral for the warm-up period, whereas padding with input values (as done for mean/max/min) would be misleading for std.

4. **deap==1.4.4 reports version as "1.4":** The `deap.__version__` string is "1.4" not "1.4.4". This is a known packaging quirk — the installed wheel from requirements-lock.txt is the correct 1.4.4 release.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Replaced lambda ephemeral constant with module-level function**
- **Found during:** Task 1 (gp_types.py implementation)
- **Issue:** The plan specified `lambda: float(random.randint(-5, 5))` for the ephemeral constant generator. DEAP raises `RuntimeWarning: Ephemeral scalar_int function cannot be pickled because its generating function is a lambda function` and pickle fails in multiprocessing.Pool workers. This would break Phase 4 parallel evaluation silently.
- **Fix:** Added `_rand_scalar_int()` module-level function in `gp_types.py`; `addEphemeralConstant` calls this function instead of a lambda.
- **Files modified:** `vgp/gp/gp_types.py`
- **Verification:** Re-ran verification with `python -W error` (warnings as errors) — no warnings raised.
- **Committed in:** `b6b2448` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 2 — missing critical pickling fix)
**Impact on plan:** Fix is required for Phase 4 correctness. No scope creep. The plan's comment noted "lambda here is ONLY used during tree initialization — it is NOT a primitive function and does not need to be module-level for pickling" but DEAP's actual implementation warns and fails to pickle lambda-based ephemeral constants in worker processes. Deviation aligns with D-08 spirit.

## Issues Encountered

- **Environment bootstrap (not a plan deviation):** No `.venv` existed; the project `.python-version` is `3.12` (pyenv 3.12.4) but deap/vectorbt were not installed. Created `.venv`, installed numpy==2.2.6 first (satisfying `<2.3` constraint), then installed from `requirements-lock.txt`, then installed vgp in editable mode with `--no-deps`. This is the Wave 0 environment setup described in RESEARCH.md Pitfall 6.

## Known Stubs

| Stub | File | Line | Reason |
|------|------|------|--------|
| `TreeEvaluator = None` | `vgp/gp/__init__.py` | 14 | Forward reference — TreeEvaluator is created in Plan 03-02; try/except allows Plan 03-01 to import cleanly |

The stub does not block this plan's goal (GP types + primitives). It will be resolved in Plan 03-02.

## Next Phase Readiness

Plan 03-02 (TreeEvaluator) can now:
- Import `from vgp.gp import Vector, Scalar, build_pset`
- Call `build_pset()` to get the fully wired PrimitiveSetTyped
- Create `vgp/gp/tree_evaluator.py` which `vgp/gp/__init__.py` already tries to import

Plan 03-03 (BacktestRunner) can import `build_pset` and the primitive functions independently.

No blockers.

---
*Phase: 03-gp-core-evaluation*
*Completed: 2026-06-09*
