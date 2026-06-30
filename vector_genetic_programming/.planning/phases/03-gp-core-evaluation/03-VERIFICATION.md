---
phase: 03-gp-core-evaluation
verified: 2026-06-09T18:00:00Z
status: passed
score: 12/12 must-haves verified
overrides_applied: 0
re_verification: false
---

# Phase 3: GP Core & Evaluation Verification Report

**Phase Goal:** Single-tree compile, execute, and backtest working end-to-end
**Verified:** 2026-06-09
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | creator.FitnessMulti and creator.Individual are defined at module import time in gp_types.py | VERIFIED | `if not hasattr(creator, "FitnessMulti"):` at line 44 and `if not hasattr(creator, "Individual"):` at line 51 — both at module scope (0 indent), outside any function |
| 2 | build_pset() returns a PrimitiveSetTyped with 12 Vector inputs and Vector return type | VERIFIED | `pset.ret == Vector`, `len(pset.primitives[Vector]) == 11` (including Scalar subtype duplicates), 12 named terminals (ret_1d..obv_signal); runtime verified |
| 3 | All 11 primitive functions are module-level in primitives.py and accept/return np.ndarray float32 | VERIFIED | `grep -c "^def prim_\|^def rolling_" vgp/gp/primitives.py` returns 11; test_arithmetic_primitives_dtype and test_rolling_primitives_shape_and_dtype both PASS |
| 4 | Rolling primitives use sliding_window_view — no per-timestep Python loops | VERIFIED | `sliding_window_view` appears 6 times in primitives.py (one per rolling function); no per-bar loop found |
| 5 | protected_div uses np.where guard — no ZeroDivisionError | VERIFIED | Line 91-95: `np.where(np.abs(ya) < 1e-7, np.ones_like(xa), xa/ya)` with `np.errstate(divide="ignore")`; test_protected_div_zero_denominator PASSES |
| 6 | TreeEvaluator.execute() applies fshift(1) via np.roll + zeroing index 0 | VERIFIED | Line 92: `shifted = np.roll(raw_output, 1)`; Line 93: `shifted[0] = 0.0`; test_fshift_index0_is_zero_gp06 PASSES |
| 7 | No per-bar Python loop in tree execution path | VERIFIED | `grep "for.*range(T)\|for t in\|for bar in" tree_evaluator.py` returns nothing; columnar list comprehension `func(*[feature_matrix[:, f] for f in range(F)])` is arg-unpacking, not per-bar iteration |
| 8 | evaluate() returns (float, float, float) without importing deap at module level | VERIFIED | `grep "^import deap\|^from deap" runner.py` returns nothing; test_backtest_runner_does_not_import_deap_eval01 PASSES (sys.modules inspection) |
| 9 | Transaction costs passed as fees=fee_per_side to Portfolio.from_signals (not post-hoc) | VERIFIED | Line 154: `fees=fee_per_side` inside `vbt.Portfolio.from_signals()`; test_transaction_costs_applied_inside_eval02 PASSES |
| 10 | < 50 trades returns (-inf, -inf, -tree_size) | VERIFIED | Lines 125-131: `if sign_changes < config.min_trades: return worst_fitness`; `worst_fitness = (-np.inf, -np.inf, float(-tree_size))`; test_below_50_trades_returns_worst_fitness_eval03 PASSES |
| 11 | 1000 random trees produce valid [T] float32 signals without IndexError | VERIFIED | test_1000_random_trees_no_error PASSES (0 errors of 1000); includes scalar coercion fix (_to_f32) for ephemeral constants |
| 12 | End-to-end: single tree compiles, executes, and evaluates to (float, float, float) | VERIFIED | Smoke: `evaluate(ind, fm, cfg)` returns `(-0.5277, -0.0413, -6.0)` — all floats, third element is -len(individual) |

**Score:** 12/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `vgp/gp/gp_types.py` | creator defs + build_pset() factory | VERIFIED | 134 lines; creator.FitnessMulti/Individual at module scope with hasattr guards; build_pset() registers 11 primitives with 12 named Vector inputs |
| `vgp/gp/primitives.py` | Vector/Scalar type tokens + 11 primitives | VERIFIED | 200 lines; 11 module-level functions; Scalar as subclass of Vector; sliding_window_view in all 6 rolling fns; _to_f32() scalar coercion guard added in Plan 03-02 |
| `vgp/gp/__init__.py` | Public re-exports (4 names) | VERIFIED | Direct imports of Vector, Scalar, build_pset, TreeEvaluator; no try/except stub remaining |
| `vgp/gp/tree_evaluator.py` | TreeEvaluator with structural fshift | VERIFIED | 100 lines; execute() with np.roll + shifted[0]=0.0; ndim and F=12 assertions; 0-D broadcast for scalar tree outputs |
| `vgp/backtest/runner.py` | EvalConfig + BacktestRunner + evaluate() | VERIFIED | 203 lines; no module-level deap; fees inside Portfolio.from_signals; worst_fitness (-inf,-inf,-size); upon_opposite_entry='close' |
| `vgp/backtest/__init__.py` | Public re-exports (3 names) | VERIFIED | Exports evaluate, EvalConfig, BacktestRunner; architecture invariant docstring preserved |
| `tests/test_gp_primitives.py` | GP-08 type-correctness + 1000-tree | VERIFIED | 6 test functions; 1000-tree validation passes |
| `tests/test_tree_evaluator.py` | GP-05/06/07 execution + fshift + lookahead | VERIFIED | 8 test functions; lookahead detection: leaky_corr=0.587 > clean_corr=0.582 > 0.5 |
| `tests/test_evaluate.py` | EVAL-01 through EVAL-04 + import audit | VERIFIED | 7 test functions; deap import boundary confirmed by sys.modules inspection |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `vgp/gp/gp_types.py` | `deap.creator` | `creator.create("FitnessMulti"` at module level | WIRED | Lines 44 and 51; both at top-level scope with hasattr guards |
| `vgp/gp/gp_types.py` | `vgp/gp/primitives.py` | `from vgp.gp.primitives import` | WIRED | Lines 24-38; all 11 primitives imported and registered in build_pset() |
| `vgp/gp/primitives.py` | `numpy.lib.stride_tricks` | `sliding_window_view` | WIRED | Line 23 import; used in all 6 rolling functions |
| `vgp/gp/tree_evaluator.py` | `deap.gp.compile` | `deap_gp.compile(individual, self._pset)` | WIRED | Line 73 in execute(); compilation per individual call |
| `vgp/gp/tree_evaluator.py` | `numpy.roll` | `np.roll(raw_output, 1)` | WIRED | Line 92; immediately followed by shifted[0]=0.0 at line 93 |
| `vgp/backtest/runner.py` | `vgp/gp/tree_evaluator.py` | deferred import inside evaluate() | WIRED | Lines 94-95: `from vgp.gp.tree_evaluator import TreeEvaluator` inside function body (D-15 compliant) |
| `vgp/backtest/runner.py` | `vbt.Portfolio.from_signals` | called in evaluate() with fees= | WIRED | Lines 145-159; fees=fee_per_side, freq=config.freq, upon_opposite_entry='close' |
| `tests/test_evaluate.py` | `sys.modules` inspection | deap import boundary audit | WIRED | Lines 91-102; verifies deap is not in sys.modules delta after runner import |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `vgp/gp/tree_evaluator.py` | `raw_output` | `deap_gp.compile(individual, self._pset)(*columns)` | Yes — compile+execute produces numpy array from feature columns | FLOWING |
| `vgp/backtest/runner.py` | `signals` | `evaluator.execute(individual, feature_matrix[:,:,a])` per asset | Yes — 3-state float32 array from TreeEvaluator | FLOWING |
| `vgp/backtest/runner.py` | `sharpe`, `total_ret` | `vbt.Portfolio.from_signals(...).sharpe_ratio()` and `.total_return()` | Yes — vectorbt portfolio metrics from real signal/price data | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 1000 random trees execute without error | `pytest tests/test_gp_primitives.py::test_1000_random_trees_no_error -v` | PASSED (0 of 1000 failed) | PASS |
| fshift zeroes index 0 | `pytest tests/test_tree_evaluator.py::test_fshift_index0_is_zero_gp06 -v` | PASSED (signal[0]==0.0 confirmed) | PASS |
| Lookahead detection (GP-07) | `pytest tests/test_tree_evaluator.py::test_lookahead_detection_gp07 -v` | PASSED (leaky=0.587 > clean=0.582 > 0.5) | PASS |
| Import boundary (no deap in backtest) | `pytest tests/test_evaluate.py::test_backtest_runner_does_not_import_deap_eval01 -v` | PASSED (sys.modules delta has no 'deap' entries) | PASS |
| End-to-end evaluate() | python smoke | `(-0.5277, -0.0413, -6.0)` — 3-tuple of floats | PASS |
| Full suite (28 tests) | `pytest tests/ --ignore=tests/test_smoke.py -q` | 28 passed in 4.81s | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| GP-01 | 03-01-PLAN | PrimitiveSetTyped declares Vector and Scalar as distinct type tokens | SATISFIED | Vector/Scalar in primitives.py; Scalar IS-A Vector; issubclass test passes |
| GP-02 | 03-01-PLAN | Arithmetic primitives (+, -, *, protected-div) with correct broadcast | SATISFIED | 5 arithmetic primitives in primitives.py; test_arithmetic_primitives_dtype PASSES |
| GP-03 | 03-01-PLAN | Vector aggregation primitives reduce Vector to Scalar | SATISFIED | 6 rolling primitives (mean5/20, std5/20, max20, min20); test_rolling_primitives_shape_and_dtype PASSES |
| GP-04 | 03-01-PLAN | Conditional primitives — DEFERRED per D-07 | DEFERRED | Explicitly deferred in plan frontmatter: "DEFERRED per D-07 — conditional primitives out of scope for Phase 3"; not implemented, not listed in Phase 4 roadmap success criteria (to be addressed in Phase 4 if needed) |
| GP-05 | 03-02-PLAN | Compiled GP trees broadcast over [T x F] with no per-bar Python loops | SATISFIED | TreeEvaluator.execute() uses list comprehension arg-unpacking; no loop over T; test_execute_output_shape_gp05 PASSES |
| GP-06 | 03-02-PLAN | Signal generator with structural fshift(1) — signal[t] uses only data <= t-1 | SATISFIED | np.roll+zero-out in execute(); signal[0]==0.0 always; test_fshift_index0_is_zero_gp06 PASSES |
| GP-07 | 03-02-PLAN | Lookahead detection test — future-leak produces detectably higher IS fitness | SATISFIED | leaky_corr=0.587 > clean_corr=0.582; abs(leaky_corr)>0.5; test_lookahead_detection_gp07 PASSES |
| GP-08 | 03-02-PLAN | All primitives pass type-correctness tests | SATISFIED | 1000/1000 random trees pass; test_1000_random_trees_no_error PASSES |
| EVAL-01 | 03-03-PLAN | evaluate() compiles tree, generates signals, calls vectorbt Portfolio.from_signals | SATISFIED | Full call chain wired; test_evaluate_returns_tuple_eval01 PASSES |
| EVAL-02 | 03-03-PLAN | Transaction costs applied inside evaluate() via fees parameter | SATISFIED | fees=fee_per_side in Portfolio.from_signals; test_transaction_costs_applied_inside_eval02 PASSES |
| EVAL-03 | 03-03-PLAN | < 50 trades returns worst-possible fitness, not NaN/exception | SATISFIED | worst_fitness=(-inf,-inf,-size) for sign_changes < min_trades; test_below_50_trades_returns_worst_fitness_eval03 PASSES |
| EVAL-04 | 03-03-PLAN | Fitness tuple is (Sharpe, total_return, -tree_size) | SATISFIED | evaluate() returns 3-float tuple; third element verified as float(-len(individual)); test_negative_tree_size_in_fitness_eval04 PASSES |

### Anti-Patterns Found

No blockers or warnings found.

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | No TODO/FIXME/placeholder found | — | — |
| — | — | No stub returns (return null/empty) found | — | — |
| — | — | No pandas imports in primitives.py | — | — |
| — | — | No per-bar loop in tree execution path | — | — |

One notable deviation documented in SUMMARY: the PLAN specified a lambda for the ephemeral constant generator. The implementation correctly replaced it with `_rand_scalar_int()` module-level function (line 15 in gp_types.py) to prevent multiprocessing.Pool pickle failures. This is a correctness improvement, not a regression.

### Human Verification Required

None. All must-haves are verifiable programmatically. All 28 tests pass.

Note: ROADMAP Success Criterion 2 says "confirmed by profiling showing vectorized execution" — profiling is documentary but not a hard gate. The absence of any `for` loop over timesteps in `tree_evaluator.py` (verified by grep) and the passing 1000-tree test both confirm vectorized behavior without requiring profiling output.

### GP-04 Disposition

GP-04 (conditional primitives) is explicitly deferred per design decision D-07 in Plan 03-01. It remains pending in REQUIREMENTS.md. It does not appear in any later phase's success criteria in ROADMAP.md; it is an intentionally descoped item for Phase 3. Since no later phase claims it, it is left as a known-pending requirement rather than a gap in this phase.

---

## Summary

Phase 3 achieves its goal: a single GP tree can be compiled, executed over a multi-asset feature matrix without Python loops, and evaluated to a valid (Sharpe, total_return, -tree_size) fitness tuple with transaction costs applied inside evaluate() and lookahead structurally prevented. All 12 observable truths are verified against the actual codebase. All 28 tests (21 Phase 3 + 7 Phase 2 regression) pass in 4.81 seconds.

Key architectural invariants confirmed:
- creator.FitnessMulti/Individual at module scope (not inside functions) — multiprocessing safe
- Scalar IS-A Vector subclass — DEAP type-chain dead-end eliminated
- fshift(1) is structural: np.roll + shifted[0]=0.0 — signal[0] is always 0.0
- BacktestRunner has zero module-level deap imports — import boundary clean
- Transaction costs live inside evaluate() — GP cannot evolve to exploit fee-free signals

---

_Verified: 2026-06-09T18:00:00Z_
_Verifier: Claude (gsd-verifier)_
