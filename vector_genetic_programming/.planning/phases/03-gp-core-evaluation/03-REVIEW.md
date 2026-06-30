---
phase: 03-gp-core-evaluation
reviewed: 2026-06-09T00:00:00Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - vgp/gp/primitives.py
  - vgp/gp/gp_types.py
  - vgp/gp/__init__.py
  - vgp/gp/tree_evaluator.py
  - vgp/backtest/runner.py
  - vgp/backtest/__init__.py
  - tests/test_gp_primitives.py
  - tests/test_tree_evaluator.py
  - tests/test_evaluate.py
findings:
  critical: 1
  warning: 5
  info: 3
  total: 9
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-06-09T00:00:00Z
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Phase 3 implements the GP core (typed primitives, tree evaluator, backtest runner) and a
comprehensive test suite. The architecture invariants — `creator.create()` at module level,
no pandas in primitives, structural `fshift(1)`, no DEAP import in `vgp.backtest` at module
level, transaction costs inside `evaluate()`, and worst-fitness tuple for sparse traders —
are all correctly implemented.

One critical issue exists: `assert` statements are used for input validation in
`TreeEvaluator.execute()`, which are silently disabled under Python's `-O` flag, removing
the only guard against malformed inputs at that layer. Five warnings cover a warm-up padding
inconsistency in the rolling-std primitives, a deferred-import architecture gap, the
`build_pset()` rebuild on every `evaluate()` call, a fragile positional-index assumption in
a test, and a test that can silently pass without exercising its target code path.

---

## Critical Issues

### CR-01: `assert` used for input validation in `TreeEvaluator.execute()`

**File:** `vgp/gp/tree_evaluator.py:60-66`

**Issue:** Both shape-validation guards use bare `assert` statements. Python's `-O`
(optimize) flag disables all `assert` statements at runtime. When Phase 4 launches workers
with `multiprocessing.Pool` it is common to use `-O` or equivalent optimizations. If a
malformed feature matrix (wrong ndim or wrong F) reaches `execute()` in an optimized
process, the assert is silently skipped, the compiled tree runs against garbage data, and
`np.sign` produces a random signal with no error — corrupting fitness silently.

The test `test_shape_assertion_wrong_ndim` and `test_shape_assertion_wrong_F` both rely on
`AssertionError` being raised, meaning those tests would also fail under `-O`.

**Fix:**
```python
# Replace both assert statements with explicit raises:
if feature_matrix.ndim != 2:
    raise ValueError(
        f"feature_matrix must be 2-D [T x F], got shape {feature_matrix.shape}"
    )
if feature_matrix.shape[1] != 12:
    raise ValueError(
        f"Expected F=12 feature columns (matching FEATURE_NAMES), "
        f"got {feature_matrix.shape[1]}"
    )
```

Also update the two tests to catch `ValueError` instead of `AssertionError`:
```python
with pytest.raises(ValueError, match="2-D"):
    ...
with pytest.raises(ValueError, match="F=12"):
    ...
```

---

## Warnings

### WR-01: Rolling-std primitives pad warm-up window with zeros, not input values

**File:** `vgp/gp/primitives.py:152-153, 163-164`

**Issue:** `rolling_std_5` pads positions 0-3 with `np.zeros(4)` and `rolling_std_20`
pads positions 0-18 with `np.zeros(19)`. Every other rolling primitive (`rolling_mean_5/20`,
`rolling_max_20`, `rolling_min_20`) pads with the corresponding input values (`x32[:n]`).

The zero-padding creates an artificial volatility discontinuity: the first `window-1` bars
report std=0.0 regardless of the actual input, then the value jumps to the true rolling std
at bar `window`. Any tree that uses `rolling_std_5` or `rolling_std_20` in a ratio or
threshold comparison will have a systematic artifact in the first bars after fshift. For
20-period std, this affects bars 0-19 (nearly 5% of a 400-bar train set).

Zero-padding is correct for std (std of fewer than 2 values is undefined or 0), but the
inconsistency with the other primitives means tree signals behave differently in their warm-up
period depending on which rolling primitive is used. The docstrings say "Pads first N
positions" without specifying the pad value, which is misleading.

**Fix:** Document the zero-padding explicitly in the docstrings, and consider whether NaN
(with downstream coercion to 0) or a partial-window std (using `ddof=0`) would be more
appropriate. At minimum, update the docstrings:

```python
def rolling_std_5(x: np.ndarray) -> np.ndarray:
    """Rolling 5-period standard deviation. Pads first 4 positions with 0.0.

    NOTE: warm-up padding is 0.0, not input values. Std of fewer than 2
    samples is undefined; 0.0 is used as a conservative neutral value.
    This differs from rolling_mean/max/min which pad with input values.
    """
```

If behavioral consistency is preferred, use the partial-window std:
```python
result[:4] = np.array([x32[:i+1].std() for i in range(4)], dtype=np.float32)
```

### WR-02: `evaluate()` rebuilds `pset` on every call — correctness risk in Phase 4

**File:** `vgp/backtest/runner.py:93-113`

**Issue:** Every call to `evaluate()` executes `build_pset()` and constructs a new
`TreeEvaluator`. In Phase 4, `evaluate()` will be called for every individual in every
generation (e.g., 200 individuals x 50 generations = 10,000 calls per run). More
importantly, `build_pset()` calls `gp.PrimitiveSetTyped(...)` and `pset.addPrimitive(...)`
multiple times on each call. In DEAP, `PrimitiveSetTyped` stores primitives in class-level
or module-level registries in some versions; repeated creation could cause silent state
contamination if the registry is not purely instance-level.

The comment at line 112 acknowledges "Phase 4 will cache" but `BacktestRunner.__init__`
(lines 196-198) already receives `config` and `feature_matrix`, making it the natural place
to cache the pset and evaluator now rather than deferring:

**Fix:** Cache in `BacktestRunner.__init__`:
```python
def __init__(self, config: EvalConfig, feature_matrix: np.ndarray) -> None:
    from vgp.gp.tree_evaluator import TreeEvaluator
    from vgp.gp.gp_types import build_pset
    self._config = config
    self._feature_matrix = feature_matrix
    self._pset = build_pset()
    self._evaluator = TreeEvaluator(self._pset)
```

Then pass the cached evaluator through to `evaluate()` or inline the logic in `run()`.

### WR-03: Deferred import of `vgp.gp.gp_types` in `evaluate()` pulls in DEAP transitively

**File:** `vgp/backtest/runner.py:94-95`

**Issue:** The architecture invariant states "This module must NOT import deap at module
level or inside any function." The module-level invariant is satisfied. However, line 95
does `from vgp.gp.gp_types import build_pset`, and `vgp/gp/gp_types.py` line 12 imports
`from deap import base, creator, gp` at its module level. This means every call to
`evaluate()` transitively imports DEAP.

The test `test_backtest_runner_does_not_import_deap_eval01` only checks that importing
`vgp.backtest.runner` does not pull in DEAP as a side effect (it checks `sys.modules` delta
around the import). It does NOT check that calling `evaluate()` leaves DEAP unimported.
The test therefore passes while the runtime transitive dependency exists.

This is not a bug today because DEAP is always available when `evaluate()` runs. But it
means the architectural boundary is a documentation claim, not an enforced contract: if
someone tries to use `BacktestRunner` in a stripped DEAP-free worker (future optimization),
it will fail at the first `evaluate()` call with `ModuleNotFoundError`.

**Fix:** Acknowledge the transitive dependency explicitly in the module docstring, or
refactor `build_pset` to accept pre-constructed primitives so `gp_types` can be imported
without pulling in DEAP. At minimum, update the invariant comment:

```python
# ARCHITECTURE INVARIANT (D-15):
#   This module does NOT import deap at module level.
#   NOTE: calling evaluate() defers-imports vgp.gp.gp_types, which imports deap
#   transitively. DEAP must therefore be present in the worker environment.
```

### WR-04: `test_lookahead_detection_gp07` uses fragile positional indexing into DEAP primitives list

**File:** `tests/test_tree_evaluator.py:178`

**Issue:** Line 178 uses `leaky_pset.primitives[Vector][-1]` to retrieve the `leak_future`
primitive by relying on it being the last element added to the `Vector` primitives list.
DEAP's `PrimitiveSetTyped.primitives` is an `OrderedDict` keyed by type; `[-1]` on the
resulting list gives the last-registered primitive. This works today because `leak_future`
is added after all standard primitives. However:

1. If `build_pset()` is ever extended (e.g., comparison primitives added in Phase 4 via
   GP-04), the new primitives become the last element and this test silently builds the
   wrong tree — testing a new primitive rather than the leak primitive.
2. The test produces no error: it just tests a non-leaky tree against itself, and the
   `abs(leaky_corr) > 0.5` assertion fails, giving a confusing error message.

**Fix:** Retrieve the primitive by name, not position:
```python
# Find leak_future by name, not positional index
leak_prim = next(
    p for p in leaky_pset.primitives[Vector] if p.name == "leak_future"
)
```

### WR-05: `test_sharpe_not_nan_valid_portfolio_eval04` can silently pass without testing anything

**File:** `tests/test_evaluate.py:135-148`

**Issue:** The test guards the assertion with `if result[0] != -np.inf`. If the random
individual generated with `random.seed(55)` (line 80) produces fewer than 50 sign changes,
`evaluate()` returns worst-fitness and the test body is never executed — it passes vacuously.
The test is supposed to verify "a valid portfolio with freq='1D' must return non-NaN Sharpe"
(i.e., guard against the silent-NaN pitfall), but it never actually confirms that a non-worst
individual is exercised.

**Fix:** Use an individual that is guaranteed to produce enough sign changes. The simplest
approach is to use a purely random signal injected directly, bypassing the GP tree:

```python
def test_sharpe_not_nan_valid_portfolio_eval04(feature_matrix, close_prices):
    """A valid portfolio with freq='1D' must return non-NaN Sharpe (EVAL-04, Pitfall 1)."""
    from vgp.backtest.runner import EvalConfig, evaluate
    from unittest.mock import MagicMock
    import random

    # Use high min_=3/max_=5 depth + many seeds until we get a trading individual
    from deap import creator, gp
    from vgp.gp.gp_types import build_pset
    pset = build_pset()
    cfg = EvalConfig(fee_bps=10.0, min_trades=50, freq="1D",
                     init_cash=10_000.0, close_prices=close_prices)
    result = None
    for seed in range(200):
        random.seed(seed)
        ind = creator.Individual(gp.genHalfAndHalf(pset, min_=3, max_=5))
        result = evaluate(ind, feature_matrix, cfg)
        if result[0] != -np.inf:
            break
    assert result is not None and result[0] != -np.inf, (
        "Could not find a trading individual in 200 seeds — cannot test Sharpe NaN guard"
    )
    assert not np.isnan(result[0]), "sharpe_ratio() returned NaN with freq='1D'"
    assert not np.isnan(result[1]), f"total_return() returned NaN: {result}"
```

---

## Info

### IN-01: Import block appears after function definition in `gp_types.py`

**File:** `vgp/gp/gp_types.py:15-38`

**Issue:** `_rand_scalar_int` is defined at lines 15-22 before the `from vgp.gp.primitives
import ...` block at lines 24-38. The standard Python convention is imports first, then
definitions. While this works correctly at runtime (Python evaluates the module sequentially
and the function definition does not depend on the imports), it is an unusual ordering that
could confuse readers who expect all imports at the top of the file.

**Fix:** Move the import block to immediately after `from deap import base, creator, gp`
(after line 12), before `_rand_scalar_int`:
```python
from deap import base, creator, gp
from vgp.gp.primitives import (
    Vector, Scalar, prim_add, ...
)

def _rand_scalar_int() -> float:
    ...
```

### IN-02: `rolling_std_5` returns `np.float32(0.0)` scalar for scalar input, inconsistent with other primitives

**File:** `vgp/gp/primitives.py:149`

**Issue:** When `x` is a scalar (0-D array), `rolling_std_5` returns `np.float32(0.0)`
(a Python scalar, not an ndarray), while `rolling_std_20` at line 164 returns
`np.float32(0.0)` as well. All other scalar-path returns (`rolling_mean_5`, `rolling_max_20`,
etc.) return the 0-D ndarray directly via `return x32`. The `np.float32(0.0)` return type
is technically compatible (TreeEvaluator coerces via `np.asarray`), but it is inconsistent.

**Fix:** For consistency with the other primitives, return a 0-D ndarray:
```python
if x32.ndim == 0:
    return np.asarray(0.0, dtype=np.float32)  # std of a constant is 0
```

### IN-03: `test_gp_primitives.py` does not test rolling primitives with scalar (0-D) input

**File:** `tests/test_gp_primitives.py:51-66`

**Issue:** `test_rolling_primitives_shape_and_dtype` only tests rolling primitives with
`[T]` array input. The scalar-input branch (lines 118-119, 133-134, etc. in primitives.py)
is untested. Since `rolling_std_5` and `rolling_std_20` return a Python scalar
(`np.float32(0.0)`) rather than a 0-D ndarray on the scalar-input path, a test for the
scalar path would surface the inconsistency noted in IN-02.

**Fix:** Add a scalar-input test:
```python
def test_rolling_primitives_scalar_input():
    """Rolling primitives with 0-D scalar input return 0-D ndarray (GP-08 edge case)."""
    from vgp.gp.primitives import (
        rolling_mean_5, rolling_mean_20, rolling_std_5,
        rolling_std_20, rolling_max_20, rolling_min_20,
    )
    scalar = np.float32(2.5)
    for fn in [rolling_mean_5, rolling_mean_20, rolling_std_5,
               rolling_std_20, rolling_max_20, rolling_min_20]:
        result = fn(scalar)
        result_arr = np.asarray(result, dtype=np.float32)
        assert result_arr.dtype == np.float32, f"{fn.__name__} scalar path dtype wrong"
```

---

_Reviewed: 2026-06-09T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
