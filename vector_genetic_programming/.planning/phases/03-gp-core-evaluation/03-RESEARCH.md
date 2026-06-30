# Phase 3: GP Core & Evaluation - Research

**Researched:** 2026-06-09
**Domain:** DEAP 1.4.4 Strongly Typed GP + vectorbt 1.0.0 Portfolio evaluation
**Confidence:** HIGH (core APIs verified via Context7 official docs + project smoke test)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**GP Input Geometry**
- D-01: A single GP tree operates on a single-asset [T x F] feature slice — the tree sees one asset's 12 features over T timesteps and produces a [T] signal array. The same compiled tree is applied independently to all 21 assets.
- D-02: The 21 per-asset signal arrays are passed to vectorbt as 21 independent columns — each asset is traded independently, vectorbt handles multi-asset portfolio construction.
- D-03: Cross-asset features remain deferred from Phase 2 D-07 — not in Phase 3 scope.

**Signal Semantics**
- D-04: Tree output is converted to long/short/flat (3-state) signals: sign(output) > 0 → long (+1), sign(output) < 0 → short (-1). Zero-crossing only — no dead band threshold. Every timestep has a direction, and a "trade" = sign change.
- D-05: fshift(1) is applied structurally in TreeEvaluator — shift happens after tree execution, before signal conversion. Signal at time t uses only output from tree execution at t-1. Hard invariant, not a parameter.

**Primitive Set**
- D-06: Minimal math core only:
  - Arithmetic: add(Vector, Vector), sub(Vector, Vector), mul(Vector, Vector), protected_div(Vector, Vector), neg(Vector) — all produce Vector
  - Scalar constants: small set of ephemeral integer constants for tree terminal leaves
  - Rolling aggregations (Vector → Scalar): rolling_mean_5, rolling_mean_20, rolling_std_5, rolling_std_20, rolling_max_20, rolling_min_20
  - Window sizes fixed at 5 and 20. No EphemeralConstant for window sizes.
- D-07: Conditional/comparison primitives (IfThenElse, GreaterThan) are deferred — minimal math core is sufficient for Phase 3.
- D-08: All primitive functions must be module-level in vgp/gp/primitives.py (not lambdas, not closures) for multiprocessing.Pool pickling.
- D-09: creator.Individual and creator.FitnessMulti defined at module level in vgp/gp/gp_types.py — not inside functions.

**Transaction Costs & Portfolio**
- D-10: 10 bps round-trip (5 bps per side) baked into evaluate(). Applied via vectorbt's built-in fees parameter. Configurable via EvalConfig for sensitivity analysis.
- D-11: Equal weight — 1/N across all active positions at each rebalance.
- D-12: Long-short portfolio — both +1 and -1 signals create positions.

**Fitness & Edge Cases**
- D-13: Fitness tuple locked at (Sharpe, total_return, -tree_size) — three objectives.
- D-14: Individuals with fewer than 50 trades receive (-np.inf, -np.inf, -tree_size) — not NaN, not exception. Must be rankable by NSGA-II.

**Architecture Invariants**
- D-15: BacktestRunner must NOT import deap. EvolutionLoop must NOT import vectorbt. Interface: numpy signal array in → fitness tuple out.
- D-16: No per-bar Python loops in the tree execution hot path.

### Claude's Discretion

- Exact numpy broadcasting in arithmetic primitives (output shape is unambiguous from types)
- protected_div behavior for near-zero denominators (small epsilon is standard)
- Exact vectorbt 1.0 Portfolio.from_signals() parameter names for fees and sizing
- How to count "trades" (entry+exit counts vs. sign-change counts)
- Tree initialization method (ramped half-and-half is DEAP default; keep it)

### Deferred Ideas (OUT OF SCOPE)

- Conditional/comparison primitives (IfThenElse, GreaterThan, LessThan) — Phase 4 expansion
- Cross-asset features as GP inputs (BTC return as global factor, cross-sectional rank)
- EphemeralConstant for window sizes — explicitly out of scope
- Domain-aware primitives (crossover, rsi_threshold) — Phase 4+
- Dead band / flat zone threshold
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GP-01 | PrimitiveSetTyped declares Vector and Scalar as distinct Python type tokens | Verified: PrimitiveSetTyped takes explicit type objects; use `np.ndarray` for Vector and a custom `Scalar` sentinel class |
| GP-02 | Arithmetic primitives (+, -, *, protected-div) operate on Vector/Scalar with correct broadcast | Verified: module-level functions accepting np.ndarray, registered with explicit [Vector, Vector] type lists |
| GP-03 | Vector aggregation primitives (mean, std, min, max, rolling variants) reduce Vector → Scalar | Verified: rolling via `numpy.lib.stride_tricks.sliding_window_view`; return last element as Scalar |
| GP-04 | Conditional primitives (if-then-else, comparison) cover all relevant type combinations | DEFERRED by D-07 — not in Phase 3 scope; requirement exists but is out of scope this phase |
| GP-05 | Compiled GP trees broadcast over full [T x F] numpy arrays with no per-bar Python loops | Verified: gp.compile() produces callable; call with per-feature column slices; no loops needed |
| GP-06 | Signal generator converts scalar tree output to directional signals; signal at time t uses only data ≤ t-1 | Verified: `np.sign()` + `np.roll(..., 1)` with index-0 zero-out in TreeEvaluator |
| GP-07 | Lookahead detection test: injects future-leak primitive and asserts fitness is worse than random | Requires injecting `leak_future(x)` primitive that returns `np.roll(x, -1)` and verifying fitness degrades |
| GP-08 | All primitives pass type-correctness unit tests (correct input/output types, no silent numpy cast) | Requires type checking 1000 random trees via PrimitiveSetTyped type annotations |
| EVAL-01 | evaluate(individual) compiles tree, generates signals, calls vectorbt Portfolio.from_signals | Verified: `gp.compile(individual, pset)` + vectorbt API confirmed via docs and smoke test |
| EVAL-02 | Transaction costs applied inside evaluate(), not post-hoc | Verified: `fees=0.0005` parameter in Portfolio.from_signals (5 bps per-side) |
| EVAL-03 | Individuals generating fewer than 50 trades receive worst-possible fitness (not disqualified) | Research clarifies: trade = sign change across [T] signal; count with `np.sum(np.diff(signal) != 0)` per asset, sum across 21 assets |
| EVAL-04 | Fitness tuple is (Sharpe, total_return, -tree_size) | Verified: pf.sharpe_ratio(freq='1D'), pf.total_return(), len(individual) |
</phase_requirements>

---

## Summary

Phase 3 delivers a complete single-tree GP evaluation loop: from tree generation with a strongly typed primitive set, through vectorized execution over a [T x F] feature matrix, to a three-objective fitness tuple computed via vectorbt with transaction costs baked in. The highest-risk items are (1) the Vector/Scalar type system in DEAP's PrimitiveSetTyped, (2) the structural lookahead prevention via fshift, and (3) extracting scalar Sharpe and total_return from vectorbt's grouped multi-asset portfolio.

The DEAP and vectorbt APIs are well-documented and have been verified via Context7 official documentation sources. The project smoke test (`tests/test_smoke.py`) already validates the core vectorbt from_signals API with fees and freq parameters, giving direct confirmation of the exact call signature. One critical environment issue exists: the system Python has numpy==2.3.2 installed (violating the <2.3 requirement), and deap/vectorbt are not installed in any accessible environment. Wave 0 of Phase 3 must create a correctly pinned virtual environment before any implementation.

GP-04 (conditional primitives) is listed in the requirements but is explicitly deferred by CONTEXT.md D-07. The planner should note this requirement as out-of-scope for Phase 3 with reference to D-07.

**Primary recommendation:** Implement in four sequential files — `vgp/gp/gp_types.py` (creator definitions), `vgp/gp/primitives.py` (all primitive functions), `vgp/gp/tree_evaluator.py` (compile + execute + fshift), `vgp/backtest/runner.py` (evaluate() with vectorbt) — with strict import-boundary enforcement between them.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Type system (Vector/Scalar tokens) | GP layer (vgp/gp/) | — | DEAP PrimitiveSetTyped owns type registration; no backend involvement |
| Primitive functions (arithmetic, rolling) | GP layer (vgp/gp/primitives.py) | — | Must be module-level for pickling; pure numpy, no vectorbt |
| Tree compilation (gp.compile) | GP layer (vgp/gp/tree_evaluator.py) | — | DEAP compile is a GP-layer concern |
| Tree execution over [T x F] | GP layer (vgp/gp/tree_evaluator.py) | — | Calls compiled function with numpy column slices; no per-bar loops |
| Signal generation + fshift | GP layer (vgp/gp/tree_evaluator.py) | — | Structurally coupled to tree execution; fshift is an invariant not a parameter |
| Portfolio backtest (vectorbt) | Backtest layer (vgp/backtest/runner.py) | — | Must NOT import deap; only receives numpy signal array |
| Transaction costs | Backtest layer (vgp/backtest/runner.py) | — | Passed as `fees` param to vectorbt; never applied post-hoc |
| Fitness tuple assembly | Backtest layer (vgp/backtest/runner.py) | — | Reads pf.sharpe_ratio(), pf.total_return(), len(individual) |
| 50-trade filter | Backtest layer (vgp/backtest/runner.py) | — | Applied inside evaluate() before calling vectorbt |
| NSGA-II fitness definition | GP layer (vgp/gp/gp_types.py) | — | creator.FitnessMulti at module level; weights define optimization direction |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| deap | 1.4.4 | GP framework: PrimitiveSetTyped, tree generation, gp.compile | Project-pinned; only supported version in requirements-lock.txt |
| vectorbt | 1.0.0 | Portfolio simulation: from_signals, sharpe_ratio, total_return | Project-pinned; 1.0 is a major rewrite — all 0.x docs are wrong API |
| numpy | >=2.0.0,<2.3 | Array operations: primitives, signal generation, fshift | pinned <2.3 for numba compatibility; current lock has 2.2.6 |
| numba | >=0.61.2 | vectorbt JIT compilation (indirect dependency via vectorbt) | Required by vectorbt 1.0; must be installed before Phase 4 parallel eval |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy.lib.stride_tricks | stdlib | Rolling window primitives without per-bar loops | sliding_window_view for rolling_mean_5/20, rolling_std_5/20 |
| operator | stdlib | attrgetter for staticLimit key function | gp.staticLimit(key=operator.attrgetter('height'), max_value=8) |
| dataclasses | stdlib | EvalConfig configuration class pattern | Consistent with Phase 2 DataConfig pattern |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| numpy.lib.stride_tricks.sliding_window_view | pandas rolling().agg() | pandas is slower inside a hot eval loop and violates the "no pandas inside primitives" invariant |
| gp.cxOnePointLeafBiased (Phase 4) | gp.cxOnePoint | For typed GP, cxOnePointLeafBiased maintains type safety; cxOnePoint may generate type-invalid trees in Phase 4 |
| np.sign(output) for 3-state signals | threshold-based discretization | zero-crossing is simpler and avoids hyperparameter; deferred to Phase 4 review |

**Installation (in a fresh venv with correct numpy):**
```bash
# CRITICAL: must pin numpy<2.3 FIRST before installing vectorbt
pip install "numpy>=2.0.0,<2.3"
pip install deap==1.4.4 vectorbt==1.0.0 "numba>=0.61.2"
pip install -r requirements-lock.txt
```

**Version verification:** [VERIFIED: requirements-lock.txt] deap==1.4.4, numpy==2.2.6 in lock file. vectorbt==1.0.0 listed as separate install due to numba dependency.

---

## Architecture Patterns

### System Architecture Diagram

```
Feature matrix [T x F x A]
         |
         | slice per asset: X_a = data[:, :, a]  shape [T x F]
         v
+---------------------------+
|  TreeEvaluator            |
|  1. gp.compile(tree, pset)|---> callable func(ARG0..ARG11)
|  2. func(*[X_a[:, f]      |
|           for f in 0..11])|---> raw_output [T]  (float32 np.ndarray)
|  3. np.roll(raw, 1)[0]=0  |---> shifted_output [T] (fshift structural)
|  4. np.sign(shifted)      |---> signal [T]  values in {-1, 0, +1}
+---------------------------+
         | signal [T] per asset
         | stack 21 signals -> signal_matrix [T x A]
         v
+---------------------------+
|  BacktestRunner           |
|  (no deap import)         |
|  1. trade_count check     |
|     if sum(sign_changes)  |
|     < 50: return worst    |
|  2. pd.DataFrame(signals) |
|  3. vbt.Portfolio.from_   |
|     signals(close,        |
|       long_entries,       |
|       long_exits,         |
|       short_entries,      |
|       short_exits,        |
|       size=1/N,           |
|       size_type='percent',|
|       fees=0.0005,        |
|       freq='1D')          |
|  4. pf.sharpe_ratio()     |
|  5. pf.total_return()     |
|     .mean() [grouped]     |
|  6. return (sharpe,       |
|             total_ret,    |
|             -tree_size)   |
+---------------------------+
         |
         v
    fitness tuple (float, float, float)
    consumed by NSGA-II in Phase 4
```

### Recommended Project Structure

```
vgp/
├── gp/
│   ├── __init__.py          # re-exports: PrimitiveSetTyped, TreeEvaluator, build_pset
│   ├── gp_types.py          # creator.FitnessMulti, creator.Individual at module level
│   ├── primitives.py        # all primitive functions (module-level, np.ndarray in/out)
│   └── tree_evaluator.py    # TreeEvaluator class: compile, execute, signal generation
├── backtest/
│   ├── __init__.py          # re-exports: evaluate, EvalConfig, BacktestRunner
│   └── runner.py            # evaluate(individual, data, config), BacktestRunner class
tests/
├── test_gp_primitives.py    # GP-08: type-correctness, 1000 random trees
├── test_tree_evaluator.py   # GP-05, GP-07: vectorized exec, lookahead detection
└── test_evaluate.py         # EVAL-01 through EVAL-04: full evaluate() contract
```

### Pattern 1: DEAP PrimitiveSetTyped with Vector/Scalar Type Tokens

**What:** Using Python class objects as STGP type tokens allows DEAP to enforce type-safe tree generation. `np.ndarray` is the Vector type; a custom `Scalar` sentinel class is the Scalar type.

**When to use:** Always — this is the only correct pattern for typed GP in DEAP 1.4.4.

**Why a custom Scalar class instead of `float`:** The primitives return `np.float32` from numpy operations, not Python `float`. Using a dedicated `Scalar` sentinel avoids type mismatches with Python builtins and keeps the type system clean.

**Example:**
```python
# Source: DEAP docs https://deap.readthedocs.io/en/master/api/gp.html
# vgp/gp/primitives.py  (note: module-level class, not lambda)

import numpy as np
from deap import gp

# Type tokens — defined once at module level
class Vector:
    """Type token for [T] numpy arrays in the GP type system."""
    pass

class Scalar:
    """Type token for scalar numpy float values."""
    pass

# Build the primitive set
def build_pset() -> gp.PrimitiveSetTyped:
    # 12 inputs = 12 feature columns from FEATURE_NAMES; all are Vector type
    pset = gp.PrimitiveSetTyped(
        "MAIN",
        in_types=[Vector] * 12,  # ARG0..ARG11 = feature columns
        ret_type=Vector,          # tree output is a Vector signal
    )
    # Arithmetic: Vector x Vector -> Vector
    pset.addPrimitive(prim_add, [Vector, Vector], Vector)
    pset.addPrimitive(prim_sub, [Vector, Vector], Vector)
    pset.addPrimitive(prim_mul, [Vector, Vector], Vector)
    pset.addPrimitive(prim_protected_div, [Vector, Vector], Vector)
    pset.addPrimitive(prim_neg, [Vector], Vector)
    # Rolling aggregations: Vector -> Scalar
    pset.addPrimitive(rolling_mean_5, [Vector], Scalar)
    pset.addPrimitive(rolling_mean_20, [Vector], Scalar)
    pset.addPrimitive(rolling_std_5, [Vector], Scalar)
    pset.addPrimitive(rolling_std_20, [Vector], Scalar)
    pset.addPrimitive(rolling_max_20, [Vector], Scalar)
    pset.addPrimitive(rolling_min_20, [Vector], Scalar)
    # Scalar constants: EphemeralConstant for integer terminal leaves
    import random
    pset.addEphemeralConstant("scalar_int", lambda: float(random.randint(-5, 5)), Scalar)
    return pset
```

**IMPORTANT:** `addEphemeralConstant` takes a lambda here only in the pset setup call — the lambda itself is not a primitive function and does not need pickling. Primitive functions do need to be module-level. [VERIFIED: Context7/DEAP docs]

### Pattern 2: Module-Level Creator Definitions

**What:** `creator.create()` must be called at module import time in `vgp/gp/gp_types.py`, not inside any function. This is mandatory for `multiprocessing.Pool` pickling.

**When to use:** Always — violation causes silent `AttributeError` in worker processes.

**Example:**
```python
# Source: DEAP docs https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html
# vgp/gp/gp_types.py

from deap import base, creator, gp

# NSGA-II weights: (Sharpe, total_return, -tree_size)
# Positive weight = maximize; negative weight = minimize.
# We MAXIMIZE Sharpe and total_return, and MINIMIZE tree_size.
# Since fitness.values = (sharpe, total_return, -tree_size), all objectives
# should have positive weight = (1.0, 1.0, 1.0) when values already encode sign.
# Alternative (more standard DEAP): weights=(1.0, 1.0, 1.0) and values=(-tree_size)
# already encodes parsimony pressure.
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)
```

### Pattern 3: Vectorized Rolling Primitives (No Per-Bar Loops)

**What:** Use `numpy.lib.stride_tricks.sliding_window_view` for O(1) memory-efficient rolling windows without Python iteration.

**When to use:** All six rolling aggregation primitives.

**Example:**
```python
# Source: numpy docs — sliding_window_view available since numpy 1.20
# vgp/gp/primitives.py

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def rolling_mean_5(x: np.ndarray) -> np.ndarray:
    """Rolling 5-period mean. Returns same-length array; first 4 values are first valid mean."""
    window = sliding_window_view(x.astype(np.float32), window_shape=5)
    result = np.empty(len(x), dtype=np.float32)
    result[:4] = x[:4]  # pad with input values for first window
    result[4:] = window.mean(axis=-1)
    return result

# IMPORTANT: rolling aggregation output is the LAST element of the result
# This preserves the [T] Vector shape — the Scalar type token in DEAP is a
# logical type tag; the actual numpy return is still [T] shaped, with each
# element being the rolling stat up to that time step.
```

**Design clarification on Scalar vs Vector:** D-06 says rolling aggregations produce `Scalar` — but in practice, the tree evaluator needs to combine rolling outputs with arithmetic primitives back up to a Vector. The correct interpretation is that `Scalar` here means "a single-value rolling statistic at the current timestep" — which, when evaluated over [T] timesteps, produces a [T] array. The Scalar type token in DEAP's type system is a logical annotation, not a literal Python scalar. Both Vector and Scalar map to `np.ndarray` at runtime. The type system exists to prevent type-unsafe tree structures (e.g., adding a rolling_mean output directly to a Vector should not be allowed unless the Scalar is treated as broadcast). [ASSUMED — this interpretation needs planner review; alternative is Scalar = numpy scalar, Vector = 1D array, with explicit broadcast operations]

### Pattern 4: Structural fshift(1) in TreeEvaluator

**What:** After tree execution and before signal conversion, shift the output array by 1 timestep and zero out index 0. This structurally prevents lookahead.

**Example:**
```python
# vgp/gp/tree_evaluator.py

import numpy as np
from deap import gp

class TreeEvaluator:
    def __init__(self, pset):
        self._pset = pset

    def execute(self, individual, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Compile and execute a GP tree over a single-asset [T x F] feature matrix.
        Returns a [T] signal array with values in {-1, 0, +1}.
        fshift(1) is applied STRUCTURALLY here — not optional.
        """
        func = gp.compile(individual, self._pset)
        T, F = feature_matrix.shape
        # Pass each feature column as a separate argument (ARG0..ARG11)
        # This is the vectorized call — no per-bar loop
        raw_output = func(*[feature_matrix[:, f] for f in range(F)])  # shape [T]

        # Structural fshift(1): signal at time t uses output from t-1
        shifted = np.roll(raw_output, 1)
        shifted[0] = 0.0  # first bar has no prior output — flat

        # Convert to 3-state signal
        signal = np.sign(shifted).astype(np.float32)
        return signal
```

### Pattern 5: vectorbt evaluate() — Correct API for 1.0.0

**What:** The verified vectorbt 1.0.0 API for long/short portfolio with per-side fees.

**Key parameters confirmed from docs + smoke test:**
- `fees=0.0005` — 5 bps per side (10 bps round-trip). Applied as fraction, not bps.
- `freq='1D'` — CRITICAL: without this, `sharpe_ratio()` returns NaN silently.
- `direction='both'` OR explicit `short_entries`/`short_exits` arrays.
- `size=1/N, size_type='percent'` — equal weight 1/N.
- `group_by=True, cash_sharing=True` — to get portfolio-level (not per-asset) metrics.

**Example:**
```python
# Source: vectorbt.dev docs + tests/test_smoke.py confirmation
# vgp/backtest/runner.py  (NO deap import in this file)

import numpy as np
import pandas as pd
import vectorbt as vbt

def evaluate(individual, feature_matrix: np.ndarray, config) -> tuple:
    """
    Returns (sharpe, total_return, -tree_size).
    individual: DEAP PrimitiveTree (passed as opaque object; tree_size = len(individual))
    feature_matrix: float32 [T x F x A] — train slice
    config: EvalConfig(fee_bps=10, min_trades=50, freq='1D', init_cash=10_000.0)
    """
    from vgp.gp.tree_evaluator import TreeEvaluator  # late import OK here
    from vgp.gp.gp_types import build_pset          # pset needed for compile

    tree_size = len(individual)
    worst_fitness = (-np.inf, -np.inf, float(-tree_size))

    T, F, A = feature_matrix.shape
    evaluator = TreeEvaluator(build_pset())

    # Execute tree on each asset independently
    signals = np.zeros((T, A), dtype=np.float32)
    for a in range(A):
        signals[:, a] = evaluator.execute(individual, feature_matrix[:, :, a])

    # Count trades: sign changes summed across all assets
    sign_changes = np.sum(np.abs(np.diff(signals, axis=0)) > 0)
    if sign_changes < config.min_trades:
        return worst_fitness

    # Convert 3-state signals to boolean long/short entry/exit arrays
    long_entries  = signals > 0   # [T x A] bool
    short_entries = signals < 0   # [T x A] bool
    # Exit when signal reverses or goes flat
    long_exits  = signals <= 0    # [T x A] bool
    short_exits = signals >= 0    # [T x A] bool

    # Wrap in DataFrame with DatetimeIndex for vectorbt
    # (close prices passed as [T x A] DataFrame from data pipeline)
    fee_per_side = config.fee_bps / 10_000.0  # 10 bps -> 0.001

    pf = vbt.Portfolio.from_signals(
        close=config.close_prices,   # [T x A] DataFrame with DatetimeIndex
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        size=1.0 / A,                # equal weight: 1/N per asset
        size_type='percent',
        fees=fee_per_side,
        freq=config.freq,            # '1D' — REQUIRED for sharpe_ratio
        init_cash=config.init_cash,
        group_by=True,               # aggregate to portfolio level
        cash_sharing=True,
    )

    sharpe = float(pf.sharpe_ratio())
    total_ret = float(pf.total_return())

    # Guard against NaN (e.g., flat portfolio, all trades rejected)
    if np.isnan(sharpe) or np.isnan(total_ret):
        return worst_fitness

    return (sharpe, total_ret, float(-tree_size))
```

**Note on size_type='percent' + position reversal:** The vectorbt docs warn that `SizeType.Percent` does not support position reversal. Use `OppositeEntryMode.Close` or separate long/short entries/exits (the pattern above uses explicit separate entry/exit arrays, which is the correct approach). [VERIFIED: vectorbt.dev docs]

### Pattern 6: NSGA-II Fitness Weights — Correct Sign Convention

**What:** DEAP's `FitnessMulti` uses `weights` to define optimization direction. Positive weight = maximize that objective. The fitness VALUES stored are the raw unweighted values; DEAP multiplies by weights internally for domination checks.

**Key insight:** The fitness tuple is `(sharpe, total_return, -tree_size)`. Setting `weights=(1.0, 1.0, 1.0)` means:
- Maximize sharpe (positive weight, positive value → maximize)
- Maximize total_return (same)
- Maximize -tree_size (same — which is equivalent to minimizing tree_size)

This is the correct convention. The `-tree_size` encoding already handles parsimony pressure within the value itself. [VERIFIED: DEAP docs, Fitness class wvalues property]

### Anti-Patterns to Avoid

- **Lambda primitives:** Lambdas cannot be pickled by `multiprocessing.Pool`. Every GP primitive must be a named module-level function in `vgp/gp/primitives.py`.
- **creator.create() inside functions:** Calling `creator.create()` inside `__init__` or any function body causes `AttributeError` in worker processes because the class is not registered at import time in child processes.
- **Missing freq in Portfolio.from_signals:** Omitting `freq='1D'` causes `pf.sharpe_ratio()` to return NaN silently. The smoke test explicitly validates this.
- **numpy .values on pandas:** All code must use `.to_numpy()`. Phase 2 established this invariant.
- **Importing deap in vgp/backtest/runner.py:** Architecture invariant D-15. Use `len(individual)` for tree size (individual is passed as opaque object, `len()` works without importing deap).
- **Per-bar Python loops in TreeEvaluator:** Execute the compiled function with full column arrays — `func(*[X[:, f] for f in range(F)])`. This is a list comprehension for argument unpacking, not a loop over T timesteps.
- **Using old vectorbt 0.x API:** The 0.x API used `Portfolio.from_signals(price, entries, exits)` with different parameter names. 1.0 uses explicit `long_entries`, `short_entries`, etc.
- **gp.cxOnePoint for typed GP:** Standard `cxOnePoint` does not guarantee type-safe subtree exchange for strongly typed GP. Use `gp.cxOnePointLeafBiased` in Phase 4. Phase 3 does not need crossover setup (that's Phase 4). [VERIFIED: DEAP docs]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Rolling window aggregation | manual loop over timesteps | `numpy.lib.stride_tricks.sliding_window_view` | Memory efficient, vectorized, no per-bar loops |
| Tree depth limiting | custom wrapper checking tree height | `gp.staticLimit(key=operator.attrgetter('height'), max_value=8)` | DEAP built-in; handles both crossover and mutation correctly |
| Strongly typed tree generation | custom generation logic | `gp.PrimitiveSetTyped` + `gp.genHalfAndHalf` | DEAP handles type-safe tree construction |
| Portfolio Sharpe/returns | custom Sharpe calculation | `pf.sharpe_ratio(freq='1D')` | vectorbt uses annualized daily Sharpe; custom implementations get annualization wrong |
| Transaction cost application | post-hoc multiplication | `fees=0.0005` in Portfolio.from_signals | Post-hoc application is a correctness bug — GP will overfit to cost-free signals |
| Fitness domination logic | custom Pareto front code | DEAP NSGA-II `selNSGA2` | DEAP implements both 'standard' and 'log' non-dominated sorting algorithms |

**Key insight:** The DEAP + vectorbt combination handles all the hard parts (type-safe tree generation, JIT-compiled portfolio simulation, Pareto domination). Custom implementations of these are almost always slower and contain subtle correctness bugs.

---

## Common Pitfalls

### Pitfall 1: Missing freq Parameter Causes Silent NaN Sharpe

**What goes wrong:** `pf.sharpe_ratio()` returns NaN without raising any error.
**Why it happens:** vectorbt requires `freq` to annualize the Sharpe ratio. Without it, the method cannot compute the annualization factor and returns NaN silently.
**How to avoid:** Always pass `freq='1D'` (or `freq='D'`) to `Portfolio.from_signals`. Make it a required field in `EvalConfig`.
**Warning signs:** Sharpe is NaN in first test run; fitness tuples all have NaN in position 0.
**[VERIFIED: tests/test_smoke.py line 62 — the existing smoke test explicitly documents this gotcha]**

### Pitfall 2: SizeType.Percent Does Not Support Position Reversal

**What goes wrong:** When a signal goes from +1 to -1 (long to short reversal), `size_type='percent'` may not fully close the long before opening the short.
**Why it happens:** `SizeType.Percent` calculates size as a fraction of available cash — but "available cash" during a reversal is not straightforward.
**How to avoid:** Use explicit separate `long_entries`/`short_entries`/`long_exits`/`short_exits` arrays AND set `upon_opposite_entry='close'` (OppositeEntryMode.Close) to ensure the existing position is closed before the new one opens.
**[VERIFIED: vectorbt.dev from_signals docs note on SizeType.Percent]**

### Pitfall 3: numpy.roll Introduces a Periodic Artifact at Index 0

**What goes wrong:** `np.roll(arr, 1)` wraps the last element to position 0. For fshift, this means `shifted[0] = arr[-1]` (the last timestep's value leaks to the first).
**Why it happens:** `np.roll` is a circular shift by design.
**How to avoid:** Always zero out index 0 immediately after rolling: `shifted[0] = 0.0`. This is the structural fshift implementation — it must be explicit, not forgotten.
**Warning signs:** Lookahead detection test GP-07 may pass even with this bug (because it catches forward-looking primitives, not boundary artifacts). Test fshift explicitly in test_tree_evaluator.py.

### Pitfall 4: 50-Trade Filter Counts Wrong

**What goes wrong:** Using `pf.orders.records_readable` count to enforce the 50-trade filter, rather than counting before calling vectorbt.
**Why it happens:** "Trade" is ambiguous — it could mean orders, round-trips, or signal changes.
**How to avoid:** D-04 defines "trade" = sign change. Count BEFORE calling vectorbt: `sign_changes = np.sum(np.abs(np.diff(signals, axis=0)) > 0)` summed across all A assets. With 21 assets and ~610 train timesteps, a strategy that never changes signal gets 0 trades. A strategy that flips every bar gets ~12,800 trades.
**Warning signs:** Strategies that are always long or always short on all assets pass the filter when they shouldn't.

### Pitfall 5: DEAP's PrimitiveTree.height vs len()

**What goes wrong:** Using `len(individual)` for tree depth limit when the intent is tree height.
**Why it happens:** `len(individual)` returns the number of nodes (tree size), not depth. `individual.height` returns the maximum depth.
**How to avoid:** For the depth limit of 8 (from CLAUDE.md), use `gp.staticLimit(key=operator.attrgetter('height'), max_value=8)`. For the `-tree_size` fitness objective, use `len(individual)` (the number of nodes is the correct parsimony measure).
**Warning signs:** Trees "depth-limited to 8" but with thousands of nodes — unlikely, but indicates the wrong key was used.

### Pitfall 6: Environment numpy Version

**What goes wrong:** System Python 3.12 has numpy==2.3.2 installed. The project requirement is numpy<2.3. Running the GP code in the wrong environment causes numba failures in Phase 4.
**Why it happens:** pyenv 3.12.4 system-level numpy was upgraded to 2.3.2 after the project was set up.
**How to avoid:** Wave 0 of Phase 3 must create a fresh virtual environment (`python -m venv .venv`) and install from requirements-lock.txt. Verify numpy==2.2.6 before proceeding.
**Warning signs:** `test_numpy_version_below_2_3` in test_smoke.py fails.

### Pitfall 7: GP-04 Scope Confusion

**What goes wrong:** Planner includes GP-04 (conditional primitives) in Phase 3 implementation plan.
**Why it happens:** GP-04 appears in REQUIREMENTS.md as a Phase 3 requirement, but CONTEXT.md D-07 explicitly defers it.
**How to avoid:** GP-04 is deferred by user decision D-07. Phase 3 should mark GP-04 as "deferred per D-07" and not implement conditional/comparison primitives. The Phase 3 primitive set is strictly: arithmetic + rolling aggregations + scalar constants.

---

## Code Examples

### Verified DEAP Patterns

**Building PrimitiveSetTyped with explicit type list:**
```python
# Source: https://deap.readthedocs.io/en/master/api/gp.html
pset = gp.PrimitiveSetTyped("MAIN", in_types=[Vector] * 12, ret_type=Vector)
pset.addPrimitive(func, [Vector, Vector], Vector)  # arity-2
pset.addPrimitive(func, [Vector], Vector)           # arity-1
pset.addPrimitive(func, [Vector], Scalar)           # aggregation
pset.addEphemeralConstant("name", lambda_fn, Scalar)
```

**Creator at module level:**
```python
# Source: https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html
from deap import base, creator, gp
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)
```

**Compile and call:**
```python
# Source: https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html
func = gp.compile(individual, pset)
result = func(*args)  # args are per-feature column arrays
```

**Depth limit (for Phase 4, but set up in gp_types or tree_evaluator):**
```python
# Source: https://deap.readthedocs.io/en/master/api/tools.html
import operator
depth_limit = gp.staticLimit(key=operator.attrgetter('height'), max_value=8)
# Applied via: toolbox.decorate("mate", depth_limit)
#              toolbox.decorate("mutate", depth_limit)
```

**Tree initialization:**
```python
# Source: https://deap.readthedocs.io/en/master/_modules/deap/gp.html
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
```

### Verified vectorbt 1.0.0 Patterns

**Long-short portfolio with fees:**
```python
# Source: vectorbt.dev API docs + tests/test_smoke.py (project smoke test confirms API)
pf = vbt.Portfolio.from_signals(
    close=close_df,             # pd.DataFrame [T x A], DatetimeIndex
    entries=long_entries,       # np.ndarray bool [T x A]
    exits=long_exits,           # np.ndarray bool [T x A]
    short_entries=short_entries,# np.ndarray bool [T x A]
    short_exits=short_exits,    # np.ndarray bool [T x A]
    size=1.0 / A,               # equal weight
    size_type='percent',
    fees=0.0005,                # 5 bps per side = 10 bps round-trip
    freq='1D',                  # REQUIRED for sharpe_ratio
    init_cash=10_000.0,
    group_by=True,
    cash_sharing=True,
)
sharpe = float(pf.sharpe_ratio())
total_return = float(pf.total_return())
```

**Extracting scalar metrics (grouped portfolio returns one value):**
```python
# Source: vectorbt.dev API docs
# When group_by=True, both methods return a single scalar (wrapped in Series/array)
sharpe = float(pf.sharpe_ratio())       # float after group_by=True
total_return = float(pf.total_return()) # float after group_by=True
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| vectorbt 0.x Portfolio.from_signals(price, entries, exits) | vectorbt 1.0 explicit long/short signal arrays | mid-2025 (1.0 release) | All pre-1.0 tutorials use wrong parameter names |
| gp.genRamped() | gp.genHalfAndHalf() | DEAP 1.0 | genRamped is deprecated, raises FutureWarning |
| Python float as Scalar type token | Custom Scalar sentinel class | Best practice for typed GP | Python float conflicts with numpy float32 return types |

**Deprecated/outdated:**
- `gp.genRamped`: renamed to `gp.genHalfAndHalf`; use the new name
- vectorbt 0.x `Portfolio.from_signals(price, entries, exits)`: wrong API for 1.0.0

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `Scalar` type token returns [T]-shaped arrays (rolling stat at each timestep), not literal Python scalars | Pattern 3 (rolling primitives), GP-03 | If Scalar must be a literal scalar, then Vector→Scalar primitives cannot combine back into Vector trees without an explicit broadcast primitive. Planner must decide whether to add a scalar_to_vector broadcast primitive or redefine Scalar=Vector |
| A2 | `group_by=True, cash_sharing=True` in Portfolio.from_signals produces a single scalar from sharpe_ratio() and total_return() | Pattern 5 (evaluate) | If multi-column result is returned, float() conversion may take first column's value rather than portfolio aggregate |
| A3 | `upon_opposite_entry='close'` (string form) is accepted by vectorbt 1.0 | Pattern 5 (evaluate) | May need `OppositeEntryMode.Close` enum value instead of string |

**If this table is empty:** All other claims in this research were verified or cited — no user confirmation needed.

---

## Open Questions (RESOLVED)

1. **Scalar type token semantics (A1)**
   - What we know: Rolling aggregation functions (rolling_mean_5 etc.) are declared as `Vector → Scalar` in the primitive set
   - What's unclear: If `Scalar` is a separate class from `Vector`, arithmetic primitives typed as `[Vector, Vector] → Vector` cannot accept the output of a rolling primitive. Trees would dead-end at Scalar leaves with no way to feed back into Vector operations.
   - Recommendation: Define `Scalar` as an alias or subtype of `Vector` (or simply use the same `Vector` type token for both), and rely on rolling primitives being Vector-returning in practice. Alternatively, add a `scalar_broadcast(Scalar) → Vector` primitive. The planner should resolve this before implementing the primitive set.
   - **RESOLVED:** `class Scalar(Vector)` — Scalar is defined as a subclass of Vector in `vgp/gp/gp_types.py`. DEAP's `issubclass()` type check accepts Scalar wherever Vector is expected, preventing dead-ends at rolling aggregation outputs. (Plan 03-01)

2. **Close price DataFrame in EvalConfig**
   - What we know: `Portfolio.from_signals` requires close prices as a DataFrame with DatetimeIndex
   - What's unclear: Should `EvalConfig` carry the close prices, or should `evaluate()` derive them from the feature matrix (feature index 3 is `log_close`, not raw close)?
   - Recommendation: Pass the raw close prices separately from the feature matrix to `evaluate()`. The feature matrix is for tree execution; vectorbt needs actual prices for PnL calculation. Add `close_prices: pd.DataFrame` to `EvalConfig` or as an additional parameter to `evaluate()`.
   - **RESOLVED:** `EvalConfig.close_prices: pd.DataFrame` field added. Feature index 3 is `log_close` (not suitable for PnL); raw close prices are passed separately. (Plan 03-03)

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | All | ✓ | 3.12.4 (pyenv) | — |
| numpy | Primitives, evaluation | ✓ | 2.3.2 (WRONG — needs <2.3) | Downgrade to 2.2.x in .venv |
| deap | GP core | ✗ | — | Install deap==1.4.4 in .venv |
| vectorbt | BacktestRunner | ✗ | — | Install vectorbt==1.0.0 in .venv |
| numba | vectorbt JIT (indirect) | ✗ | — | Install numba>=0.61.2 after numpy<2.3 |
| pytest | Testing | ✓ | 9.0.3 (requirements-lock.txt) | — |

**Missing dependencies with no fallback:**
- deap==1.4.4: Wave 0 must create .venv and install before any implementation
- vectorbt==1.0.0: Same
- numba>=0.61.2: Same (install AFTER confirming numpy<2.3)

**Critical environment issue:**
- System-level numpy==2.3.2 violates the project constraint numpy<2.3. A fresh `.venv` must be created with `pip install "numpy>=2.0.0,<2.3"` BEFORE vectorbt or numba. The existing smoke test `test_numpy_version_below_2_3` will fail if run against the system Python.

---

## Security Domain

Security enforcement is not applicable to this phase. The GP core and evaluation layer operates entirely in-process with no external network calls, user input, or authentication boundaries. The only external dependency is reading from a pre-validated feature matrix (validated by Phase 2 pipeline). No ASVS categories apply.

---

## Sources

### Primary (HIGH confidence)
- `/websites/deap_readthedocs_io_en_master` (Context7) — PrimitiveSetTyped, creator.create, gp.compile, staticLimit, genHalfAndHalf, cxOnePointLeafBiased, addEphemeralConstant
- `/websites/vectorbt_dev` (Context7) — Portfolio.from_signals full parameter list, SizeType enum, fees parameter, sharpe_ratio, total_return, freq requirement, direction='both'
- `tests/test_smoke.py` (project codebase) — Directly validates vectorbt 1.0.0 from_signals call signature with fees and freq parameters; confirms sharpe_ratio() NaN behavior when freq missing

### Secondary (MEDIUM confidence)
- `vgp/data/feature_engine.py` (project codebase) — FEATURE_NAMES list (12 features), float32 dtype, .to_numpy() idiom
- `vgp/data/splitter.py` (project codebase) — WalkForwardSplitter interface; evaluate() receives pre-sliced train data
- `pyproject.toml` (project codebase) — Pinned dependency versions
- `requirements-lock.txt` (project codebase) — deap==1.4.4, numpy==2.2.6 confirmed

### Tertiary (LOW confidence)
- Rolling window Scalar/Vector type ambiguity (A1) — interpretation based on DEAP type system behavior, not explicitly documented for this exact use case

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all package versions from project files; APIs verified via Context7
- Architecture: HIGH — patterns verified against official docs; smoke test confirms vectorbt API
- Pitfalls: HIGH for documented gotchas (NaN Sharpe, fshift boundary); MEDIUM for type system ambiguity (A1)
- Environment: HIGH — directly probed via bash commands

**Research date:** 2026-06-09
**Valid until:** 2026-09-09 (stable: DEAP 1.4.4 has not been updated; vectorbt 1.0 API is stable)
