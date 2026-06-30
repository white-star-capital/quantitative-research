# Phase 3: GP Core & Evaluation - Pattern Map

**Mapped:** 2026-06-09
**Files analyzed:** 10 (7 new source files + 3 new test files)
**Analogs found:** 9 / 10

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `vgp/gp/gp_types.py` | config/type-def | — | `vgp/data/config.py` | role-match (module-level definitions) |
| `vgp/gp/primitives.py` | utility | transform | `vgp/data/feature_engine.py` | role-match (numpy array transforms, float32 invariant) |
| `vgp/gp/tree_evaluator.py` | service | transform | `vgp/data/splitter.py` | role-match (stateful class, public `.execute()` method) |
| `vgp/backtest/runner.py` | service | request-response | `vgp/data/feature_engine.py` + `tests/test_smoke.py` | partial-match (numpy in → result out; vectorbt API verified in smoke) |
| `vgp/gp/__init__.py` | config | — | `vgp/data/__init__.py` | exact (re-export pattern) |
| `vgp/backtest/__init__.py` | config | — | `vgp/data/__init__.py` | exact (re-export pattern) |
| `tests/test_gp_primitives.py` | test | transform | `tests/test_data_pipeline.py` | exact (pytest structure, import-from-vgp pattern) |
| `tests/test_tree_evaluator.py` | test | transform | `tests/test_data_pipeline.py` | exact (pytest structure) |
| `tests/test_evaluate.py` | test | request-response | `tests/test_smoke.py` + `tests/test_data_pipeline.py` | role-match |
| `vgp/data/config.py` (read-only ref) | config | — | — | source analog only |

---

## Pattern Assignments

### `vgp/gp/gp_types.py` (type-definitions, module-level)

**Analog:** `vgp/data/config.py`

**Rationale:** Both files define module-level objects consumed by the rest of the package. `DataConfig` is a dataclass instantiated at call time; `creator.create()` in `gp_types.py` must be called at *import time* (DEAP pickling requirement D-09). The analogy is structural: one module owns canonical definitions for the whole layer.

**Imports pattern** (`vgp/data/config.py` lines 1-7):
```python
"""Pipeline-wide configuration for data fetching, feature engineering, and splitting."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
```
Apply the same `from __future__ import annotations` header and module-level docstring convention. For `gp_types.py` replace with:
```python
"""DEAP creator definitions — FitnessMulti and Individual at module level.

ARCHITECTURE INVARIANT: creator.create() must be called at module import time,
not inside functions. multiprocessing.Pool workers re-import this module; if
creator.create() is inside a function it will not run in workers, causing
AttributeError on Individual.fitness.
"""
from __future__ import annotations

from deap import base, creator, gp
```

**Core pattern — module-level creator.create() calls** (RESEARCH.md Pattern 2):
```python
# NSGA-II weights: positive = maximize.
# values=(sharpe, total_return, -tree_size) — all three positive weights
# because -tree_size already encodes parsimony pressure in the value itself.
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)
```

**No error handling needed** — `creator.create()` raises `TypeError` if called twice with conflicting definitions. A module-level `try/except AttributeError` guard is the correct defensive pattern:
```python
# Guard: repeated import in interactive sessions should not raise
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)
```

---

### `vgp/gp/primitives.py` (utility, transform)

**Analog:** `vgp/data/feature_engine.py`

**Rationale:** Both are pure numpy transform modules — `feature_engine.py` maps OHLCV DataFrames to float32 arrays; `primitives.py` maps float32 arrays to float32 arrays. Key shared conventions: float32 throughout, `.to_numpy()` not `.values`, defensive zero-division guards, module-level functions (not lambdas).

**Imports pattern** (`vgp/data/feature_engine.py` lines 17-23):
```python
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
```
For `primitives.py` drop `pandas` (no pandas inside primitives — CLAUDE.md invariant) and add stride_tricks:
```python
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
```

**Float32 enforcement** (`vgp/data/feature_engine.py` line 172):
```python
arrays = [
    feat_df.to_numpy(dtype=np.float32) for feat_df in per_asset_trimmed
]
```
All primitive return arrays must be `.astype(np.float32)` — maintain dtype throughout, matching `FeatureEngine` output dtype.

**Division-by-zero guard pattern** (`vgp/data/feature_engine.py` lines 261-264):
```python
roll_range = roll_max - roll_min
norm_close = (close - roll_min) / roll_range
norm_close = norm_close.where(roll_range != 0.0, other=0.5)
```
Apply the same defensive pattern in `protected_div`:
```python
def protected_div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Protected division: returns x/y, substituting 1.0 where |y| < epsilon."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(y) < 1e-7, np.ones_like(x, dtype=np.float32), x / y)
    return result.astype(np.float32)
```

**Rolling window pattern** (`vgp/data/feature_engine.py` lines 231-232 — pandas rolling; replace with stride_tricks per RESEARCH.md Pattern 3):
```python
# Pandas original (for reference):
vol_5d = close.pct_change(1).rolling(5).std()

# Primitives equivalent — numpy only, no pandas:
def rolling_mean_5(x: np.ndarray) -> np.ndarray:
    """Rolling 5-period mean. Pads first 4 values with input values."""
    window = sliding_window_view(x.astype(np.float32), window_shape=5)
    result = np.empty(len(x), dtype=np.float32)
    result[:4] = x[:4].astype(np.float32)
    result[4:] = window.mean(axis=-1).astype(np.float32)
    return result
```

**NaN guard pattern** (`vgp/data/feature_engine.py` lines 179-183):
```python
if np.isnan(result).any():
    raise ValueError(
        "FeatureEngine output contains NaN after lookback trim. "
        "Check rolling windows and ffill logic."
    )
```
Primitives do NOT raise on NaN — that would kill the GP loop. Instead, NaN propagation is acceptable within primitives; the `evaluate()` function guards NaN in fitness values (runner.py).

**Type token definitions** — place at the top of `primitives.py` so `build_pset()` can import them:
```python
class Vector:
    """Type token for [T] numpy arrays in the DEAP PrimitiveSetTyped type system."""

class Scalar:
    """Type token for rolling aggregation outputs (also [T]-shaped at runtime; logical distinction only)."""
```

**FEATURE_NAMES terminal naming** (`vgp/data/feature_engine.py` lines 27-40):
```python
FEATURE_NAMES: list[str] = [
    "ret_1d",       # index 0  -> ARG0
    "ret_5d",       # index 1  -> ARG1
    "ret_20d",      # index 2  -> ARG2
    "log_close",    # index 3  -> ARG3
    "vol_5d",       # index 4  -> ARG4
    "vol_20d",      # index 5  -> ARG5
    "atr_14",       # index 6  -> ARG6
    "parkinson_14", # index 7  -> ARG7
    "rsi_14",       # index 8  -> ARG8
    "norm_close",   # index 9  -> ARG9
    "vol_ratio_20d",# index 10 -> ARG10
    "obv_signal",   # index 11 -> ARG11
]
```
`build_pset()` must rename ARG0..ARG11 to match FEATURE_NAMES via `pset.renameArguments(ARG0="ret_1d", ...)` for readable tree string representation.

---

### `vgp/gp/tree_evaluator.py` (service, transform)

**Analog:** `vgp/data/splitter.py`

**Rationale:** Both are stateful classes with a single dominant public method: `WalkForwardSplitter.split()` → `TreeEvaluator.execute()`. Both accept a primary data argument and perform a structural invariant (ordering assertion / fshift). Both use clear docstrings documenting the invariant behavior.

**Class structure pattern** (`vgp/data/splitter.py` lines 20-36):
```python
class WalkForwardSplitter:
    """
    Slice data along the time axis into non-overlapping train / val / test sets.
    ...
    Parameters
    ----------
    None — all configuration is passed per ``split()`` call.
    """

    def split(
        self,
        data,
        train_end: str,
        ...
    ) -> tuple:
        """
        Split data into (train, val, test) along the time axis.

        Structural assertions
        ---------------------
        * ``val_start`` must be strictly after ``train_end`` — enforced as
          ``AssertionError`` (D-11 requirement).
        ...
        """
```
Apply the same docstring + "Structural invariants" pattern in `TreeEvaluator.execute()` to document fshift:
```python
class TreeEvaluator:
    """
    Compile and execute a GP tree over a single-asset [T x F] feature matrix.

    Parameters
    ----------
    pset : gp.PrimitiveSetTyped
        The primitive set used for tree compilation.
    """

    def __init__(self, pset) -> None:
        self._pset = pset

    def execute(self, individual, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Execute a GP tree and return a [T] signal array.

        Structural invariant (D-05)
        ---------------------------
        fshift(1) is applied INSIDE this method after tree execution.
        Signal at time t uses only tree output from t-1. This is structural,
        not configurable.

        Parameters
        ----------
        individual : creator.Individual (opaque — no deap import needed by caller)
            Compiled GP tree.
        feature_matrix : np.ndarray
            Shape [T x F], dtype float32. Single-asset feature slice.

        Returns
        -------
        np.ndarray
            Shape [T], dtype float32, values in {-1.0, 0.0, +1.0}.
        """
```

**Logging pattern** (`vgp/data/splitter.py` lines 17-18, 99-104):
```python
logger = logging.getLogger(__name__)
...
logger.info(
    "Split summary — train: %d rows, val: %d rows, test: %d rows",
    len(train), len(val), len(test),
)
```
`TreeEvaluator` does not need per-call logging (hot path — Phase 4 parallel eval). Add only module-level `logger = logging.getLogger(__name__)` for future use.

**Structural invariant enforcement** (`vgp/data/splitter.py` lines 81-85):
```python
assert pd.Timestamp(val_start) > pd.Timestamp(train_end), (
    f"val_start ({val_start}) must be strictly after train_end ({train_end})"
)
```
Apply the same assertion pattern for shape validation:
```python
assert feature_matrix.ndim == 2, (
    f"feature_matrix must be 2-D [T x F], got shape {feature_matrix.shape}"
)
assert feature_matrix.shape[1] == 12, (
    f"Expected F=12 feature columns, got {feature_matrix.shape[1]}"
)
```

**fshift core pattern** (RESEARCH.md Pattern 4):
```python
from deap import gp as deap_gp
import numpy as np

func = deap_gp.compile(individual, self._pset)
T, F = feature_matrix.shape
# Vectorized call — pass each feature column as a separate argument
raw_output = func(*[feature_matrix[:, f] for f in range(F)])  # shape [T]
# Ensure float32
raw_output = np.asarray(raw_output, dtype=np.float32)
# Structural fshift(1): np.roll wraps last->first; zero out index 0 explicitly
shifted = np.roll(raw_output, 1)
shifted[0] = 0.0  # no prior output on first bar — flat
signal = np.sign(shifted).astype(np.float32)
return signal
```

---

### `vgp/backtest/runner.py` (service, request-response)

**Analog:** `tests/test_smoke.py` (vectorbt API pattern) + `vgp/data/config.py` (EvalConfig structure)

**Rationale:** `runner.py` contains no close Phase 2 structural analog (it's the first vectorbt-calling production file). The smoke test (`test_smoke.py`) provides the verified vectorbt 1.0.0 call signature. `DataConfig` provides the dataclass config pattern for `EvalConfig`.

**ARCHITECTURE INVARIANT** (`vgp/backtest/__init__.py` line 3):
```
ARCHITECTURE INVARIANT: This module must NOT import deap.
```
`runner.py` must never have `import deap` or `from deap import ...` at module level. `len(individual)` works without importing deap (PrimitiveTree implements `__len__`).

**EvalConfig pattern** — copy `DataConfig` from `vgp/data/config.py` lines 8-29:
```python
@dataclass
class DataConfig:
    # Date boundaries (D-10)
    start_date: str = "2021-01-01"
    train_end: str = "2023-12-31"
    ...
    # Fetcher params
    interval: str = "1d"
    cache_dir: Path = field(default_factory=lambda: Path("vgp/data/cache"))
```
Apply the same `@dataclass` + grouped-by-concern + inline-comment pattern for `EvalConfig`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

@dataclass
class EvalConfig:
    """Configuration for evaluate() — backtest parameters and constraints."""

    # Transaction costs (D-10)
    fee_bps: float = 10.0           # round-trip fee in basis points (5 bps per side)

    # Trade filter (D-14)
    min_trades: int = 50            # individuals below this receive worst fitness

    # Portfolio parameters (D-11, D-12)
    freq: str = "1D"                # REQUIRED for sharpe_ratio() — must not be omitted
    init_cash: float = 10_000.0

    # Close prices for vectorbt (required — not derivable from feature matrix)
    close_prices: pd.DataFrame = field(default_factory=pd.DataFrame)
```

**vectorbt call pattern** (`tests/test_smoke.py` lines 71-88 — verified working API):
```python
pf = vbt.Portfolio.from_signals(
    price,
    entries,
    exits,
    size=1,
    direction="longonly",
    fees=0.001,
    freq="1D",
    init_cash=10_000.0,
)
sharpe = pf.sharpe_ratio()
total_return = pf.total_return()
assert not np.isnan(float(sharpe)), "sharpe_ratio() returned NaN — freq='1D' must be set"
```
Extend to long-short multi-asset (RESEARCH.md Pattern 5):
```python
fee_per_side = config.fee_bps / 10_000.0  # 10 bps → 0.001

pf = vbt.Portfolio.from_signals(
    close=config.close_prices,        # [T x A] DataFrame with DatetimeIndex
    entries=long_entries,             # [T x A] bool
    exits=long_exits,                 # [T x A] bool
    short_entries=short_entries,      # [T x A] bool
    short_exits=short_exits,          # [T x A] bool
    size=1.0 / A,                     # equal weight 1/N
    size_type="percent",
    fees=fee_per_side,
    freq=config.freq,                 # "1D" — REQUIRED
    init_cash=config.init_cash,
    group_by=True,                    # aggregate to portfolio level
    cash_sharing=True,
)
```

**50-trade filter and worst-fitness pattern** (RESEARCH.md EVAL-03):
```python
tree_size = len(individual)          # no deap import needed; PrimitiveTree.__len__
worst_fitness = (-np.inf, -np.inf, float(-tree_size))

sign_changes = int(np.sum(np.abs(np.diff(signals, axis=0)) > 0))
if sign_changes < config.min_trades:
    return worst_fitness
```

**NaN guard pattern** — modelled after `vgp/data/feature_engine.py` lines 179-183 (raise on NaN in data; return worst fitness on NaN in metrics):
```python
if np.isnan(sharpe) or np.isnan(total_ret):
    return worst_fitness
```

**Imports for runner.py** (no deap — invariant D-15):
```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import vectorbt as vbt

from vgp.gp.tree_evaluator import TreeEvaluator
from vgp.gp.primitives import build_pset
```

---

### `vgp/gp/__init__.py` (re-export module)

**Analog:** `vgp/data/__init__.py` lines 1-21 (exact pattern)

**Pattern to copy:**
```python
"""Data pipeline: DataLoader, FeatureEngine, WalkForwardSplitter."""

from .universe import UNIVERSE_30, get_binance_symbols
from .fetcher import BinanceFetcher
from .feature_engine import FeatureEngine
from .splitter import WalkForwardSplitter
from .config import DataConfig

DataLoader = BinanceFetcher

__all__ = [
    "UNIVERSE_30",
    ...
]
```
Apply exactly — replace with GP exports:
```python
"""GP core: PrimitiveSetTyped, primitives, tree evaluation, signal generation.

IMPORTANT: All primitive functions must be defined at module level (not inside
functions or lambdas) for multiprocessing.Pool pickling compatibility.
All primitives accept and return np.ndarray — no pandas objects inside primitives.
"""

from .gp_types import build_toolbox         # when added in Phase 4
from .primitives import build_pset, Vector, Scalar
from .tree_evaluator import TreeEvaluator

__all__ = [
    "build_pset",
    "Vector",
    "Scalar",
    "TreeEvaluator",
]
```
Note: the existing `vgp/gp/__init__.py` docstring is already correct — preserve it and add the exports below it.

---

### `vgp/backtest/__init__.py` (re-export module)

**Analog:** `vgp/data/__init__.py` (exact pattern)

The existing stub docstring is correct. Add exports following the same `__all__` pattern:
```python
"""Backtest runner: vectorbt integration, evaluate(), fitness functions.

ARCHITECTURE INVARIANT: This module must NOT import deap.
Interface from GP: numpy array in -> fitness tuple out.
"""

from .runner import evaluate, EvalConfig, BacktestRunner

__all__ = [
    "evaluate",
    "EvalConfig",
    "BacktestRunner",
]
```

---

### `tests/test_gp_primitives.py` (test, GP-08)

**Analog:** `tests/test_data_pipeline.py` — exact structure pattern

**File header pattern** (`tests/test_data_pipeline.py` lines 1-15):
```python
"""
Data pipeline integration tests -- DATA-04.

Uses data_pipeline_example/cache/ parquet files as the test fixture.
No network access required: all tests run against pre-cached data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURE_CACHE = Path(__file__).parent.parent / "data_pipeline_example" / "cache"
```
Apply same style for GP test header:
```python
"""
GP primitive unit tests -- GP-08.

Type-correctness validation: all primitives accept np.ndarray and return np.ndarray
with dtype float32. 1000 random trees are generated and executed to verify
PrimitiveSetTyped enforces type safety throughout.
No network access. No deap evolution — only tree generation and execution.
"""
from __future__ import annotations

import numpy as np
import pytest
```

**Test function naming pattern** (`tests/test_data_pipeline.py` lines 18-32):
```python
def test_pipeline_no_nan():
    """Full pipeline: fetch_ohlcv -> fit_transform -> assert zero NaN (DATA-04)."""
    from vgp.data import BinanceFetcher, FeatureEngine
    ...
    assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"
```
Apply same `assert ... , f"..."` style — the `f"..."` error message pattern is consistent across all Phase 2 tests.

**Import-inside-test pattern** (`tests/test_data_pipeline.py` line 21):
```python
from vgp.data import BinanceFetcher, FeatureEngine
```
Use the same late-import pattern in test functions to ensure ImportError is caught as a test failure, not a collection error.

**pytest.raises pattern** (`tests/test_data_pipeline.py` lines 52-64):
```python
def test_splitter_ordering_assertion():
    """WalkForwardSplitter raises AssertionError when val_start <= train_end (DATA-03)."""
    from vgp.data import WalkForwardSplitter

    splitter = WalkForwardSplitter()
    with pytest.raises(AssertionError):
        splitter.split(
            data=pd.DataFrame(),
            train_end="2024-01-01",
            val_start="2023-06-01",   # before train_end -- must raise
            ...
        )
```
Apply for GP-08 — e.g., test that a primitive called with wrong shape raises, or that a 0-length array is handled.

---

### `tests/test_tree_evaluator.py` (test, GP-05 + GP-07)

**Analog:** `tests/test_data_pipeline.py` + `tests/test_smoke.py`

**File header pattern** — same as `test_gp_primitives.py` above.

**Lookahead detection test structure** (GP-07, RESEARCH.md requirement):
```python
def test_lookahead_detection_gp07():
    """GP-07: injecting a future-leak primitive produces worse fitness than a valid tree."""
    # Step 1: define a leak_future primitive that returns np.roll(x, -1) — future data
    # Step 2: build a pset that includes leak_future
    # Step 3: construct a trivial tree that uses leak_future
    # Step 4: evaluate both valid and leaking trees
    # Step 5: assert leaking tree has higher (better) in-sample fitness
    #         (it "sees" the future so it should trade perfectly in-sample)
    # This test PASSES when the leaking tree fitness > valid tree fitness
    # i.e., the test confirms that lookahead is detectable and leads to overfit
```

**Vectorized execution test** (GP-05):
```python
def test_vectorized_execution_no_loops_gp05():
    """GP-05: tree evaluation must not iterate per-bar; output shape is [T]."""
    # Execute tree on [T x F] array and assert output.shape == (T,)
    # Time the execution and assert < threshold (e.g., 1s for T=1000)
```

---

### `tests/test_evaluate.py` (test, EVAL-01 through EVAL-04)

**Analog:** `tests/test_smoke.py` (request-response test pattern)

**smoke test NaN assertion pattern** (`tests/test_smoke.py` lines 83-88):
```python
assert not np.isnan(float(sharpe)), (
    "sharpe_ratio() returned NaN — this is the silent failure mode when freq= is missing. "
    "Check that freq='1D' is passed to Portfolio.from_signals."
)
```
Apply this exact NaN-guard-as-test pattern throughout `test_evaluate.py`.

**Single-import-per-test pattern** (`tests/test_smoke.py` lines 53-55):
```python
def test_deap_imports():
    """Verify all required DEAP sub-modules import cleanly."""
    import deap  # noqa: F401
    from deap import algorithms, base, creator, gp, tools  # noqa: F401
```
Apply for EVAL-01 (import boundary check):
```python
def test_backtest_runner_does_not_import_deap_eval01():
    """EVAL-01: runner.py must not import deap at module level (D-15)."""
    import importlib, sys
    # Ensure deap is NOT imported as a side effect of importing runner
    before = set(sys.modules.keys())
    import vgp.backtest.runner  # noqa: F401
    after = set(sys.modules.keys())
    new_imports = after - before
    assert not any("deap" in m for m in new_imports), (
        f"vgp.backtest.runner imported deap: {[m for m in new_imports if 'deap' in m]}"
    )
```

---

## Shared Patterns

### from __future__ import annotations
**Source:** Every Phase 2 file (`vgp/data/config.py` line 1, `vgp/data/feature_engine.py` line 17, `vgp/data/splitter.py` line 11)
**Apply to:** All new source files
```python
from __future__ import annotations
```

### Float32 dtype enforcement
**Source:** `vgp/data/feature_engine.py` line 172
```python
arrays = [
    feat_df.to_numpy(dtype=np.float32) for feat_df in per_asset_trimmed
]
result = np.stack(arrays, axis=2)  # shape: [T, F, A]
```
**Apply to:** All primitive functions in `primitives.py`, all array operations in `tree_evaluator.py`
Convention: `.astype(np.float32)` at the return boundary; intermediate operations may upcast but must return float32.

### Module-level logger
**Source:** `vgp/data/feature_engine.py` line 23, `vgp/data/splitter.py` line 17
```python
import logging
logger = logging.getLogger(__name__)
```
**Apply to:** `vgp/gp/tree_evaluator.py`, `vgp/backtest/runner.py` (not primitives.py — hot path)

### ValueError for data integrity violations
**Source:** `vgp/data/feature_engine.py` lines 179-183, `vgp/data/splitter.py` lines 112-116
```python
if np.isnan(result).any():
    raise ValueError(
        "FeatureEngine output contains NaN after lookback trim. "
        "Check rolling windows and ffill logic."
    )
```
```python
if dates is None:
    raise ValueError(
        "dates kwarg is required when data is np.ndarray. "
        "Pass dates=engine.dates_ from FeatureEngine.fit_transform()."
    )
```
**Apply to:** `vgp/gp/tree_evaluator.py` shape assertions (use `AssertionError` per splitter pattern), `vgp/backtest/runner.py` input validation.
**Exception:** GP fitness edge cases (NaN fitness, < 50 trades) return worst_fitness tuple — NOT raise. Only data integrity failures should raise.

### Defensive zero-division with np.where
**Source:** `vgp/data/feature_engine.py` lines 261-264
```python
norm_close = (close - roll_min) / roll_range
norm_close = norm_close.where(roll_range != 0.0, other=0.5)
```
**Apply to:** `protected_div` in `primitives.py` — use `np.where(np.abs(y) < 1e-7, ...)` for numpy arrays.

### f-string error messages in assertions
**Source:** `tests/test_data_pipeline.py` throughout (lines 29-33, 44-47, etc.)
```python
assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"
assert arr.shape[1] == 12, f"Expected F=12 features, got {arr.shape[1]}"
```
**Apply to:** All test assertions in the three new test files.

### Late imports in test functions (no module-level vgp imports)
**Source:** `tests/test_data_pipeline.py` lines 21, 41, 53, etc.
```python
def test_pipeline_no_nan():
    from vgp.data import BinanceFetcher, FeatureEngine
```
**Apply to:** All three new test files — import `from vgp.gp import ...` inside each test function body, not at module top-level. This ensures ImportError manifests as a test failure, not a collection-time error.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (none) | — | — | All files have analogs; RESEARCH.md code examples fill any remaining gaps |

**Closest gaps (partial matches only):**
- `vgp/backtest/runner.py` has no production analog for the vectorbt portfolio call. The smoke test (`tests/test_smoke.py` lines 58-90) provides the verified API call pattern but is a test, not a service. The RESEARCH.md Pattern 5 code example is the primary reference for the multi-asset long-short call.
- `vgp/gp/gp_types.py` has no analog for `creator.create()` — the RESEARCH.md Pattern 2 code example is authoritative.

---

## Metadata

**Analog search scope:** `/Users/ale/Documents/quantitative-research/vector_genetic_programming/vgp/`, `tests/`
**Files scanned:** 11 (all Python files in vgp/ and tests/)
**Pattern extraction date:** 2026-06-09
