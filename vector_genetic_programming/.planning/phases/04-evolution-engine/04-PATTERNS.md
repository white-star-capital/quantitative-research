# Phase 4: Evolution Engine - Pattern Map

**Mapped:** 2026-06-09
**Files analyzed:** 8 new/modified files
**Analogs found:** 8 / 8

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `vgp/evolution/config.py` | config | — | `vgp/backtest/runner.py` (`EvalConfig`) | exact |
| `vgp/evolution/loop.py` | service | event-driven (per-generation) | `vgp/backtest/runner.py` (`evaluate`) | role-match |
| `vgp/evolution/checkpoint.py` | utility | file-I/O | `vgp/backtest/runner.py` (module structure) | partial |
| `vgp/evolution/__init__.py` | config | — | `vgp/backtest/__init__.py` | exact |
| `vgp/gp/primitives.py` (MODIFY) | utility | transform | `vgp/gp/primitives.py` (existing primitives) | exact |
| `vgp/gp/gp_types.py` (MODIFY) | config | — | `vgp/gp/gp_types.py` (`build_pset`) | exact |
| `pyproject.toml` (MODIFY) | config | — | `pyproject.toml` (`[tracking]` extra) | exact |
| `tests/test_evolution.py` | test | request-response | `tests/test_evaluate.py` | exact |

---

## Pattern Assignments

### `vgp/evolution/config.py` (config dataclass)

**Analog:** `vgp/backtest/runner.py` lines 27–51 (`EvalConfig`)

**Imports pattern** (runner.py lines 11–14):
```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
```

**Core dataclass pattern** (runner.py lines 27–51):
```python
@dataclass
class EvalConfig:
    """Configuration for evaluate() — backtest parameters and trade constraints.

    All parameters have safe defaults matching the Phase 3 design decisions.
    Override fee_bps for sensitivity analysis in Phase 4/5.
    """

    # Transaction costs (D-10): 10 bps round-trip = 5 bps per side
    fee_bps: float = 10.0

    # Trade filter (D-14): individuals below min_trades receive worst fitness.
    min_trades: int = 50

    # Portfolio parameters
    freq: str = "1D"
    init_cash: float = 10_000.0

    # Close prices for vectorbt (D-12).
    close_prices: pd.DataFrame = field(default_factory=pd.DataFrame)
```

**EvolutionConfig must follow this pattern exactly** — `field(default_factory=lambda: max(1, os.cpu_count() - 1))` for `n_jobs`, inline comments explaining each constraint decision (D-13 through D-16), and `from __future__ import annotations` at line 1. Do NOT use `pd.DataFrame` as a field type — `EvolutionConfig` has no mutable default container fields except `n_jobs`.

**Module docstring pattern** (runner.py lines 1–10): docstring opens with one-line description, then ARCHITECTURE INVARIANT block if an import boundary exists.

---

### `vgp/evolution/loop.py` (NSGA-II evolution loop)

**Analog:** `vgp/backtest/runner.py` — module-level function with config + deferred imports

**Imports pattern** (runner.py lines 11–20 — mirror structure but swap vectorbt for deap):
```python
from __future__ import annotations

import functools
import logging
import multiprocessing
import operator
import random
from pathlib import Path

import numpy as np
from deap import algorithms, base, gp, tools

from vgp.backtest.runner import evaluate, EvalConfig
from vgp.gp.gp_types import build_pset, creator  # noqa: F401 — side effect: registers creator
from vgp.evolution.config import EvolutionConfig
from vgp.evolution.checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)
```

**Architecture invariant docstring** (runner.py lines 1–10 — copy the pattern):
```python
"""NSGA-II evolution loop — DEAP toolbox, eaMuPlusLambda-equivalent varOr loop.

ARCHITECTURE INVARIANT (D-15 / CLAUDE.md §Architecture Invariants):
  This module must NOT import vectorbt at module level or inside any function.
  The interface is: numpy feature_matrix in -> (population, hof, logbook) out.
  Backtest evaluation is accessed only via functools.partial(evaluate, ...).
"""
```

**Module-level JIT warmup function** — must be at module level (not inside `run_evolution`) for spawn pickling:
```python
def _jit_warmup() -> None:
    """Trigger numba JIT compilation in spawn worker before evaluation begins.

    CLAUDE.md constraint #8: warmup MUST run in worker initializer, not main process.
    Each spawn worker gets a fresh interpreter with uncompiled JIT cache.
    """
    import vectorbt as vbt  # deferred — only runs inside worker process
    import pandas as pd
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    close = pd.DataFrame({"a": np.ones(10) * 100.0}, index=dates)
    entries = pd.DataFrame({"a": np.array([True, False] * 5)}, index=dates)
    exits = pd.DataFrame({"a": np.array([False, True] * 5)}, index=dates)
    vbt.Portfolio.from_signals(
        close=close, entries=entries, exits=exits,
        fees=0.001, freq="1D", init_cash=1000.0,
        group_by=True, cash_sharing=True,
    )
```

**Core function signature pattern** (runner.py lines 58–62 — mirror the style):
```python
def run_evolution(
    config: EvolutionConfig,
    feature_matrix: np.ndarray,
    eval_config: EvalConfig,
    tracker=None,           # NoOpTracker or MLflowTracker (duck-typed)
    resume_checkpoint: str | None = None,
) -> tuple:
    """Run NSGA-II GP evolution and return (population, hof, logbook).
    ...
    """
```

**Error handling pattern** (runner.py lines 96–109 — validate inputs early):
```python
    if feature_matrix.ndim != 3:
        raise ValueError(
            f"feature_matrix must be 3-D [T x F x A], got shape {feature_matrix.shape}"
        )
    T, F, A = feature_matrix.shape
    if F != 12:
        raise ValueError(
            f"Expected F=12 feature columns (FEATURE_NAMES), got {F}"
        )
```

**Logger usage** (runner.py lines 20, 128–131):
```python
logger = logging.getLogger(__name__)
# ...
logger.debug(
    "Individual (size=%d) has only %d sign changes — below min_trades=%d. "
    "Returning worst fitness.",
    tree_size, sign_changes, config.min_trades,
)
```

---

### `vgp/evolution/checkpoint.py` (pickle-based checkpoint save/resume)

**Analog:** `vgp/backtest/runner.py` — module structure; `vgp/data/config.py` — Path usage

**Imports pattern:**
```python
from __future__ import annotations

import random
from pathlib import Path

import dill
import numpy as np
```

**Module docstring pattern** (data/config.py line 1):
```python
"""Checkpoint save/load for NSGA-II evolution state — pickle-based, dill serializer."""
```

**Path creation pattern** (data/config.py lines 22–24 — field default_factory for Path):
```python
# In save_checkpoint:
p = Path(path)
p.parent.mkdir(parents=True, exist_ok=True)
```

**No logging inside checkpoint.py** — checkpoint functions are called from within the evolution loop that already has a logger. Keep checkpoint.py as pure I/O.

---

### `vgp/evolution/__init__.py` (public exports)

**Analog:** `vgp/backtest/__init__.py` (lines 1–13) — exact pattern to copy

```python
"""Evolution engine: DEAP toolbox, NSGA-II loop, checkpointing.

ARCHITECTURE INVARIANT: This module must NOT import vectorbt.
Interface to backtesting: numpy array in -> fitness tuple out.
"""

from .config import EvolutionConfig
from .loop import run_evolution
from .checkpoint import save_checkpoint, load_checkpoint

__all__ = [
    "EvolutionConfig",
    "run_evolution",
    "save_checkpoint",
    "load_checkpoint",
]
```

The existing stub at `vgp/evolution/__init__.py` (lines 1–5) already has the correct docstring and invariant comment — preserve those lines and add the imports below them.

---

### `vgp/gp/primitives.py` — ADD `gt`, `lt`, `if_then_else` (MODIFY)

**Analog:** `vgp/gp/primitives.py` lines 67–101 (existing arithmetic primitives) — exact pattern

**Module-level function pattern** (primitives.py lines 67–69):
```python
def prim_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Element-wise addition. Handles scalar (ephemeral constant) inputs via numpy broadcast."""
    return (np.asarray(x, dtype=np.float32) + np.asarray(y, dtype=np.float32)).astype(np.float32)
```

**New primitives must follow this exact style:**
- One-line docstring describing semantics
- `np.asarray(x, dtype=np.float32)` coercion at entry (matches `_to_f32` convention without calling `_to_f32` directly in these two-arg functions — look at how `prim_add` uses inline `np.asarray`)
- Return type: `.astype(np.float32)` at the end
- Module-level (not inside any class or function)

**Append after** the rolling aggregation block (after line 199). Add a section header comment:
```python
# ---------------------------------------------------------------------------
# Conditional/comparison primitives: [Vector, Vector] -> Vector  [D-10]
# Added in Phase 4. Returns 0.0/1.0 float32 arrays — compose naturally
# with arithmetic primitives without needing a boolean type in the type system.
# ---------------------------------------------------------------------------
```

**Pattern for `if_then_else`** — uses `np.where` like `prim_protected_div` uses `np.where` (lines 89–96):
```python
def prim_protected_div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xa, ya = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            np.abs(ya) < 1e-7,
            np.ones_like(xa, dtype=np.float32),
            xa / ya,
        )
    return np.asarray(result, dtype=np.float32)
```

---

### `vgp/gp/gp_types.py` — ADD 3 primitive registrations (MODIFY)

**Analog:** `vgp/gp/gp_types.py` lines 24–38 (import block) and lines 93–105 (addPrimitive calls)

**Import addition** — extend the existing import from `vgp.gp.primitives` at lines 24–38:
```python
from vgp.gp.primitives import (
    Vector,
    Scalar,
    prim_add,
    prim_sub,
    prim_mul,
    prim_protected_div,
    prim_neg,
    rolling_mean_5,
    rolling_mean_20,
    rolling_std_5,
    rolling_std_20,
    rolling_max_20,
    rolling_min_20,
    # Phase 4 additions:
    gt,
    lt,
    if_then_else,
)
```

**Registration pattern** (gp_types.py lines 93–95 — copy this exact style):
```python
pset.addPrimitive(prim_add, [Vector, Vector], Vector, name="add")
pset.addPrimitive(prim_sub, [Vector, Vector], Vector, name="sub")
```

**New registrations append after** the existing `pset.addPrimitive(rolling_min_20, ...)` block and before `pset.addEphemeralConstant(...)`. Add comment:
```python
    # Conditional/comparison primitives: Phase 4 additions (D-10)
    pset.addPrimitive(gt, [Vector, Vector], Vector, name="gt")
    pset.addPrimitive(lt, [Vector, Vector], Vector, name="lt")
    pset.addPrimitive(if_then_else, [Vector, Vector, Vector], Vector, name="if_then_else")
```

**Also update the docstring** of `build_pset()` (lines 56–82) — the `NOTE on GP-04` (line 75–76) says conditionals are deferred. Replace that note:
```python
    # OLD (remove): NOTE on GP-04: Conditional/comparison primitives are deferred per D-07.
    # NEW: NOTE on Phase 4 (D-10): gt, lt, if_then_else added as comparison primitives.
```

---

### `pyproject.toml` — UPDATE `[tracking]` optional extra (MODIFY)

**Analog:** `pyproject.toml` lines 63–66 (existing tracking extra)

```toml
# Current (line 63-66):
tracking = [
  "mlflow>=2.14",
]

# Update to:
tracking = [
  "mlflow>=3.0",
]
```

The comment above the extra (lines 60–63) explains the pandas<3 conflict — preserve it and update the version reference from "2.14" to "3.0". The comment text should also note Phase 4 resolved tracking approach (NoOpTracker + optional MLflowTracker).

---

### `tests/test_evolution.py` (NEW — EVO-01 through EVO-07, EXP-01 through EXP-03)

**Analog:** `tests/test_evaluate.py` (full file) — exact structural template

**File header pattern** (test_evaluate.py lines 1–12):
```python
"""
evaluate() contract tests — EVAL-01, EVAL-02, EVAL-03, EVAL-04.

Tests use synthetic data: no parquet files, no network access.
The feature matrix is a random [T x F x A] float32 array.
Close prices are a synthetic pd.DataFrame with a DatetimeIndex.

EVAL-01: ...
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
```

**Shared fixture pattern** (test_evaluate.py lines 26–81):
```python
_T = 400     # timesteps
_F = 12      # feature columns
_A = 3       # assets

@pytest.fixture(scope="module")
def pset():
    from vgp.gp.gp_types import build_pset as _build
    return _build()

@pytest.fixture(scope="module")
def feature_matrix():
    """Random [T x F x A] float32 feature matrix."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((_T, _F, _A)).astype(np.float32)

@pytest.fixture(scope="module")
def close_prices():
    rng = np.random.default_rng(7)
    dates = pd.date_range(start="2024-01-01", periods=_T, freq="D")
    prices = 100.0 * np.exp(np.cumsum(rng.standard_normal((_T, _A)) * 0.01, axis=0))
    return pd.DataFrame(
        prices.astype(np.float64),
        index=dates,
        columns=[f"asset_{i}" for i in range(_A)],
    )
```

**Import boundary test pattern** (test_evaluate.py lines 88–103 — test_backtest_runner_does_not_import_deap_eval01):
```python
def test_evolution_loop_does_not_import_vectorbt_evo01():
    """vgp.evolution.loop must NOT import vectorbt at module level (D-15)."""
    if "vgp.evolution.loop" in sys.modules:
        del sys.modules["vgp.evolution.loop"]

    mods_before = set(sys.modules.keys())
    import vgp.evolution.loop  # noqa: F401
    mods_after = set(sys.modules.keys())

    new_vbt_mods = [m for m in (mods_after - mods_before) if "vectorbt" in m]
    assert not new_vbt_mods, (
        f"vgp.evolution.loop imported vectorbt modules: {new_vbt_mods}. "
        f"Architecture invariant D-15 violated."
    )
```

**skipif pattern for optional dependency** (D-02 — new in Phase 4, not in existing tests):
```python
try:
    import mlflow  # noqa: F401
    mlflow_available = True
except ImportError:
    mlflow_available = False

@pytest.mark.skipif(not mlflow_available, reason="mlflow not installed")
def test_mlflow_tracker_logs_params_exp01():
    ...
```

**Reproducibility test pattern** (EXP-03 — no exact existing analog, but follows same fixture + assertion style):
```python
def test_seed_reproducibility_exp03(feature_matrix, close_prices):
    """Two runs with same seed must produce identical Pareto fronts (EXP-03)."""
    from vgp.backtest.runner import EvalConfig
    from vgp.evolution.config import EvolutionConfig
    from vgp.evolution.loop import run_evolution

    cfg = EvolutionConfig(pop_size=10, n_generations=2, seed=42, n_jobs=1, checkpoint_freq=999)
    eval_cfg = EvalConfig(close_prices=close_prices, min_trades=1)
    tracker = None  # NoOpTracker default

    _, hof1, _ = run_evolution(cfg, feature_matrix, eval_cfg, tracker)
    _, hof2, _ = run_evolution(cfg, feature_matrix, eval_cfg, tracker)

    # Same seed must produce identical HoF
    assert len(hof1) == len(hof2), "HoF sizes differ across identical-seed runs"
    for ind1, ind2 in zip(hof1, hof2):
        assert str(ind1) == str(ind2), "HoF individuals differ across identical-seed runs (EXP-03)"
```

**Test section comment pattern** (test_evaluate.py lines 86–87):
```python
# ---------------------------------------------------------------------------
# EVO-01: DEAP toolbox wires selNSGA2, cxOnePoint, mutUniform
# ---------------------------------------------------------------------------
```

---

## Shared Patterns

### Module-Level Function Requirement (multiprocessing.Pool pickling)
**Source:** `vgp/gp/primitives.py` lines 1–19 (module docstring invariants), `vgp/gp/gp_types.py` lines 14–22 (`_rand_scalar_int`)
**Apply to:** `vgp/evolution/loop.py` (`_jit_warmup`), `vgp/gp/primitives.py` (`gt`, `lt`, `if_then_else`)
```python
# From gp_types.py lines 14-22:
def _rand_scalar_int() -> float:
    """Ephemeral constant generator: random integer in [-5, 5] as float.

    Defined at module level (not as a lambda) so it can be pickled by
    multiprocessing.Pool workers. DEAP's addEphemeralConstant requires a
    callable; using a lambda here causes a RuntimeWarning and pickle failure.
    """
    return float(random.randint(-5, 5))
```
Every callable passed to or registered in a Pool must be a module-level named function. This applies to `_jit_warmup`, `gt`, `lt`, `if_then_else`, and `evaluate` (already correct).

### Architecture Invariant Docstring
**Source:** `vgp/backtest/runner.py` lines 1–10, `vgp/evolution/__init__.py` lines 1–5
**Apply to:** `vgp/evolution/loop.py`, `vgp/evolution/__init__.py`
```python
"""BacktestRunner and evaluate() — vectorbt integration for GP fitness evaluation.

ARCHITECTURE INVARIANT (D-15):
  This module must NOT import deap at module level or inside any function.
  The interface is: numpy signal array in -> fitness tuple out.
  tree_size = len(individual) works without deap (PrimitiveTree implements __len__).
"""
```

### `from __future__ import annotations`
**Source:** Every existing `.py` file in `vgp/` — line 1
**Apply to:** All new files (`config.py`, `loop.py`, `checkpoint.py`, `tracker.py`)

### Dataclass with `field(default_factory=...)` for Mutable Defaults
**Source:** `vgp/backtest/runner.py` lines 51–52, `vgp/data/config.py` lines 22–24
**Apply to:** `vgp/evolution/config.py` (`n_jobs` default via lambda)
```python
# runner.py line 51:
close_prices: pd.DataFrame = field(default_factory=pd.DataFrame)

# data/config.py line 22:
cache_dir: Path = field(default_factory=lambda: Path("vgp/data/cache"))
```
For `EvolutionConfig.n_jobs`: `n_jobs: int = field(default_factory=lambda: max(1, os.cpu_count() - 1))`

### float32 Throughout
**Source:** `vgp/gp/primitives.py` lines 67–101 — every arithmetic primitive returns `.astype(np.float32)`
**Apply to:** `gt`, `lt`, `if_then_else` in `vgp/gp/primitives.py` — all must return `.astype(np.float32)` on the final line.

### Deferred Import Pattern (import boundary enforcement)
**Source:** `vgp/backtest/runner.py` lines 93–96
**Apply to:** `_jit_warmup()` in `loop.py` (imports vectorbt only inside the function, which runs only in worker process)
```python
# runner.py lines 93-96:
# Import GP layer here (deferred — maintains architectural separation D-15)
from vgp.gp.tree_evaluator import TreeEvaluator
from vgp.gp.gp_types import build_pset
```

### Test File Structure
**Source:** `tests/test_evaluate.py` (full file)
**Apply to:** `tests/test_evolution.py`
- `from __future__ import annotations` at line 1
- Module docstring with requirement IDs and data invariants (lines 1–12)
- `_T`, `_F`, `_A` constants for synthetic data dimensions
- `scope="module"` fixtures for expensive objects (pset, feature_matrix, close_prices)
- `pytest.skip(...)` for inconclusive paths
- Section dividers as `# ---` comment blocks with requirement ID
- Imports inside test functions (not at module level) for `vgp.*` modules

### Logger Initialization
**Source:** `vgp/backtest/runner.py` line 20, `vgp/data/splitter.py` line 17
**Apply to:** `vgp/evolution/loop.py`
```python
logger = logging.getLogger(__name__)
```

---

## No Analog Found

All files have close analogs in the existing codebase. No files require fallback to RESEARCH.md patterns exclusively.

However, the following patterns have **no existing code analog** and must be implemented from RESEARCH.md patterns directly:

| Pattern | Apply To | RESEARCH.md Reference |
|---|---|---|
| `NoOpTracker` / `MLflowTracker` duck-typed tracker | `vgp/evolution/tracker.py` (new file per architecture map) | RESEARCH.md Pattern 6 (lines 388–426) |
| `multiprocessing.get_context("spawn").Pool(initializer=...)` | `vgp/evolution/loop.py` | RESEARCH.md Pattern 3 (lines 276–308) |
| `tools.MultiStatistics` + `tools.Logbook` | `vgp/evolution/loop.py` | RESEARCH.md Pattern 4 (lines 316–336) |
| `dill.dump` / `dill.load` checkpoint | `vgp/evolution/checkpoint.py` | RESEARCH.md Pattern 5 (lines 338–380) |
| `gp.staticLimit` decorator on `mate` and `mutate` | `vgp/evolution/loop.py` | RESEARCH.md Pattern 1 (lines 197–222) |

Note: `vgp/evolution/tracker.py` is identified in the RESEARCH.md architecture map but not in CONTEXT.md §Integration Points. It should be created as a companion to `loop.py` — `loop.py` imports `NoOpTracker` from `tracker.py` as its default tracker.

---

## Metadata

**Analog search scope:** `vgp/`, `tests/`, `pyproject.toml`
**Files scanned:** 15 source files + 5 test files + pyproject.toml
**Pattern extraction date:** 2026-06-09
