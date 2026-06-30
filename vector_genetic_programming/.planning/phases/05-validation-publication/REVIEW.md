---
phase: 05-validation-publication
reviewed: 2026-06-15T00:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - vgp/analysis/runner.py
  - vgp/analysis/dsr.py
  - vgp/analysis/plots.py
  - vgp/analysis/__init__.py
  - tests/test_validation.py
  - tests/test_plots.py
  - README.md
  - CONTRIBUTING.md
findings:
  critical: 2
  warning: 3
  info: 3
  total: 8
status: issues_found
---

# Phase 05: Code Review Report

**Reviewed:** 2026-06-15
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

The Phase 5 analysis module is well-structured and correctly enforces the OOS holdout invariant (test_fm is a local variable, consumed exactly once). The `matplotlib.use('Agg')` placement is correct. No `.values` violations found. No circular imports. No DEAP imports in `vgp/backtest/`.

Two critical issues were found:

1. **`n_trials` understates the multiple-testing correction** — `compute_dsr()` is documented to take `seeds x windows` but `runner.py` passes only `len(seeds)` per window, ignoring the window dimension. This silently inflates DSR (makes strategies appear more significant than they are), which is the opposite of the statistical safety the DSR is meant to provide.

2. **`periods_per_year` parameter accepted but never used** — the DSR variance formula uses the raw annualized `sr_hat` against per-period returns' moments without scaling. This creates a unit mismatch that biases `sr_var` and thus the DSR probability.

Three warnings cover: a potential `IndexError` on empty logbook, an architecture boundary violation (`plots.py` imports `deap` and `vectorbt` at module level making those imports unavoidable when `vgp.analysis` is imported), and a logic ambiguity in the `is_sharpe` source when the evolution loop resumes from checkpoint.

---

## Critical Issues

### CR-01: `n_trials` passed to `compute_dsr()` is too small — DSR is inflated

**File:** `vgp/analysis/runner.py:277`

**Issue:** `compute_dsr()` docstring (and `dsr.py` line 40) specify that `n_trials` must equal `seeds × windows` — the total number of independent strategy evaluations — so that `E[SR_max]` under H0 correctly adjusts for the full search multiplicity. In `runner.py`, `n_trials = max(1, len(seeds))` is computed per call to `run_window()`. If the caller runs 4 windows × 3 seeds = 12 trials, each window receives `n_trials=3` instead of `12`. This makes `E[SR_max]` smaller, `sr_hat - E[SR_max]` larger, and the reported DSR probability closer to 1.0. Strategies appear more statistically significant than they are — the exact failure mode DSR was designed to prevent.

**Fix:** Pass the total trial count into `run_window()` and forward it to `compute_dsr()`:

```python
# In runner.py: run_window() signature
def run_window(
    self,
    window: WindowSpec,
    feature_matrix: np.ndarray,
    close_prices: pd.DataFrame,
    base_eval_config: EvalConfig,
    seeds: list[int],
    evo_config_kwargs: dict,
    n_total_trials: int | None = None,   # ADD this parameter
) -> list[dict]:
    ...
    # Inside the per-seed loop, replace:
    #   n_trials = max(1, len(seeds))
    # with:
    n_trials = max(1, n_total_trials if n_total_trials is not None else len(seeds))
```

The caller (e.g. a top-level experiment script) should pass `n_total_trials = len(seeds) * len(windows)`. This is a single-line caller-side change once the parameter exists.

---

### CR-02: `periods_per_year` accepted but never used — DSR variance formula has unit mismatch

**File:** `vgp/analysis/dsr.py:30,74-75`

**Issue:** `compute_dsr()` accepts `periods_per_year: int = 252` and documents it as "252 for daily data", but the parameter is never referenced in the function body. The DSR variance formula (line 74-75) uses `sr_hat` — which is an **annualized** Sharpe from vectorbt — in a formula whose derivation assumes `sr_hat` is a **per-period** (non-annualized) value matching the scale of `returns`. This mismatch biases `sr_var`. The correct formula requires de-annualizing `sr_hat` before plugging it into the variance term:

```
sr_per_period = sr_hat / sqrt(periods_per_year)
sr_var = (1 + 0.5*sr_per_period**2 - skew*sr_per_period + (kurt/4)*sr_per_period**2) / (T - 1)
```

Then the DSR numerator uses `sr_hat` (annualized) minus `expected_max_sr` (also in annualized units, since the formula normalizes by its own `sqrt(sr_var * periods_per_year)` implicitly) or both are kept in per-period units consistently.

**Fix:** De-annualize before the variance computation and use the per-period SR throughout:

```python
# dsr.py — inside compute_dsr(), after the T check
sr_per_period = sr_hat / np.sqrt(periods_per_year)

sr_var = (
    1
    + (0.5 * sr_per_period**2)
    - ret_skew * sr_per_period
    + ((ret_kurt / 4) * sr_per_period**2)
) / (T - 1)

if sr_var <= 0.0:
    return 0.0

# Scale expected_max_sr to per-period units for the numerator
expected_max_sr_per_period = expected_max_sr / np.sqrt(periods_per_year)

dsr = float(norm.cdf((sr_per_period - expected_max_sr_per_period) / np.sqrt(sr_var)))
```

---

## Warnings

### WR-01: `logbook.chapters["fitness"][-1]` raises `IndexError` if logbook is empty

**File:** `vgp/analysis/runner.py:266`

**Issue:** `is_sharpe = float(logbook.chapters["fitness"][-1]["sharpe_max"])` will raise `IndexError: list index out of range` if `run_evolution()` is called with `resume_checkpoint` pointing to a checkpoint at the final generation and `n_generations=0`, or if a future refactor skips gen-0 recording. In `loop.py`, generation 0 is only recorded when `resume_checkpoint is None` (line 305). A resumed run that re-enters `run_evolution()` with `start_gen > n_generations` exits the `for` loop immediately without recording anything into the in-memory `logbook` (though the checkpoint's logbook is restored, `chapters` may be empty if the checkpoint was written before any `record()` call). The error will surface only at runtime, not in any current test.

**Fix:** Guard with a length check:

```python
fitness_chapter = logbook.chapters.get("fitness", [])
if not fitness_chapter:
    logger.warning(
        "Window %d seed %d: logbook.chapters['fitness'] is empty — "
        "using is_sharpe=0.0",
        window.window_id, seed,
    )
    is_sharpe = 0.0
else:
    is_sharpe = float(fitness_chapter[-1]["sharpe_max"])
```

---

### WR-02: `plots.py` imports `deap` and `vectorbt` at module level — architecture boundary violated

**File:** `vgp/analysis/plots.py:25-26`

**Issue:** `vgp/analysis/__init__.py` imports `plots.py` unconditionally (line 4). This means `import vgp.analysis` (or any `from vgp.analysis import ...`) always triggers `import vectorbt as vbt` and `from deap import gp as deap_gp` at module load time — before any function is called. The CLAUDE.md architecture invariant states `EvolutionLoop` must not import `vectorbt`. While the violation is in `plots.py` (not `loop.py`), the transitive import chain `vgp.analysis → plots.py → vectorbt` means any module that imports `vgp.analysis` (including `runner.py`) will load vectorbt unconditionally. This is not currently a correctness bug but it violates the layering contract and will cause problems if `vgp.analysis` is ever imported in a worker process before JIT warmup.

**Fix:** Move vectorbt and deap imports inside `plot_*` function bodies (deferred imports, consistent with the `D-15` pattern used in `runner.py`'s `_get_is_returns`):

```python
# plots.py — remove module-level imports of vectorbt and deap_gp
# Inside plot_equity_curves():
    import vectorbt as vbt  # noqa: PLC0415 — deferred; D-15 pattern

# Inside plot_tree_graph():
    from deap import gp as deap_gp  # noqa: PLC0415 — deferred; D-15 pattern
```

---

### WR-03: `is_sharpe` source is the final-generation **population** maximum, not the best individual's IS Sharpe

**File:** `vgp/analysis/runner.py:266`

**Issue:** `logbook.chapters["fitness"][-1]["sharpe_max"]` is the maximum Sharpe across the **entire population** in the last generation — not the Sharpe of `hof[0]` (the best individual from the Pareto front). For most runs these will be the same individual, but they can diverge when the Pareto front's top individual has lower Sharpe but better total_return or smaller tree size. Using the population maximum inflates `sr_hat` in the DSR calculation, making DSR appear higher than justified by the specific individual being evaluated OOS.

**Fix:** Re-extract IS Sharpe from `_get_is_returns` result rather than the logbook population statistic:

```python
is_returns = _get_is_returns(best_ind, train_fm, train_eval_cfg)
# Compute IS Sharpe from the actual IS backtest of the best individual
_std = np.std(is_returns)
if _std > 0:
    is_sharpe = float(np.mean(is_returns) / _std * np.sqrt(periods_per_year))
else:
    is_sharpe = 0.0
dsr = compute_dsr(is_returns, sr_hat=is_sharpe, n_trials=n_trials)
```

This also eliminates the logbook dependency for IS Sharpe, removing WR-01 as a concern.

---

## Info

### IN-01: CONTRIBUTING.md `WalkForwardRunner` constructor signature is wrong

**File:** `CONTRIBUTING.md:46-54`

**Issue:** The documented constructor `WalkForwardRunner(feature_matrix=fm, windows=windows, eval_cfg=eval_cfg, evo_kwargs=evo_kwargs, seeds=..., output_dir=...)` does not match the actual constructor at `runner.py:168`, which only accepts `dates: pd.DatetimeIndex`. The documented `runner.run()` method does not exist — the actual API is `runner.run_window(window, feature_matrix, close_prices, base_eval_config, seeds, evo_config_kwargs)`. A contributor following CONTRIBUTING.md will get an immediate `TypeError`.

**Fix:** Update CONTRIBUTING.md to show the actual API pattern:

```python
runner = WalkForwardRunner(dates=fe.dates_)
for window in windows:
    results = runner.run_window(
        window=window,
        feature_matrix=fm,
        close_prices=close_df,
        base_eval_config=eval_cfg,
        seeds=[0, 1, 2],
        evo_config_kwargs=evo_kwargs,
    )
```

---

### IN-02: CONTRIBUTING.md `EvalConfig` example uses non-existent parameters

**File:** `CONTRIBUTING.md:40-42`

**Issue:** The documented call `EvalConfig(fee_pct=0.001, min_trades=50, periods_per_year=252)` uses three parameter names that do not exist in `EvalConfig` (`backtest/runner.py:27-51`). The correct parameter is `fee_bps: float = 10.0` (basis points, not fraction). `periods_per_year` is not an `EvalConfig` field. A contributor copy-pasting this example will get `TypeError: unexpected keyword argument 'fee_pct'`.

**Fix:**

```python
eval_cfg = EvalConfig(
    fee_bps=10.0,     # 10 bps round-trip (5 bps per side)
    min_trades=50,
    freq="1D",
    init_cash=10_000.0,
    close_prices=close_df,   # set to actual data before use
)
```

---

### IN-03: CONTRIBUTING.md results CSV table omits `train_end`, `test_start`, `test_end`, `n_nodes_best` columns

**File:** `CONTRIBUTING.md:72-80`

**Issue:** The results CSV table documents only `window_id`, `seed`, `is_sharpe`, `oos_sharpe`, `dsr`. The actual dict appended at `runner.py:288-298` contains 9 keys: additionally `train_end`, `test_start`, `test_end`, and `n_nodes_best`. The omission is documentation-only and does not affect correctness, but users relying on the table for downstream analysis will miss four columns.

**Fix:** Add the missing columns to the table in CONTRIBUTING.md.

---

_Reviewed: 2026-06-15_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
