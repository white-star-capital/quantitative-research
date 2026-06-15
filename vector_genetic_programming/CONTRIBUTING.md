# Contributing to VGP

VGP is a research framework for evolving trading strategies using genetic programming. Contributions are welcome. This guide covers the three most common contributor tasks.

## Adding a GP Primitive

Primitives live in `vgp/gp/` (to be populated in Phase 3). Before adding a primitive, read the architecture constraints in `CLAUDE.md` carefully — violating any of the three rules below will cause silent failures in multiprocessing workers.

**Three mandatory rules:**

1. All primitives MUST be module-level functions (not lambdas). `multiprocessing.Pool` pickles worker functions by name; lambdas and closures defined inside other functions are not picklable and cause `AttributeError` in workers with no obvious error message.

2. All primitives MUST accept and return `np.ndarray`. No pandas objects inside primitives. The GP evaluation loop passes raw numpy arrays — introducing a DataFrame conversion inside a primitive creates a serialization boundary and kills performance.

3. Primitives operate on vectors: shape `[T]` (time series) or `[T, F]` (feature window), depending on the primitive type. `PrimitiveSetTyped` distinguishes `Vector` (array) and `Scalar` (float) types — only matching types can be composed. Both type categories will be documented here once Phase 3 is complete.

See `vgp/gp/primitives.py` for examples (added in Phase 3).

## Running an Experiment

`WalkForwardRunner` is the high-level entry point for running a full multi-seed, walk-forward experiment. It handles window generation, per-seed evolution, OOS evaluation, and DSR computation.

```python
from vgp.analysis import generate_windows, WalkForwardRunner
from vgp.backtest.runner import EvalConfig
from vgp.evolution.config import EvolutionConfig

# Generate 4 walk-forward windows from available data
windows = generate_windows("2024-01-01", "2026-04-01")

# Configure evolution (small config for testing)
evo_kwargs = dict(
    pop_size=50, n_generations=10, cxpb=0.7, mutpb=0.2,
    n_jobs=1, tree_height_limit=8, checkpoint_freq=5,
    checkpoint_dir="checkpoints",
)

# EvalConfig controls backtest parameters (fees, min trades)
eval_cfg = EvalConfig(
    fee_bps=10.0,         # 10 bps round-trip taker fee (default)
    min_trades=50,        # individuals with fewer trades get worst fitness
)

# Run one walk-forward window across 3 seeds
# WalkForwardRunner takes the DatetimeIndex from FeatureEngine.dates_
runner = WalkForwardRunner(dates=fe.dates_)
window_results = runner.run_window(
    window=windows[0],
    feature_matrix=fm,          # float32 [T×F×A] from FeatureEngine
    close_prices=close_prices,  # pd.DataFrame [T×A] with DatetimeIndex
    base_eval_config=eval_cfg,
    seeds=[0, 1, 2],
    evo_config_kwargs=evo_kwargs,
)
all_results = window_results  # list of dicts, one per seed
```

Results are saved to `results/results.csv` automatically. Each row is one (window, seed) combination. To load a pre-existing feature matrix, use `DataLoader` and `FeatureEngine`:

```python
from vgp.data import DataLoader, FeatureEngine, WalkForwardSplitter

loader = DataLoader(cache_dir="data/cache")
ohlcv = loader.load()                     # dict[str, pd.DataFrame]
fe = FeatureEngine()
fm = fe.transform(ohlcv)                  # float32 [T×F×A]
```

See `vgp/evolution/` for the NSGA-II loop and `vgp/evolution/config.py` for all `EvolutionConfig` parameters.

## Interpreting Results

After a run, `results/results.csv` contains one row per (window, seed) combination with these columns:

| Column | Description |
|--------|-------------|
| `window_id` | Integer index of the walk-forward window (0-indexed) |
| `seed` | Random seed used for this evolution run |
| `train_end` | Last date of in-sample training period (ISO format) |
| `test_start` | First date of out-of-sample test period (ISO format) |
| `test_end` | Last date of out-of-sample test period (ISO format) |
| `is_sharpe` | In-sample Sharpe ratio of the best individual (annualized, with fees) |
| `oos_sharpe` | Out-of-sample Sharpe ratio on the held-out test split |
| `dsr` | Deflated Sharpe Ratio — significance probability after correcting for multiple testing |
| `n_nodes_best` | Node count of the best individual's GP tree |

**Key interpretation rules:**

- **DSR > 0.95** means the strategy is statistically significant at the 5% level after correcting for the number of strategies tested (Bailey & Lopez de Prado, 2014). Prioritize DSR over raw OOS Sharpe.
- **`median_oos_sharpe` across seeds** is more reliable than any single-seed OOS Sharpe. Aggregate results with `aggregate_seeds()` from `vgp.analysis`.
- **IS Sharpe >> OOS Sharpe** indicates overfitting. Remedies: reduce `n_generations`, increase `min_trades` threshold, or increase the `tree_height_limit` penalty weight.
- **Negative OOS Sharpe** is a valid result — the strategy found no edge on held-out data. Run more seeds before concluding (3-5 minimum).

Aggregate across seeds programmatically:

```python
from vgp.analysis import aggregate_seeds

# all_results is list of dicts from runner.run()
window_0_results = [r for r in all_results if r["window_id"] == 0]
summary = aggregate_seeds(window_0_results)
# summary keys: median_oos_sharpe, iqr_oos_sharpe, median_dsr, n_seeds_positive_oos
```

## Visualizations

Three plots are generated automatically in `results/` when using `WalkForwardRunner`. To regenerate them manually:

```python
from vgp.analysis import plot_pareto_front, plot_equity_curves, plot_tree_graph

# Pareto front scatter (3D: Sharpe vs total_return vs tree_size)
plot_pareto_front(hof, "results/pareto_front.png")

# IS/OOS equity curves with train/test boundary overlay
# plot_equity_curves(individuals, fm, eval_cfg, train_end, "results/equity_curves.png")

# GP tree graph (networkx, no graphviz required)
plot_tree_graph(hof[0], "results/tree_graph.png", title="Best Individual")
```

All three functions write PNG files and return `None`. They use the `Agg` matplotlib backend (headless-safe — no display required). Plots are saved at 150 dpi.

## Updating the Dependency Lock File

VGP uses `pip-tools` to maintain a reproducible lock file.

1. Install pip-tools:
   ```
   pip install pip-tools
   ```

2. Compile the lock file from pyproject.toml:
   ```
   pip-compile pyproject.toml -o requirements-lock.txt
   ```

3. Commit the updated `requirements-lock.txt`.

**Important:** The `numpy<2.3` upper bound in `pyproject.toml` must remain. NumPy 2.3 hard-breaks numba's internal C extension APIs, which makes the entire evolution engine inoperable. Do not remove or relax this pin. If numba adds NumPy 2.3 support in a future release, update the pin in `pyproject.toml` and re-run the smoke tests (`test_numba_jit_compiles`) before committing.

## Code Conventions

**pandas 3.0 idioms** — pandas 3.0 makes Copy-on-Write the default and removes several deprecated patterns. Follow these rules in all DataFrame code:
- Use `.to_numpy()` (not `.values`) to extract numpy arrays from DataFrames
- Use explicit `.copy()` before mutating a DataFrame slice
- Use `.loc[]` for all index-based assignment (no chained indexing)

**Future annotations** — All new Python files must start with:
```python
from __future__ import annotations
```
This enables PEP 563 postponed evaluation for forward references in type hints and is required for consistency across the codebase.

**Module-level loggers** — Declare loggers at module level, not inside functions:
```python
import logging
logger = logging.getLogger(__name__)
```

**Run tests:**
```
python -m pytest tests/ -v
```
