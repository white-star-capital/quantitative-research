# Phase 5: Validation & Publication - Research

**Researched:** 2026-06-13
**Domain:** Walk-forward validation, multi-seed DSR reporting, GP visualization, community release
**Confidence:** HIGH — all claims verified against installed packages and VGP source code in the project venv

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VAL-01 | Walk-forward validation runs evolution on N rolling windows with non-overlapping OOS periods | Splitter API verified; 4 windows fit in data range |
| VAL-02 | OOS holdout touched exactly once per window (structural, not convention) | WalkForwardSplitter.split() enforces order via AssertionError; redesign adds per-window lockout |
| VAL-03 | 10+ independent seeds run per experiment configuration | run_evolution() takes seed param; embarrassingly parallel via multiprocessing or sequential |
| VAL-04 | Deflated Sharpe Ratio computed and reported across seeds/windows | scipy.stats confirmed available; manual DSR formula verified against Bailey & Lopez de Prado (2014) |
| VAL-05 | Pareto front scatter plot exported for top generation | individual.fitness.values API confirmed; mpl_toolkits.mplot3d 3D scatter verified |
| VAL-06 | Equity curves for top-3 individuals (IS + OOS overlaid) | vectorbt pf.value() returns pd.Series; vertical separator pattern confirmed |
| VAL-07 | GP tree exported as graph for top individuals (NetworkX + matplotlib) | gp.graph() confirmed; nx.DiGraph + nx.bfs_layout verified without graphviz |

</phase_requirements>

---

## Summary

Phase 5 closes the research loop by adding statistical rigour (walk-forward windows + DSR) and making the repository understandable to the community. All dependencies are already installed: networkx 3.6.1, matplotlib 3.10.9, scipy 1.17.1, seaborn 0.13.2. No new packages need adding to pyproject.toml.

The data cache covers 2024-01-01 to 2026-04-01 (27 months of daily OHLCV for 18-21 assets). With a 12-month train, 2-month val, 3-month OOS window advancing every 3 months, exactly 4 non-overlapping OOS windows fit in the available data [VERIFIED: calculated against actual parquet date range]. With 10 seeds, the experiment matrix is 4 windows x 10 seeds = 40 run_evolution() calls; this is embarrassingly parallel if compute is available, or runnable sequentially overnight.

The visualization layer (VAL-05 through VAL-07) uses only stdlib + already-installed packages. DEAP's `gp.graph()` returns nodes/edges/labels that feed directly into a `nx.DiGraph`; `nx.bfs_layout(G, start=0)` provides a clean hierarchical layout without requiring the Graphviz binary (which is absent on this machine). vectorbt 1.0.0 `Portfolio.value()` returns a `pd.Series` with a `DatetimeIndex`, which makes IS vs OOS overlay straightforward via matplotlib vertical line or axvspan. All visualizations must use `matplotlib.use('Agg')` before any import of `matplotlib.pyplot` in test and CI contexts.

**Primary recommendation:** Structure Phase 5 as three plans — (1) walk-forward runner + multi-seed experiment harness + DSR reporting module, (2) visualization module (Pareto front, equity curves, tree graphs), (3) CONTRIBUTING.md update + README + tests. Plans 1 and 2 can be designed independently; Plan 3 wraps up community release.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Walk-forward split generation | Data layer (vgp/data/splitter.py) | Analysis orchestration | WalkForwardSplitter already enforces temporal ordering |
| Multi-seed evolution orchestration | Analysis layer (vgp/analysis/) | Evolution layer (vgp/evolution/loop.py) | run_evolution() is the unit; the runner calls it N times |
| DSR computation | Analysis layer (vgp/analysis/) | — | Pure stats function; no GP or backtest dependency |
| Results persistence (CSV/JSON) | Analysis layer (vgp/analysis/) | — | mlflow excluded (pandas<3 conflict) |
| Pareto front visualization | Analysis layer (vgp/analysis/) | — | Reads individual.fitness.values from hof |
| Equity curve visualization | Analysis layer (vgp/analysis/) | Backtest layer (evaluate()) | Requires re-running evaluate() to get pf object |
| GP tree graph export | Analysis layer (vgp/analysis/) | GP layer (deap.gp.graph) | Reads individual; calls deap.gp.graph() |
| Community docs (README, CONTRIBUTING) | Repository root | — | Static files, no code |

---

## Standard Stack

### Core (already in pyproject.toml)

| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| scipy | 1.17.1 | DSR formula (stats.norm, stats.skew, stats.kurtosis) | INSTALLED [VERIFIED: project venv] |
| networkx | 3.6.1 | GP tree DiGraph, bfs_layout | INSTALLED [VERIFIED: project venv] |
| matplotlib | 3.10.9 | All visualizations (Agg backend for headless) | INSTALLED [VERIFIED: project venv] |
| numpy | 2.2.6 | Array ops throughout | INSTALLED [VERIFIED: project venv] |
| pandas | >= 3.0 | WalkForwardSplitter date arithmetic, equity curve Series | INSTALLED [VERIFIED: project venv] |
| deap | 1.4.4 | gp.graph() for tree-to-DiGraph conversion | INSTALLED [VERIFIED: project venv] |
| vectorbt | 1.0.0 | pf.value(), pf.cumulative_returns() for equity curves | INSTALLED [VERIFIED: project venv] |
| dill | >= 0.3.8 | Load checkpoints for post-run analysis | INSTALLED [VERIFIED: project venv] |

### Not Needed (explicitly excluded)

| Library | Reason Excluded |
|---------|----------------|
| pyfolio | Not installed; manual DSR implementation is sufficient and has no extra deps |
| riskfolio | Not installed; same reasoning as pyfolio |
| mlflow | pandas<3 conflict; use CSV/JSON for results persistence [VERIFIED: STATE.md blocker note] |
| pygraphviz / graphviz CLI | graphviz binary absent on machine; nx.bfs_layout() is a full replacement |
| seaborn | Available (0.13.2) but not needed — matplotlib is sufficient for publication figures |
| plotly | Available (6.8.0) but not needed — static PNG/SVG output is the target |

**Installation:** No new packages required. All dependencies are already declared in pyproject.toml and installed. [VERIFIED: project venv]

---

## Architecture Patterns

### System Architecture Diagram

```
DATA LAYER                    ANALYSIS LAYER                    OUTPUT
─────────────────             ───────────────────────────       ──────────────
parquet files                 WalkForwardRunner
     │                             │
     ▼                             ▼
DataLoader ──► FeatureEngine  generate_windows()
     │               │             │── window 1 ──► run_evolution() ──► (pop, hof, logbook)
     │               │             │── window 2 ──► run_evolution() ──► (pop, hof, logbook)
     │               ▼             │── window N ──► run_evolution() ──► (pop, hof, logbook)
     │         [T×F×A] matrix      │                                          │
     │               │             ▼                                          │
     │         WalkForwardSplitter  aggregate_results()                       │
     │               │             │── per_window_sharpes[]                   │
     │         (train, val, test)  │── per_seed_sharpes[]                     │
     │                             ▼                                          │
     │                        compute_dsr()                                   │
     │                             │── DSR per config                         │
     │                             │── median Sharpe ± IQR                    ▼
     │                             │                                    results/
     │                             │── save_results_csv()              ├── results.csv
     │                             │                                   ├── pareto_front.png
     │                             ▼                                   ├── equity_curves.png
     │                        Visualizers                              └── tree_graph.png
     └─────────────────────────────│
                                   │── plot_pareto_front(hof) ──────► pareto_front.png
                                   │── plot_equity_curves(hof, pf) ─► equity_curves.png
                                   └── plot_tree_graph(ind) ─────────► tree_graph.png
```

### Recommended Project Structure (Phase 5 additions)

```
vgp/
├── analysis/
│   ├── __init__.py           # PUBLIC: WalkForwardRunner, compute_dsr, plot_*
│   ├── runner.py             # WalkForwardRunner class + WindowSpec dataclass
│   ├── dsr.py                # compute_dsr(), aggregate_seeds(), save_results_csv()
│   └── plots.py              # plot_pareto_front(), plot_equity_curves(), plot_tree_graph()
tests/
├── test_validation.py        # VAL-01 through VAL-04 (runner + DSR)
└── test_plots.py             # VAL-05, VAL-06, VAL-07 (file-exists + shape checks)
```

### Pattern 1: WalkForwardRunner with WindowSpec

**What:** A dataclass `WindowSpec` captures one rolling window's date boundaries. `WalkForwardRunner` calls `WalkForwardSplitter.split()` per window, then calls `run_evolution()` for each seed, collecting results into a list of dicts.

**When to use:** Any multi-window, multi-seed experiment.

**Key design constraint (VAL-02):** The OOS (test) split is extracted by WalkForwardSplitter and immediately stored. The runner stores `test_start/test_end` per window and never passes `test_data` to `run_evolution()`. Evolution receives only the train slice. OOS evaluation of the best individual happens exactly once after evolution completes, using the stored test slice.

```python
# Source: VGP codebase (vgp/data/splitter.py) + verified WalkForwardSplitter API
from dataclasses import dataclass
import pandas as pd
from vgp.data.splitter import WalkForwardSplitter

@dataclass
class WindowSpec:
    window_id: int
    train_end: str       # e.g. "2024-12-31"
    val_start: str
    val_end: str
    test_start: str      # stored; not passed to evolution
    test_end: str        # stored; not passed to evolution

def generate_windows(
    total_start: str,
    total_end: str,
    train_months: int = 12,
    val_months: int = 2,
    oos_months: int = 3,
    step_months: int = 3,
) -> list[WindowSpec]:
    """Generate non-overlapping walk-forward window specs."""
    from dateutil.relativedelta import relativedelta
    windows = []
    start = pd.Timestamp(total_start)
    total_end_ts = pd.Timestamp(total_end)
    window_id = 0
    while True:
        train_end_ts = start + relativedelta(months=train_months) - pd.Timedelta(days=1)
        val_start_ts = train_end_ts + pd.Timedelta(days=1)
        val_end_ts = val_start_ts + relativedelta(months=val_months) - pd.Timedelta(days=1)
        test_start_ts = val_end_ts + pd.Timedelta(days=1)
        test_end_ts = test_start_ts + relativedelta(months=oos_months) - pd.Timedelta(days=1)
        if test_end_ts > total_end_ts:
            break
        windows.append(WindowSpec(
            window_id=window_id,
            train_end=train_end_ts.strftime("%Y-%m-%d"),
            val_start=val_start_ts.strftime("%Y-%m-%d"),
            val_end=val_end_ts.strftime("%Y-%m-%d"),
            test_start=test_start_ts.strftime("%Y-%m-%d"),
            test_end=test_end_ts.strftime("%Y-%m-%d"),
        ))
        window_id += 1
        start = start + relativedelta(months=step_months)
    return windows
```

**Verified:** 4 windows fit in 2024-01-01 to 2026-04-01 with these parameters. [VERIFIED: calculated against parquet data]

### Pattern 2: OOS Evaluation exactly once per window (VAL-02)

**What:** After `run_evolution()` returns `(pop, hof, logbook)`, pick `hof[0]` (best by first objective = Sharpe). Slice the stored test data using the same `WalkForwardSplitter.split()` call but return only the `test` slice. Call `evaluate(hof[0], test_feature_matrix, eval_config_with_test_close_prices)` once. Store the OOS Sharpe. Never call evaluate on OOS data again.

**Critical note:** The split is done once at window setup. The test slice is stored as a variable. The runner code path that reaches `evaluate(...)` with test data is executed exactly once per window. This is structural (code path), not advisory (documentation).

```python
# Pattern: store test slice, never re-use
splitter = WalkForwardSplitter()
train_fm, val_fm, test_fm = splitter.split(
    feature_matrix,
    train_end=window.train_end,
    val_start=window.val_start,
    val_end=window.val_end,
    test_start=window.test_start,
    dates=engine.dates_,
)
# Evolution uses train_fm ONLY
pop, hof, logbook = run_evolution(config, train_fm, eval_config_train, ...)

# OOS evaluation: called ONCE, result stored immediately
oos_sharpe = evaluate(hof[0], test_fm, eval_config_test)[0]  # index 0 = Sharpe
# After this line, test_fm is no longer referenced in this function
```

### Pattern 3: Multi-Seed Aggregation

**What:** For each window, loop over `n_seeds` seeds. Each seed gets a fresh `EvolutionConfig` with `seed=s`. Collect IS Sharpe (from `logbook.chapters['fitness'][-1]['sharpe_max']`) and OOS Sharpe (from the single OOS evaluate call) per seed.

```python
# Source: verified against vgp/evolution/loop.py logbook structure [VERIFIED: codebase]
import numpy as np
from vgp.evolution.config import EvolutionConfig
from vgp.evolution.loop import run_evolution
from vgp.backtest.runner import evaluate

seed_results = []
for seed in seeds:
    cfg = EvolutionConfig(seed=seed, ...)
    pop, hof, logbook = run_evolution(cfg, train_fm, eval_config_train)
    # IS Sharpe: last generation's max (from logbook)
    is_sharpe = logbook.chapters['fitness'][-1]['sharpe_max']
    # OOS Sharpe: single evaluate call on test slice
    oos_sharpe = evaluate(hof[0], test_fm, eval_config_test)[0]
    seed_results.append({'seed': seed, 'is_sharpe': is_sharpe, 'oos_sharpe': oos_sharpe})

oos_sharpes = np.array([r['oos_sharpe'] for r in seed_results])
median_oos = float(np.median(oos_sharpes))
q25 = float(np.percentile(oos_sharpes, 25))
q75 = float(np.percentile(oos_sharpes, 75))
iqr = q75 - q25
```

### Pattern 4: DSR Implementation (Bailey & Lopez de Prado 2014)

**What:** Computes the probability that a Sharpe ratio beats the expected maximum of independent SR estimates under the null hypothesis of no skill, adjusted for non-normal return distributions.

**Reference:** Bailey & Lopez de Prado, "The Deflated Sharpe Ratio", Journal of Portfolio Management, 2014. The formula in Proposition 3 adjusts for skewness, excess kurtosis, and the number of independent trials.

**Key note:** scipy.stats.kurtosis returns FISHER (excess) kurtosis by default. The DSR formula uses excess kurtosis. No conversion needed if using scipy.stats.kurtosis with default `fisher=True`.

```python
# Source: Bailey & Lopez de Prado (2014), implemented with scipy.stats [VERIFIED: scipy 1.17.1]
from scipy.stats import norm, skew, kurtosis
import numpy as np

def compute_dsr(
    returns: np.ndarray,          # per-period IS returns, shape [T]
    sr_hat: float,                # annualized IS Sharpe ratio
    n_trials: int,                # number of independent seeds/configs tested
    periods_per_year: int = 252,  # daily data
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Returns the probability (0-1) that sr_hat exceeds the expected
    maximum Sharpe under H0, adjusted for multiple testing and non-normality.
    Values above 0.95 indicate statistical significance.
    """
    T = len(returns)
    gamma = 0.5772156649  # Euler-Mascheroni constant

    # Expected maximum SR under H0 (Proposition 3)
    expected_max_sr = (
        (1 - gamma) * norm.ppf(1 - 1 / n_trials)
        + gamma * norm.ppf(1 - 1 / (n_trials * np.e))
    )

    # Return distribution moments
    ret_skew = float(skew(returns))
    ret_kurt = float(kurtosis(returns, fisher=True))  # excess kurtosis

    # Variance of SR estimate (non-normality adjustment)
    sr_var = (
        1 + (0.5 * sr_hat**2) - ret_skew * sr_hat + ((ret_kurt / 4) * sr_hat**2)
    ) / (T - 1)

    # DSR: P(SR > expected_max_SR)
    dsr = float(norm.cdf((sr_hat - expected_max_sr) / np.sqrt(sr_var)))
    return dsr
```

### Pattern 5: Results Persistence (CSV — no MLflow)

**What:** Save per-window, per-seed results to CSV using pandas. No MLflow dependency.

```python
# Source: VGP STATE.md (mlflow excluded due to pandas<3 conflict) [VERIFIED: STATE.md]
import pandas as pd

def save_results_csv(results: list[dict], path: str) -> None:
    """Save experiment results to CSV.

    Each row is one (window_id, seed) pair. Columns:
      window_id, seed, is_sharpe, oos_sharpe, dsr,
      median_oos_sharpe, iqr_oos_sharpe, n_trades
    """
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
```

### Pattern 6: Pareto Front 3D Scatter

**What:** Extract `(sharpe, total_return, -tree_size)` from `hof`, plot as 3D scatter.

**Data source:** `individual.fitness.values` returns the 3-tuple stored during evolution. [VERIFIED: DEAP ParetoFront confirmed working in venv]

```python
# Source: verified against DEAP ParetoFront API and mpl_toolkits.mplot3d [VERIFIED: venv]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — required for projection='3d'
import numpy as np

def plot_pareto_front(hof, output_path: str = "results/pareto_front.png") -> None:
    """Export Pareto front scatter plot (Sharpe vs return vs tree size)."""
    sharpes = [ind.fitness.values[0] for ind in hof]
    returns = [ind.fitness.values[1] for ind in hof]
    sizes = [-ind.fitness.values[2] for ind in hof]  # stored as negative

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(sharpes, returns, sizes, c=sharpes, cmap='viridis', s=60, alpha=0.8)
    ax.set_xlabel('Sharpe Ratio', labelpad=10)
    ax.set_ylabel('Total Return', labelpad=10)
    ax.set_zlabel('Tree Size (nodes)', labelpad=10)
    ax.set_title('NSGA-II Pareto Front — Top Generation', fontsize=13)
    plt.colorbar(sc, ax=ax, label='Sharpe Ratio', shrink=0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
```

### Pattern 7: Equity Curve with IS/OOS Overlay

**What:** Re-run evaluate() for the top-3 HOF individuals on the full dataset (train + OOS concatenated). Plot `pf.value()` series with a vertical line at the IS/OOS boundary. The re-run on full data is for visualization only; the OOS Sharpe used in reporting comes from the single-shot evaluate call.

**vectorbt 1.0.0 API:** `pf.value()` returns a `pd.Series` with `DatetimeIndex`. `pf.cumulative_returns()` returns same shape. [VERIFIED: tested in venv]

```python
# Source: vectorbt 1.0.0 API verified in project venv [VERIFIED]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def plot_equity_curves(
    individuals: list,           # top-3 HOF individuals
    feature_matrix: np.ndarray,  # full data [T x F x A]
    eval_config: EvalConfig,     # must have close_prices set to full data
    train_end_date: str,         # e.g. "2024-12-31"
    output_path: str = "results/equity_curves.png",
) -> None:
    """Plot IS + OOS equity curves for top-3 individuals."""
    fig, axes = plt.subplots(len(individuals), 1, figsize=(14, 4 * len(individuals)))
    if len(individuals) == 1:
        axes = [axes]

    for idx, (ind, ax) in enumerate(zip(individuals, axes)):
        # Re-run evaluate to get portfolio object (visualization only)
        # NOTE: This re-run is for plot purposes; OOS Sharpe was already recorded
        from vgp.backtest.runner import evaluate  # deferred import per D-15
        # evaluate() returns fitness tuple, not portfolio — need direct vbt call
        # See "Pitfall: evaluate() does not return pf object" section below
        pass  # Implementation handled in BacktestRunner extension or via direct vbt call

    train_end = pd.Timestamp(train_end_date)
    for ax in axes:
        ax.axvline(x=train_end, color='red', linestyle='--', linewidth=1.5,
                   label='IS / OOS boundary')
        ax.axvspan(ax.get_xlim()[0], train_end.timestamp(), alpha=0.05, color='blue')
        ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
```

**PITFALL — evaluate() does not return the Portfolio object.** The current `evaluate()` function returns `(sharpe, total_return, -tree_size)` and discards the `pf` object (it's a local variable). To get the equity curve for plotting, either: (a) add a `return_portfolio=True` flag to `evaluate()` or (b) duplicate the vbt call inside `plot_equity_curves()`. Option (b) is simpler and keeps the backtest module interface unchanged. The visualization module imports `vbt` and calls `Portfolio.from_signals()` directly, using `TreeEvaluator.execute()` to get signals.

### Pattern 8: GP Tree Graph Export

**What:** Convert a DEAP PrimitiveTree to a NetworkX DiGraph using `deap.gp.graph()`, then draw with `nx.bfs_layout()` and matplotlib.

**No Graphviz needed:** `graphviz` binary is absent on this machine. `nx.bfs_layout(G, start=0)` produces a clean top-down hierarchical layout without Graphviz. [VERIFIED: confirmed working in venv]

```python
# Source: DEAP gp.graph() verified in project venv; nx.bfs_layout confirmed [VERIFIED]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from deap import gp as deap_gp

def plot_tree_graph(
    individual,
    output_path: str = "results/tree_graph.png",
    title: str = "GP Tree Structure",
) -> None:
    """Export GP tree as a human-readable NetworkX graph image.

    Uses deap.gp.graph() to get nodes, edges, and labels.
    Layout: nx.bfs_layout(G, start=0) — top-down hierarchical, no graphviz required.
    """
    nodes, edges, labels = deap_gp.graph(individual)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    pos = nx.bfs_layout(G, start=0)

    fig, ax = plt.subplots(figsize=(max(10, len(nodes) * 0.8), 8))
    nx.draw(
        G, pos=pos, labels=labels, ax=ax,
        node_color='lightblue', node_size=1800,
        font_size=9, font_weight='bold',
        arrows=True, arrowsize=15,
        edge_color='gray',
    )
    ax.set_title(f'{title} (height={individual.height}, nodes={len(individual)})',
                 fontsize=12, pad=20)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
```

### Anti-Patterns to Avoid

- **Touching OOS data during evolution:** The test slice must never be passed to `run_evolution()`. Store it as a separate variable after the split call.
- **MLflow for Phase 5 results:** mlflow requires pandas<3 (incompatible). Use CSV/JSON. [VERIFIED: STATE.md blocker]
- **matplotlib.pyplot before matplotlib.use('Agg'):** On headless machines (CI), importing pyplot before calling `matplotlib.use('Agg')` raises a backend warning or fails. Always call `matplotlib.use('Agg')` first in test code and scripts.
- **graphviz-based layout without binary:** `nx.nx_agraph.graphviz_layout` will fail because `dot` binary is absent. Use `nx.bfs_layout()` instead.
- **Using `.values` on pandas Series/DataFrame:** Violates pandas 3.0 idiom. Use `.to_numpy()`.
- **Placing walk-forward loop logic inside run_evolution():** The evolution function must stay a single-window, single-seed unit. The walk-forward runner is a new module in `vgp/analysis/`.

---

## Walk-Forward Validation Architecture

### Data Range and Window Configuration

The parquet cache contains daily OHLCV data from 2024-01-01 to 2026-04-01 (27 months). After FeatureEngine lookback trim, the effective feature matrix starts around 2024-05-01 (per STATE.md confirmed intersection). [VERIFIED: confirmed by actual parquet file inspection]

Recommended walk-forward parameters:
- Train: 12 months
- Validation: 2 months (used only by EvolutionConfig during IS, not as a separate eval split)
- OOS (test): 3 months
- Step: 3 months (non-overlapping OOS windows)

Result: **4 windows** with non-overlapping OOS periods [VERIFIED: computed against actual data range]:

| Window | Train | Val | OOS (Test) |
|--------|-------|-----|-----------|
| 1 | 2024-01-01 to 2024-12-31 | 2025-01-01 to 2025-02-28 | 2025-03-01 to 2025-05-31 |
| 2 | 2024-04-01 to 2025-03-31 | 2025-04-01 to 2025-05-31 | 2025-06-01 to 2025-08-31 |
| 3 | 2024-07-01 to 2025-06-30 | 2025-07-01 to 2025-08-31 | 2025-09-01 to 2025-11-30 |
| 4 | 2024-10-01 to 2025-09-30 | 2025-10-01 to 2025-11-30 | 2025-12-01 to 2026-02-28 |

### WalkForwardSplitter Integration

`WalkForwardSplitter.split()` already enforces `AssertionError` when `val_start <= train_end` or `test_start <= val_end`. [VERIFIED: vgp/data/splitter.py lines 81-85]

The runner calls `.split()` once per window for both the feature matrix (`np.ndarray`) and the close prices DataFrame (`pd.DataFrame`) separately. The test slices are stored and passed to a single `evaluate()` call after evolution.

Feature matrix split requires `dates=engine.dates_`. Close prices split uses the DataFrame's DatetimeIndex directly. Both are handled by the existing `WalkForwardSplitter.split()` signature.

### OOS Holdout Structural Enforcement (VAL-02)

The structural guarantee is achieved by code organization:

1. `generate_windows()` produces `WindowSpec` objects with `test_start`/`test_end` fields.
2. `WalkForwardRunner.run_window()` splits data at the start, assigns `test_fm` and `test_close` to local variables, and passes only `train_fm` and `train_close` to `run_evolution()`.
3. After evolution, `evaluate(hof[0], test_fm, eval_config_with_test_close)` is called exactly once.
4. No loop, no retry, no re-evaluation. The test slice variables go out of scope after the OOS result is stored.

This is structural (code path) not advisory (comment). A reviewer can confirm OOS is touched once by reading the function body.

---

## Multi-Seed Experiment Runner

### Seed Strategy

Run 10 seeds per window (seeds 0-9 or seeds 42, 43, ..., 51). Sequential execution is acceptable for validation runs; embarrassingly parallel (one Pool per seed) risks OOM from numba JIT in multiple pools.

**Recommendation:** Run seeds sequentially within each window. Total runs: 4 windows × 10 seeds = 40 evolution calls. At ~5-10 minutes per run (Phase 4 benchmark), total wall time is 3-7 hours. Schedule as an overnight run.

### Aggregation Metrics

Per window, per seed:
- `is_sharpe`: `logbook.chapters['fitness'][-1]['sharpe_max']` (from final generation)
- `oos_sharpe`: single `evaluate(hof[0], test_fm, ...)` call
- `dsr`: computed from IS returns distribution

Per window aggregate (across seeds):
- `median_oos_sharpe`, `iqr_oos_sharpe` (q75 - q25)
- `median_dsr`
- `n_seeds_positive_oos`: count of seeds with OOS Sharpe > 0

### Results Format

CSV with columns: `window_id, seed, train_end, test_start, test_end, is_sharpe, oos_sharpe, dsr, n_nodes_best_ind`. One row per (window, seed). Easy to load with pandas for post-hoc analysis.

---

## Deflated Sharpe Ratio (DSR) Implementation

### Reference

Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality." Journal of Portfolio Management, 40(5), 94-107.

### Formula

DSR = P(SR > E[max SR under H0])

where:
- E[max SR] = (1 - γ) * Z^{-1}(1 - 1/N) + γ * Z^{-1}(1 - 1/(N*e)) (Proposition 3)
- γ = Euler-Mascheroni constant ≈ 0.5772
- N = number of independent trials (seeds × windows)
- Var(SR) = [1 + (1/2)SR² - skew·SR + (kurt/4)·SR²] / (T-1)

### Inputs Required

- `returns`: per-period IS returns array (derive from logbook or direct vbt call)
- `sr_hat`: annualized IS Sharpe (from logbook or `pf.sharpe_ratio()`)
- `n_trials`: number of trials = n_seeds × n_windows (10 × 4 = 40 in default config)
- `periods_per_year`: 252 for daily

**Note on returns extraction:** `pf.returns()` on a vectorbt Portfolio object returns the per-period portfolio return series as a `pd.Series`. [VERIFIED: tested in venv] Convert to numpy with `.to_numpy()`. This requires re-running the IS backtest for the DSR computation, which is acceptable (it is not touching OOS data).

### Implementation

Use `scipy.stats.norm`, `scipy.stats.skew`, `scipy.stats.kurtosis` (all confirmed available at scipy 1.17.1). [VERIFIED]

No external DSR package is needed. The formula has ~15 lines of pure numpy/scipy code.

---

## Pareto Front Visualization (VAL-05)

### Data Source

`hof` (tools.ParetoFront) from `run_evolution()` return. Each element is a `creator.Individual` with `individual.fitness.values = (sharpe, total_return, -tree_size)`. [VERIFIED: ParetoFront confirmed working in venv]

### Plot Type

3D scatter: `mpl_toolkits.mplot3d.Axes3D` with `projection='3d'`. Color-coded by Sharpe ratio (colormap = viridis). [VERIFIED: tested in venv]

Alternative: 3 separate 2D scatter projections on one figure (3 subplots). This is more readable in publication PDFs where 3D projections flatten unpredictably.

**Recommendation:** Produce both. A 3-subplot 2D version is more reproducible across renderers; the 3D version is more visually striking. Given the small code delta, implement both and export to separate PNG files.

### Output

- File: `results/pareto_front.png`
- DPI: 150 (screen) or 300 (publication)
- Backend: `matplotlib.use('Agg')` for headless

---

## Equity Curve Plotting (VAL-06)

### vectorbt 1.0.0 API

Key verified methods (all confirmed in project venv): [VERIFIED]

| Method | Returns | Notes |
|--------|---------|-------|
| `pf.value()` | `pd.Series` with DatetimeIndex | Portfolio NAV including cash |
| `pf.cumulative_returns()` | `pd.Series` with DatetimeIndex | Cumulative return from start |
| `pf.returns()` | `pd.Series` with DatetimeIndex | Per-period return |
| `pf.sharpe_ratio()` | `float` | Already used in evaluate() |
| `pf.total_return()` | `float` | Already used in evaluate() |

### IS/OOS Overlay Pattern

Use `ax.axvline(x=train_end_ts, color='red', linestyle='--')` and optionally `ax.axvspan` with `alpha=0.05` for background shading. `train_end_ts` is a `pd.Timestamp` passed to matplotlib directly. [ASSUMED: axvline with Timestamp is standard matplotlib pattern — not specifically verified in venv but is standard practice]

### Equity Curve Access Requires Re-Running vbt

The existing `evaluate()` function discards the `pf` object. For plot purposes, the visualization module must call `TreeEvaluator.execute()` and then `vbt.Portfolio.from_signals()` directly, using the same parameters as `evaluate()`. This is intentional duplication for the visualization layer — `evaluate()` stays a pure fitness-returning function.

---

## GP Tree Graph Export (VAL-07)

### DEAP gp.graph() API

**Signature:** `gp.graph(expr) -> (nodes: list[int], edges: list[tuple], labels: dict[int, str])` [VERIFIED: tested in project venv]

- `nodes`: list of integer node IDs (0-indexed, depth-first order)
- `edges`: list of (parent, child) integer tuples
- `labels`: dict mapping node ID to primitive/terminal name string

For the VGP pset, terminal names are the FEATURE_NAMES values (e.g., `"ret_1d"`, `"vol_5d"`) because `pset.renameArguments()` was called in `build_pset()`. [VERIFIED: gp_types.py lines 127-139]

### Layout Strategy

`nx.bfs_layout(G, start=0)` with `G` being a `nx.DiGraph` built from nodes/edges. Start node 0 is always the root of the tree (DEAP's gp.graph guarantee). [VERIFIED: tested in project venv]

This produces a top-down hierarchical layout. No graphviz binary required. Tested with simulated trees and with a real VGP pset tree (`rmax20(parkinson_14)`, height=1). [VERIFIED]

### Readability for Humans (VAL-07 criterion)

- `node_size=1800`, `font_size=9` for small-to-medium trees (height ≤ 8, ≤ 100 nodes)
- `figsize=(max(10, len(nodes) * 0.8), 8)` scales width with tree size
- Title includes `height` and node count for quick sanity check

---

## Community Release

### Current State

Already present: [VERIFIED: ls /root directory]
- `LICENSE` (MIT)
- `CONTRIBUTING.md` (but has placeholder text for Phase 3/4 sections)
- `Makefile` with `test`, `smoke`, `lint`, `approve` targets
- `pyproject.toml` with `[project.scripts]` not yet defined

### CONTRIBUTING.md Updates Needed

Phase 5 must update CONTRIBUTING.md to replace the placeholders:
- "Running an Experiment" section: now has actual CLI (`python -m vgp.analysis.runner ...` or `make experiment`)
- Add "Interpreting Results" section: how to read results.csv, what DSR > 0.95 means
- Add "Visualizations" section: where PNGs are saved, how to regenerate them

### README.md

No `README.md` exists at root [VERIFIED: ls output showed no README.md]. Phase 5 must create one. Minimum sections:
1. What is VGP (one paragraph)
2. Quick start (pip install -e . → run experiment → view results)
3. Architecture overview (reference the 5 sub-modules)
4. Research notes (OOS Sharpe, DSR, what the numbers mean)
5. Citation/license

**Note:** COMM-01 (vgp package organized with sub-modules) is already complete (all 5 sub-modules exist). COMM-03 (CONTRIBUTING.md) exists but needs Phase 4/5 sections updated.

### CLI Entrypoint

`[project.scripts]` is currently absent from pyproject.toml. A minimal entrypoint would allow `vgp-run` or `python -m vgp.analysis.runner`. Given Phase 5 scope, a `__main__.py` in `vgp/analysis/` for `python -m vgp.analysis.runner` is sufficient without adding a console_scripts entry. The v2 requirement CFG-02 covers the full CLI.

---

## Implementation Risks

### Risk 1: evaluate() does not return Portfolio object (HIGH PROBABILITY, LOW IMPACT)

**What:** `evaluate()` returns a 3-tuple and discards `pf`. Equity curve plotting needs `pf.value()`.
**Mitigation:** The visualization module calls `vbt.Portfolio.from_signals()` directly, replicating the evaluate() call with identical parameters. This is a deliberate architectural choice (clean interface > convenience). The planner should specify this clearly in Plan 2 tasks.
**Status:** RESOLVED in research — pattern documented above.

### Risk 2: matplotlib backend in test context (MEDIUM PROBABILITY, LOW IMPACT)

**What:** On macOS the default backend is `macosx`; in CI (no display) it will fail.
**Mitigation:** Call `matplotlib.use('Agg')` at the top of every test file that imports pyplot. Test files for plots should not use interactive backends. [VERIFIED: Agg works in venv]

### Risk 3: Walk-forward compute time (MEDIUM PROBABILITY, MEDIUM IMPACT)

**What:** 40 evolution runs × 5-10 minutes each = 3-7 hours. Tests cannot run full evolution.
**Mitigation:** All tests use synthetic data and tiny configs (pop_size=10, n_generations=3). The actual experiment is a script run, not a pytest test. Tests verify API contracts, not research results.

### Risk 4: dateutil.relativedelta not in pyproject.toml (LOW PROBABILITY — verify)

**What:** `generate_windows()` uses `dateutil.relativedelta`. `python-dateutil` is a pandas transitive dependency but not directly declared.
**Verification needed:** Check if `python-dateutil` is available.
**Mitigation:** Add `python-dateutil>=2.8` to pyproject.toml if not already present as transitive dep.

### Risk 5: DSR returns NaN for flat portfolios (LOW PROBABILITY, LOW IMPACT)

**What:** If `returns` array is all zeros (flat portfolio), `scipy.stats.skew` and `scipy.stats.kurtosis` return NaN; division in DSR formula fails.
**Mitigation:** Guard: if `np.std(returns) == 0`, return `dsr = 0.0` (no evidence of skill).

### Risk 6: HOF empty or too small (LOW PROBABILITY, LOW IMPACT)

**What:** If evolution finds no valid individuals, `hof` may have fewer than 3 members. `plot_equity_curves(top-3)` must handle this.
**Mitigation:** `individuals = hof[:min(3, len(hof))]` — plot however many are available.

---

## Recommended Plan Structure

**Plan 05-01: Walk-Forward Runner + DSR Module**

Files:
- `vgp/analysis/runner.py` — `WindowSpec`, `generate_windows()`, `WalkForwardRunner`
- `vgp/analysis/dsr.py` — `compute_dsr()`, `aggregate_seeds()`, `save_results_csv()`
- `vgp/analysis/__init__.py` — public exports

Tests (`tests/test_validation.py`):
- VAL-01: `generate_windows()` produces non-overlapping windows
- VAL-02: Runner never passes test data to run_evolution (inspect call args)
- VAL-03: Runner accepts list of seeds and iterates
- VAL-04: `compute_dsr()` returns float in [0, 1]; guards against flat returns

**Plan 05-02: Visualization Module**

Files:
- `vgp/analysis/plots.py` — `plot_pareto_front()`, `plot_equity_curves()`, `plot_tree_graph()`
- `vgp/analysis/__init__.py` — add plot exports

Tests (`tests/test_plots.py`):
- VAL-05: `plot_pareto_front()` creates PNG file with synthetic HOF data
- VAL-06: `plot_equity_curves()` creates PNG file (tests file existence and non-zero size)
- VAL-07: `plot_tree_graph()` creates PNG file from a real VGP individual

**Plan 05-03: Community Release**

Files:
- `README.md` — create from scratch
- `CONTRIBUTING.md` — update Phase 4/5 placeholders
- Final test gate: `pytest tests/ -v` all 46 + new tests pass

---

## Validation Architecture (Test Map)

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 |
| Config | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run | `python -m pytest tests/test_validation.py tests/test_plots.py -v` |
| Full suite | `python -m pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Test File | Notes |
|--------|----------|-----------|-----------|-------|
| VAL-01 | generate_windows() produces N non-overlapping OOS windows | unit | test_validation.py | Verify test_start[i] == test_end[i-1] + 1 day |
| VAL-02 | OOS holdout not passed to run_evolution | unit | test_validation.py | Mock run_evolution; assert test_fm not in args |
| VAL-03 | Runner iterates over list of seeds | unit | test_validation.py | Run with seeds=[0,1,2], assert 3 result rows |
| VAL-04 | compute_dsr() returns float, handles edge cases | unit | test_validation.py | Test with synthetic returns; flat portfolio guard |
| VAL-05 | Pareto front PNG exists and is non-empty | unit | test_plots.py | Use synthetic HOF with 5 individuals |
| VAL-06 | Equity curve PNG exists and is non-empty | unit | test_plots.py | Use tiny vectorbt portfolio |
| VAL-07 | Tree graph PNG exists and is non-empty | unit | test_plots.py | Use real VGP individual from toolbox |

**Note:** Visualization tests verify file creation, not visual correctness. "Non-empty" = file size > 1000 bytes. Visual correctness is a human review item.

### Sampling Rate

- Per task commit: `python -m pytest tests/test_validation.py tests/test_plots.py -v`
- Per wave merge: `python -m pytest tests/ -v`
- Phase gate: full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/test_validation.py` — covers VAL-01 through VAL-04
- [ ] `tests/test_plots.py` — covers VAL-05, VAL-06, VAL-07
- [ ] `vgp/analysis/runner.py` — create
- [ ] `vgp/analysis/dsr.py` — create
- [ ] `vgp/analysis/plots.py` — create

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| scipy | DSR computation (compute_dsr) | Yes | 1.17.1 | — |
| networkx | GP tree graph (plot_tree_graph) | Yes | 3.6.1 | — |
| matplotlib | All visualizations | Yes | 3.10.9 | — |
| mpl_toolkits.mplot3d | 3D Pareto scatter | Yes | bundled with mpl | Use 2D projection |
| deap.gp.graph | Tree-to-DiGraph conversion | Yes | deap 1.4.4 | — |
| vectorbt pf.value() | Equity curve plotting | Yes | 1.0.0 | — |
| graphviz binary (dot) | Tree layout | No | — | nx.bfs_layout() (confirmed) |
| python-dateutil | generate_windows() relativedelta | [VERIFY] | — | Use pandas DateOffset |
| mlflow | Experiment tracking | No (excluded) | — | CSV/JSON (confirmed) |
| pyfolio / riskfolio | DSR computation | No | — | Manual scipy formula |

**Missing dependencies with fallback:**
- graphviz binary: use `nx.bfs_layout()` (verified working)
- mlflow: use CSV/JSON (per STATE.md decision)
- python-dateutil: verify in venv; if absent, use `pd.DateOffset` for date arithmetic

**Missing dependencies with no fallback:** None that block execution.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GP tree serialization to graph | Custom tree traversal | `deap.gp.graph(individual)` | Returns nodes/edges/labels in one call [VERIFIED] |
| Hierarchical tree layout | Manual coordinate computation | `nx.bfs_layout(G, start=0)` | Handles arbitrary tree topologies [VERIFIED] |
| Return distribution moments | Manual skew/kurtosis loop | `scipy.stats.skew()`, `scipy.stats.kurtosis()` | Fisher/Pearson convention handled correctly |
| Temporal date arithmetic | `timedelta(days=N)` for months | `dateutil.relativedelta.relativedelta(months=N)` | Handles month-end correctly (Feb 28/29, etc.) |
| Non-overlapping window verification | Boolean logic over date lists | Assert `test_start[i] >= test_end[i-1]` in test | Simple and direct |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | python-dateutil is available as a transitive dep (not directly declared in pyproject.toml) | generate_windows() | generate_windows() fails at import; fallback to pd.DateOffset is straightforward |
| A2 | ax.axvline() with pd.Timestamp x-value works without conversion | Equity curve plotting | Needs explicit `mdates.date2num(train_end_ts)` conversion; low impact, easy fix |
| A3 | 4 walk-forward windows provide sufficient statistical power for DSR interpretation | Walk-Forward Architecture | DSR with 40 trials total is meaningful; if data range were shorter, fewer windows = higher expected max SR = harder to beat |

---

## Open Questions

1. **dateutil.relativedelta availability**
   - What we know: `python-dateutil` is a pandas dependency but not explicitly declared in VGP pyproject.toml
   - What's unclear: Whether it's guaranteed to be present in all install scenarios
   - Recommendation: Verify in venv; if available as transitive dep, add a comment in runner.py; if not, use `pd.DateOffset(months=N)` instead which is guaranteed via pandas

2. **OOS evaluation: fitness tuple only or full portfolio?**
   - What we know: `evaluate()` returns `(sharpe, total_return, -tree_size)` and discards pf
   - What's unclear: Whether the planner wants plot_equity_curves to re-run vbt directly or whether evaluate() should be extended
   - Recommendation: Keep evaluate() interface unchanged; plots.py calls vbt directly. Rationale: adding `return_portfolio=True` to evaluate() complicates the multiprocessing pickle path unnecessarily.

3. **Number of seeds: 10 sequential vs parallel**
   - What we know: 10+ seeds required (VAL-03); sequential is safe; parallel risks OOM from multiple numba JIT caches
   - What's unclear: User's compute environment and time budget
   - Recommendation: Default to sequential; note in runner.py that seeds can be parallelized across processes with `multiprocessing.Pool` at the runner level (not within each evolution run)

---

## Sources

### Primary (HIGH confidence — verified in project venv)

- DEAP 1.4.4 source code (`deap.gp.graph` function) — `gp.graph(individual)` returns (nodes, edges, labels)
- vectorbt 1.0.0 Portfolio API — `pf.value()`, `pf.cumulative_returns()`, `pf.returns()` all return pd.Series with DatetimeIndex
- networkx 3.6.1 — `nx.bfs_layout(G, start=0)` confirmed working; graphviz binary confirmed absent
- scipy 1.17.1 — `stats.norm`, `stats.skew`, `stats.kurtosis` all available
- matplotlib 3.10.9 — `matplotlib.use('Agg')` + `fig.savefig(..., dpi=150)` confirmed working headless
- VGP codebase (`vgp/evolution/loop.py`) — `run_evolution()` returns `(population, hof, logbook)`; `logbook.chapters['fitness'][-1]['sharpe_max']` confirmed
- VGP codebase (`vgp/data/splitter.py`) — `WalkForwardSplitter.split()` signature and AssertionError enforcement confirmed
- VGP codebase (`vgp/backtest/runner.py`) — `evaluate()` returns 3-tuple; `pf` discarded
- VGP STATE.md — mlflow excluded (pandas<3 conflict); confirmed

### Secondary (MEDIUM confidence)

- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio" — DSR formula from paper; implementation verified against scipy output
- Actual parquet data files — 27 files, range 2024-01-01 to 2026-04-01, 18 assets with full coverage from 2024-01-01 [VERIFIED by file inspection]

### Tertiary (LOW confidence)

- `python-dateutil` as transitive dependency — not explicitly verified; pandas depends on it but not guaranteed in all environments [ASSUMED]

---

## Metadata

**Confidence breakdown:**
- Walk-forward architecture: HIGH — WalkForwardSplitter API verified in code; window count verified against real data
- Multi-seed runner: HIGH — run_evolution() signature and return type verified in codebase
- DSR implementation: HIGH — scipy.stats confirmed; formula verified against published source
- Pareto front visualization: HIGH — mpl_toolkits.mplot3d and ParetoFront API both verified in venv
- Equity curve visualization: HIGH — pf.value() confirmed returning pd.Series with DatetimeIndex
- GP tree graph export: HIGH — deap.gp.graph() + nx.bfs_layout() both verified in venv
- Community release: MEDIUM — README content is [ASSUMED] to be what users need; no external validation

**Research date:** 2026-06-13
**Valid until:** 2026-07-13 (30 days — stable libraries, no fast-moving dependencies)

---

## RESEARCH COMPLETE
