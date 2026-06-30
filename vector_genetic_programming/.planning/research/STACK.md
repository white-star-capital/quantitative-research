# Stack Research: VGP for Crypto Trading

**Project:** Vector Genetic Programming (VGP)
**Researched:** 2026-06-03
**Mode:** Ecosystem

---

## Recommended Stack (2025/2026)

### GP Framework

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| DEAP | ==1.4.4 | Core GP engine, NSGA-II, strongly-typed GP | Only mature Python library with native `PrimitiveSetTyped`, `selNSGA2`, and a single-line parallel evaluation API (`toolbox.map`). Actively maintained — 1.4.4 released April 2026, 1.4.3 May 2025. |

**DEAP wins for this project because:**

1. `PrimitiveSetTyped` enforces vector/scalar type separation at tree construction time — prevents invalid trees without runtime checks.
2. `tools.selNSGA2` is the standard multi-objective selection operator for GP; implemented and battle-tested inside DEAP.
3. Parallelism is a single-line swap: `toolbox.register("map", pool.map)` — no architectural change required.
4. `gp.graph(expr)` returns nodes/edges/labels directly, enabling tree visualization with NetworkX or pygraphviz without custom serialization.
5. `algorithms.eaMuPlusLambda` and `eaMuCommaLambda` support the (mu+lambda) and (mu,lambda) NSGA-II lifecycle out of the box.

**Confidence:** HIGH — verified against DEAP GitHub, ReadTheDocs 1.4.3 docs, and PyPI release history.

---

### Backtesting

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| vectorbt | ==1.0.0 | Vectorized portfolio simulation for GP fitness evaluation | Only open-source backtester that fully vectorizes across thousands of strategy variants simultaneously. 1M orders in 70–100ms on M1. |

**vectorbt 1.0.0 API — What you must know:**

The 1.0.0 release (April 2026, PyPI) is a **major rewrite** from 0.28.x. Key patterns for GP evaluation:

```python
# Core evaluation pattern for a GP individual
import vectorbt as vbt

pf = vbt.Portfolio.from_signals(
    price,           # pd.Series or pd.DataFrame of close prices
    entries,         # boolean array/Series: buy signals
    exits,           # boolean array/Series: sell signals
    size=1,
    direction="longonly",   # or "both" for long+short
    fees=0.001,             # 0.1% per trade
    slippage=0.0005,
    init_cash=10_000.0,
    freq="1D",              # REQUIRED for annualized metrics
    seed=42,
)

# Fitness objectives — all available as direct attributes
sharpe  = pf.sharpe_ratio()     # annualized, requires freq
maxdd   = pf.max_drawdown()     # peak-to-trough fraction
returns = pf.total_return()

# Comprehensive stats dict for logging
stats = pf.stats()
```

**Key 1.0.0 changes from 0.x:**

- Optional **Rust engine** (`pip install "vectorbt[rust]"`) as an alternative to Numba for the hottest paths (indicators, portfolio simulation). Auto-dispatch: `engine="auto"` uses Rust when available, falls back to Numba.
- `use_numba=True` is still the default; Rust is additive.
- `from_signals` signature is **stable and unchanged** between 0.x and 1.0.0 for core parameters — the `group_by`, `cash_sharing`, `direction`, `upon_opposite_entry` parameters all exist in 1.0.0 as documented.
- `Portfolio.stats()` is the standard stats builder; custom metrics use `pf.stats(metrics=[...])`.
- `pf.sharpe_ratio()` requires `freq` to be set (either at construction time or passed as kwarg).

**Confidence:** HIGH for core `from_signals` API. MEDIUM for 1.0.0 vs 0.x delta (GitHub release notes not detailed; PyPI confirms April 2026 release; Rust engine addition is the primary documented change).

---

### Data & Numerics

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| pandas | >=3.0.0 | DataFrame operations, time series | Project pinned. CoW semantics are a hard requirement in 3.0 (released Jan 2026). |
| numpy | >=2.0.0 | Array operations, GP node evaluation | Project pinned. Numba 0.61.x supports NumPy 2.1/2.2 (confirmed binary compatible). |
| pyarrow | >=18.0 | Parquet I/O — default engine for `pd.read_parquet` | Default engine in pandas 3.0. Faster than fastparquet, supports column pushdown for large parquet datasets. |
| numba | >=0.61.2 | JIT acceleration for vectorbt hot paths | 0.61.2 (April 2025) explicitly adds NumPy 2.2 support. 0.61.0 adds NumPy 2.1. Pin to >=0.61.2 to be safe with NumPy 2.x. |
| scipy | >=1.13 | Statistical tests, rolling correlations, optimization utilities | Standard quant research toolkit; stable long-term. |

**Pandas 3.0 mandatory idioms for this project:**

```python
# CORRECT — Copy-on-Write compliant
arr = df["col"].to_numpy()          # not .values
subset = df.loc[mask].copy()        # explicit copy when needed
df.loc[mask, "col"] = value         # no chained assignment

# WRONG — breaks silently or raises in pandas 3.0
arr = df["col"].values              # deprecated, use .to_numpy()
df["foo"][df["bar"] > 5] = 100      # chained assignment — dead
```

**Why NOT polars here:** vectorbt's internal engine, numba JITs, and DEAP all operate on numpy arrays and pandas DataFrames. Polars is 10–50x faster on ETL workloads, but the GP evaluation loop's bottleneck is the backtesting simulation (Numba/Rust), not DataFrame manipulation. Introducing Polars would require constant `.to_pandas()` conversion at the boundary. Keep pandas 3.0 throughout; accept the performance trade-off.

**Confidence:** HIGH for pandas/numpy/pyarrow. HIGH for numba >=0.61.2 (official release notes verified).

---

### Parallel Evaluation

| Library | Version | Pattern | Why |
|---------|---------|---------|-----|
| multiprocessing (stdlib) | Python 3.11 stdlib | `Pool.map` → `toolbox.map` | Zero-dependency, DEAP-native pattern. Single line change to parallelize fitness evaluation across CPU cores. Best for populations of 100–500 where per-individual evaluation > ~10ms. |
| joblib | >=1.4 | `Parallel(n_jobs=-1)(delayed(evaluate)(ind) for ind in pop)` | Drop-in when you need finer control (backend switching, memory mapping, progress). Use `backend="loky"` (default). Do NOT use when evaluation returns large DataFrames — serialization overhead dominates. |

**Recommended pattern for GP fitness evaluation:**

```python
import multiprocessing

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=None)  # uses os.cpu_count()
    toolbox.register("map", pool.map)
    # ... evolutionary loop ...
    pool.close()
    pool.join()
```

**Why NOT Ray for v1:**
- Ray adds cluster infrastructure complexity (actor system, object store) that is not justified for single-workstation runs of 100–500 individuals.
- Ray's serialization of DEAP `Individual` objects (which carry fitness objects and tree structures) can cause pickling issues with `creator`-registered types.
- Ray shines for >10,000 individuals or multi-node execution. Flag for v2 when population sizes scale.

**Why NOT SCOOP:**
- SCOOP is DEAP's documented distributed evaluation framework, but it requires a separate deployment (`python -m scoop script.py`) and is primarily useful for cluster execution. Adds operational overhead not warranted for v1.

**Serialization warning with multiprocessing:** DEAP `creator` classes (e.g., `creator.Individual`, `creator.FitnessMulti`) must be defined at module level (not inside functions) for Python's `pickle` to serialize them correctly across process boundaries. This is a common silent failure.

**Confidence:** HIGH for multiprocessing pattern (DEAP official docs). MEDIUM for joblib (documented, but loky serialization overhead with large numpy payloads is a real risk — measure before committing).

---

### Visualization & Analysis

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| matplotlib | >=3.9 | GP tree visualization, fitness convergence plots, Pareto front scatter | Standard; integrates with networkx for tree drawing via `nx.draw`. |
| networkx | >=3.3 | GP tree graph construction from `gp.graph(expr)` | DEAP's canonical tree viz pipeline: `gp.graph()` → NetworkX → matplotlib/graphviz. |
| pygraphviz | >=1.13 | High-quality GP tree rendering (dot layout) | DEAP official example uses pygraphviz for publication-quality tree diagrams. Requires system-level Graphviz install. |
| plotly | >=5.22 | Interactive Pareto front visualization, multi-run comparisons | vectorbt uses plotly for its native charts. Pareto front scatter with hover tooltips is essential for multi-objective result analysis. |
| seaborn | >=0.13 | Distribution plots, correlation heatmaps for feature importance | Quant research standard for statistical visualization. |

**GP-specific visualization patterns:**

```python
# 1. GP tree visualization (requires networkx + matplotlib)
import networkx as nx
import matplotlib.pyplot as plt
from deap import gp

nodes, edges, labels = gp.graph(individual)
g = nx.DiGraph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
pos = nx.nx_agraph.graphviz_layout(g, prog="dot")  # requires pygraphviz
nx.draw(g, pos, labels=labels, with_labels=True, node_size=800)
plt.savefig("tree.png", dpi=150)

# 2. Pareto front scatter (2-objective: Sharpe vs complexity)
import plotly.express as px
fig = px.scatter(
    pareto_df, x="sharpe", y="complexity",
    color="generation", hover_data=["max_drawdown"],
    title="NSGA-II Pareto Front"
)
fig.show()

# 3. Fitness convergence over generations
fig, ax = plt.subplots()
ax.plot(gen_numbers, best_sharpe_per_gen)
ax.set_xlabel("Generation"); ax.set_ylabel("Best Sharpe (in-sample)")
```

**For experiment tracking and reproducibility:**

Use **MLflow** (>=2.14) as the experiment tracker. It logs per-generation metrics (best Sharpe, Pareto size, population diversity), saves evolved tree strings as artifacts, and provides a local UI. No external server needed for solo research. W&B is an alternative but adds account dependency — avoid for an open-source project.

```python
import mlflow

with mlflow.start_run(run_name=f"gen_{gen}"):
    mlflow.log_metric("best_sharpe", best_ind.fitness.values[0])
    mlflow.log_metric("pareto_size", len(pareto_front))
    mlflow.log_param("seed", SEED)
    mlflow.log_text(str(best_ind), "best_individual.txt")
```

**Confidence:** HIGH for matplotlib/networkx/plotly (all stable, widely used). MEDIUM for pygraphviz (requires system Graphviz binary; installation can be painful on some platforms, especially Apple Silicon — test early).

---

### Dev Tooling

| Tool | Version | Purpose | Why |
|------|---------|---------|-----|
| uv | >=0.4 | Package/environment management | Fastest resolver; replaces pip+virtualenv. Handles complex pin graphs (numba/numpy/vectorbt) reliably. |
| pytest | >=8.2 | Unit/integration tests | Standard. Use `pytest-benchmark` for performance regression testing of evaluation loop. |
| jupyter | >=7.0 | Research notebooks | Project requirement. Use `nbstripout` in git hooks to prevent notebook output from bloating the repo. |
| black | >=24.0 | Code formatting | Non-negotiable for community-ready open source. |
| ruff | >=0.5 | Linting | Replaces flake8/isort/pylint in one tool. Fast. |
| mypy | >=1.10 | Static type checking | The strongly-typed GP primitives make type correctness critical. |
| dill | >=0.3.8 | Extended pickle for DEAP serialization | DEAP checkpointing uses pickle; dill handles lambda functions and closures that standard pickle cannot serialize. Required for checkpoint/resume. |

---

## What NOT to Use (and Why)

| Library | Verdict | Reason |
|---------|---------|--------|
| **gplearn** | Do not use | Implements only symbolic regression with a fixed sklearn-compatible API. Does not support strongly-typed GP, vector/window inputs, or multi-objective fitness. Last major release 0.4.3; actively maintained for sklearn regression only. Cannot serve as a GP engine for this project. |
| **PyGAD** | Do not use | Genetic Algorithm framework, not GP. No tree representation, no `PrimitiveSet`, no type enforcement. Designed for GA over fixed-length chromosomes, not program synthesis. |
| **EvoGP** | Do not use in v1 | GPU-accelerated tree GP (IEEE TEVC 2025, 304x speedup). Requires CUDA + PyTorch environment. Architecture forces tensorized fixed-shape tree representations that conflict with variable-length DEAP trees. Promising for v2 with GPU hardware. |
| **Backtrader** | Do not use | Event-driven backtester — iterates one bar at a time in Python. Cannot evaluate 100–500 GP individuals at vectorized speed. Order-of-magnitude slower than vectorbt for large-population GP fitness evaluation. |
| **Zipline/Zipline-Reloaded** | Do not use | Event-driven, single-strategy focus. Poor pandas 3.0 compatibility. Community maintenance stalled. |
| **SCOOP** | Defer to v2 | Excellent for cluster-scale parallelism but adds deployment overhead. `multiprocessing.Pool` is sufficient for single-workstation populations of 100–500. |
| **Ray** | Defer to v2 | Justified only for >10,000 individuals or multi-node execution. DEAP creator class pickling with Ray requires extra setup. |
| **polars** | Do not use | vectorbt, numba, and DEAP all consume pandas/numpy; polars-at-boundary requires constant `.to_pandas()` conversion. Bottleneck is GP tree evaluation and Numba simulation, not DataFrame manipulation. Re-evaluate if data preprocessing becomes a bottleneck. |
| **Optuna** | Optional for GP hyperparameters only | Optuna has NSGA-II sampler for hyperparameter search, but this project uses DEAP's own NSGA-II for the evolutionary loop. Optuna would be useful only for tuning GP hyperparameters (population size, crossover rate, etc.) — not for running the GP itself. |
| **fastparquet** | Do not use | Slower than pyarrow for most read patterns; pyarrow is the pandas 3.0 default engine. No reason to prefer it. |

---

## Version Compatibility Landmines

### 1. numba / numpy 2.x — MOST CRITICAL

**The problem:** numba has historically lagged NumPy major versions. NumPy 2.0 broke many internal APIs.

**Current status (verified):**
- numba 0.61.0 (Jan 2025): Added NumPy 2.1 support
- numba 0.61.2 (Apr 2025): Added NumPy 2.2 support, fixed regressions
- numba is binary compatible with both NumPy 1.x and 2.x as of 0.61.x

**Required pin:** `numba>=0.61.2` when `numpy>=2.0` is in the environment.

**Smoke test — run this before writing any backtesting code:**
```python
import numpy as np
import numba
import vectorbt as vbt

print(f"numpy: {np.__version__}")       # expect 2.x
print(f"numba: {numba.__version__}")    # expect >=0.61.2

# Force JIT compilation
import numpy as np
@numba.njit
def _smoke(x): return np.sum(x)
arr = np.ones(100, dtype=np.float64)
result = _smoke(arr)
assert result == 100.0, "numba/numpy JIT broken"

# vectorbt smoke test
pf = vbt.Portfolio.from_signals(
    pd.Series([1.0, 2.0, 3.0, 2.0, 1.0]),
    pd.Series([True, False, False, False, False]),
    pd.Series([False, False, True, False, False]),
    freq="1D"
)
assert pf.sharpe_ratio() is not None
print("All smoke tests passed")
```

### 2. pandas 3.0 Copy-on-Write (CoW)

**The problem:** pandas 3.0 (Jan 2026) makes CoW the default. Code written for pandas 2.x that uses chained assignment silently fails or raises.

**Hard rules:**
- Never use `.values` — use `.to_numpy()` instead.
- Never use chained assignment (`df["a"][mask] = x`) — use `.loc` instead.
- Defensive `.copy()` calls that were added to silence `SettingWithCopyWarning` are now harmless but unnecessary; they were needed in 2.x, not 3.x.
- `dtype == "object"` checks for string columns are now wrong — use `pd.api.types.is_string_dtype()`.

**Migration path:** Run code on pandas 2.3 first (released Jun 2025) — it emits deprecation warnings for all patterns that will silently break in 3.0.

### 3. vectorbt 1.0 `from_signals` — freq parameter

**The problem:** Sharpe ratio and annualized metrics require `freq` to be set. If omitted, `pf.sharpe_ratio()` returns `NaN` silently.

**Rule:** Always pass `freq` at construction time or verify it is set:
```python
pf = vbt.Portfolio.from_signals(..., freq="1D")   # daily data
pf = vbt.Portfolio.from_signals(..., freq="1H")   # hourly data
# Or after creation:
pf = pf.replace(freq="1D")
```

### 4. DEAP creator — multiprocessing pickling

**The problem:** `creator.create("Individual", ...)` and `creator.create("FitnessMulti", ...)` must be called at module level (top of the file, not inside `if __name__ == "__main__":` or inside functions). Python's `multiprocessing` serializes workers via pickle; if creator classes are not importable at the top level, worker processes raise `AttributeError`.

**Rule:** Put all `creator.create()` calls in a dedicated module (e.g., `gp_types.py`) and import it everywhere. Do not create types inside functions.

### 5. vectorbt Rust engine — optional but must be explicit

**The problem:** `pip install vectorbt` installs the Numba-only version. The Rust engine is opt-in:
```bash
pip install "vectorbt[rust]"
```
If you check for Rust availability in code, use:
```python
import vectorbt as vbt
has_rust = vbt.settings.get("engine") == "rust" or vbt._has_rust_engine
```
Do not assume Rust is available in CI or on collaborators' machines. Always gate Rust-specific code paths.

### 6. pygraphviz — system dependency

**The problem:** `pip install pygraphviz` requires the Graphviz C library installed at system level. On macOS Apple Silicon:
```bash
brew install graphviz
pip install pygraphviz --global-option=build_ext \
    --global-option="-I$(brew --prefix graphviz)/include/" \
    --global-option="-L$(brew --prefix graphviz)/lib/"
```
On Linux: `apt install graphviz libgraphviz-dev`. This will fail in CI without the system dependency. Make pygraphviz optional in the codebase; fall back to networkx + matplotlib for tree rendering when it is absent.

### 7. dill — DEAP checkpoint/resume

DEAP's checkpoint pattern uses `pickle`. Standard `pickle` cannot serialize lambda functions used as primitives or closures used as fitness functions. Use `dill` as a drop-in replacement:
```python
import dill as pickle   # not import pickle
with open("checkpoint.pkl", "wb") as f:
    pickle.dump({"population": pop, "logbook": log, "gen": gen}, f)
```

---

## Confidence Notes

| Area | Confidence | Evidence Basis |
|------|------------|----------------|
| DEAP 1.4.4 as GP framework | HIGH | PyPI confirms 1.4.4 (Apr 2026); DEAP docs confirm PrimitiveSetTyped, selNSGA2, Pool.map pattern |
| vectorbt 1.0.0 API stability | HIGH for from_signals core | vectorbt.dev API docs + PyPI 1.0.0 release (Apr 2026); Rust engine addition confirmed |
| vectorbt 1.0.0 breaking changes from 0.x | MEDIUM | No official migration guide found; core from_signals signature appears stable; Rust engine is additive |
| numba >=0.61.2 for numpy 2.x | HIGH | numba release notes 0.61.0 and 0.61.2 explicitly state NumPy 2.1/2.2 support |
| pandas 3.0 CoW semantics | HIGH | pandas official release notes (Jan 2026), multiple verified sources |
| Parallel evaluation via multiprocessing | HIGH | DEAP official docs, confirmed working pattern |
| Ray as v2 parallel option | MEDIUM | General Ray docs; DEAP+Ray integration not officially documented |
| pygraphviz system dependency issues | HIGH | Widely reported; official DEAP examples acknowledge it |
| dill for DEAP checkpointing | HIGH | DEAP community standard; pickle limitations with closures well-documented |
| polars exclusion rationale | MEDIUM | Performance argument is sound; boundary conversion cost is real but not benchmarked for this specific workload |

---

## Sources

- DEAP GitHub + ReadTheDocs: https://github.com/DEAP/deap / https://deap.readthedocs.io/en/master/
- DEAP PyPI (version history): https://pypi.org/project/deap/
- vectorbt PyPI: https://pypi.org/project/vectorbt/
- vectorbt.dev API docs: https://vectorbt.dev/api/portfolio/base/
- vectorbt features: https://vectorbt.dev/getting-started/features/
- numba 0.61.0 release notes: https://numba.readthedocs.io/en/stable/release/0.61.0-notes.html
- numba 0.61.2 release notes: https://numba.readthedocs.io/en/stable/release/0.61.2-notes.html
- pandas 3.0 what's new: https://pandas.pydata.org/docs/whatsnew/v3.0.0.html
- Vectorial GP trading paper (arxiv 2504.05418): https://arxiv.org/abs/2504.05418
- EvoGP GPU paper (arxiv 2501.17168): https://arxiv.org/abs/2501.17168
- MLflow: https://mlflow.org/
