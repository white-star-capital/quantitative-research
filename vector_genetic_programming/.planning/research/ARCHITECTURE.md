# Architecture Research: VGP for Crypto Trading

**Researched:** 2026-06-03
**Confidence:** HIGH (primary source: arxiv:2504.05418 VGP paper + DEAP official docs + vectorbt docs)

---

## Component Map

Seven distinct components with hard boundaries. Each has a single owner and a typed interface
to its neighbors. Violations of these boundaries are the primary cause of "big ball of mud"
in research GP systems.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                                 │
│  ┌──────────────────────┐  ┌────────────────────────────────────────────┐  │
│  │  DataLoader          │→ │  FeatureEngine                             │  │
│  │  (Parquet → pandas)  │  │  (OHLCV + indicators → aligned DataFrame) │  │
│  └──────────────────────┘  └───────────────────────┬────────────────────┘  │
└──────────────────────────────────────────────────── │ ────────────────────-┘
                                                       │ np.ndarray[T × F × A]
                                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  GP ENGINE                                                                  │
│  ┌──────────────────────┐  ┌─────────────────────────────────────────────┐ │
│  │  PrimitiveRegistry   │→ │  EvolutionLoop                              │ │
│  │  (typed primitives,  │  │  (DEAP toolbox, NSGA-II, pop management,   │ │
│  │   terminals, ephem.) │  │   selection, crossover, mutation)           │ │
│  └──────────────────────┘  └──────────────────┬──────────────────────────┘ │
└─────────────────────────────────────────────── │ ───────────────────────────┘
                                                  │ List[Individual] (GP trees)
                                                  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  EVALUATION LAYER                                                           │
│  ┌──────────────────────┐  ┌─────────────────────────────────────────────┐ │
│  │  TreeEvaluator       │→ │  BacktestRunner                             │ │
│  │  (compile, execute   │  │  (vectorbt Portfolio.from_signals,          │ │
│  │   tree over data,    │  │   Sharpe, drawdown, return calculation)     │ │
│  │   emit signal arrays)│  └──────────────────────────────────────────── │ │
│  └──────────────────────┘                                                 │ │
└─────────────────────────────────────────────────────────────────────────── │ ┘
                                                  │ FitnessVector (NSGA-II tuple)
                                                  ↑
                                    returned to EvolutionLoop
┌─────────────────────────────────────────────────────────────────────────────┐
│  INFRASTRUCTURE LAYER                                                       │
│  ┌──────────────────────┐  ┌─────────────────────────────────────────────┐ │
│  │  ExperimentConfig    │  │  RunTracker                                 │ │
│  │  (Hydra / dataclass, │  │  (MLflow run, generation logbook, HoF,     │ │
│  │   seeds, HP space)   │  │   pickle checkpoint, artifact logging)      │ │
│  └──────────────────────┘  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Owns | Does NOT Own |
|-----------|------|--------------|
| DataLoader | Reading parquet, schema validation, asset alignment | Feature computation |
| FeatureEngine | Technical indicator computation, normalization, rolling windows | Data loading, tree evaluation |
| PrimitiveRegistry | Typed primitive definitions, terminal set, ephemeral constants | Evolution strategy, data |
| EvolutionLoop | NSGA-II execution, selection, genetic operators, population state | Fitness computation |
| TreeEvaluator | Tree compilation, signal array generation, signal interpretation | Backtesting, metrics |
| BacktestRunner | vectorbt execution, fitness metric extraction | Signal generation, logging |
| RunTracker | Checkpointing, logging, experiment metadata | Any GP or data logic |

The critical rule: **EvolutionLoop must not import vectorbt. BacktestRunner must not import DEAP.**
This boundary enforces replaceability — you can swap the backtesting engine without touching GP code.

---

## Data Flow

### Full Pipeline

```
Parquet files (on disk)
    │
    ↓ DataLoader.load(assets, date_range)
    │
pandas DataFrame [T × (F×A)]          ← aligned multi-asset OHLCV
    │
    ↓ FeatureEngine.transform(df)
    │
np.ndarray [T × F × A], dtype=float32  ← all features, all assets, windowed
    │                                      shape example: [2000 × 13 × 8]
    ↓ (sliced by WalkForwardSplitter)
    │
train_data, test_data (np.ndarray views, not copies)
    │
    ↓ passed once to EvaluationContext at run start (shared read-only)
    │
    ┌── EvolutionLoop (per generation)
    │       │
    │       ↓ for each Individual in population
    │           │
    │           ↓ TreeEvaluator.compile(individual) → Callable
    │           ↓ TreeEvaluator.execute(callable, train_data) → signal_array [T × A]
    │           ↓ SignalInterpreter.to_entries_exits(signal_array) → bool[T×A], bool[T×A]
    │           ↓ BacktestRunner.run(entries, exits, prices) → FitnessVector
    │           ↓ individual.fitness.values = FitnessVector
    │
    ↓ (after N generations)
    │
Hall of Fame individuals → selected by Pareto rank
    │
    ↓ final evaluation on test_data (OOS)
    │
OOS FitnessVector + strategy metadata → RunTracker.log_result()
```

### Performance Bottlenecks (in order of severity)

1. **vectorbt JIT warmup on first call** — Numba compiles `Portfolio.from_signals` on the first
   invocation. This is a one-time ~30-60 second cost per process. Mitigation: run a dummy
   backtest as a warmup probe before evolution starts. Source: vectorbt/discussions/447.

2. **Tree compilation overhead per individual** — `deap.gp.compile()` re-evaluates the
   expression string for every tree on every fitness call. Mitigation: cache compiled callables
   keyed by tree hash; invalidate on genetic operator modification.

3. **Pickle serialization in multiprocessing.Pool** — When using `pool.map(evaluate, population)`,
   each individual and the feature data are pickled for each worker call. Mitigation: use
   `multiprocessing.shared_memory` to place feature arrays in shared memory once; workers
   attach by name. Only GP tree strings (small) cross the pickle boundary.

4. **Feature recomputation** — If FeatureEngine runs inside the evaluation loop, it repeats
   expensive rolling calculations. Mitigation: compute features once before evolution begins
   and hold in shared memory.

5. **Signal array allocation** — Creating a new `np.ndarray` per tree evaluation per generation
   is GC pressure. Mitigation: pre-allocate a signal buffer and write in-place.

---

## Individual Evaluation Pipeline

### Tree Representation

DEAP `PrimitiveTree` is a depth-first list of nodes. Each node has an `arity` attribute.
The tree `add(mul(x, sin(y)), z)` is stored as `[add, mul, x, sin, y, z]`.

For VGP specifically, use `PrimitiveSetTyped` with two concrete types:

```python
# Type tokens — use plain classes, not strings
class Vector: pass   # np.ndarray of shape (window_size,)
class Scalar: pass   # float

pset = gp.PrimitiveSetTyped("main",
    in_types=[Vector] * n_features,  # one Vector per feature per asset
    ret_type=Scalar                   # final output must be a Scalar
)

# Vector → Vector (element-wise)
pset.addPrimitive(np.add,      [Vector, Vector], Vector, name="vadd")
pset.addPrimitive(np.multiply, [Vector, Vector], Vector, name="vmul")
pset.addPrimitive(np.sin,      [Vector],         Vector, name="vsin")

# Vector → Scalar (aggregation)
pset.addPrimitive(np.mean,     [Vector],         Scalar, name="vmean")
pset.addPrimitive(np.std,      [Vector],         Scalar, name="vstd")

# Scalar → Scalar (standard arithmetic)
pset.addPrimitive(operator.add, [Scalar, Scalar], Scalar, name="sadd")

# Mixed (scalar broadcast — replicate scalar to vector size)
def broadcast_mul(v: Vector, s: Scalar) -> Vector:
    return v * s
pset.addPrimitive(broadcast_mul, [Vector, Scalar], Vector, name="vsmul")
```

Source: arxiv:2504.05418 — "if one of the arguments is a scalar and the other is a
21-dimensional vector, the scalar is replicated 21 times."

### Compilation to Callable

```python
func = gp.compile(expr=individual, pset=pset)
# func is now a Python callable: func(*feature_windows) → float
```

The compiled function takes `n_features` arguments (one per input terminal), each being
a numpy array of shape `(window_size,)` for a single bar on a single asset.

### Vectorized Execution Over a Time Series

The VGP key insight: rather than calling `func(window)` in a Python loop for each bar,
pass 2D arrays (full time series per feature) and let numpy ufuncs vectorize automatically:

```python
# feature_matrix: np.ndarray [T × n_features]
# Transpose so each feature is a row: [n_features × T]
signals = func(*feature_matrix.T)  # returns np.ndarray [T,] if primitives broadcast
```

This works because numpy ufuncs are already vectorized — `np.add(a, b)` works on scalars
or arrays of any shape. Tree-compiled functions inherit this property automatically.

**Caveat:** Not all primitives are numpy ufuncs. Any primitive using a Python loop or
pandas operation kills vectorization. Every primitive in the registry must accept and
return numpy arrays, not Python scalars.

### Signal Interpretation

Output from `func(*features)` is a float array of length T. Convert to entries/exits:

```python
# Threshold-based (VGP style, confirmed by arxiv:2504.05418)
buy_signal  = signal_array >= threshold_buy   # e.g. >= 1.0
sell_signal = signal_array <= threshold_sell  # e.g. <= -1.0

# Boolean-output (STVGP style — direct boolean output)
entries = func(*features)  # already bool array
exits   = ~entries
```

For multi-asset, execute tree independently per asset using same tree, different
feature inputs. This gives a `[T × A]` entries/exits matrix ready for vectorbt.

### Feeding to vectorbt

```python
import vectorbt as vbt

pf = vbt.Portfolio.from_signals(
    close=price_data,   # pd.DataFrame or np.ndarray [T × A]
    entries=entries,    # bool np.ndarray [T × A]
    exits=exits,        # bool np.ndarray [T × A]
    freq="1D",
    init_cash=10_000,
)

# Extract fitness metrics
sharpe   = pf.sharpe_ratio()    # Series per asset or scalar
max_dd   = pf.max_drawdown()
total_ret = pf.total_return()
```

Note: vectorbt 1.0 API differs from 0.x. The argument names and call signatures were
rewritten in the 1.0 release. Target only 1.0 docs. Source: PROJECT.md constraint note.

---

## Parallelism Pattern

### Recommended: multiprocessing.Pool with shared feature data

```
Main process:
  1. Load and compute feature_matrix (np.ndarray, float32)
  2. Place in multiprocessing.shared_memory.SharedMemory block
  3. Create Pool(n_workers)
  4. Register toolbox.map = pool.map
  5. Each worker call receives:
       - Individual (GP tree as DEAP list — small, fast to pickle)
       - SharedMemory block name + shape + dtype (tiny metadata)
     Worker reconstructs numpy view from shared block (zero-copy)
  6. Worker runs compile → execute → backtest → return FitnessVector
  7. Pool collects fitness tuples, main process assigns to individuals
```

### Why not SCOOP for v1

SCOOP is the DEAP-recommended distributed option. It is appropriate for cluster/grid
environments. For a single workstation, `multiprocessing.Pool` has lower overhead and
no external daemon dependency. SCOOP becomes relevant when you want to run evolution
across multiple machines. Defer to a later phase.

### Worker count guidance

```python
import os
n_workers = os.cpu_count() - 1   # leave 1 core for main process + vectorbt JIT
```

With 100-500 individuals, a 10-core machine evaluates a generation in roughly:
- Single-threaded: 100 × 50ms/eval = 5 seconds/generation
- 9 workers: ~0.6 seconds/generation + ~5s JIT warmup on first call per worker

JIT warmup is per-worker-process. With 9 workers, expect ~270 seconds one-time cost
on first generation. Run warmup probes in all workers before starting evolution.

### Lambda / closure pickling constraint

DEAP's official documentation states: "Lambda functions cannot be pickled in any Python
version." Every primitive function in the registry must be a named module-level function
or a `functools.partial` of a named function. Closures that capture local variables
also cannot be pickled. This is a hard constraint, not a style preference.

---

## Experiment Management

### Run Config (Hydra recommended, dataclass acceptable)

```yaml
# config/run.yaml
experiment:
  name: "vgp_btc_eth_v1"
  seed: 42
  seeds: [42, 137, 271]    # for multi-seed reproducibility

evolution:
  population_size: 200
  n_generations: 100
  crossover_prob: 0.8
  mutation_prob: 0.2
  tree_max_depth: 7
  objectives: [sharpe, max_drawdown, tree_size]  # NSGA-II fitness tuple

data:
  assets: ["BTC", "ETH"]
  window_size: 21
  train_start: "2019-01-01"
  train_end:   "2022-12-31"
  test_start:  "2023-01-01"
  test_end:    "2024-12-31"

backtester:
  init_cash: 10_000
  freq: "1D"
  threshold_buy:  1.0
  threshold_sell: -1.0
```

### Checkpointing (DEAP native pattern)

DEAP's documented pattern saves a pickle dict every N generations containing:
- `population`: current List[Individual] with fitness values
- `generation`: int (resume from here)
- `halloffame`: tools.HallOfFame object
- `logbook`: tools.Logbook (generation statistics history)
- `rndstate`: random.getstate() (deterministic resumption)

```python
import pickle, random

# Save
checkpoint = dict(
    population=pop,
    generation=gen,
    halloffame=hof,
    logbook=logbook,
    rndstate=random.getstate(),
)
with open(f"checkpoint_gen{gen:04d}.pkl", "wb") as f:
    pickle.dump(checkpoint, f)

# Load and resume
with open("checkpoint_gen0050.pkl", "rb") as f:
    cp = pickle.load(f)
pop, start_gen, hof, logbook = cp["population"], cp["generation"], cp["halloffame"], cp["logbook"]
random.setstate(cp["rndstate"])
```

Source: deap.readthedocs.io/en/master/tutorials/advanced/checkpoint.html

### Experiment Tracking (MLflow)

MLflow is the recommended tracker for this project because:
1. Local-first — no external service required, runs on workstation
2. Artifact store handles pickle files naturally
3. Parent/child run structure maps cleanly to "experiment → seed run"
4. Compare runs across hyperparameter configs via MLflow UI

```python
import mlflow

with mlflow.start_run(run_name=config.experiment.name) as parent_run:
    mlflow.log_params(config_flat_dict)

    for seed in config.experiment.seeds:
        with mlflow.start_run(run_name=f"seed_{seed}", nested=True):
            # run evolution
            mlflow.log_metric("oos_sharpe", oos_sharpe, step=0)
            mlflow.log_metric("oos_max_drawdown", oos_dd, step=0)
            mlflow.log_artifact("checkpoint_final.pkl")
            mlflow.log_artifact("best_strategy_expression.txt")

            # per-generation metrics logged inside evolution loop
            for gen, stats in enumerate(logbook):
                mlflow.log_metric("train_sharpe_max", stats["max_sharpe"], step=gen)
                mlflow.log_metric("population_avg_depth", stats["avg_depth"], step=gen)
```

### Hall of Fame Strategy

Use `tools.HallOfFame(n)` to preserve the top `n` Pareto-front individuals across all
generations. For NSGA-II, "best" is ambiguous (multi-objective), so maintain a hall of
all non-dominated individuals at the final generation instead of a strict top-N.

Export each Hall of Fame individual as:
- The DEAP expression string (human-readable)
- The compiled Python source (via `ast.unparse` or `str(individual)`)
- The OOS fitness vector

---

## Build Order (Phase Implications)

### Dependency graph (what blocks what)

```
Phase 1: Data Layer
    DataLoader (parquet → DataFrame)
    FeatureEngine (indicators → np.ndarray)
    WalkForwardSplitter (train/test views)
    ─── Nothing blocks this. Build first. ───

Phase 2: Primitive Registry + Tree Mechanics
    PrimitiveSetTyped (Vector/Scalar types)
    Primitive functions (numpy ufuncs, safe division)
    Tree generation (ramped half-and-half)
    ─── Requires: no data dependency. But define types here. ───
    ─── Blocks: EvolutionLoop, TreeEvaluator ───

Phase 3: TreeEvaluator (the hardest single component)
    compile() → Callable
    execute() over full T dimension
    SignalInterpreter (threshold / boolean)
    ─── Requires: Phase 1 (data shape), Phase 2 (types) ───
    ─── Blocks: BacktestRunner integration, parallelism ───

Phase 4: BacktestRunner + Fitness
    vectorbt 1.0 integration
    FitnessVector extraction (Sharpe, drawdown, tree_size)
    NSGA-II fitness tuple format
    ─── Requires: Phase 3 (signal arrays) ───
    ─── Blocks: EvolutionLoop fitness assignment ───

Phase 5: EvolutionLoop
    DEAP toolbox wiring (NSGA-II, selNSGA2, varOr)
    Population init, generation loop
    Bloat control (static depth limit + NSGA-II size objective)
    ─── Requires: Phases 2, 3, 4 ───
    ─── This is the integration phase — components snap together ───

Phase 6: Parallelism
    Shared memory setup for feature arrays
    Pool wiring via toolbox.register("map", pool.map)
    JIT warmup probes
    ─── Requires: Phase 5 working single-threaded ───
    ─── Do not parallelize before single-threaded is validated ───

Phase 7: Experiment Infrastructure
    Config schema (dataclass or Hydra)
    Checkpointing (pickle pattern)
    MLflow run tracking + artifact logging
    ─── Can be scaffolded in Phase 1 but fully wired in Phase 5+ ───

Phase 8: Walk-Forward Validation + OOS Reporting
    WalkForwardSplitter with multiple windows
    Multi-seed run orchestration
    OOS result aggregation and reporting
    ─── Requires: all preceding phases ───
```

**Critical path:** Data → Primitives → TreeEvaluator → BacktestRunner → EvolutionLoop.

TreeEvaluator is the highest-risk component. It is where the research novelty lives
(vectorized tree execution over multi-asset time series) and where integration bugs
between DEAP's compile() and numpy's broadcasting are most likely to surface.

**Do not parallelize until a single-threaded evolution run completes one generation
end-to-end.** Parallelism adds pickling and shared-memory complexity that obscures
correctness bugs.

---

## Critical Interface Contracts

These are the typed boundaries that must remain stable across phase work.

### Contract 1: FeatureEngine → TreeEvaluator

```python
# FeatureEngine.transform() returns:
FeatureMatrix = np.ndarray   # shape: (T, n_features, n_assets), dtype=float32
# Index 0 is time. Always float32, never float64 (memory + numba compatibility).
# Feature order is fixed and documented in FeatureEngine.FEATURE_NAMES: List[str]
# No NaNs — filled or trimmed at construction time.
```

### Contract 2: TreeEvaluator → BacktestRunner

```python
# TreeEvaluator.execute() returns:
SignalResult = dataclass(
    entries: np.ndarray,   # shape (T, n_assets), dtype=bool
    exits:   np.ndarray,   # shape (T, n_assets), dtype=bool
    prices:  np.ndarray,   # shape (T, n_assets), dtype=float32 (close price passthrough)
)
# Invariant: np.sum(entries & exits) == 0 (no simultaneous entry+exit on same bar/asset)
```

### Contract 3: BacktestRunner → EvolutionLoop (fitness)

```python
# BacktestRunner.evaluate() returns:
FitnessVector = tuple[float, ...]
# For NSGA-II with 3 objectives:
#   (sharpe_ratio, -max_drawdown, -tree_size)
# All values are floats. Signs chosen so that NSGA-II maximizes all objectives.
# tree_size is len(individual) (number of nodes), negated to penalize bloat.
# DEAP NSGA-II requires: individual.fitness.weights = (1.0, 1.0, 1.0) (all maximize)
```

### Contract 4: EvolutionLoop → RunTracker

```python
# At end of each generation, EvolutionLoop calls:
RunTracker.log_generation(
    gen: int,
    logbook_record: dict,      # DEAP Logbook stats for this generation
    halloffame: HallOfFame,    # current HoF (DEAP object)
    population: List[Individual],  # full population for checkpoint
    rndstate: tuple,           # random.getstate() snapshot
)
```

### Contract 5: WalkForwardSplitter → EvolutionLoop

```python
# WalkForwardSplitter.get_fold(i) returns:
WalkForwardFold = dataclass(
    train: FeatureMatrix,   # np.ndarray, read-only view (no copy)
    test:  FeatureMatrix,   # np.ndarray, read-only view (no copy)
    train_dates: pd.DatetimeIndex,
    test_dates:  pd.DatetimeIndex,
    fold_id: int,
)
# Train data is placed in shared memory by the caller (EvolutionLoop orchestrator).
# WalkForwardSplitter only produces views, never copies.
```

### Contract 6: PrimitiveRegistry → TreeEvaluator (compatibility invariant)

All primitives registered in PrimitiveRegistry must satisfy:

1. Accept and return `np.ndarray` or Python `float` — no pandas, no lists
2. Be module-level named functions or `functools.partial` of named functions — no lambdas
3. Handle divide-by-zero and NaN without raising exceptions (return 0.0 or fill)
4. For Vector-typed primitives: work on arrays of shape `(T,)` where T is variable
5. For Scalar-typed: accept `float` and return `float`

Violations of rule 2 cause silent `pool.map` failures (pickle error at worker boundary).
Violations of rule 3 cause NaN poisoning that propagates to fitness and breaks NSGA-II.

---

## Architecture Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| vectorbt 1.0 API underdocumented | HIGH | Write a BacktestRunner smoke test before Phase 4 begins; pin exact call signature from 1.0 docs |
| numba ↔ numpy 2.x incompatibility | HIGH | Run import + JIT warmup probe in Phase 1; fail fast before writing any GP code |
| Primitive pickling failures in Pool | HIGH | Unit test every primitive's picklability before wiring Pool |
| NaN propagation through tree | MEDIUM | Wrap primitives with nan_to_num guard; add fitness validator |
| Tree bloat (code bloat) | MEDIUM | Dual control: static max_depth=7 + tree_size as NSGA-II 3rd objective |
| JIT recompile per-worker-process | MEDIUM | Warmup probe in worker initializer function (Pool initializer arg) |
| Walk-forward data leakage | MEDIUM | Assert test_start > train_end in WalkForwardSplitter constructor |
| Overfitting to in-sample period | HIGH | Walk-forward + multiple seeds + report OOS Sharpe as primary metric |

---

## Sources

- Vectorial GP for trading architecture (arxiv:2504.05418): https://arxiv.org/html/2504.05418v1
- DEAP GP API and PrimitiveSetTyped: https://deap.readthedocs.io/en/stable/api/gp.html
- DEAP parallel evaluation patterns: https://deap.readthedocs.io/en/master/tutorials/basic/part4.html
- DEAP checkpointing: https://deap.readthedocs.io/en/master/tutorials/advanced/checkpoint.html
- vectorbt JIT warmup discussion: https://github.com/polakowo/vectorbt/discussions/447
- vectorbt base portfolio API: https://vectorbt.dev/api/portfolio/base/
- GP bloat control for trading: https://fabian-kostadinov.github.io/2014/11/01/evolving-trading-strategies-with-genetic-programming-gp-parameters-and-operators/
- GP trading overview and parsimony pressure: https://fabian-kostadinov.github.io/2014/09/01/evolving-trading-strategies-with-genetic-programming-an-overview/
- GPU population-level parallelism (EvoGP, 2025): https://arxiv.org/html/2501.17168v3
- Shared memory numpy multiprocessing: https://luis-sena.medium.com/sharing-big-numpy-arrays-across-python-processes-abf0dc2a0ab2
- Walk-forward validation patterns: https://arxiv.org/html/2512.12924v1
- NSGA-II multi-objective GP for trading: https://link.springer.com/article/10.1007/s10462-025-11390-9
