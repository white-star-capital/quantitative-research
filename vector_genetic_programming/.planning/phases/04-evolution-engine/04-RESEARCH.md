# Phase 4: Evolution Engine — Research

**Researched:** 2026-06-09
**Domain:** DEAP 1.4.4 NSGA-II evolution loop, multiprocessing on macOS (spawn), pickle checkpointing, MLflow optional tracking
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** MLflow is an optional extra (`pip install vgp[tracking]`). The pandas<3.0 conflict means mlflow cannot be in the core dependency set.
- **D-02:** When mlflow is NOT installed, all tracking calls are silent no-ops. Tests for EXP-01/EXP-02/EXP-03 use `@pytest.mark.skipif(not mlflow_available, reason="mlflow not installed")`.
- **D-03:** The tracking layer must be fully decoupled — a single `tracker` object (duck-typed: real MLflow client or no-op stub) is passed in. Evolution loop calls `tracker.log_params(...)`, `tracker.log_metrics(...)` without knowing which implementation it has.
- **D-04:** `evaluate()` wired via `functools.partial`: `toolbox.register("evaluate", functools.partial(evaluate, feature_matrix=X, config=cfg))`. DEAP calls `toolbox.evaluate(ind)` — partial captures `feature_matrix` and `config` at setup time.
- **D-05:** JIT warmup via Pool initializer: `multiprocessing.Pool(initializer=_jit_warmup)`. `_jit_warmup()` runs a 10-row dummy `Portfolio.from_signals` in each worker before evaluation begins.
- **D-06:** `n_jobs` parameter in `EvolutionConfig`, default `os.cpu_count()-1`. `n_jobs=1` runs single-threaded (no Pool).
- **D-07:** Checkpoint format: pickle. Stored as `{run_id}/gen_{N:04d}.pkl`. Contents: `{'population', 'halloffame', 'logbook', 'rng_state', 'np_rng_state', 'generation', 'seed'}`.
- **D-08:** Checkpoint frequency: every N generations, configurable via `EvolutionConfig(checkpoint_freq=5)`.
- **D-09:** Reproducibility requires capturing and restoring BOTH Python `random` state AND numpy rng state.
- **D-10:** Phase 4 adds `gt`, `lt`, `if_then_else` to `vgp/gp/primitives.py` as module-level functions.
- **D-11:** All new primitives must be module-level functions (not lambdas) for multiprocessing.Pool pickling.
- **D-12:** The type system change is additive — existing Phase 3 primitives unchanged.
- **D-13:** `EvolutionConfig` dataclass in `vgp/evolution/config.py` with `pop_size=100`, `n_generations=10`, `cxpb=0.7`, `mutpb=0.2`, `n_jobs=os.cpu_count()-1`, `checkpoint_freq=5`, `seed=42`, `hof_size=20`.
- **D-14:** Hall-of-fame uses `tools.ParetoFront` (not `HallOfFame`).
- **D-15:** `vgp/evolution/loop.py` must NOT import vectorbt.
- **D-16:** Tree depth hard-limited to 8 via DEAP's `staticLimit` applied to BOTH `toolbox.mate` AND `toolbox.mutate` from generation 0.

### Claude's Discretion

- Exact DEAP `tools.Statistics` configuration (which stats per generation — mean/max/min Sharpe and mean tree_size are the minimum)
- Logbook formatting for MLflow logging (flatten per-gen dict to individual metrics)
- `run_id` generation strategy (timestamp + seed suffix is standard)
- Whether `EvolutionLoop` is a class or a module-level function
- Pool context manager vs. explicit `.close()` / `.join()` pattern

### Deferred Ideas (OUT OF SCOPE)

- Domain-aware primitives (crossover indicator, RSI threshold) — Phase 5 or future PR
- Walk-forward multi-seed runs — Phase 5 scope
- YAML-based experiment configuration (CFG-01) — v2 requirement
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVO-01 | DEAP toolbox wires selNSGA2, cxOnePoint crossover, mutUniform mutation with configurable probabilities | VERIFIED: all functions in DEAP 1.4.4; exact registration pattern confirmed |
| EVO-02 | eaMuPlusLambda runs a complete evolution loop; population size and generation count are configurable | VERIFIED: exact signature confirmed in installed DEAP 1.4.4; custom loop with varOr required for checkpointing |
| EVO-03 | GP tree depth hard-limited to 8 via DEAP staticLimit decorator | VERIFIED: `toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=8))` pattern confirmed |
| EVO-04 | Hall-of-fame tracks the top-N non-dominated individuals across all generations | VERIFIED: `tools.ParetoFront` confirmed available; `hof.update(offspring)` called per-generation in the custom loop |
| EVO-05 | Generation-level checkpoints written to disk (pickle: population + rng state); resumable from any checkpoint | VERIFIED: Python random and numpy RNG state pickle/restore works; dill already in deps |
| EVO-06 | DEAP Statistics and Logbook capture per-generation metrics (mean/max/min Sharpe, mean tree size) | VERIFIED: `tools.MultiStatistics` pattern confirmed; fields available at `mstats.fields` |
| EVO-07 | Parallel evaluation via multiprocessing.Pool with vectorbt JIT warmup in worker initializer | VERIFIED: spawn context (macOS default) works; Pool(initializer=_jit_warmup) confirmed; module-level functions required |
| EXP-01 | MLflow experiment run logs all hyperparameters | VERIFIED: pyproject.toml `[tracking]` extra already in place; mlflow 3.x still requires pandas<3 confirming D-01 |
| EXP-02 | MLflow logs per-generation statistics from Logbook | VERIFIED: logbook records are dicts; can be flattened to `mlflow.log_metrics(flat_dict, step=gen)` |
| EXP-03 | Experiment is reproducible: given the same seed, two runs produce identical Pareto fronts | VERIFIED: Python random and numpy RNG state serialize and restore identically via pickle |
</phase_requirements>

---

## Summary

Phase 4 wires the Phase 3 `evaluate()` function into a full NSGA-II evolution loop using DEAP 1.4.4. The core challenge is integrating four concerns that have different ownership: the evolution algorithm (DEAP), parallel evaluation (multiprocessing), checkpointing (dill/pickle), and optional experiment tracking (MLflow). Research confirms all four are solvable with established patterns, and the locked decisions in CONTEXT.md are all technically sound.

The most important architectural finding is that `eaMuPlusLambda` cannot be used directly when per-generation checkpointing is required, because it has no callback hook. The correct pattern is to replicate its body using `algorithms.varOr` in a custom outer loop, calling `hof.update(offspring)`, `toolbox.select(pop + offspring, mu)`, and writing checkpoints at the required frequency. This is the same logic as `eaMuPlusLambda` but with full control over the generation boundary.

macOS Python 3.12 defaults to `spawn` for multiprocessing (not `fork`). Spawn re-imports the entire module in each worker, which is why `creator.create()` at module level (already done in `gp_types.py`) and module-level primitive functions are non-negotiable. All DEAP individuals (PrimitiveTree), `functools.partial` wrapping evaluate(), and dill checkpoints are confirmed pickle-safe with spawn. The JIT warmup via `Pool(initializer=_jit_warmup)` is the correct location — not in the main process — because each spawn worker gets a fresh Python interpreter with uncompiled numba JIT cache.

**Primary recommendation:** Implement `run_evolution()` as a module-level function in `vgp/evolution/loop.py` using a manual `varOr`-based loop (not `eaMuPlusLambda` directly) to support per-generation checkpointing. Use `multiprocessing.get_context("spawn")` explicitly rather than relying on the default, so behavior is consistent across platforms.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Toolbox wiring (selNSGA2, cxOnePoint, mutUniform) | Evolution (loop.py) | GP layer (gp_types.py) | Toolbox built in loop.py, calls build_pset() from gp layer |
| Population initialization | Evolution (loop.py) | — | toolbox.population() using genHalfAndHalf |
| Fitness evaluation | Backtest (runner.py) | Evolution via functools.partial | Backtest layer owns evaluate(); evolution wires it via partial |
| Parallel execution | Evolution (loop.py) | OS (spawn Pool) | Pool created in loop.py; workers are OS processes |
| JIT warmup | Worker initializer (_jit_warmup) | — | Must run inside each spawn worker before any evaluation |
| staticLimit enforcement | Evolution (loop.py) | — | toolbox.decorate() called immediately after register() |
| Per-generation statistics | Evolution (loop.py) | — | MultiStatistics.compile() called after select() |
| Hall-of-fame tracking | Evolution (loop.py) | — | hof.update(offspring) called per-generation |
| Checkpointing | Evolution (checkpoint.py) | — | Separate module so loop.py stays clean |
| Experiment tracking | Evolution (tracker.py) | MLflow (optional) | Duck-typed tracker; NoOpTracker is default |
| Conditional primitives (gt, lt, if_then_else) | GP layer (primitives.py) | — | Module-level functions, registered in build_pset() |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| deap | 1.4.4 [VERIFIED: importlib.metadata] | NSGA-II, toolbox, tree operators | Only mature GP framework with NSGA-II + typed GP |
| multiprocessing (stdlib) | Python 3.12 [VERIFIED: .python-version] | Parallel evaluation pool | stdlib; DEAP's documented parallel approach |
| dill | 0.4.1 [VERIFIED: project venv] | Deep pickle for checkpoints (already in core deps) | Handles DEAP PrimitiveTree, closures; already installed |
| numpy | 2.2.6 [VERIFIED: project venv] | RNG state serialization | `np.random.get_state()` / `np.random.set_state()` |
| functools (stdlib) | Python 3.12 | Partial application for evaluate() wiring | DEAP-documented pattern for parallel evaluation |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| mlflow | 3.x (optional extra) [VERIFIED: pip index] | Experiment tracking | Only when `pip install vgp[tracking]`; still requires pandas<3 (separate venv) |
| operator (stdlib) | Python 3.12 | `operator.attrgetter('height')` for staticLimit key | Required by staticLimit decorator |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom varOr loop | `eaMuPlusLambda` directly | eaMuPlusLambda has no checkpoint callback hook; varOr loop is identical logic with full control |
| `tools.MultiStatistics` | Simple `tools.Statistics` | MultiStatistics separates fitness stats from tree size stats cleanly in Logbook chapters |
| `dill` for checkpoints | `pickle` | Both work for DEAP; dill is already in deps and handles edge cases pickle can miss |
| `multiprocessing.get_context("spawn")` | Default `multiprocessing.Pool()` | Explicit spawn context makes macOS/Linux behavior identical and documents the intent |

**Installation:** All core dependencies are already installed. MLflow optional extra is already defined in pyproject.toml.

---

## Architecture Patterns

### System Architecture Diagram

```
EvolutionConfig (config.py)
        |
        v
run_evolution(config, feature_matrix, close_prices, tracker)
  [loop.py — no vectorbt import]
        |
        +---> build_pset() + register conditional primitives  [gp_types.py + primitives.py]
        |
        +---> build_toolbox(pset, evaluate_fn)
        |       |--- toolbox.register("evaluate", partial(evaluate, X, cfg))
        |       |--- toolbox.register("select", selNSGA2)
        |       |--- toolbox.register("mate", cxOnePoint)
        |       |--- toolbox.register("mutate", mutUniform, expr, pset)
        |       |--- toolbox.decorate("mate", staticLimit(height <= 8))
        |       |--- toolbox.decorate("mutate", staticLimit(height <= 8))
        |       |--- toolbox.register("map", pool.map)  [if n_jobs > 1]
        |
        +---> tracker.start_run(...)             [tracker.py — NoOpTracker or MLflowTracker]
        +---> tracker.log_params(config)
        |
        +---> resume OR init population
        |       |--- load_checkpoint(path) if checkpoint exists
        |       |--- toolbox.population(n=pop_size) if fresh start
        |
        +---> initial evaluation (invalid_ind -> toolbox.map(toolbox.evaluate, ...))
        +---> hof.update(population)
        +---> stats.compile(population) -> logbook.record(gen=0, ...)
        |
        +---> for gen in range(start_gen, n_generations + 1):
        |       |--- offspring = varOr(pop, toolbox, lambda_, cxpb, mutpb)
        |       |--- evaluate invalid offspring via toolbox.map
        |       |--- hof.update(offspring)
        |       |--- pop[:] = toolbox.select(pop + offspring, mu)
        |       |--- record = stats.compile(pop)
        |       |--- logbook.record(gen=gen, nevals=..., **record)
        |       |--- tracker.log_metrics(flatten(record), step=gen)
        |       |--- if gen % checkpoint_freq == 0: save_checkpoint(...)
        |
        +---> tracker.end_run()
        +---> return pop, hof, logbook
              |
              v
        caller uses hof (ParetoFront of non-dominated individuals)
        checkpoints/{run_id}/gen_NNNN.pkl on disk
```

### Recommended Project Structure

```
vgp/
├── evolution/
│   ├── __init__.py          # public: run_evolution, EvolutionConfig, load_checkpoint
│   ├── config.py            # EvolutionConfig dataclass
│   ├── loop.py              # run_evolution() — MUST NOT import vectorbt
│   ├── checkpoint.py        # save_checkpoint() / load_checkpoint()
│   └── tracker.py           # NoOpTracker, MLflowTracker (duck-typed)
├── gp/
│   ├── primitives.py        # ADD: gt, lt, if_then_else (module-level)
│   └── gp_types.py          # ADD: register new primitives in build_pset()
tests/
└── test_evolution.py        # EVO-01 through EVO-07, EXP-01 through EXP-03
```

### Pattern 1: staticLimit on Both Operators

**What:** Apply depth limit to BOTH `toolbox.mate` and `toolbox.mutate` immediately after registration. If applied to only one, the other operator can produce oversized trees.

**When to use:** Always, for every GP run.

```python
# Source: https://deap.readthedocs.io/en/master/api/tools.html
import operator
from deap import gp

TREE_HEIGHT_LIMIT = 8  # CLAUDE.md constraint #5

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# CRITICAL: decorate BOTH, not just one
toolbox.decorate(
    "mate",
    gp.staticLimit(key=operator.attrgetter("height"), max_value=TREE_HEIGHT_LIMIT),
)
toolbox.decorate(
    "mutate",
    gp.staticLimit(key=operator.attrgetter("height"), max_value=TREE_HEIGHT_LIMIT),
)
```

### Pattern 2: Custom varOr Loop with Checkpoint Hook

**What:** Replicate `eaMuPlusLambda` body manually to enable per-generation checkpoints. `eaMuPlusLambda` has no callback; this loop is identical logic with full control.

**When to use:** Always when checkpointing is required.

```python
# Source: deap.algorithms.eaMuPlusLambda source (verified in installed DEAP 1.4.4)
from deap import algorithms, tools

for gen in range(start_gen, config.n_generations + 1):
    # varOr: each offspring is product of CX OR mutation (not both)
    # varOr clones individuals and deletes fitness.values on modified ones
    offspring = algorithms.varOr(
        population, toolbox, config.pop_size, config.cxpb, config.mutpb
    )

    # Evaluate only individuals with invalidated fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Update ParetoFront with NEW offspring (not combined pop)
    hof.update(offspring)

    # NSGA-II selection from combined parent + offspring pool
    population[:] = toolbox.select(population + offspring, config.pop_size)

    # Statistics and logging
    record = mstats.compile(population)
    logbook.record(gen=gen, nevals=len(invalid_ind), **record)
    tracker.log_metrics(_flatten_record(record), step=gen)

    # Checkpoint every N generations
    if gen % config.checkpoint_freq == 0:
        save_checkpoint(
            path=f"checkpoints/{run_id}/gen_{gen:04d}.pkl",
            population=population,
            halloffame=hof,
            logbook=logbook,
            generation=gen,
            seed=config.seed,
        )
```

### Pattern 3: Pool with spawn Context and JIT Warmup

**What:** On macOS, `multiprocessing` defaults to `spawn` (Python 3.12 confirmed). Use explicit spawn context and register `pool.map` with the toolbox. The `_jit_warmup` initializer runs `Portfolio.from_signals` on dummy data to trigger numba JIT compilation in each worker before evolution starts.

**When to use:** When `n_jobs > 1`.

```python
# Source: https://deap.readthedocs.io/en/master/tutorials/basic/part4.html
import multiprocessing
import numpy as np
import pandas as pd
import vectorbt as vbt

def _jit_warmup() -> None:
    """Trigger numba JIT compilation in spawn worker before evaluation begins.

    CLAUDE.md constraint #8: warmup MUST run in worker initializer, not main process.
    Each spawn worker gets a fresh interpreter with uncompiled JIT cache.
    10 rows x 1 asset is the minimal call that exercises Portfolio.from_signals.
    """
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    close = pd.DataFrame({"a": np.ones(10) * 100.0}, index=dates)
    entries = pd.DataFrame({"a": np.array([True, False] * 5)}, index=dates)
    exits = pd.DataFrame({"a": np.array([False, True] * 5)}, index=dates)
    vbt.Portfolio.from_signals(
        close=close, entries=entries, exits=exits,
        fees=0.001, freq="1D", init_cash=1000.0,
        group_by=True, cash_sharing=True,
    )


# In run_evolution():
ctx = multiprocessing.get_context("spawn")
pool = ctx.Pool(processes=config.n_jobs, initializer=_jit_warmup)
toolbox.register("map", pool.map)
# ... evolution loop ...
pool.close()
pool.join()
```

### Pattern 4: MultiStatistics Configuration

**What:** Capture Sharpe (max, mean, min) and tree size (mean, max) per generation in separate Logbook chapters.

**When to use:** Always — EVO-06 requirement.

```python
# Source: https://deap.readthedocs.io/en/master/api/tools.html (MultiStatistics)
from deap import tools
import numpy as np
from operator import attrgetter

# Fitness stats: extract first component (Sharpe) from 3-tuple fitness values
fit_stats = tools.Statistics(key=attrgetter("fitness.values"))
fit_stats.register("sharpe_max", lambda vals: float(max(v[0] for v in vals)))
fit_stats.register("sharpe_mean", lambda vals: float(np.mean([v[0] for v in vals])))
fit_stats.register("sharpe_min", lambda vals: float(min(v[0] for v in vals)))

# Tree size stats: len(ind) gives node count
size_stats = tools.Statistics(key=len)
size_stats.register("size_mean", lambda vals: float(np.mean(vals)))
size_stats.register("size_max", float(max))

mstats = tools.MultiStatistics(fitness=fit_stats, size=size_stats)
# mstats.fields == ['fitness', 'size']
# After compile(), record is {'fitness': {'sharpe_max': ..., ...}, 'size': {'size_mean': ...}}
```

### Pattern 5: Checkpoint Save / Load

**What:** Serialize full evolution state including both RNG states for exact resume.

**When to use:** Every `checkpoint_freq` generations and at evolution end.

```python
# Source: Python docs (random.getstate, numpy.random.get_state) — verified in project venv
import dill  # already in core deps (pyproject.toml)
import random
import numpy as np
from pathlib import Path

def save_checkpoint(path: str, *, population, halloffame, logbook, generation: int, seed: int) -> None:
    checkpoint = {
        "population": population,
        "halloffame": halloffame,
        "logbook": logbook,
        "rng_state": random.getstate(),          # Python random module state
        "np_rng_state": np.random.get_state(),   # numpy legacy RNG state
        "generation": generation,
        "seed": seed,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        dill.dump(checkpoint, f)


def load_checkpoint(path: str) -> dict:
    with open(path, "rb") as f:
        return dill.load(f)


# On resume:
ckpt = load_checkpoint(path)
random.setstate(ckpt["rng_state"])
np.random.set_state(ckpt["np_rng_state"])
population = ckpt["population"]
hof = ckpt["halloffame"]
logbook = ckpt["logbook"]
start_gen = ckpt["generation"] + 1
```

### Pattern 6: Duck-Typed Tracker

**What:** NoOpTracker and MLflowTracker implement the same interface. Evolution loop receives a `tracker` argument, never imports mlflow directly.

**When to use:** Always — D-03 requirement.

```python
# Source: CONTEXT.md D-03 decision
class NoOpTracker:
    """Silent no-op tracker for when mlflow is not installed."""
    def start_run(self, run_name: str = "") -> None: pass
    def log_params(self, params: dict) -> None: pass
    def log_metrics(self, metrics: dict, step: int = 0) -> None: pass
    def end_run(self) -> None: pass
    def log_artifact(self, path: str) -> None: pass


class MLflowTracker:
    """Real MLflow tracker. Only importable when mlflow[tracking] extra is installed."""
    def __init__(self, experiment_name: str) -> None:
        import mlflow  # deferred import — fails gracefully if not installed
        self._mlflow = mlflow
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str = "") -> None:
        self._mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int = 0) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def end_run(self) -> None:
        self._mlflow.end_run()

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)


def make_tracker(use_mlflow: bool = False, experiment_name: str = "vgp") -> NoOpTracker:
    if use_mlflow:
        return MLflowTracker(experiment_name)
    return NoOpTracker()
```

### Pattern 7: functools.partial for evaluate() Wiring

**What:** Capture `feature_matrix` and `config` at toolbox registration time. DEAP passes only `individual` to `toolbox.evaluate(ind)`.

**When to use:** Always for DEAP + multiprocessing evaluate() wiring.

```python
# Source: https://deap.readthedocs.io/en/master/tutorials/basic/part4.html
# + CONTEXT.md D-04
import functools
from vgp.backtest.runner import evaluate, EvalConfig

feature_matrix: np.ndarray  # [T x F x A] float32 from Phase 2
config = EvalConfig(close_prices=close_prices_df, fee_bps=10.0)

toolbox.register(
    "evaluate",
    functools.partial(evaluate, feature_matrix=feature_matrix, config=config),
)
# Verified pickle-safe for spawn: functools.partial with numpy array + dataclass pickles OK
```

### Pattern 8: New Conditional Primitives

**What:** Three new module-level primitive functions added to `vgp/gp/primitives.py`. Registered in `build_pset()` in `gp_types.py`. All return float32 arrays.

```python
# Source: CONTEXT.md D-10 + D-11 + D-12
# Must be module-level (not lambda) for spawn pickling — same as existing primitives

def gt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise greater-than. Returns 0.0/1.0 float32 array."""
    return (np.asarray(a, dtype=np.float32) > np.asarray(b, dtype=np.float32)).astype(np.float32)


def lt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise less-than. Returns 0.0/1.0 float32 array."""
    return (np.asarray(a, dtype=np.float32) < np.asarray(b, dtype=np.float32)).astype(np.float32)


def if_then_else(cond: np.ndarray, true_val: np.ndarray, false_val: np.ndarray) -> np.ndarray:
    """Element-wise conditional. Selects true_val where cond > 0, else false_val."""
    c = np.asarray(cond, dtype=np.float32)
    t = np.asarray(true_val, dtype=np.float32)
    f = np.asarray(false_val, dtype=np.float32)
    return np.where(c > 0, t, f).astype(np.float32)


# In gp_types.py build_pset() — after existing primitives:
pset.addPrimitive(gt, [Vector, Vector], Vector, name="gt")
pset.addPrimitive(lt, [Vector, Vector], Vector, name="lt")
pset.addPrimitive(if_then_else, [Vector, Vector, Vector], Vector, name="if_then_else")
```

### Anti-Patterns to Avoid

- **Applying staticLimit to only one operator:** `toolbox.decorate("mate", ...)` without also decorating `mutate` allows mutation to bypass the depth limit. Both must be decorated. [VERIFIED: CLAUDE.md #5, CONTEXT.md D-16]
- **Passing lambdas to multiprocessing.Pool:** Lambdas are not picklable by Python's standard pickle. All functions passed to or registered in the Pool must be module-level. This includes primitive functions, evaluate, and the JIT warmup. [VERIFIED: spawn test confirmed lambda failure]
- **Calling `eaMuPlusLambda` directly when checkpointing:** `eaMuPlusLambda` runs all `ngen` generations without a callback. Checkpoint at generation N is impossible without reimplementing the loop body with `varOr`. [VERIFIED: eaMuPlusLambda source has no hook]
- **Putting `_jit_warmup()` in the main process:** numba JIT compiles per-process. Warming up in main does nothing for spawn workers. The initializer runs once per worker process at Pool creation time, before any `pool.map` calls. [VERIFIED: CLAUDE.md #8]
- **Running Pool with `fork` on macOS:** Python 3.12 on macOS defaults to `spawn`. Do not override to `fork` — spawn is safer for numba's internal state and is the expected behavior. [VERIFIED: `multiprocessing.get_start_method()` returns 'spawn' on macOS Python 3.12]
- **Using `creator.create()` inside functions:** Spawn workers re-import modules from scratch. If `creator.create()` is inside a function, it won't run in workers, causing `AttributeError: Individual has no attribute fitness`. This is already correct in `gp_types.py` but must not be changed. [VERIFIED: module-level guard confirmed in gp_types.py]
- **Missing `if __name__ == "__main__"` guard at script level:** Any script that creates a Pool must guard the top-level code. Not needed inside a function (`run_evolution()`), but required for any `__main__` script calling it. [CITED: DEAP tutorials/basic/part4.html]
- **Saving checkpoints with standard `pickle` instead of `dill`:** Both work for simple cases, but `dill` is already in core deps and handles edge cases (e.g., if ParetoFront contains items with closures). Use `dill` consistently. [ASSUMED — dill confirmed in deps; using it is safer]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Non-dominated selection | Custom Pareto sorting | `tools.selNSGA2` | Implements fast non-dominated sort + crowding distance (Deb 2002); correct dominance semantics for multi-objective |
| Non-dominated HoF tracking | Custom tracking list | `tools.ParetoFront` | Handles incremental dominance checks, removes newly dominated members per-update |
| GP crossover type-safety | Custom typed crossover | `gp.cxOnePoint` | Already handles strongly-typed GP — exchanges subtrees with compatible return types |
| GP mutation | Custom random subtree replace | `gp.mutUniform` | Handles typed GP — generates replacement subtree of correct type |
| Population initialization | Custom tree generator | `gp.genHalfAndHalf` + `tools.initIterate` | Ramped half-and-half is standard for balanced initial depth distribution |
| Statistics aggregation | Manual dict building | `tools.MultiStatistics` + `tools.Logbook` | Automatic chapter-based recording; Logbook supports `stream` property for live printing |
| Depth enforcement | Manual post-generation height check | `gp.staticLimit` decorator | Replaces oversized children at birth with a parent copy; correct Koza semantics |

**Key insight:** DEAP 1.4.4 provides all required evolution primitives. The only reason to write custom code is the checkpoint outer loop (which wraps `varOr`, a DEAP function) and the tracker decoupling (which is project-specific). Everything else is a DEAP registration call.

---

## Common Pitfalls

### Pitfall 1: spawn Pool Fails When Code Runs from stdin / -c

**What goes wrong:** `multiprocessing.Pool` with `spawn` raises `FileNotFoundError: [Errno 2] No such file or directory: '<stdin>'` when the Pool is created in code passed via `python -c "..."` or a heredoc.

**Why it happens:** `spawn` serializes the "main module" path to send to workers. When running from stdin, there is no path. [VERIFIED: reproduced in research]

**How to avoid:** Evolution code always runs from a `.py` file or as an installed module, never from `python -c`. For tests that test the Pool directly, write them as proper `.py` test files.

**Warning signs:** The error appears immediately at `Pool.__init__` or first `pool.map()` call.

### Pitfall 2: ParetoFront Grows Without Bound

**What goes wrong:** `tools.ParetoFront` has no `maxsize` parameter. After many generations with a diverse population, it can contain thousands of individuals, consuming significant memory and slowing `hof.update()` calls.

**Why it happens:** `ParetoFront` stores every non-dominated individual ever seen. With 3 objectives and a large population over many generations, the Pareto front can be large. [CITED: deap.readthedocs.io/en/master/api/tools.html — "can become very large"]

**How to avoid:** For production runs, trim the ParetoFront to the top-N individuals by Sharpe after evolution, rather than capping during evolution (capping would miss valid non-dominated individuals). `EvolutionConfig.hof_size=20` is a post-run filter, not a cap on `ParetoFront`.

**Warning signs:** Memory growth across generations; `hof.update()` latency increasing.

### Pitfall 3: Statistics Lambda Captures Wrong Variable

**What goes wrong:** Using a lambda in `stats.register()` that captures a loop variable captures the reference, not the value. This is the classic Python closure-in-loop bug.

**Why it happens:** `stats.register("sharpe_max", lambda vals: max(v[0] for v in vals))` is safe because `v` is a generator variable, not a captured outer variable. But if registering multiple stats in a loop over a list of indices, lambdas would all capture the last value.

**How to avoid:** Register each stat explicitly with a named function or generator expression, not a loop-captured variable. The patterns above (named lambdas per-stat) are safe.

**Warning signs:** All registered stats return the same value.

### Pitfall 4: Logbook MLflow Flattening Loses Chapter Structure

**What goes wrong:** `logbook[-1]` returns `{'gen': N, 'nevals': M, 'fitness': {'sharpe_max': ...}, 'size': {...}}` with nested dicts. `mlflow.log_metrics()` requires a flat dict with string keys.

**Why it happens:** `tools.MultiStatistics` creates chapter-based nesting in the Logbook record.

**How to avoid:** Flatten the record before passing to `tracker.log_metrics()`:

```python
def _flatten_record(record: dict) -> dict:
    """Flatten MultiStatistics logbook record to {chapter__key: value} dict."""
    flat = {}
    for key, val in record.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                flat[f"{key}__{subkey}"] = float(subval)
        else:
            flat[key] = float(val)
    return flat
```

**Warning signs:** `TypeError: The value must be of type Union[int, float]` from mlflow.

### Pitfall 5: Resume Diverges After One Generation

**What goes wrong:** A resumed run produces different individuals than the continuous run after the first resumed generation.

**Why it happens:** DEAP genetic operators use Python's `random` module. Population initialization may use numpy's RNG. Restoring only one of the two states causes divergence immediately. [VERIFIED: both states tested — Python random restore confirmed True, numpy restore confirmed True]

**How to avoid:** Checkpoint and restore BOTH: `random.setstate()` AND `np.random.set_state()`. Restore BEFORE any call to `toolbox.population()`, `varOr()`, or `toolbox.evaluate()`.

**Warning signs:** Resumed run produces same first generation but diverges from generation 2 onward.

### Pitfall 6: worker_init (JIT warmup) Re-runs Every pool.map Call

**What goes wrong:** Assuming `initializer` runs before each `pool.map` call. It only runs once per worker at Pool creation time.

**Why it happens:** This is actually correct behavior and not a pitfall — the initializer runs once at worker startup, numba compiles once, and all subsequent `pool.map` calls in that worker use the compiled cache.

**Warning signs (of the opposite bug):** JIT warmup happening inside `evaluate()` instead of in the initializer — causes 30-60s delay on the first individual evaluated per worker, every time.

---

## Code Examples

### Complete Toolbox Build (verified against DEAP 1.4.4)

```python
# Source: DEAP 1.4.4 installed — all symbols verified present
import operator
import functools
import random
import numpy as np
from deap import base, gp, tools

from vgp.gp.gp_types import build_pset, creator  # noqa: F401 — side effect: registers creator
from vgp.backtest.runner import evaluate, EvalConfig

def build_toolbox(pset, feature_matrix: np.ndarray, eval_config: EvalConfig) -> base.Toolbox:
    """Build DEAP toolbox for NSGA-II GP evolution."""
    toolbox = base.Toolbox()

    # Population init (ramped half-and-half, max initial depth 4)
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation: partial captures feature_matrix and config; only individual is passed per call
    toolbox.register(
        "evaluate",
        functools.partial(evaluate, feature_matrix=feature_matrix, config=eval_config),
    )

    # NSGA-II operators
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

    # Depth limit — applied to BOTH operators (CLAUDE.md #5, D-16)
    toolbox.decorate(
        "mate",
        gp.staticLimit(key=operator.attrgetter("height"), max_value=8),
    )
    toolbox.decorate(
        "mutate",
        gp.staticLimit(key=operator.attrgetter("height"), max_value=8),
    )

    return toolbox
```

### EvolutionConfig Dataclass

```python
# Source: CONTEXT.md D-13 — verified against EvalConfig pattern in runner.py
import os
from dataclasses import dataclass, field

@dataclass
class EvolutionConfig:
    """NSGA-II hyperparameters and run configuration."""
    pop_size: int = 100
    n_generations: int = 10
    cxpb: float = 0.7
    mutpb: float = 0.2
    n_jobs: int = field(default_factory=lambda: max(1, os.cpu_count() - 1))
    checkpoint_freq: int = 5
    seed: int = 42
    hof_size: int = 20
    tree_height_limit: int = 8    # matches CLAUDE.md constraint #5
    checkpoint_dir: str = "checkpoints"
```

### RNG Seeding Pattern

```python
# Source: Python docs + numpy docs — verified round-trip in project venv
import random
import numpy as np

def seed_all(seed: int) -> None:
    """Seed all RNGs that affect DEAP operator randomness."""
    random.seed(seed)      # DEAP's genetic operators use Python random
    np.random.seed(seed)   # population init may use numpy RNG
```

---

## Runtime State Inventory

This is a greenfield phase (new files in `vgp/evolution/`). No runtime state inventory needed.

The only modifications to existing files are:
- `vgp/gp/primitives.py` — additive (new functions appended)
- `vgp/gp/gp_types.py` — additive (`build_pset()` registers 3 new primitives)
- `pyproject.toml` — `[tracking]` extra already present with `mlflow>=2.14`; update to `mlflow>=3.0` if desired

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `eaMuPlusLambda` direct call | Custom varOr loop wrapping same internals | N/A — design choice for checkpointing | Must reimplement loop body; varOr is a DEAP function |
| `tools.HallOfFame` | `tools.ParetoFront` | N/A — domain choice for multi-objective | ParetoFront tracks all non-dominated, not just top-N by single metric |
| `multiprocessing.Pool()` default | `multiprocessing.get_context("spawn").Pool()` | Python 3.12 macOS default is spawn | Explicit context prevents behavior differences across platforms |
| `pickle` for checkpoints | `dill` | N/A — already in deps | dill is a strict superset of pickle; handles DEAP edge cases |
| mlflow 2.x | mlflow 3.x (latest: 3.13.0) | 2026 | Still requires pandas<3; optional-extra approach unchanged |

**Deprecated/outdated:**
- `multiprocessing.Pool()` without explicit context: On macOS Python 3.12, this is already spawn, but it is best to be explicit. fork is available but unsafe for numba state.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | All | ✓ | 3.12.4 | — |
| DEAP | EVO-01 through EVO-07 | ✓ | 1.4.4 | — |
| numpy | RNG state, conditionals | ✓ | 2.2.6 | — |
| dill | Checkpointing | ✓ | 0.4.1 | — |
| multiprocessing spawn | EVO-07 | ✓ | stdlib | — |
| vectorbt | JIT warmup (in worker) | ✓ | 1.0.0 | — |
| mlflow | EXP-01/02/03 | ✗ (optional) | — | NoOpTracker |

**Missing dependencies with no fallback:** None. All required packages are installed.

**Missing dependencies with fallback:** mlflow — NoOpTracker provides silent no-ops when not installed. Tests guarded with `pytest.mark.skipif`.

---

## Open Questions

1. **run_evolution() as function vs. EvolutionLoop as class**
   - What we know: CONTEXT.md marks this as Claude's discretion
   - What's unclear: Whether stateful (class) or functional (run_evolution fn) is better for test isolation
   - Recommendation: Module-level function `run_evolution()` with explicit parameters. Classes add state management complexity with no benefit when all state is in the checkpoint dict. Matches the existing pattern (`evaluate()` in runner.py is a function, not a class method).

2. **Pool context manager vs. explicit close/join**
   - What we know: Python context manager (`with pool:`) calls `pool.terminate()` on exception and `pool.join()` on exit
   - What's unclear: Whether `terminate()` is preferred over `close()` on exit
   - Recommendation: Explicit `pool.close(); pool.join()` in a try/finally block. `terminate()` kills workers abruptly; `close()` waits for in-flight work to complete, which is correct behavior for a graceful shutdown. Context manager uses `terminate()` on exception, which is fine — use `with pool:` for simplicity since the difference only matters on exception paths.

3. **Statistics: mean/max/min from MultiStatistics vs. simple Statistics**
   - What we know: CONTEXT.md marks statistics config as Claude's discretion; MultiStatistics verified working
   - Recommendation: Use `MultiStatistics` with two `Statistics` objects (fitness and size). This produces a clean Logbook with `{'fitness': {...}, 'size': {...}}` chapters that flatten naturally for MLflow. Avoid `stats.register("size_mean", ...)` inside a fitness Statistics object — key function is called per-individual and can't access both fitness values and tree size in one pass.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | dill handles DEAP PrimitiveTree edge cases better than pickle for checkpoint serialization | Common Pitfalls, Pattern 5 | Low — if pickle works, dill also works (strict superset); dill is already in core deps |
| A2 | MLflow 3.x API (`mlflow.log_params`, `mlflow.log_metrics`, `mlflow.start_run`) is stable and compatible with the NoOpTracker interface | Pattern 6 | Medium — MLflow 3.x changed to experiment-centric API; if API differs, MLflowTracker implementation needs adjustment. Does not affect NoOpTracker or core evolution. |

**All critical claims (DEAP API, spawn behavior, pickle safety, RNG restore) were verified by running code in the project venv.**

---

## Security Domain

Security enforcement is not applicable to this phase. This is a local research computation library with no network services, user input, authentication, or sensitive data handling.

---

## Sources

### Primary (HIGH confidence)

- DEAP 1.4.4 installed in project `.venv` — all API calls verified by running code
- `/websites/deap_readthedocs_io_en_master` via Context7 — `eaMuPlusLambda`, `staticLimit`, `MultiStatistics`, `ParetoFront`, `selNSGA2`, `cxOnePoint`, `mutUniform`, `toolbox.decorate` documentation
- `vgp/backtest/runner.py` — Phase 3 `evaluate()` interface (the function Phase 4 wraps)
- `vgp/gp/gp_types.py` — existing `creator` definitions and `build_pset()` Phase 4 extends
- `pyproject.toml` — confirmed `[tracking]` optional extra already present; dill already in core deps
- `.python-version` — Python 3.12 confirmed
- `multiprocessing.get_start_method()` in project venv — confirmed `spawn` on macOS

### Secondary (MEDIUM confidence)

- CONTEXT.md D-01 through D-16 — user decisions that constrain implementation
- ROADMAP.md Phase 4 — five verifiable success criteria

### Tertiary (LOW confidence — marked [ASSUMED] in Assumptions Log)

- dill vs pickle edge case advantage for DEAP checkpoints (A1)
- MLflow 3.x API stability (A2) — not verified by installing mlflow due to pandas<3 conflict

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages verified in project venv
- Architecture patterns: HIGH — varOr loop, staticLimit, MultiStatistics, RNG state all verified by running code
- Pitfalls: HIGH — spawn stdin limitation verified by reproducing; others verified via source inspection

**Research date:** 2026-06-09
**Valid until:** 2026-09-09 (DEAP 1.4.4 is stable; macOS spawn behavior is OS-level stable)
