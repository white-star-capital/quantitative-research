# Phase 4: Evolution Engine - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning

<domain>
## Phase Boundary

A complete NSGA-II evolution run executes for multiple generations with parallel evaluation, per-generation metrics logged to MLflow (optional extra), and full reproducibility — given the same seed, two runs produce identical Pareto fronts. This phase delivers: DEAP toolbox wiring, eaMuPlusLambda loop, parallel Pool evaluation with JIT warmup, pickle-based checkpointing, generation-level statistics, and MLflow experiment logging as an optional dependency.

Walk-forward validation and publication artifacts are Phase 5.

</domain>

<decisions>
## Implementation Decisions

### Experiment Tracking
- **D-01:** MLflow is an **optional extra** (`pip install vgp[tracking]`). The pandas<3.0 conflict means mlflow cannot be in the core dependency set. It installs in a separate environment or via the optional extra.
- **D-02:** When mlflow is NOT installed, all tracking calls are **silent no-ops** — the evolution run proceeds without error or warning. Tests for EXP-01/EXP-02/EXP-03 use `@pytest.mark.skipif(not mlflow_available, reason="mlflow not installed")`.
- **D-03:** The tracking layer must be fully decoupled from the evolution loop — a single `tracker` object (duck-typed: real MLflow client or no-op stub) is passed in rather than imported directly in the evolution loop.

### Parallel Evaluation
- **D-04:** `evaluate()` receives `(individual, feature_matrix, config)` but `pool.map` passes only `individual`. Wiring via **`functools.partial`** at toolbox registration: `toolbox.register("evaluate", functools.partial(evaluate, feature_matrix=X, config=cfg))`. DEAP's `eaMuPlusLambda` calls `toolbox.evaluate(ind)` — the partial captures `feature_matrix` and `config` at setup time. Pickle-safe (numpy arrays and dataclasses are pickle-safe).
- **D-05:** JIT warmup via **Pool initializer**: `multiprocessing.Pool(initializer=_jit_warmup)`. `_jit_warmup()` runs a 10-row dummy `Portfolio.from_signals` call inside each worker process before any evaluation begins. The warmup must be in the initializer — not in the main process — because worker processes fork fresh and each needs its own numba compilation (CLAUDE.md #8).
- **D-06:** Worker count via `n_jobs` parameter in `EvolutionConfig`, **default `os.cpu_count() - 1`**. Setting `n_jobs=1` runs single-threaded (no Pool) for debugging. Matches scikit-learn/joblib convention.

### Checkpoints & Reproducibility
- **D-07:** Checkpoint format: **pickle**. Stored as `{run_id}/gen_{N:04d}.pkl`. Each checkpoint contains: `{'population': pop, 'halloffame': hof, 'logbook': log, 'rng_state': random.getstate(), 'np_rng_state': np.random.get_state(), 'generation': gen, 'seed': seed}`.
- **D-08:** Checkpoint frequency: **every N generations**, configurable via `EvolutionConfig(checkpoint_freq=5)`. Default is every 5 generations. Checkpoints written to `checkpoints/{run_id}/` directory.
- **D-09:** Reproducibility (EXP-03) requires capturing and restoring **both** Python `random` state AND numpy rng state. On resume: call `random.setstate()` AND `np.random.set_state()` before continuing the loop. DEAP's genetic operators use Python's `random` module; population initialization may use numpy's rng — missing either causes divergence after resume.

### Primitive Set Expansion
- **D-10:** Phase 4 **adds conditional primitives** to `vgp/gp/primitives.py`:
  - `gt(a, b) -> Vector`: element-wise `(a > b).astype(np.float32)` — returns 0.0/1.0 array
  - `lt(a, b) -> Vector`: element-wise `(a < b).astype(np.float32)`
  - `if_then_else(cond, true_branch, false_branch) -> Vector`: `np.where(cond > 0, true_branch, false_branch)`
  - All registered as `[Vector, Vector] -> Vector` or `[Vector, Vector, Vector] -> Vector`
- **D-11:** All new primitives must be **module-level functions** (not lambdas) for multiprocessing.Pool pickling — same requirement as Phase 3 D-08.
- **D-12:** The type system change (conditionals produce Vector, not bool) is additive — existing Phase 3 primitives are unchanged. `gt`/`lt` return float arrays (0.0/1.0), not Python booleans, so they compose naturally with arithmetic primitives.

### Evolution Loop Configuration
- **D-13:** `EvolutionConfig` dataclass in `vgp/evolution/config.py` holds all NSGA-II hyperparameters with sensible defaults:
  - `pop_size=100`, `n_generations=10` (for validation runs; scale up after first successful run)
  - `cxpb=0.7` (crossover probability), `mutpb=0.2` (mutation probability)
  - `n_jobs=os.cpu_count()-1`, `checkpoint_freq=5`, `seed=42`
  - `hof_size=20` (top-N non-dominated individuals in Hall of Fame)
- **D-14:** Hall-of-fame uses **DEAP's `tools.ParetoFront`** (not `HallOfFame`) — `ParetoFront` tracks all non-dominated individuals across all generations, which aligns with EVO-04's "top-N non-dominated" requirement.

### Architecture
- **D-15:** `vgp/evolution/loop.py` contains `EvolutionLoop` class (or module-level `run_evolution()` function). It must **NOT import vectorbt** — the backtest layer is accessed only via `functools.partial(evaluate, ...)`.
- **D-16:** Tree depth hard-limited to 8 via DEAP's `staticLimit` decorator applied to both crossover and mutation operators from generation 0 (CLAUDE.md #5). This must be enforced before the first generation, not just monitored.

### Claude's Discretion
- Exact DEAP `tools.Statistics` configuration (which stats per generation — mean/max/min Sharpe and mean tree_size are the minimum)
- Logbook formatting for MLflow logging (flatten per-gen dict to individual metrics)
- `run_id` generation strategy (timestamp + seed suffix is standard)
- Whether `EvolutionLoop` is a class or a module-level function
- Pool context manager vs. explicit `.close()` / `.join()` pattern

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Evolution & Requirements
- `.planning/REQUIREMENTS.md` §Evolution Loop — EVO-01 through EVO-07: exact acceptance criteria for toolbox wiring, eaMuPlusLambda, depth limit, HoF, checkpointing, statistics, and parallel evaluation.
- `.planning/REQUIREMENTS.md` §Experiment Tracking — EXP-01 through EXP-03: MLflow hyperparameters, per-generation stats, and seed reproducibility.
- `.planning/ROADMAP.md` §Phase 4 — 5 success criteria to verify at phase end.

### Architecture Constraints (MANDATORY)
- `CLAUDE.md` §Critical Technical Constraints — especially: #5 (tree depth ≤8 via staticLimit), #7 (creator.create() at module level), #8 (vectorbt JIT warmup in worker initializer).
- `CLAUDE.md` §Architecture Invariants — EvolutionLoop must NOT import vectorbt; interface is numpy array in → fitness tuple out.

### Phase 3 Output (interface Phase 4 consumes)
- `vgp/backtest/runner.py` — `evaluate(individual, feature_matrix, config)` signature and `EvalConfig` dataclass. Phase 4 wraps this with `functools.partial`.
- `vgp/gp/gp_types.py` — `creator.Individual`, `creator.FitnessMulti`, `build_pset()`. Phase 4 calls `build_pset()` and extends with conditional primitives.
- `vgp/gp/primitives.py` — Existing primitive functions. Phase 4 adds `gt`, `lt`, `if_then_else` here.
- `.planning/phases/03-gp-core-evaluation/03-CONTEXT.md` — D-09 (creator at module level), D-13 (fitness tuple), D-14 (50-trade filter), D-15 (import boundary), D-16 (no Python loops in evaluation).

### Data Interface
- `.planning/phases/02-data-pipeline/02-CONTEXT.md` — D-10 (train/val/test date ranges: actual cache window 2024-05-01–2025-12-31 for 21 assets).
- `vgp/data/splitter.py` — `WalkForwardSplitter.split()` returns train/val/test slices. Phase 4 evolution runs on the train split.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `vgp/backtest/runner.py` `EvalConfig`: Pattern for config dataclasses. Extend with `EvolutionConfig` in `vgp/evolution/config.py` following same style.
- `vgp/gp/gp_types.py` `build_pset()`: Returns configured `PrimitiveSetTyped`. Phase 4 calls this and then adds conditional primitives to the returned pset before building the toolbox.
- `vgp/gp/primitives.py`: All module-level functions. Add `gt`, `lt`, `if_then_else` following the same module-level pattern.

### Established Patterns
- Module-level DEAP creator definitions (gp_types.py) — must not move into functions.
- `EvalConfig` dataclass pattern with `field(default_factory=...)` for mutable defaults.
- float32 throughout — maintain float32 in new `gt`/`lt`/`if_then_else` outputs via `.astype(np.float32)`.
- All tests under `tests/` — evolution tests go in `tests/test_evolution.py`.

### Integration Points
- Phase 4 creates: `vgp/evolution/loop.py`, `vgp/evolution/config.py`, `vgp/evolution/checkpoint.py`, `vgp/evolution/__init__.py` (public exports).
- Phase 4 extends: `vgp/gp/primitives.py` (add gt, lt, if_then_else), `vgp/gp/gp_types.py` (register new primitives in build_pset), `pyproject.toml` ([tracking] optional extra for mlflow).
- Tests: `tests/test_evolution.py` covering EVO-01 through EVO-07 and EXP-01 through EXP-03.

</code_context>

<specifics>
## Specific Ideas

- The no-op tracker pattern: define `class NoOpTracker` with same methods as the mlflow logging wrapper — this is the duck-typed fallback when mlflow is not installed. The evolution loop receives a `tracker` argument and calls `tracker.log_params(...)`, `tracker.log_metrics(...)` without knowing which implementation it has.
- For the `_jit_warmup()` initializer: use the smallest possible portfolio (10 rows, 1 asset, constant signal) to minimize warmup time. The goal is numba compilation, not a real backtest.
- `staticLimit` must be applied to both `toolbox.mate` (crossover) and `toolbox.mutate` (mutation). Applying to only one allows the other operator to generate oversized trees.

</specifics>

<deferred>
## Deferred Ideas

- Domain-aware primitives (crossover indicator, RSI threshold) — Phase 5 or future PR after first evolution run validates the conditional primitives
- Walk-forward multi-seed runs — Phase 5 scope
- YAML-based experiment configuration (CFG-01) — v2 requirement, out of scope for v1

</deferred>

---

*Phase: 04-evolution-engine*
*Context gathered: 2026-06-09*
