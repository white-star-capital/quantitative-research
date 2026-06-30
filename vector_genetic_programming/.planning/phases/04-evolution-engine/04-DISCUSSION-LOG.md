# Phase 4: Evolution Engine - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-09
**Phase:** 04-evolution-engine
**Areas discussed:** Experiment tracking, Parallel eval wiring, Checkpoint format & frequency, Primitive expansion

---

## Experiment Tracking

| Option | Description | Selected |
|--------|-------------|----------|
| MLflow in [tracking] optional extra | Install mlflow only in a separate venv or via pip install vgp[tracking]. Core vgp runs without it. Tests get skip markers. | ✓ |
| JSONL file-based logging only | No mlflow at all. Each run writes hyperparams + per-gen stats to .jsonl. Zero deps. | |
| Weights & Biases (wandb) | pandas-3.x compatible, richer UI, but requires account + internet. | |

**User's choice:** MLflow in [tracking] optional extra
**Notes:** pandas<3 conflict documented in STATE.md. Core evolution unaffected if mlflow absent.

---

## Experiment Tracking — Fallback Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Silent no-op — log calls are no-ops, run proceeds normally | If mlflow not importable, tracking silently skips. No warning, no crash. | ✓ |
| Log to JSONL as fallback | If mlflow not importable, auto-fallback to writing a .jsonl file. | |
| Raise ImportError | If tracking requested but mlflow not installed, raise a clear error. | |

**User's choice:** Silent no-op
**Notes:** Duck-typed tracker pattern (NoOpTracker class) decouples evolution loop from mlflow dependency entirely.

---

## Parallel Eval Wiring — Argument Passing

| Option | Description | Selected |
|--------|-------------|----------|
| functools.partial at toolbox registration | toolbox.register('evaluate', functools.partial(evaluate, feature_matrix=X, config=cfg)). Clean, no globals, pickle-safe. | ✓ |
| Worker globals via Pool initializer | Pool(initializer=init_worker, initargs=(X, cfg)). Stores globals in workers. Implicit state. | |
| Wrapper class with __call__ | class EvalWrapper with __call__. Clean OOP but adds a class boundary. | |

**User's choice:** functools.partial
**Notes:** feature_matrix and config are captured in the partial at setup time. DEAP's eaMuPlusLambda calls toolbox.evaluate(ind) — the partial transparently provides the remaining args.

---

## Parallel Eval Wiring — JIT Warmup

| Option | Description | Selected |
|--------|-------------|----------|
| Warmup in Pool initializer with tiny dummy backtest | Pool(initializer=_jit_warmup). Runs 10-row dummy Portfolio.from_signals in each worker. | ✓ |
| Warmup before Pool creation in main process only | Does NOT warm up workers — they fork fresh and still need to compile. | |

**User's choice:** Pool initializer warmup
**Notes:** CLAUDE.md #8 is explicit — warmup must be in worker initializer, not main process.

---

## Parallel Eval Wiring — Worker Count

| Option | Description | Selected |
|--------|-------------|----------|
| n_jobs param in EvolutionConfig, default=os.cpu_count()-1 | Explicit config, n_jobs=1 for single-threaded debugging. | ✓ |
| Always os.cpu_count()-1, not configurable | Hard-coded. Harder to debug. | |
| n_jobs=-1 means all cores | Risks starving OS threads. | |

**User's choice:** n_jobs param, default os.cpu_count()-1
**Notes:** Matches scikit-learn/joblib convention. n_jobs=1 disables Pool entirely for debugging.

---

## Checkpoint Format

| Option | Description | Selected |
|--------|-------------|----------|
| pickle | pickle.dump({'population', 'halloffame', 'logbook', 'rng_state', 'np_rng_state', 'generation', 'seed'}). DEAP standard, full fidelity. | ✓ |
| JSON metadata + pickle population | Two files per checkpoint. | |
| HDF5 (h5py) | Requires custom serialization for DEAP Individuals. Overhead not justified. | |

**User's choice:** pickle
**Notes:** Must capture both Python random state AND numpy rng state for exact reproduction after resume.

---

## Checkpoint Frequency

| Option | Description | Selected |
|--------|-------------|----------|
| Every N generations (configurable, default=5) to checkpoints/ dir | checkpoint_freq=5 in EvolutionConfig. Low overhead for long runs. | ✓ |
| Every generation, always | Maximum durability but can dominate wall clock for large runs. | |
| Only on KeyboardInterrupt / exception | Loses generations since last checkpoint. Not resumable for planned partial runs. | |

**User's choice:** Every N generations, configurable, default=5

---

## Checkpoint Reproducibility Content

| Option | Description | Selected |
|--------|-------------|----------|
| seed + numpy rng state + Python random state + population | Captures both random sources. Required for exact resume. | ✓ |
| Seed only (re-run from scratch) | Not a true resume — must re-run all prior generations. | |
| Python random state only | Missing numpy rng state causes divergence after resume. | |

**User's choice:** Full state capture (seed + numpy rng + Python random + population + HoF + logbook + gen number)

---

## Primitive Expansion

| Option | Description | Selected |
|--------|-------------|----------|
| Add conditionals in Phase 4 | IfThenElse, gt, lt. Adds regime-switching expressiveness. Type system validated in Phase 3. | ✓ |
| Keep minimal math core only | No new primitives. Evolution engine is the focus. | |
| Add domain-aware primitives instead | crossover indicator, rsi_threshold. More interpretable but tighter domain coupling. | |

**User's choice:** Add conditionals (gt, lt, if_then_else) in Phase 4
**Notes:** Domain-aware primitives deferred to Phase 5 / future PR.

---

## Primitive Types for Conditionals

| Option | Description | Selected |
|--------|-------------|----------|
| gt/lt return Vector (1.0/0.0 array), IfThenElse(Vector, Vector, Vector) -> Vector | Per-timestep comparison. Fully composable with existing math primitives. | ✓ |
| gt/lt return Scalar (scalar 1.0 or 0.0) | Loses per-timestep semantics. Wrong for time-series GP. | |

**User's choice:** Vector (0.0/1.0 array) outputs for all conditional primitives

---

## Claude's Discretion

- Exact DEAP `tools.Statistics` configuration
- Logbook formatting for MLflow logging
- `run_id` generation strategy
- Whether `EvolutionLoop` is a class or module-level function
- Pool context manager vs. explicit .close()/.join() pattern

## Deferred Ideas

- Domain-aware primitives (crossover, rsi_threshold) — Phase 5 or future PR
- Walk-forward multi-seed runs — Phase 5 scope
- YAML experiment configuration (CFG-01) — v2 requirement
