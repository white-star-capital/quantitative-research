---
plan: "04-02"
phase: "04-evolution-engine"
status: complete
completed: "2026-06-10"
duration: ~7min
tasks_total: 2
tasks_completed: 2
---

# Plan 04-02: NSGA-II Evolution Loop + Checkpointing

## What Was Built

**Task 1 — checkpoint.py (EVO-05, D-07/D-09):**
- `save_checkpoint(path, *, population, halloffame, logbook, generation, seed)` — serializes complete evolution state using dill. Creates parent directories. Captures both Python (`random.getstate()`) and numpy (`np.random.get_state()`) RNG states.
- `load_checkpoint(path)` — deserializes checkpoint dict. Docstring documents the 3-step restore protocol (setstate both RNGs, set start_gen = generation+1) callers must follow for reproducible resume.
- Uses `dill` (not pickle) — required for DEAP PrimitiveTree and creator.Individual serialization.

**Task 2 — loop.py (EVO-02, EVO-04, EVO-05, EVO-06, EVO-07):**
- `run_evolution(config, feature_matrix, eval_config, tracker, resume_checkpoint)` — complete NSGA-II GP evolution returning `(population, hof, logbook)`.
- `_jit_warmup()` — module-level Pool initializer that triggers numba JIT compilation once per worker via a dummy vectorbt call (CLAUDE.md #8). vectorbt import is INSIDE this function body only (D-15).
- `_build_toolbox()` — wires NSGA-II operators: genHalfAndHalf init, selNSGA2, cxOnePoint, mutUniform. `gp.staticLimit` applied to BOTH `mate` AND `mutate` enforcing `tree_height_limit=8` (CLAUDE.md #5).
- `_build_stats()` — MultiStatistics with `fitness` chapter (sharpe_max/mean/min) and `size` chapter (size_mean/max).
- `_flatten_record()` — flattens MultiStatistics records for MLflow-compatible flat dict format.
- Loop uses `algorithms.varOr` (not `eaMuPlusLambda`) for per-generation checkpoint hooks.
- `multiprocessing.get_context("spawn")` with `_jit_warmup` as initializer when `n_jobs > 1`.
- `n_jobs=1` bypasses Pool entirely, uses built-in `map`.
- Checkpoint written every `checkpoint_freq` generations.
- Resume restores both RNG states before first operator call (D-09).

## Key Files

- `vgp/evolution/checkpoint.py` — new file: save_checkpoint + load_checkpoint (80 lines)
- `vgp/evolution/loop.py` — new file: run_evolution + helpers (373 lines)

## Self-Check: PASSED

- checkpoint round-trip preserves both Python and numpy RNG states ✓
- No `import vectorbt` at module level in loop.py ✓ (`grep "^import vectorbt" loop.py` returns nothing)
- `gp.staticLimit(` appears exactly twice (mate + mutate) ✓
- `algorithms.varOr` used, `eaMuPlusLambda` only in comments ✓
- `tools.ParetoFront()` used as hof ✓
- `multiprocessing.get_context("spawn")` ✓
- `pool.close()` + `pool.join()` in finally block ✓
- 2 commits: `feat(04-02): checkpoint.py` + `feat(04-02): evolution loop` ✓
