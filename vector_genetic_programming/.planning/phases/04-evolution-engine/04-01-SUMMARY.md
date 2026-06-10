---
plan: "04-01"
phase: "04-evolution-engine"
status: complete
completed: "2026-06-10"
duration: ~5min
tasks_total: 2
tasks_completed: 2
---

# Plan 04-01: Conditional Primitives + EvolutionConfig + Tracker

## What Was Built

Added conditional/comparison primitives to the GP layer and created the two foundational evolution modules that Plan 02 depends on.

**Task 1 — Conditional primitives (GP-04, D-10):**
- `gt(a, b)` — element-wise greater-than → 0.0/1.0 float32 array
- `lt(a, b)` — element-wise less-than → 0.0/1.0 float32 array
- `if_then_else(cond, t, f)` — selects t where cond > 0, else f
- All three are module-level functions (pickle-safe for multiprocessing.Pool)
- Registered in `build_pset()` with `[Vector, Vector] -> Vector` type signatures
- GP-04 "deferred per D-07" note removed from docstring

**Task 2 — Evolution infrastructure (EVO-01, EVO-03, EXP-01/02/03):**
- `EvolutionConfig` dataclass: all D-13 NSGA-II hyperparameters with defaults (pop_size=100, n_generations=10, cxpb=0.7, mutpb=0.2, seed=42, tree_height_limit=8, n_jobs=cpu-1)
- `NoOpTracker`: silent no-op tracker, zero dependencies
- `MLflowTracker`: real tracker with deferred `import mlflow` inside `__init__` (not at module level)
- `make_tracker(use_mlflow=False)` factory for duck-typed dispatch
- `pyproject.toml` tracking extra updated to `mlflow>=3.0`

## Key Files

- `vgp/gp/primitives.py` — +24 lines: gt, lt, if_then_else functions
- `vgp/gp/gp_types.py` — +7 lines: import + registration of 3 new primitives
- `vgp/evolution/config.py` — new file: EvolutionConfig dataclass (40 lines)
- `vgp/evolution/tracker.py` — new file: NoOpTracker + MLflowTracker + make_tracker (79 lines)
- `pyproject.toml` — tracking extra mlflow>=2.14 → mlflow>=3.0

## Self-Check: PASSED

- `gt`, `lt`, `if_then_else` in `pset.primitives[Vector]` ✓
- All conditional functions return float32 arrays ✓
- No `import mlflow` at module level in tracker.py ✓
- No vectorbt imported by any modified module ✓
- `EvolutionConfig()` defaults match D-13 spec ✓
- 2 commits: `feat(04-01): add gt, lt, if_then_else` + `feat(04-01): add EvolutionConfig, trackers` ✓
