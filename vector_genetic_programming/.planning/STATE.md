---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: complete
stopped_at: Phase 5 complete — milestone v1.0 achieved. All 5 phases done, 14 plans, 57 tests passing.
last_updated: "2026-06-13T00:00:00.000Z"
last_activity: 2026-06-13
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 14
  completed_plans: 14
  percent: 100
---

# Project State: VGP

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-13)

**Core value:** Evolve strategies with demonstrable positive out-of-sample Sharpe ratio
**Current focus:** Milestone v1.0 complete

## Current Position

Phase: 5 of 5 (Validation & Publication) — COMPLETE
Plan: 3 of 3 in final phase — COMPLETE
Status: Milestone v1.0 achieved
Last activity: 2026-06-13

Progress: [████████████] 100%

## Performance Metrics

**Velocity:**

- Total plans completed: 14
- Average duration: ~6 min/plan
- Total execution time: ~84 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 — Foundation & Environment | 2 | ~3 min | 1.5 min |
| 2 — Data Pipeline | 3 | ~22 min | 7 min |
| 3 — GP Core & Evaluation | 3 | ~14 min | 4.7 min |
| 4 — Evolution Engine | 3 | ~18 min | 6 min |
| 5 — Validation & Publication | 3 | ~25 min | 8 min |

*Updated after each plan completion*
| Phase 3 P1 | 4min | 2 tasks | 4 files |
| Phase 3 P2 | 5min | 2 tasks | 5 files |
| Phase 03 P03 | 5min | 2 tasks | 3 files |
| Phase 4 P1 | 6min | 2 tasks | 5 files |
| Phase 4 P2 | 6min | 3 tasks | 4 files |
| Phase 4 P3 | 6min | 2 tasks | 2 files |
| Phase 5 P1 | 8min | 2 tasks | 4 files |
| Phase 5 P2 | 8min | 3 tasks | 4 files |
| Phase 5 P3 | 10min | 2+1 tasks | 2 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Day one: `numpy<2.3` pin is mandatory — numba raises ImportError on numpy 2.3+
- Day one: `creator.Individual` and `creator.FitnessMulti` must be defined at module level in `gp_types.py` for multiprocessing pickling
- Day one: Signal shift `fshift(1)` must be a structural invariant in TreeEvaluator, not a config option
- [Phase 3 P1]: Scalar defined as subclass of Vector (class Scalar(Vector)) to prevent DEAP typed tree dead-ends at rolling aggregation outputs
- [Phase 3 P1]: Ephemeral constant generator _rand_scalar_int() must be module-level function (not lambda) for multiprocessing.Pool pickling in Phase 4
- [Phase 3 P2]: _to_f32() returns 0-D array (not 1-D via atleast_1d) to preserve numpy broadcast semantics; arithmetic ops work naturally, rolling ops guard ndim==0 and return early
- [Phase 3 P2]: TreeEvaluator broadcasts 0-D scalar output to [T] via np.broadcast_to after compile() result check
- [Phase 3 P2]: GP-07 lookahead detection uses correlation proxy (no vectorbt) — leaky_corr=0.587 > clean_corr=0.582 > 0.5, confirming future-leak primitive inflates IS fitness
- [Phase 4 P1]: Deferred GP imports inside evaluate() body preserves D-15 (no deap in backtest layer)
- [Phase 4 P1]: fee_per_side = (fee_bps / 2.0) / 10_000.0 — round-trip bps to per-side decimal for vectorbt fees parameter
- [Phase 4 P1]: upon_opposite_entry='close' required with size_type='percent' to prevent double-position accumulation on long-to-short reversals
- [Phase 4 P2]: dill (not pickle) used for checkpoints — required for DEAP PrimitiveTree/creator.Individual serialization
- [Phase 4 P2]: multiprocessing.get_context("spawn") + _jit_warmup() in initializer — numba JIT compiles in workers, not main process
- [Phase 4 P2]: DEAP Logbook stores MultiStatistics chapters in logbook.chapters['fitness'/'size'], not per-entry dicts
- [Phase 4 P2]: DEAP toolbox.register partial sets __name__ = alias; underlying function accessed via .func for identity checks
- [Phase 4 P3]: D-15 check uses source-code inspection (not sys.modules) — transitive vectorbt import via runner.py always populates sys.modules
- [Phase 5 P1]: OOS touchonce structural enforcement — test_fm is a local variable in _run_window(), not stored on runner; one evaluate() call per window after run_evolution() returns
- [Phase 5 P1]: _get_is_returns() extracted as module-level function for test patching; compute_dsr imported at module level for same reason
- [Phase 5 P2]: plot_equity_curves() calls vbt.Portfolio.from_signals() directly — evaluate() interface unchanged; Portfolio object is created fresh for visualization only
- [Phase 5 P2]: deap.gp.graph() + nx.bfs_layout(G, start=0) replaces graphviz — no system dependency required
- [Phase 5 P2]: matplotlib.use('Agg') at module top in plots.py — headless rendering, no display required
- [Phase 5 P3]: Human checkpoint placed after Task 1+2 complete — visual sign-off on plots before SUMMARY.md created
- [Phase 5 P3]: README honest caveat: positive OOS Sharpe is the goal, not a guarantee — research tool framing preserved

### Pending Todos

None.

### Blockers/Concerns

- ✓ numpy<2.3 pin is in pyproject.toml — day-one blocker resolved
- ✓ requirements-lock.txt committed — reproducibility confirmed
- ✓ Cache data starts 2024-01-01 (not 2021); FeatureEngine intersection produces dates 2024-05-01 to 2025-12-31; test splits calibrated to actual cache window
- ✓ TreeEvaluator structural fshift(1) verified — GP-06 test passes (signal[0]==0.0 always)
- ✓ 1000-tree validation passes (GP-08) — ephemeral constant scalar coercion fixed
- ✓ Parallel eval confirmed working (make approve) — n_jobs=2 spawn pool, JIT warmup fires, no errors
- ✓ mlflow deferred to optional extra — Phase 5 uses CSV/JSON for experiment logs
- ✓ VAL-01 through VAL-07 all verified — 57 tests passing, 0 failed
- ✓ Human checkpoint approved — pareto_front.png and tree_graph.png visually correct

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-06-13T00:00:00.000Z
Stopped at: Milestone v1.0 complete. Phase 5 done. 57 tests pass. README.md and CONTRIBUTING.md published. Human checkpoint approved.
Resume file: None
