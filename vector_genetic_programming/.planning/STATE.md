---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: "Phase 3 Plan 02 complete. TreeEvaluator + GP tests. Advancing to Plan 03-03 (BacktestRunner)."
last_updated: "2026-06-09T16:47:00Z"
last_activity: 2026-06-09
progress:
  total_phases: 5
  completed_phases: 2
  total_plans: 8
  completed_plans: 7
  percent: 88
---

# Project State: VGP

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-03)

**Core value:** Evolve strategies with demonstrable positive out-of-sample Sharpe ratio
**Current focus:** Phase 3 — GP Core & Evaluation

## Current Position

Phase: 3 of 5 (GP Core & Evaluation)
Plan: 3 of 3 in current phase
Status: Ready to execute Plan 03-03
Last activity: 2026-06-09

Progress: [████████░░] 75%

## Performance Metrics

**Velocity:**

- Total plans completed: 7
- Average duration: ~5 min/plan
- Total execution time: ~30 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 — Foundation & Environment | 2 | ~3 min | 1.5 min |
| 2 — Data Pipeline | 3 | ~22 min | 7 min |
| 3 — GP Core & Evaluation | 2/3 | ~9 min | 4.5 min |

**Recent Trend:**

- Last 5 plans: 02-02 (~7min), 02-03 (~8min), 03-01 (~4min), 03-02 (~5min)
- Trend: Fast (well-specified plans, minimal surprises)

*Updated after each plan completion*
| Phase 3 P1 | 4min | 2 tasks | 4 files |
| Phase 3 P2 | 5min | 2 tasks | 5 files |

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

### Pending Todos

None.

### Blockers/Concerns

- ✓ numpy<2.3 pin is in pyproject.toml — day-one blocker resolved
- ✓ requirements-lock.txt committed — reproducibility confirmed
- ✓ Cache data starts 2024-01-01 (not 2021); FeatureEngine intersection produces dates 2024-05-01 to 2025-12-31; test splits calibrated to actual cache window
- ✓ TreeEvaluator structural fshift(1) verified — GP-06 test passes (signal[0]==0.0 always)
- ✓ 1000-tree validation passes (GP-08) — ephemeral constant scalar coercion fixed
- vectorbt 1.0.0 has no official migration guide from 0.x — all API calls must be verified against 1.0 docs directly
- mlflow requires pandas<3 (incompatible with pandas>=3.0.0); moved to [tracking] optional extra — resolve tracking strategy in Phase 4
- On-chain data availability in parquet files unknown; if absent, defer on-chain terminals and constrain to price/volume primitives
- 3/30 assets geo-restricted (HYPE, AERO, FLUID — 451 Binance error); 21/30 pass min_obs_fraction filter after intersection

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-06-09T16:47:00Z
Stopped at: Phase 3 Plan 02 complete. TreeEvaluator + GP tests. Advancing to Plan 03-03 (BacktestRunner).
Resume file: None
