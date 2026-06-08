# Project State: VGP

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-03)

**Core value:** Evolve strategies with demonstrable positive out-of-sample Sharpe ratio
**Current focus:** Phase 3 — GP Core & Evaluation

## Current Position

Phase: 3 of 5 (GP Core & Evaluation)
Plan: 0 of TBD in current phase
Status: Phase 2 complete — ready to discuss/plan Phase 3
Last activity: 2026-06-08 — Phase 2 execution complete (3/3 plans, verification passed, all 6 requirements satisfied)

Progress: [████░░░░░░] 40%

## Performance Metrics

**Velocity:**
- Total plans completed: 5
- Average duration: ~5 min/plan
- Total execution time: ~25 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 — Foundation & Environment | 2 | ~3 min | 1.5 min |
| 2 — Data Pipeline | 3 | ~22 min | 7 min |

**Recent Trend:**
- Last 5 plans: 02-01 (~7min), 02-02 (~7min), 02-03 (~8min)
- Trend: Fast (implementation from well-defined specs)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Day one: `numpy<2.3` pin is mandatory — numba raises ImportError on numpy 2.3+
- Day one: `creator.Individual` and `creator.FitnessMulti` must be defined at module level in `gp_types.py` for multiprocessing pickling
- Day one: Signal shift `fshift(1)` must be a structural invariant in TreeEvaluator, not a config option

### Pending Todos

None yet.

### Blockers/Concerns

- ✓ numpy<2.3 pin is in pyproject.toml — day-one blocker resolved
- ✓ requirements-lock.txt committed — reproducibility confirmed
- ✓ Cache data starts 2024-01-01 (not 2021); FeatureEngine intersection produces dates 2024-05-01 to 2025-12-31; test splits calibrated to actual cache window
- vectorbt 1.0.0 has no official migration guide from 0.x — all API calls must be verified against 1.0 docs directly
- mlflow requires pandas<3 (incompatible with pandas>=3.0.0); moved to [tracking] optional extra — resolve tracking strategy in Phase 4
- On-chain data availability in parquet files unknown; if absent, defer on-chain terminals and constrain to price/volume primitives
- 3/30 assets geo-restricted (HYPE, AERO, FLUID — 451 Binance error); 21/30 pass min_obs_fraction filter after intersection

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-06-08
Stopped at: Phase 2 complete. Verification passed (10/10). Advancing to Phase 3 GP Core.
Resume file: .planning/phases/ (Phase 3 directory to be created on plan)
