# Project State: VGP

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-03)

**Core value:** Evolve strategies with demonstrable positive out-of-sample Sharpe ratio
**Current focus:** Phase 2 — Data Pipeline

## Current Position

Phase: 2 of 5 (Data Pipeline)
Plan: 0 of TBD in current phase
Status: Phase 1 complete — Phase 2 not yet planned
Last activity: 2026-06-08 — Phase 1 complete (verified 5/5, all requirements satisfied)

Progress: [██░░░░░░░░] 20%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: —
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 — Foundation & Environment | 2 | ~3h | 1.5h |

**Recent Trend:**
- Last 5 plans: 01-01 (~1min), 01-02 (~2min)
- Trend: Fast (structure-only phase)

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
- vectorbt 1.0.0 has no official migration guide from 0.x — all API calls must be verified against 1.0 docs directly
- mlflow requires pandas<3 (incompatible with pandas>=3.0.0); moved to [tracking] optional extra — resolve tracking strategy in Phase 4
- On-chain data availability in parquet files unknown; if absent, defer on-chain terminals and constrain to price/volume primitives

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-06-08
Stopped at: Phase 1 complete. Phase 2 (Data Pipeline) not yet planned.
Resume file: None
