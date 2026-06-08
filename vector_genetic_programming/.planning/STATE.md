# Project State: VGP

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-03)

**Core value:** Evolve strategies with demonstrable positive out-of-sample Sharpe ratio
**Current focus:** Phase 1 — Foundation & Environment

## Current Position

Phase: 1 of 5 (Foundation & Environment)
Plan: 1 of 2 in current phase
Status: Executing — Wave 2 in progress
Last activity: 2026-06-08 — Plan 01-01 complete (package skeleton, pyproject.toml, vgp/ stubs)

Progress: [█████░░░░░] 50%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: —
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: —
- Trend: —

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
- vectorbt 1.0.0 has no official migration guide from 0.x — all API calls must be verified against 1.0 docs directly
- On-chain data availability in parquet files unknown; if absent, defer on-chain terminals and constrain to price/volume primitives

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-06-08
Stopped at: Phase 1 Wave 2 in progress — Plan 01-02 (smoke tests + CI) is next.
Resume file: None
