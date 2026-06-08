---
phase: 01-foundation-environment
plan: 02
subsystem: testing
tags: [smoke-test, pytest, numba, numpy, vectorbt, github-actions, ci, pandas]

# Dependency graph
requires:
  - phase: 01-01
    provides: pyproject.toml with pinned dependencies (numpy<2.3, numba, deap, vectorbt==1.0.0)
provides:
  - pytest smoke test gating numba/numpy/vectorbt compatibility on every push
  - GitHub Actions CI workflow running smoke tests on push and pull_request to main
  - Runtime assertion that numpy < 2.3 (FOUND-03 structural enforcement)
  - CI pipeline enforcing numpy<2.3 constraint automatically (FOUND-02)
affects: [all-phases, evolution, backtest, data-pipeline]

# Tech tracking
tech-stack:
  added: [pytest, github-actions]
  patterns: [smoke-test-as-gate, double-gate-numpy-version-check, ci-pip-cache-pyproject]

key-files:
  created:
    - tests/test_smoke.py
    - .github/workflows/ci.yml
  modified: []

key-decisions:
  - "Double-gate numpy<2.3: both pytest (test_numpy_version_below_2_3) and standalone CI step assert version constraint"
  - "freq='1D' required for vectorbt 1.0.0 sharpe_ratio() to return non-NaN — documented as 1.0 gotcha, enforced in smoke test"
  - "Smoke test is the gate: must pass before any backtest code is added (structural, not convention)"

patterns-established:
  - "Gate pattern: smoke test must pass before advancing to next phase's backtest code"
  - "pandas 3.0 idioms tested: .to_numpy() not .values, .loc[] for assignment, explicit .copy()"
  - "CI pip cache keyed on pyproject.toml hash for fast dependency restoration"

requirements-completed: [FOUND-02, FOUND-03]

# Metrics
duration: 2min
completed: 2026-06-08
---

# Phase 1 Plan 02: CI + Smoke Test Summary

**pytest smoke test with 5 numba/numpy/vectorbt compatibility guards plus GitHub Actions CI that enforces the numpy<2.3 constraint on every push to main**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-06-08T14:47:08Z
- **Completed:** 2026-06-08T14:48:28Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `tests/test_smoke.py` with 5 test functions covering the numpy<2.3 gate, numba JIT compilation, DEAP imports, vectorbt Portfolio.from_signals end-to-end, and pandas 3.0 idioms
- `.github/workflows/ci.yml` triggering on push and pull_request to main with Python 3.12, pip cache, and a belt-and-suspenders numpy version check independent of pytest
- FOUND-02 satisfied: CI runs import smoke tests on every push to main
- FOUND-03 satisfied: numba/numpy compatibility is verified by a smoke test acting as the gate before backtest code is added

## Task Commits

Each task was committed atomically:

1. **Task 1: Write tests/test_smoke.py** - `6084fbd` (test)
2. **Task 2: Create .github/workflows/ci.yml** - `b10c8f0` (feat)

## Files Created/Modified

- `tests/test_smoke.py` - 5 pytest smoke tests: numpy version guard, numba JIT compilation, DEAP imports, vectorbt from_signals, pandas 3.0 idioms
- `.github/workflows/ci.yml` - GitHub Actions CI workflow (push + PR to main, Python 3.12, pip cache, pytest + standalone numpy check)

## Decisions Made

- Double-gate for numpy<2.3: pytest test_numpy_version_below_2_3 plus a separate CI step both assert the constraint independently. This ensures the gate holds even if pytest itself is skipped or misconfigured.
- `freq="1D"` explicitly set in the vectorbt smoke test because omitting it causes `sharpe_ratio()` to return NaN silently — documented 1.0.0 gotcha that this test guards against.
- Smoke test framed as the structural gate in its module docstring: "This test file is the GATE that must pass before any backtest code is added."

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The `pyyaml` module was not available in system Python for YAML validation, but an equivalent Python check verified all required CI content directly.

## User Setup Required

None - no external service configuration required. CI will activate automatically when pushed to GitHub.

## Next Phase Readiness

- Smoke test and CI gate are in place. Phase 2 (Data Pipeline) can proceed.
- The numpy<2.3 constraint is now enforced at two levels: pyproject.toml pin (install-time) and smoke test + CI step (runtime).
- No blockers from this phase.

## Known Stubs

None - both files are complete and fully wired. The smoke test will run real imports and assertions; CI will run the real test suite on push.

---
*Phase: 01-foundation-environment*
*Completed: 2026-06-08*
