---
phase: 01-foundation-environment
plan: 01
subsystem: infra
tags: [pyproject, setuptools, numpy, deap, vectorbt, numba, pandas, python312]

# Dependency graph
requires: []
provides:
  - pyproject.toml with all 13 pinned dependencies including numpy>=2.0.0,<2.3 constraint
  - vgp package skeleton with 5 sub-module stubs (data, gp, evolution, backtest, analysis)
  - Architecture invariant comments embedded in evolution/ and backtest/ stubs
  - .python-version pinning Python 3.12
  - MIT LICENSE
affects:
  - 01-02-ci-smoke-test
  - all subsequent phases (every plan depends on installable vgp package)

# Tech tracking
tech-stack:
  added:
    - deap==1.4.4
    - vectorbt==1.0.0
    - numpy>=2.0.0,<2.3
    - numba>=0.61.2
    - pandas>=3.0.0,<4.0
    - setuptools>=70.0 (build backend)
    - pytest>=8.2 (dev extra)
    - ruff, black, mypy (dev extras)
  patterns:
    - numpy<2.3 upper-bound pin as day-one constraint (numba compatibility gate)
    - Architecture boundary enforcement via docstring invariants in module stubs
    - Strongly-typed module skeleton with explicit cross-import prohibitions

key-files:
  created:
    - pyproject.toml
    - vgp/__init__.py
    - vgp/data/__init__.py
    - vgp/gp/__init__.py
    - vgp/evolution/__init__.py
    - vgp/backtest/__init__.py
    - vgp/analysis/__init__.py
    - tests/__init__.py
    - .python-version
    - LICENSE
  modified: []

key-decisions:
  - "numpy>=2.0.0,<2.3 pinned in pyproject.toml as day-one blocker — NumPy 2.3+ hard-breaks numba"
  - "numba>=0.61.2 required: first release with explicit NumPy 2.2 support"
  - "vectorbt==1.0.0 hard-pinned (major API rewrite from 0.x; old tutorials reference wrong API)"
  - "deap==1.4.4 hard-pinned (GP framework; only tested version for PrimitiveSetTyped + NSGA-II)"
  - "Architecture invariants documented in module stubs: evolution must not import vectorbt, backtest must not import deap"
  - "dill>=0.3.8 added alongside joblib — handles closures that pickle cannot (needed for DEAP checkpoints)"

patterns-established:
  - "Dependency pinning: exact pins (==) for framework packages, range pins (>=,<) for utilities where patch updates are safe"
  - "Architecture boundary enforcement: cross-import prohibitions stated explicitly in module docstrings from day one"
  - "numpy array interface: all GP primitives must accept/return np.ndarray, no pandas inside GP"

requirements-completed: [FOUND-01, FOUND-04, COMM-02]

# Metrics
duration: 1min
completed: 2026-06-08
---

# Phase 1 Plan 01: Package Skeleton & Pinned Dependencies Summary

**Installable vgp package with 5 sub-module stubs, setuptools build config, and numpy>=2.0.0,<2.3 day-one constraint locked in pyproject.toml**

## Performance

- **Duration:** ~1 min
- **Started:** 2026-06-08T14:39:23Z
- **Completed:** 2026-06-08T14:40:41Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments

- Created pyproject.toml with all 13 pinned dependencies matching CLAUDE.md exactly, including the critical numpy<2.3 upper-bound to prevent numba breakage
- Created vgp package skeleton with 5 sub-module stubs embedding architecture invariant comments (evolution must not import vectorbt, backtest must not import deap)
- Created .python-version (3.12) and MIT LICENSE, satisfying COMM-02 and reproducibility requirements

## Task Commits

Each task was committed atomically:

1. **Task 1: Create pyproject.toml with pinned dependencies** - `53df9b1` (chore)
2. **Task 2: Create vgp package skeleton, .python-version, and MIT LICENSE** - `5dd197e` (chore)

**Plan metadata:** (committed with SUMMARY.md)

## Files Created/Modified

- `pyproject.toml` - Full build config with all 13 pinned dependencies; numpy>=2.0.0,<2.3 is the critical numba-compatibility constraint
- `vgp/__init__.py` - Package entry point with version and architecture invariants comment block
- `vgp/data/__init__.py` - Data pipeline stub (DataLoader, FeatureEngine, WalkForwardSplitter)
- `vgp/gp/__init__.py` - GP core stub with module-level primitive requirement documented
- `vgp/evolution/__init__.py` - Evolution engine stub with "must NOT import vectorbt" invariant
- `vgp/backtest/__init__.py` - Backtest runner stub with "must NOT import deap" invariant
- `vgp/analysis/__init__.py` - Analysis stub (Pareto front, equity curves, DSR reporting)
- `tests/__init__.py` - Empty file for pytest package discovery
- `.python-version` - Pins Python 3.12 for pyenv/uv
- `LICENSE` - MIT License, Copyright (c) 2026 VGP Contributors

## Decisions Made

- numpy>=2.0.0,<2.3 pinned as the single most important constraint per CLAUDE.md — NumPy 2.3 hard-breaks numba
- deap==1.4.4 and vectorbt==1.0.0 exact-pinned (major API boundaries, not interchangeable with adjacent versions)
- dill>=0.3.8 included in core dependencies (not dev extras) because DEAP checkpoint pickling requires it at runtime
- Architecture cross-import prohibitions documented in stubs from day one, before any implementation — establishes the boundary as a convention not just a constraint

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Python 3.9.6 is active in the worktree shell environment (system Python); tomllib is only available in Python 3.12+. TOML validity was verified via python3.12 -c "import tomllib; ..." which succeeded — pyproject.toml is valid TOML as parsed by the standard library's tomllib.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Package skeleton ready: Plan 01-02 (CI smoke test) can proceed immediately
- The numpy<2.3 pin is in place before any backtest code — day-one blocker resolved
- Architecture invariant comments established in module stubs — boundary enforcement starts from first line of implementation code
- No blockers for Phase 1 Plan 02

---
*Phase: 01-foundation-environment*
*Completed: 2026-06-08*
