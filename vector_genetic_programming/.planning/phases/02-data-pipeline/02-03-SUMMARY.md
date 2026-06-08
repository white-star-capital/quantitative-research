---
phase: 02-data-pipeline
plan: 03
subsystem: data
tags: [integration-tests, public-api, contributing, fixture, walk-forward, data-loader]

# Dependency graph
requires:
  - phase: 02-data-pipeline
    plan: 01
    provides: "BinanceFetcher, UNIVERSE_30, DataConfig"
  - phase: 02-data-pipeline
    plan: 02
    provides: "FeatureEngine, WalkForwardSplitter"
provides:
  - "DataLoader alias exported from vgp.data (BinanceFetcher is DataLoader for Phase 2)"
  - "Full-pipeline fixture tests in tests/test_data_pipeline.py (DATA-04 success criterion)"
  - "CONTRIBUTING.md at repo root documenting primitive rules, experiment CLI, lock file steps"
  - "Complete vgp.data public API: DataLoader, BinanceFetcher, FeatureEngine, WalkForwardSplitter, DataConfig, UNIVERSE_30"
affects:
  - 03-gp-core

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "DataLoader = BinanceFetcher alias pattern: Phase 3 imports DataLoader; concrete class is BinanceFetcher"
    - "FIXTURE_CACHE = Path(__file__).parent.parent / 'data_pipeline_example' / 'cache' — fixture-relative path for no-network CI tests"
    - "Fixture-based integration test pattern: full pipeline (fetch -> features -> split) against cached parquets"
    - "pytest.raises(AssertionError) for structural contract enforcement tests (D-11)"

key-files:
  created:
    - tests/test_data_pipeline.py
    - CONTRIBUTING.md
  modified:
    - vgp/data/__init__.py

key-decisions:
  - "DataLoader alias pattern: BinanceFetcher is exposed both as BinanceFetcher (concrete) and DataLoader (abstract alias for Phase 3 compatibility). Phase 3 GP layer imports DataLoader; swapping the concrete fetcher only requires one line change in __init__.py."
  - "Fixture date boundaries adjusted to cache window: the plan showed dates like train_end='2023-12-31' for the full-pipeline test, but the actual cache covers 2024-05-01 to 2025-12-31. test_full_pipeline_split_array() uses cache-appropriate dates (train_end='2025-01-31', val_start='2025-02-01', val_end='2025-06-30', test_start='2025-07-01') producing train=276, val=150, test=184 rows."
  - "test_splitter_valid_split_dataframe() uses a synthetic DatetimeIndex from 2021 to 2025, so D-10 date boundaries (train_end='2023-12-31', val_start='2024-01-01') work correctly without cache data."

patterns-established:
  - "All 7 data pipeline tests pass against data_pipeline_example/cache/ without network access"
  - "vgp.data full public API verified: DataLoader is BinanceFetcher, len(UNIVERSE_30)==30"
  - "CONTRIBUTING.md cites three mandatory primitive rules (module-level, np.ndarray, vector shape)"

requirements-completed: [DATA-04, COMM-01, COMM-03]

# Metrics
duration: 8min
completed: 2026-06-08
---

# Phase 2 Plan 03: Public API, Pipeline Tests, and CONTRIBUTING.md Summary

**DataLoader alias wired in vgp/data/__init__.py; 7 fixture-based pipeline integration tests passing without network access; CONTRIBUTING.md at repo root with primitive rules, experiment CLI stub, lock file pip-compile steps, and code conventions.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-06-08T20:52:00Z
- **Completed:** 2026-06-08T21:01:45Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- `vgp/data/__init__.py`: Added `DataLoader = BinanceFetcher` alias and updated `__all__` to include it alongside BinanceFetcher; all 7 public names exported (UNIVERSE_30, get_binance_symbols, DataLoader, BinanceFetcher, FeatureEngine, WalkForwardSplitter, DataConfig)
- `tests/test_data_pipeline.py`: 7 integration tests covering: zero-NaN pipeline output, DatetimeIndex on DataLoader, AssertionError on inverted dates (both val and test ordering), valid DataFrame splits, full array split with cache-appropriate dates, and all 5 sub-module imports — all passing without network access
- `CONTRIBUTING.md`: 4 sections — Adding a GP Primitive (three mandatory rules), Running an Experiment (Phase 4 placeholder), Updating the Dependency Lock File (pip-compile steps + numpy<2.3 warning), Code Conventions (pandas 3.0 idioms, from __future__, module-level loggers, test command)

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire vgp/data/__init__.py public exports** — `170181a` (feat)
2. **Task 2: Write tests/test_data_pipeline.py** — `4513d81` (feat)
3. **Task 3: Create CONTRIBUTING.md** — `bb0b6d8` (docs)

## Files Created/Modified

- `vgp/data/__init__.py` — Added DataLoader alias and updated __all__
- `tests/test_data_pipeline.py` — 7 fixture-based integration tests (DATA-04 success criterion); FIXTURE_CACHE = Path(__file__).parent.parent / "data_pipeline_example" / "cache"
- `CONTRIBUTING.md` — Contributor guide: Adding a GP Primitive, Running an Experiment, Updating the Dependency Lock File, Code Conventions

## Decisions Made

- **DataLoader alias pattern**: `DataLoader = BinanceFetcher` in `__init__.py` makes Phase 3's GP layer import-agnostic. The GP core will `from vgp.data import DataLoader` without knowing the concrete fetcher class. If a future phase swaps BinanceFetcher for a different fetcher, only one line changes.

- **Fixture date boundaries adjusted for actual cache window**: The plan's action block showed `train_end="2023-12-31"` for `test_full_pipeline_split_array()`. The actual cache covers 2024-05-01 to 2025-12-31 (not 2021 as originally assumed). Using 2023-12-31 as a boundary with the real fixture produces an empty train slice. The critical context in the task summary specified the correct dates: `train_end="2025-01-31"`, `val_start="2025-02-01"`, `val_end="2025-06-30"`, `test_start="2025-07-01"`. These produce train=276, val=150, test=184 rows as verified against the actual cache.

- **Synthetic DataFrame for test_splitter_valid_split_dataframe()**: This test uses a synthetic DatetimeIndex from 2021 to 2025, allowing D-10 default dates (train_end="2023-12-31") to work correctly in isolation from the cache data limitation. This keeps the splitter contract test independent of the fixture cache range.

## Deviations from Plan

None — plan executed as written. The date boundary adjustment in `test_full_pipeline_split_array()` was specified in the task's `<critical_context>` override and is not a deviation.

## Known Stubs

None — all three files are fully implemented. CONTRIBUTING.md intentionally contains Phase 3 and Phase 4 placeholders, but these are documented as future sections, not stubs that block the plan's goal (COMM-03).

## Threat Flags

None — no new network endpoints, auth paths, or schema changes introduced. FIXTURE_CACHE is a local path with no network access possible from the test suite (T-02-07 mitigated). CONTRIBUTING.md content is public by design (T-02-08 accepted).

## Self-Check: PASSED

- FOUND: vgp/data/__init__.py (DataLoader alias and full __all__)
- FOUND: tests/test_data_pipeline.py (7 tests collected and passing)
- FOUND: CONTRIBUTING.md (all 4 sections with pip-compile, numpy<2.3, module-level, np.ndarray)
- FOUND: 170181a (Task 1 commit)
- FOUND: 4513d81 (Task 2 commit)
- FOUND: bb0b6d8 (Task 3 commit)
- All 7 data pipeline tests PASSED against data_pipeline_example/cache/ without network calls
- Note: tests/test_smoke.py requires numba and vectorbt which are not installed in the local environment (documented as "install separately" in requirements-lock.txt). This is a pre-existing environment limitation, not introduced by Plan 02-03. The smoke tests run in CI via GitHub Actions where the full environment is set up.

---
*Phase: 02-data-pipeline*
*Completed: 2026-06-08*
