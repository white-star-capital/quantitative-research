# Roadmap: Vector Genetic Programming (VGP)

**Milestone:** v1.0 — Research-grade VGP system with positive OOS Sharpe
**Status:** Not started
**Phases:** 5
**Requirements mapped:** 40/40

---

## Overview

The build follows a strict dependency order: environment first, then data, then the GP primitive and evaluation layer (highest risk), then the full evolution loop with experiment tracking, then walk-forward validation and publication artifacts. Each phase delivers a coherent, independently verifiable capability. No phase begins until the prior phase's success criteria are confirmed.

---

## Phases

- [x] **Phase 1: Foundation & Environment** - Reproducible Python environment with verified numba/numpy/vectorbt compatibility (complete 2026-06-08)
- [ ] **Phase 2: Data Pipeline** - Validated multi-asset feature matrix with enforced train/OOS split
- [ ] **Phase 3: GP Core & Evaluation** - Single-tree compile, execute, and backtest working end-to-end
- [ ] **Phase 4: Evolution Engine** - Complete multi-generation NSGA-II evolution with parallel evaluation and experiment tracking
- [ ] **Phase 5: Validation & Publication** - Walk-forward OOS results, DSR reporting, and community-ready repository

---

## Phase Details

### Phase 1: Foundation & Environment
**Goal**: A reproducible Python environment exists where numba, numpy, and vectorbt all import and interoperate correctly, CI runs on every push, and the repo is licensed for community release.
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01, FOUND-02, FOUND-03, FOUND-04, COMM-02
**Success Criteria** (what must be TRUE):
  1. `pip install -e .` on a clean Python 3.12 environment installs all pinned dependencies without conflict, and `import vectorbt; import numba; import deap` all succeed
  2. A smoke test that imports numba and asserts `numpy.__version__ < "2.3"` passes in CI on every push to main
  3. GitHub Actions workflow runs the import smoke test and reports pass/fail on every commit to main
  4. `python-version` file and requirements-lock.txt are committed; a second developer can reproduce the environment from these files alone
  5. MIT LICENSE file is present at repo root
**Plans:** 2 plans

Plans:
- [x] 01-01-PLAN.md — pyproject.toml with pinned deps, vgp package skeleton, .python-version, MIT LICENSE (complete 2026-06-08)
- [x] 01-02-PLAN.md — smoke test (tests/test_smoke.py) and GitHub Actions CI workflow (complete 2026-06-08)

### Phase 2: Data Pipeline
**Goal**: A validated multi-asset feature matrix is produced from parquet files, the train/validation/test split is structurally enforced before any evolution code is written, and the package module layout is established.
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, COMM-01, COMM-03
**Success Criteria** (what must be TRUE):
  1. `DataLoader` reads multi-asset parquet files and returns a DataFrame with a verified DatetimeIndex and no NaN values in the output
  2. `FeatureEngine` produces a float32 `[T × F × A]` numpy array from the loaded DataFrame; shape can be inspected and confirmed correct for the fixture dataset
  3. `WalkForwardSplitter` raises an assertion error if `test_start <= train_end`, confirming temporal ordering is structurally enforced rather than advisory
  4. A known-good parquet fixture runs through the full pipeline (load → features → split) and produces the expected output schema with zero NaNs
  5. The `vgp/` package is importable with sub-modules `data`, `gp`, `evolution`, `backtest`, `analysis`; CONTRIBUTING.md documents how to add a primitive and run an experiment
**Plans:** 3 plans

Plans:
- [ ] 02-01-PLAN.md — universe.py (UNIVERSE_30), config.py (DataConfig), fetcher.py (BinanceFetcher with fetch_ohlcv)
- [ ] 02-02-PLAN.md — feature_engine.py (FeatureEngine: [T×F×A] float32, 12 features, NaN guard), splitter.py (WalkForwardSplitter with AssertionError enforcement)
- [ ] 02-03-PLAN.md — vgp/data/__init__.py (public exports), tests/test_data_pipeline.py (DATA-04 fixture test), CONTRIBUTING.md (COMM-03)

### Phase 3: GP Core & Evaluation
**Goal**: A single evolved GP tree can be compiled, executed over a full multi-asset feature matrix without Python loops, and evaluated to a three-objective fitness tuple via vectorbt — with lookahead structurally prevented and all edge cases handled.
**Depends on**: Phase 2
**Requirements**: GP-01, GP-02, GP-03, GP-04, GP-05, GP-06, GP-07, GP-08, EVAL-01, EVAL-02, EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):
  1. 1,000 randomly generated GP trees all pass type-correctness checks with zero `IndexError` or silent numpy cast failures, confirming `PrimitiveSetTyped` Vector/Scalar separation works
  2. A compiled tree executes over a `[T × F]` array and produces a signal array with no per-bar Python loops (confirmed by profiling showing vectorized execution)
  3. A deliberately injected future-leak primitive produces a fitness that is not better than a randomly-initialized individual, confirming the lookahead detection test catches the exploit
  4. `evaluate(individual)` returns a `(sharpe, total_return, -tree_size)` tuple with transaction costs baked in; an individual with fewer than 50 trades receives the worst-possible fitness tuple rather than a NaN or exception
  5. The `BacktestRunner` and `EvolutionLoop` modules have no cross-imports (confirmed by import audit); their interface is a numpy array in, fitness tuple out
**Plans**: TBD
**UI hint**: no

### Phase 4: Evolution Engine
**Goal**: A complete NSGA-II evolution run executes for multiple generations with parallel evaluation, per-generation metrics logged to MLflow, and full reproducibility — given the same seed, two runs produce identical Pareto fronts.
**Depends on**: Phase 3
**Requirements**: EVO-01, EVO-02, EVO-03, EVO-04, EVO-05, EVO-06, EVO-07, EXP-01, EXP-02, EXP-03
**Success Criteria** (what must be TRUE):
  1. `eaMuPlusLambda` runs a configurable number of generations end-to-end (e.g., 10 generations, 100 individuals) and terminates without error; the hall-of-fame contains non-dominated individuals
  2. Parallel evaluation via `multiprocessing.Pool` runs faster than single-threaded evaluation, with vectorbt JIT warmup confirmed complete in worker initializers before any evaluation begins
  3. A checkpoint written at generation N can be resumed from disk; the resumed run produces the same hall-of-fame as a continuous run given the same seed
  4. MLflow experiment logs show all hyperparameters and per-generation Sharpe/tree-size statistics for a completed run; two runs with the same seed produce identical logged Pareto fronts
  5. GP tree depth is hard-limited to 8; no individual in any generation exceeds this depth (confirmed by post-run assertion over the full population)
**Plans**: TBD

### Phase 5: Validation & Publication
**Goal**: Walk-forward OOS results are computed across multiple windows and seeds with DSR reported, publication-quality visualizations are exported, and the repository is ready for community release.
**Depends on**: Phase 4
**Requirements**: VAL-01, VAL-02, VAL-03, VAL-04, VAL-05, VAL-06, VAL-07
**Success Criteria** (what must be TRUE):
  1. Walk-forward validation runs evolution on N rolling windows; each OOS period is non-overlapping and the holdout is provably touched exactly once per window (enforced by the splitter, not by convention)
  2. 10+ independent seeds run per experiment configuration; median OOS Sharpe ± IQR is reported, and the Deflated Sharpe Ratio is computed and shown alongside the raw IS Sharpe
  3. Pareto front scatter plot (Sharpe vs. return vs. tree size) is exported as an image for the top generation of a completed run
  4. Equity curves for the top-3 individuals are plotted with in-sample and OOS periods overlaid and visually distinguishable
  5. GP tree structure for top individuals is exported as a graph (NetworkX + matplotlib); a human can read the tree and verify it contains no obvious lookahead patterns
**Plans**: TBD

---

## Progress

**Execution Order:** 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation & Environment | 2/2 | Complete | 2026-06-08 |
| 2. Data Pipeline | 0/3 | Planning done | - |
| 3. GP Core & Evaluation | 0/? | Not started | - |
| 4. Evolution Engine | 0/? | Not started | - |
| 5. Validation & Publication | 0/? | Not started | - |

---

*Roadmap created: 2026-06-03*
*Last updated: 2026-06-08 — Phase 2 planned (3 plans, 3 waves)*
