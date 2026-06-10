# Requirements: Vector Genetic Programming (VGP)

**Defined:** 2026-06-03
**Core Value:** Evolve strategies with demonstrable positive out-of-sample Sharpe ratio

## v1 Requirements

### Foundation

- [x] **FOUND-01**: pyproject.toml installs cleanly on Python 3.11+ with pinned deps (deap==1.4.4, vectorbt==1.0.0, numpy<2.3)
- [x] **FOUND-02**: GitHub Actions CI runs import smoke tests on every push to main
- [x] **FOUND-03**: numba/numpy compatibility verified by smoke test that must pass before any backtest code is added
- [x] **FOUND-04**: `.python-version` file pins Python 3.12; requirements-lock.txt committed for reproducible installs

### Data Pipeline

- [x] **DATA-01**: DataLoader ingests multi-asset parquet files into a standardized DataFrame with DatetimeIndex — Validated in Phase 2
- [x] **DATA-02**: FeatureEngine computes rolling window features (returns, volatility, normalized OHLCV, on-chain metrics) — Validated in Phase 2
- [x] **DATA-03**: Time-based train / validation / test split is defined before first evolution run and enforced structurally — Validated in Phase 2
- [x] **DATA-04**: Data pipeline validated with a known-good parquet fixture (correct schema, no NaNs in output) — Validated in Phase 2

### GP Primitives & Representation

- [x] **GP-01**: PrimitiveSetTyped declares Vector and Scalar as distinct Python type tokens
- [x] **GP-02**: Arithmetic primitives (+, -, *, protected-div) operate on both Vector and Scalar with correct broadcast
- [x] **GP-03**: Vector aggregation primitives (mean, std, min, max, rolling-stat variants) reduce Vector → Scalar
- [x] **GP-04**: Conditional primitives (if-then-else, comparison) cover all relevant type combinations — Validated in Phase 4 Plan 01 (gt, lt, if_then_else added to pset)
- [x] **GP-05**: Compiled GP trees broadcast over full [T × F] numpy arrays with no per-bar Python loops — Validated in Phase 3 Plan 02
- [x] **GP-06**: Signal generator converts scalar tree output to directional signals; signal at time t uses only data ≤ t-1 — Validated in Phase 3 Plan 02 (structural fshift(1) in TreeEvaluator, signal[0]==0.0 always)
- [x] **GP-07**: Lookahead detection test: injects a future-leak primitive and asserts fitness is worse than random — Validated in Phase 3 Plan 02 (leaky_corr=0.587 > clean_corr=0.582 > 0.5)
- [x] **GP-08**: All primitives pass type-correctness unit tests (correct input/output types, no silent numpy cast) — Validated in Phase 3 Plan 02 (1000/1000 random trees pass)

### Evaluation & Fitness

- [x] **EVAL-01**: evaluate(individual) compiles the tree, generates signals, and calls vectorbt Portfolio.from_signals
- [x] **EVAL-02**: Transaction costs (configurable bps) applied inside evaluate(), not post-hoc
- [x] **EVAL-03**: Individuals generating fewer than 50 trades receive worst-possible fitness (not disqualified, ranked last)
- [x] **EVAL-04**: Fitness tuple is (Sharpe, total_return, -tree_size) — three objectives for NSGA-II

### Evolution Engine

- [x] **EVO-01**: DEAP toolbox wires selNSGA2, cxOnePoint crossover, mutUniform mutation with configurable probabilities — Validated in Phase 4 Plan 02
- [x] **EVO-02**: eaMuPlusLambda runs a complete evolution loop; population size and generation count are configurable — Validated in Phase 4 Plan 02
- [x] **EVO-03**: GP tree depth hard-limited to 8 via DEAP staticLimit decorator — Validated in Phase 4 Plan 02 (staticLimit on both mate and mutate)
- [x] **EVO-04**: Hall-of-fame tracks the top-N non-dominated individuals across all generations — Validated in Phase 4 Plan 02
- [x] **EVO-05**: Generation-level checkpoints written to disk (dill: population + rng state); resumable from any checkpoint — Validated in Phase 4 Plan 02
- [x] **EVO-06**: DEAP Statistics and Logbook capture per-generation metrics (mean/max/min Sharpe, mean tree size) — Validated in Phase 4 Plan 03 (logbook.chapters['fitness'/'size'])
- [x] **EVO-07**: Parallel evaluation via multiprocessing.Pool with vectorbt JIT warmup in worker initializer — Validated in Phase 4 Plan 02 (spawn context + _jit_warmup)

### Experiment Tracking

- [x] **EXP-01**: MLflow experiment run logs all hyperparameters (pop size, generations, mutation rate, depth limit, fitness weights) — Validated in Phase 4 Plan 03 (MLflowTracker, skipped if mlflow not installed)
- [x] **EXP-02**: MLflow logs per-generation statistics from Logbook — Validated in Phase 4 Plan 03
- [x] **EXP-03**: Experiment is reproducible: given the same seed, two runs produce identical Pareto fronts — Validated in Phase 4 Plan 03 (EXP-03 test passes)

### Validation & Analysis

- [ ] **VAL-01**: Walk-forward validation runs evolution on N rolling windows with non-overlapping OOS periods
- [ ] **VAL-02**: OOS holdout is touched exactly once per window (used only for final performance reporting)
- [ ] **VAL-03**: 10+ independent seeds run per experiment configuration
- [ ] **VAL-04**: Deflated Sharpe Ratio (DSR) computed and reported across seeds/windows
- [ ] **VAL-05**: Pareto front scatter plot exported (Sharpe vs. return vs. tree size) for top generation
- [ ] **VAL-06**: Equity curves plotted for top-3 individuals (in-sample + OOS overlaid)
- [ ] **VAL-07**: GP tree structure exported as graph for top individuals (NetworkX + matplotlib layout)

### Community & Repository

- [ ] **COMM-01**: Package organized under vgp/ with sub-modules: data, gp, evolution, backtest, analysis
- [x] **COMM-02**: MIT license file present at repo root
- [ ] **COMM-03**: CONTRIBUTING.md documents: how to add a primitive, how to run an experiment, how to update the lock file

## v2 Requirements

### Configuration

- **CFG-01**: YAML-based experiment configuration (primitives, evolution params, fitness weights) with no Python edits required
- **CFG-02**: CLI entrypoint (`vgp run config.yaml`) to launch experiments from terminal

### Documentation & Community

- **DOC-01**: Quickstart Jupyter notebook: load parquet → run VGP → inspect results
- **DOC-02**: Results analysis notebook: read checkpoint → plot Pareto front → export best tree
- **DOC-03**: Data schema documentation: expected parquet columns, types, frequency

### Data Pipeline

- **DATA-V2-01**: Parquet schema validation with clear error messages before evolution starts
- **DATA-V2-02**: Caching layer (hdf5 or parquet cache) for preprocessed feature matrices

### Validation

- **VAL-V2-01**: Bootstrap significance tests on OOS returns
- **VAL-V2-02**: Ensemble top-K individuals and report ensemble OOS performance
- **VAL-V2-03**: Sensitivity analysis on primitive set (ablation: which primitives contribute most)

### Monitoring

- **MON-01**: Paper trading hooks (live signal generation from evolved tree on new data)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Live trading infrastructure | Research only — no trading pipeline in v1 or v2 |
| Real-time API data ingestion | Data comes from parquet files; API complexity deferred indefinitely |
| Streamlit / web dashboard | Jupyter notebooks sufficient for research output |
| Backtrader / Zipline validation | vectorbt is the single source of truth for v1 |
| Mobile or cloud deployment | Local workstation + optional cloud for large runs |
| OAuth / user accounts | Not a multi-user product |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | Phase 1 | ✓ Complete (2026-06-08) |
| FOUND-02 | Phase 1 | ✓ Complete (2026-06-08) |
| FOUND-03 | Phase 1 | ✓ Complete (2026-06-08) |
| FOUND-04 | Phase 1 | ✓ Complete (2026-06-08) |
| COMM-02 | Phase 1 | ✓ Complete (2026-06-08) |
| DATA-01 | Phase 2 | Pending |
| DATA-02 | Phase 2 | Pending |
| DATA-03 | Phase 2 | Pending |
| DATA-04 | Phase 2 | Pending |
| COMM-01 | Phase 2 | Pending |
| COMM-03 | Phase 2 | Pending |
| GP-01 | Phase 3 | Complete |
| GP-02 | Phase 3 | Complete |
| GP-03 | Phase 3 | Complete |
| GP-04 | Phase 3 | Pending |
| GP-05 | Phase 3 | Pending |
| GP-06 | Phase 3 | Pending |
| GP-07 | Phase 3 | Pending |
| GP-08 | Phase 3 | Pending |
| EVAL-01 | Phase 3 | Complete |
| EVAL-02 | Phase 3 | Complete |
| EVAL-03 | Phase 3 | Complete |
| EVAL-04 | Phase 3 | Complete |
| EVO-01 | Phase 4 | Pending |
| EVO-02 | Phase 4 | Pending |
| EVO-03 | Phase 4 | Pending |
| EVO-04 | Phase 4 | Pending |
| EVO-05 | Phase 4 | Pending |
| EVO-06 | Phase 4 | Pending |
| EVO-07 | Phase 4 | Pending |
| EXP-01 | Phase 4 | Pending |
| EXP-02 | Phase 4 | Pending |
| EXP-03 | Phase 4 | Pending |
| VAL-01 | Phase 5 | Pending |
| VAL-02 | Phase 5 | Pending |
| VAL-03 | Phase 5 | Pending |
| VAL-04 | Phase 5 | Pending |
| VAL-05 | Phase 5 | Pending |
| VAL-06 | Phase 5 | Pending |
| VAL-07 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 40 total
- Mapped to phases: 40 ✓
- Unmapped: 0

---
*Requirements defined: 2026-06-03*
*Last updated: 2026-06-03 after roadmap initialization — all 40 requirements mapped to phases*
