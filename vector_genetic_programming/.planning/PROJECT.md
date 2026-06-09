# Vector Genetic Programming (VGP)

## What This Is

An open-source framework that uses genetic programming (DEAP) to evolve trading and investment strategies from on-chain and market data for multi-asset crypto. The system handles vector inputs natively (rolling windows of features), evaluates evolved programs via vectorbt's vectorized backtesting engine, and produces reproducible, documented research artifacts — strategies, indicators, and findings — intended for community publication.

## Core Value

The system must evolve strategies that demonstrate positive out-of-sample Sharpe ratio — if the GP engine produces nothing better than noise on held-out data, the project has failed regardless of how clean the code is.

## Requirements

### Validated

- [x] Reproducible Python 3.12 environment with all pinned deps (`numpy>=2.0.0,<2.3`, `deap==1.4.4`, `vectorbt==1.0.0`) — Validated in Phase 1
- [x] numba/numpy compatibility gate: smoke test asserts `numpy<2.3` on every push — Validated in Phase 1
- [x] CI pipeline runs smoke tests on all branches — Validated in Phase 1
- [x] MIT license committed — Validated in Phase 1
- [x] Multi-asset OHLCV DataLoader (`BinanceFetcher`) reads cached parquets and returns dict[str, pd.DataFrame] with DatetimeIndex — Validated in Phase 2
- [x] FeatureEngine produces float32 [T×F×A] array (T=610, F=12, A=21) with zero NaN from fixture cache — Validated in Phase 2
- [x] WalkForwardSplitter enforces train/val/test temporal ordering structurally (AssertionError, not advisory) — Validated in Phase 2
- [x] Full pipeline fixture test (DATA-04): load → features → split runs without network access — Validated in Phase 2
- [x] All 5 vgp sub-modules importable; DataLoader alias and public API wired in vgp.data — Validated in Phase 2
- [x] CONTRIBUTING.md documents how to add a GP primitive, run an experiment, and update the lock file — Validated in Phase 2

### Validated (continued)

- [x] GP PrimitiveSetTyped with Vector/Scalar type tokens; 1,000 random trees all compile and execute without error — Validated in Phase 3
- [x] TreeEvaluator.execute() applies structural fshift(1) (np.roll + zero index 0); GP-07 lookahead detection test passes — Validated in Phase 3
- [x] evaluate() returns (Sharpe, total_return, -tree_size); fees baked in; <50 trades → (-inf, -inf, -size); no DEAP import in vgp.backtest — Validated in Phase 3

### Active

- [ ] Evolve GP trees that operate on vector inputs (rolling windows) as first-class primitives
- [ ] Evaluate populations via vectorbt at speed sufficient for hundreds of individuals per generation
- [ ] Multi-objective fitness via NSGA-II (return, Sharpe, drawdown, complexity)
- [ ] Demonstrate positive out-of-sample Sharpe on held-out multi-asset crypto data
- [ ] Walk-forward validation with multiple random seeds for reproducibility
- [ ] Modular, extensible primitive set (easy to add new operators and features)
- [ ] Reproducible experiment runs (seeded, logged, checkpointed)
- [ ] Community-ready: documented GitHub repo, MIT license, example notebooks

### Out of Scope

- Live or paper trading integration — research output only, no trading infrastructure
- Real-time API data ingestion (Glassnode, Dune, exchange APIs) — v1 data comes from parquet files provided externally
- Web/dashboard UI (Streamlit etc.) — Jupyter notebooks are sufficient for v1
- Mobile or cloud deployment — local workstation + optional cloud for large runs
- Multi-framework validation (Backtrader/Zipline) — vectorbt is the single source of truth for v1

## Context

- **Data source**: User-provided parquet files. Schema (columns, frequency, assets covered) will be specified when the data pipeline phase begins. No external API integrations required for v1.
- **Target universe**: Multi-asset crypto — BTC, ETH, and broader altcoin/DeFi tokens. Asset selection governed by what's in the parquet files.
- **vectorbt 1.0 API**: The 1.0 release (2025) is a major rewrite from the 0.x series most tutorials and papers reference. All implementation must target 1.0 docs directly — old `Portfolio.from_signals` call signatures are invalid.
- **NumPy 2.x / numba**: vectorbt uses numba for JIT compilation. numba has historically lagged NumPy major versions. The import smoke test must verify numba ↔ NumPy 2.x compatibility before any backtesting code is written.
- **pandas 3.0 idioms**: Pandas 3.0 (Jan 2026) mandates Copy-on-Write and removes several deprecated patterns. All DataFrame code must use `.to_numpy()` (not `.values`), explicit `.copy()`, and current aggregation patterns.
- **Solo build**: All development is done by the AI assistant. Role breakdown in the original plan (DE, GP Specialist, BQ, SE, Researcher) maps to sequential focus areas rather than parallel human contributors.
- **Research framing**: Primary output is reproducible findings — evolved strategies with measurable OOS performance — plus a documented, forkable codebase. Not a trading system.

## Constraints

- **Python**: ≥3.11, <3.14 — required by pandas 3.0
- **DEAP**: ==1.4.4 — GP framework, pinned hard
- **vectorbt**: ==1.0.0 — backtesting engine, pinned hard (major API rewrite at 1.0)
- **Data format**: Parquet files only — no live API integrations in v1
- **Evolution scope**: Start with populations of 100–500 individuals; scale to larger runs only after single-run performance is validated
- **Reproducibility**: All experiments must be seeded and produce identical results given the same seed

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| DEAP as GP framework | Mature, flexible, supports strongly-typed GP and NSGA-II natively | — Pending |
| vectorbt 1.0 for backtesting | Vectorized speed essential for evaluating large populations; 1.0 is the current stable | — Pending |
| Strongly-typed GP (PrimitiveSetTyped) | Prevents type-invalid trees at construction time; critical for vector/scalar mixing | — Pending |
| NSGA-II multi-objective | Balances return vs. complexity vs. risk simultaneously; avoids single-objective overfitting | — Pending |
| Parquet-first data pipeline | Eliminates API complexity from v1 scope; user controls data quality upstream | — Pending |
| Research-only output | No live trading infrastructure — keeps scope bounded and failure modes tractable | — Pending |
| MIT license | Community release; permissive for academic and commercial reuse | ✓ LICENSE committed (Phase 1) |
| mlflow deferred to optional extra | mlflow requires pandas<3, conflicts with pandas>=3.0.0 core dep; experiment tracking strategy to be decided in Phase 4 | Moved to [tracking] optional extra (Phase 1) |

## Evolution

This document evolves at phase transitions and milestone boundaries.

Last updated: 2026-06-09 — Phase 3 complete (GP Core & Evaluation)

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-08 — Phase 2 complete (Data Pipeline verified — 21-asset [T×F×A] feature matrix, structural OOS split, 7 pipeline tests passing)*
