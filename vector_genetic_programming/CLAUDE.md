# Vector Genetic Programming (VGP)

> Open-source framework using DEAP + vectorbt to evolve trading strategies from multi-asset crypto data.
> Research output — positive out-of-sample Sharpe is the core success criterion.

## GSD Workflow

This project uses **Get Shit Done (GSD)** for structured phase-based development.

**Current status:** Phase 1 not started. Run `/gsd-discuss-phase 1` to begin.

**Planning artifacts:**
- `.planning/PROJECT.md` — project context and decisions
- `.planning/ROADMAP.md` — 5-phase plan, 40 requirements
- `.planning/REQUIREMENTS.md` — full requirements with traceability
- `.planning/research/` — stack, features, architecture, pitfalls research
- `.planning/config.json` — workflow config (YOLO, coarse, parallel, balanced models)

**GSD commands:**
```
/gsd-discuss-phase 1   # Gather context before planning (recommended)
/gsd-plan-phase 1      # Create PLAN.md for Phase 1
/gsd-execute-phase 1   # Execute the plan
/gsd-progress          # Check current status
/gsd-next              # Advance to next logical step
```

## Project Structure (target)

```
vgp/
├── data/          # DataLoader, FeatureEngine, WalkForwardSplitter
├── gp/            # PrimitiveSetTyped, primitives, tree evaluation, signal generation
├── evolution/     # DEAP toolbox, NSGA-II loop, checkpointing
├── backtest/      # vectorbt integration, evaluate(), fitness functions
└── analysis/      # Pareto front viz, equity curves, tree graphs, DSR reporting
tests/
.planning/
pyproject.toml
CLAUDE.md
```

## Critical Technical Constraints

These are non-negotiable — violating any of these silently destroys results:

1. **`numpy<2.3`** — Pin this from day one. NumPy 2.3 (mid-2026) hard-breaks numba. The smoke test must verify this before any backtest code is added.

2. **No lookahead in signals** — Signal at time `t` may only use data from `t-1` and earlier. `fshift(1)` must be structural, not optional. The lookahead detection test (GP-07) must pass before any evolution runs.

3. **Transaction costs inside `evaluate()`** — Never applied post-hoc. GP will evolve strategies that exploit the absence of costs if they're not in the fitness signal.

4. **50-trade minimum hard filter** — Individuals with < 50 trades receive worst-possible fitness tuple, not NaN or exclusion. They must be rankable by NSGA-II.

5. **Tree depth ≤ 8** — DEAP default of 17 is too permissive. Enforce via `staticLimit` from generation 0.

6. **OOS holdout touched once** — The test split is defined before the first evolution run and used only for final reporting. Any "look at OOS to adjust" invalidates the results.

7. **`DEAP creator.create()` at module level** — Not inside functions. `multiprocessing.Pool` pickles these; function-level definitions cause silent `AttributeError` in workers.

8. **vectorbt JIT warmup in worker initializer** — numba compiles `Portfolio.from_signals` on first call (~30-60s). Run a dummy backtest in the Pool initializer before evolution starts.

9. **vectorbt 1.0.0 API** — The 1.0 release is a major rewrite from 0.x. All tutorials before mid-2025 reference the wrong API. Use vectorbt.dev docs for 1.0 only.

10. **pandas 3.0 idioms** — No `.values` (use `.to_numpy()`). No chained assignment. Explicit `.copy()`. Mandatory from first line of data pipeline code.

## Pinned Dependencies

```toml
deap==1.4.4
vectorbt==1.0.0
numpy>=2.0.0,<2.3    # <2.3 required for numba compatibility
pandas>=3.0.0,<4.0
numba>=0.61.2
scikit-learn>=1.7.0,<2.0
joblib>=1.4.0,<2.0
matplotlib>=3.9.0,<4.0
```

## Architecture Invariants

- `EvolutionLoop` must NOT import `vectorbt`
- `BacktestRunner` must NOT import `deap`
- Interface between them: numpy array in → fitness tuple out
- All GP primitive functions must be module-level (not lambdas) for pickle compatibility
- All primitives accept and return `np.ndarray` — no pandas inside primitives

## Roadmap Summary

| Phase | Goal | Key Risk |
|-------|------|----------|
| 1 | Foundation & Environment | numpy<2.3 + numba compat |
| 2 | Data Pipeline | Enforced OOS split (structural, not convention) |
| 3 | GP Core & Evaluation | Vectorized tree exec + lookahead prevention |
| 4 | Evolution Engine | JIT warmup trap in parallel eval |
| 5 | Validation & Publication | Multi-seed DSR, not raw OOS Sharpe |
