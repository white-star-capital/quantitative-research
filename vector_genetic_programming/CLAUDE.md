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

1. **numpy must match what the installed numba supports** — numba binds numpy's C extension APIs and pins them tightly. The current chain is `vectorbt 1.1.0` → `numpy>=2.4.6`, `numba 0.67.0` → `numpy<2.6`, so numpy lands in `[2.4.6, 2.6)`. Do **not** hardcode a numpy ceiling in a test: the original `numpy<2.3` assertion was numba 0.61's limit, numba 0.67 raised it to `<2.6`, and the test then failed on a valid environment. `tests/test_smoke.py` reads the bound from numba's own metadata and additionally asserts that the pyproject set both resolves and is self-consistent.

2. **No lookahead in signals** — Signal at time `t` may only use data from `t-1` and earlier. `fshift(1)` must be structural, not optional. The lookahead detection test (GP-07) must pass before any evolution runs.

3. **Transaction costs inside `evaluate()`** — Never applied post-hoc. GP will evolve strategies that exploit the absence of costs if they're not in the fitness signal.

4. **50-trade minimum hard filter** — Individuals with < 50 trades receive worst-possible fitness tuple, not NaN or exclusion. They must be rankable by NSGA-II. That tuple is a **ranking sentinel, not a measurement**: reporting code must call `evaluate_with_status()` and record NaN plus a status, never write `-inf` out as a Sharpe. The OOS threshold is scaled to the OOS window length (`oos_min_trades`) because 50 is a trade *rate* set for a ~12-month train window.

5. **Tree depth ≤ 8** — DEAP default of 17 is too permissive. Enforce via `staticLimit` from generation 0.

6. **OOS holdout touched once** — The test split is defined before the first evolution run and used only for final reporting. Any "look at OOS to adjust" invalidates the results.

7. **`DEAP creator.create()` at module level** — Not inside functions. `multiprocessing.Pool` pickles these; function-level definitions cause silent `AttributeError` in workers.

8. **vectorbt JIT warmup in worker initializer** — numba compiles `Portfolio.from_signals` on first call (~30-60s). Run a dummy backtest in the Pool initializer before evolution starts.

9. **vectorbt 1.x API** — The 1.0 release is a major rewrite from 0.x. All tutorials before mid-2025 reference the wrong API. Use vectorbt.dev docs for 1.x only. Pinned at `>=1.1.0` because 1.0.0 caps `pandas<3.0`; also requires `plotly<6`, since plotly 6 removed `scattermapbox` and vectorbt registers it in its figure templates at import time.

10. **pandas 3.0 idioms** — No `.values` (use `.to_numpy()`). No chained assignment. Explicit `.copy()`. Mandatory from first line of data pipeline code.

## Pinned Dependencies

`pyproject.toml` is the single source of truth; this block is a summary. The pins
are mutually satisfiable and CI-verified — change them together, never one at a time.

```toml
deap==1.4.4
vectorbt>=1.1.0,<2.0   # 1.0.0 caps pandas<3.0 — cannot coexist with pandas 3
numpy>=2.4.6,<2.6      # floor from vectorbt 1.1.0, ceiling from numba 0.67
numba>=0.67.0          # 0.61.x caps numpy<2.3
pandas>=3.0.3,<4.0     # required by vectorbt>=1.1.0
plotly>=5.22,<6.0      # plotly 6 removed scattermapbox; breaks vectorbt import
scikit-learn>=1.7.0,<2.0
joblib>=1.4.0,<2.0
matplotlib>=3.9.0,<4.0
```

**A green test suite does not prove the project installs.** For most of this
project's life `pip install -e .` failed for every user — `vectorbt==1.0.0`
requires `pandas<3.0` while the project declared `pandas>=3.0.0` — because tests
were run against a hand-built environment that ignored the declared pins.
`tests/test_smoke.py` now checks the declared set against the live environment.

## Statistical Invariants

These govern what may be claimed from a run, and are as load-bearing as the
technical constraints above.

1. **DSR needs the whole trial set** — `E[SR_max]` is `sigma_SR × bracket(N)`, where `sigma_SR` is the cross-sectional spread of trial Sharpes. Dropping `sigma_SR` puts the hurdle at ~24 annualized and every DSR collapses to zero. It cannot be computed per row: `run_window()` leaves `dsr` NaN, `attach_dsr()` fills it after all windows and seeds have run.

2. **N is every individual evaluated, not the reported winners** — thousands, not the row count. At N=9 the multiplier is 1.52; at N=3600 it is 3.24, and that gap is the difference between certifying and rejecting a noise-derived strategy. `vgp/trials.py` accumulates this in O(1) memory. Report `dsr` (all evaluations, conservative) and `dsr_bests_only` (winners, optimistic) together; if they straddle 0.95 the question is unsettled.

3. **DSR cannot detect bias shared by all trials** — a lookahead, one reused training window, a survivor-biased universe all shift every trial together and leave `sigma_SR` unchanged. No DSR refinement catches them. The **null control** (`vgp/analysis/null_control.py`) is the only check that does: re-run the identical pipeline on block-bootstrapped surrogates and require the real result to clear the null distribution. A run with `N_NULL_RUNS = 0` must not be described as validated.

4. **NaN means "not measured"** — never "bad result", and never 0.0 for a probability. `aggregate_seeds()` excludes unmeasured seeds and reports `n_seeds_valid_oos` against `n_seeds` so the denominator stays visible.

5. **A missing asset is a change to the experiment, not a warning** — `fetch_ohlcv()` RAISES `FetchError` when any symbol fails. A caller willing to run on fewer assets must declare it (`allow_partial=True`) and set a `min_assets` floor; an empty universe always raises. The realized universe is then recorded in `results/universe.json` and stamped onto every result row (`n_assets`, `universe_fingerprint`), because two stages narrow it — fetch failures, then `min_obs_fraction` — and results computed on different compositions are not comparable. Compare fingerprints, not asset counts. This is the survivorship channel DSR and the null control cannot see: every trial and every surrogate inherits the same universe, so nothing in the statistics reveals that it moved.

6. **Null control block size must be shorter than the effect horizon** — a block bootstrap only breaks dependence at block boundaries, so structure shorter than the block survives into the surrogate. The 20-bar default will not flag a 1–5 day momentum artifact.

## Architecture Invariants

- `EvolutionLoop` must NOT import `vectorbt`
- `BacktestRunner` must NOT import `deap`
- Interface between them: numpy array in → fitness tuple out
- All GP primitive functions must be module-level (not lambdas) for pickle compatibility
- All primitives accept and return `np.ndarray` — no pandas inside primitives
- `vgp/trials.py` is numpy-only and sits below both `vgp.evolution` and `vgp.analysis`, so the evolution loop can record trials without importing `vgp.analysis` (which would pull vectorbt in via `vgp.analysis.runner` and break D-15)

## Roadmap Summary

| Phase | Goal | Key Risk |
|-------|------|----------|
| 1 | Foundation & Environment | numpy/numba/vectorbt pin coherence |
| 2 | Data Pipeline | Enforced OOS split + recorded realized universe |
| 3 | GP Core & Evaluation | Vectorized tree exec + lookahead prevention |
| 4 | Evolution Engine | JIT warmup trap in parallel eval |
| 5 | Validation & Publication | Multi-seed DSR + null control, not raw OOS Sharpe |
