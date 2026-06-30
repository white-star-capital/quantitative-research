# Phase 3: GP Core & Evaluation - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning

<domain>
## Phase Boundary

A single evolved GP tree can be compiled, executed over a full multi-asset feature matrix without Python loops, and evaluated to a three-objective fitness tuple via vectorbt — with lookahead structurally prevented and all edge cases handled.

This phase delivers: PrimitiveSetTyped with Vector/Scalar type system, a minimal math primitive library, a vectorized tree evaluator with structural fshift(1), and an evaluate() function that calls vectorbt and returns (Sharpe, total_return, -tree_size) with transaction costs baked in. The evolution loop and parallel evaluation are Phase 4.

</domain>

<decisions>
## Implementation Decisions

### GP Input Geometry
- **D-01:** A single GP tree operates on a **single-asset [T × F] feature slice** — the tree sees one asset's 12 features over T timesteps and produces a [T] signal array. The same compiled tree is applied independently to all 21 assets.
- **D-02:** The 21 per-asset signal arrays are passed to vectorbt as **21 independent columns** — each asset is traded independently, vectorbt handles multi-asset portfolio construction. No signal averaging or cross-sectional ranking in Phase 3.
- **D-03:** Cross-asset features (BTC as global factor, cross-sectional rank inputs) remain deferred from Phase 2 D-07 — not in Phase 3 scope.

### Signal Semantics
- **D-04:** Tree output is converted to **long/short/flat (3-state) signals**: sign(output) > 0 → long (+1), sign(output) < 0 → short (-1). Zero-crossing only — no dead band threshold. Every timestep has a direction (long or short), and a "trade" = sign change.
- **D-05:** fshift(1) is applied **structurally in TreeEvaluator** — the shift happens after tree execution, not as an optional primitive or config flag. Signal at time t uses only output from tree execution at t-1. This is a hard invariant, not a parameter.

### Primitive Set
- **D-06:** Minimal math core only for Phase 3:
  - **Arithmetic**: add(Vector, Vector), sub(Vector, Vector), mul(Vector, Vector), protected_div(Vector, Vector), neg(Vector) — all produce Vector
  - **Scalar constants**: small set of ephemeral integer constants for tree terminal leaves
  - **Rolling aggregations** (Vector → Scalar): rolling_mean_5, rolling_mean_20, rolling_std_5, rolling_std_20, rolling_max_20, rolling_min_20
  - **Note**: window sizes are fixed at 5 and 20, matching Phase 2's lookback choices (vol_5d, vol_20d, ret_5d, ret_20d features). No EphemeralConstant for window sizes.
- **D-07:** Conditional/comparison primitives (IfThenElse, GreaterThan) are deferred — minimal math core is sufficient for Phase 3 validation and keeps the type system clean.
- **D-08:** All primitive functions must be **module-level** in `vgp/gp/primitives.py` (not lambdas, not closures) for multiprocessing.Pool pickling in Phase 4.
- **D-09:** `creator.Individual` and `creator.FitnessMulti` defined at **module level** in `vgp/gp/gp_types.py` — not inside functions. This is a DEAP pickling requirement.

### Transaction Costs & Portfolio
- **D-10:** Transaction costs: **10 bps round-trip (5 bps per side)** baked into evaluate(). Applied via vectorbt's built-in fees parameter, not post-hoc. This is the default — make it a configurable parameter in EvalConfig so sensitivity analysis is possible in Phase 4/5.
- **D-11:** Portfolio sizing: **equal weight — 1/N across all active positions** at each rebalance. vectorbt handles this via `size='value_percent'` or equivalent in the 1.0 API.
- **D-12:** Portfolio style: **long-short** — both +1 and -1 signals create positions. No long-only restriction. Crypto instruments support short exposure.

### Fitness & Edge Cases
- **D-13:** Fitness tuple locked at **(Sharpe, total_return, -tree_size)** — three objectives, consistent with EVAL-04. Maximizing all three means: maximize Sharpe, maximize return, minimize tree size (parsimony pressure).
- **D-14:** Individuals with **fewer than 50 trades** (sign changes across all assets combined) receive the worst-possible fitness tuple — e.g., `(-np.inf, -np.inf, -tree_size)` — not NaN, not exception, not exclusion. They must be rankable by NSGA-II. This is a hard filter in evaluate().

### Architecture Invariants (from CLAUDE.md)
- **D-15:** `BacktestRunner` (`vgp/backtest/`) must NOT import deap. `EvolutionLoop` (`vgp/evolution/`) must NOT import vectorbt. The interface between them is: numpy signal array in → fitness tuple out.
- **D-16:** Vectorized tree execution (GP-05): no per-bar Python loops in the hot path. The tree evaluator must call numpy operations on full [T] arrays, not iterate timestep-by-timestep.

### Claude's Discretion
- Exact numpy broadcasting in arithmetic primitives (where output shape is unambiguous from types)
- protected_div behavior for near-zero denominators (small epsilon is standard)
- Exact vectorbt 1.0 Portfolio.from_signals() parameter names for fees and sizing
- How to count "trades" (entry+exit counts vs. sign-change counts) — implement what aligns cleanly with the 3-state signal
- Tree initialization method (ramped half-and-half is DEAP default; keep it)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### GP & Evaluation Requirements
- `.planning/REQUIREMENTS.md` §GP Primitive Set — GP-01 through GP-08: exact acceptance criteria for PrimitiveSetTyped, type correctness, vectorized execution, and lookahead detection test.
- `.planning/REQUIREMENTS.md` §Evaluation & Fitness — EVAL-01 through EVAL-04: evaluate() signature, transaction cost placement, 50-trade filter, fitness tuple structure.

### Phase 3 Success Criteria (from ROADMAP.md)
- `.planning/ROADMAP.md` §Phase 3 — 5 success criteria including: 1000-tree type-correctness validation, vectorized execution confirmation, lookahead detection test, and BacktestRunner/EvolutionLoop import audit.

### Architecture Constraints
- `CLAUDE.md` §Critical Technical Constraints — All 10 constraints apply; especially: creator.create() at module level (#7), vectorbt JIT warmup in worker initializer (#8), vectorbt 1.0 API (#9), numpy<2.3 pin (#1), no lookahead (#2), transaction costs inside evaluate() (#3), 50-trade filter (#4), tree depth ≤8 (#5).
- `vgp/gp/__init__.py` — Docstring states all primitive functions must be module-level and accept/return np.ndarray.
- `vgp/backtest/__init__.py` — Docstring states this module must NOT import deap.
- `vgp/evolution/__init__.py` — Docstring states this module must NOT import vectorbt.

### Data Interface (Phase 2 output)
- `vgp/data/feature_engine.py` — FEATURE_NAMES list (12 features, F-axis layout). Phase 3 tree execution must align with this axis ordering.
- `vgp/data/splitter.py` — WalkForwardSplitter interface: train/val/test slices. evaluate() runs on train slice only during evolution.
- `.planning/phases/02-data-pipeline/02-CONTEXT.md` — D-05 (float32 [T×F×A] shape), D-10 (train/val/test date ranges: train 2021-01-01–2023-12-31, val 2024-01-01–2024-06-30, test 2024-07-01–2025-12-31). Note: actual cache window is 2024-05-01–2025-12-31 for the 21 passing assets.

### vectorbt 1.0 (NOT 0.x)
- All vectorbt usage must target the 1.0 API. Old tutorials use Portfolio.from_signals with different parameter names. Use vectorbt.dev docs directly.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `vgp/data/feature_engine.py` `FEATURE_NAMES`: The 12-feature ordered list is the single source of truth for the F-axis. TreeEvaluator must use this ordering when slicing [T×F] for individual features.
- `vgp/data/splitter.py` `WalkForwardSplitter.split()`: Returns (train, val, test) slices. evaluate() should receive the train slice already cut — it does not re-slice.
- `vgp/data/config.py` `DataConfig`: Pattern for config dataclasses. Extend this pattern for `EvalConfig` (fee_bps, min_trades, etc.).

### Established Patterns
- All numpy operations use `.to_numpy()` not `.values` (pandas 3.0 requirement from Phase 2).
- Float32 arrays throughout — maintain float32 inside primitives to avoid dtype promotion issues.
- Defensive ValueError for data integrity violations (e.g., NaN in input array should raise, not silently produce bad fitness).

### Integration Points
- Phase 3 → Phase 4: `evaluate(individual) → (float, float, float)` is the complete interface. Phase 4 imports only this function from `vgp/backtest/`.
- Phase 3 files to create: `vgp/gp/primitives.py`, `vgp/gp/gp_types.py`, `vgp/gp/tree_evaluator.py`, `vgp/backtest/runner.py`.
- Tests to add: `tests/test_gp_primitives.py` (GP-08), `tests/test_tree_evaluator.py` (GP-05, GP-07 lookahead detection), `tests/test_evaluate.py` (EVAL-01 through EVAL-04).

</code_context>

<specifics>
## Specific Ideas

- The 12 features in FEATURE_NAMES (feature_engine.py) directly inform useful terminal nodes — individual feature columns are the natural GP terminals (e.g., `X[:, 0]` = ret_1d, `X[:, 4]` = vol_5d). The terminal set should expose named feature columns.
- fshift(1) structural placement: apply `np.roll(signal, 1)` and zero-out index 0 after tree execution, before passing to vectorbt. This is in TreeEvaluator, not in the primitive definitions.
- The 50-trade filter counts sign changes per asset per run; with 21 assets and ~610 timesteps, even a trivially alternating strategy generates thousands of "trades" — the real risk is strategies that stay flat (0 sign changes). Filter should check non-flat periods.

</specifics>

<deferred>
## Deferred Ideas

- Conditional/comparison primitives (IfThenElse, GreaterThan, LessThan) — Phase 4 primitive expansion, after Phase 3 minimal core is validated.
- Cross-asset features as GP inputs (BTC return as global factor, cross-sectional rank) — carried forward from Phase 2 D-07; revisit before Phase 4 if primitive expansion is in scope.
- EphemeralConstant for window sizes — explicitly out of scope (fixed windows only in Phase 3).
- Domain-aware primitives (crossover, rsi_threshold) — Phase 4+.
- Dead band / flat zone threshold — "you decide" if needed; zero-crossing is the starting point.

</deferred>

---

*Phase: 03-gp-core-evaluation*
*Context gathered: 2026-06-09*
