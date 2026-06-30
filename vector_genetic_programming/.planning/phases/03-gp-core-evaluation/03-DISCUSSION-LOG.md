# Phase 3: GP Core & Evaluation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-09
**Phase:** 03-gp-core-evaluation
**Areas discussed:** GP input geometry, Signal semantics, Primitive set scope, Transaction costs & portfolio setup

---

## GP Input Geometry

| Option | Description | Selected |
|--------|-------------|----------|
| Single-asset [T×F] — same tree for all assets | Tree operates on one asset's [T×F] slice, applied to all 21 assets identically | ✓ |
| Per-feature [T×A] — one feature row, all assets | Cross-sectional; tree sees all assets for one feature at each time | |
| Full tensor [T×F×A] — complete cross-asset view | Tree sees full feature tensor | |

**User's choice:** Single-asset [T×F]

**Follow-up: How do 21 signal arrays combine?**

| Option | Description | Selected |
|--------|-------------|----------|
| Run independently — 21 separate positions, vectorbt handles portfolio | Each asset traded independently | ✓ |
| Average the signals — one blended signal per asset timestep | Aggregate signals before portfolio construction | |
| Cross-sectional rank — long top-N, short bottom-N | Equity-style long-short by signal rank | |

**User's choice:** Run independently

---

## Signal Semantics

| Option | Description | Selected |
|--------|-------------|----------|
| Long/short/flat (3-state: +1/0/-1) | sign(output) → direction, every timestep has a direction | ✓ |
| Long-only binary (0/1) | Output > threshold → long, else flat/cash | |
| Continuous fractional position | Map to [-1, +1] via sigmoid/tanh | |

**User's choice:** Long/short/flat

**Follow-up: Flat zone threshold?**

| Option | Description | Selected |
|--------|-------------|----------|
| Zero-crossing only — sign(output), no dead zone | Long if >0, short if <0, no flat state | ✓ |
| Symmetric dead band — flat when |output| < threshold | Reduces trades, introduces hyperparameter | |
| Claude's discretion on threshold | Start with zero-crossing, adjust if needed | |

**User's choice:** Zero-crossing only

---

## Primitive Set Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal math core | add, sub, mul, protected_div, neg, rolling_mean/std/max/min at windows 5 & 20 | ✓ |
| Math core + domain primitives | Above + crossover, rsi_crossover, momentum_rank | |
| Math core + comparisons | Above + GreaterThan, LessThan, IfThenElse | |

**User's choice:** Minimal math core

**Follow-up: Rolling window parameterization?**

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed windows as separate named primitives (rolling_mean_5, rolling_mean_20) | Predefine specific sizes; GP selects which | ✓ |
| Configurable window via EphemeralConstant | GP evolves window size; more flexible | |
| Claude's discretion on window sizes | Align with Phase 2 feature lookbacks | |

**User's choice:** Fixed windows as separate named primitives

---

## Transaction Costs & Portfolio Setup

| Option | Description | Selected |
|--------|-------------|----------|
| 10 bps round-trip (5 bps per side) | Conservative, realistic for Binance maker pricing | ✓ |
| 20 bps round-trip (10 bps per side) | More conservative, taker pricing / lower liquidity | |
| Configurable via EvalConfig, default 10 bps | Parameterized from day 1 for sensitivity analysis | |

**User's choice:** 10 bps round-trip
**Notes:** Make it a configurable parameter in EvalConfig even though default is 10 bps.

**Follow-up: Portfolio sizing?**

| Option | Description | Selected |
|--------|-------------|----------|
| Equal weight — 1/N across all active positions | Standard, no parameter tuning | ✓ |
| Fixed single-asset weight (e.g., 5% per asset) | Fixed fraction, exposure varies by signal count | |
| Claude's discretion on sizing | Start with equal-weight | |

**User's choice:** Equal weight — 1/N

---

## Claude's Discretion

- protected_div behavior for near-zero denominators
- Exact vectorbt 1.0 Portfolio.from_signals() parameter names
- How "trade count" is computed in the 50-trade filter
- Tree initialization method (ramped half-and-half default)

## Deferred Ideas

- Conditional/comparison primitives (IfThenElse, GreaterThan) — Phase 4 expansion
- Cross-asset features as GP inputs — carried from Phase 2 D-07
- EphemeralConstant for window sizes — out of scope
- Domain-aware primitives (crossover, rsi_threshold) — Phase 4+
- Dead band flat zone threshold — deferred, zero-crossing is baseline
