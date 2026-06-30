# Phase 2: Data Pipeline - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-08
**Phase:** 02-data-pipeline
**Areas discussed:** Parquet schema & multi-asset layout, Feature set scope, Train/val/test split design, NaN handling & lookback strategy

---

## Parquet Schema & Multi-Asset Layout

| Option | Description | Selected |
|--------|-------------|----------|
| One file per asset (btc.parquet, eth.parquet...) | DataLoader globs a directory and loads each asset from its own file. | ✓ (via playbook) |
| Single file, wide format (columns: btc_close, eth_close...) | One file with all assets as column prefixes. | |
| Single file, multi-index | One file with MultiIndex columns — (asset, ohlcv_field). | |
| Single file, long format | Rows are per (timestamp, asset), then pivot on asset. | |

**User's choice:** One file per asset following the `rp_pca/data` playbook pattern (`{SYMBOL}_{interval}.parquet`). User provided a comprehensive pipeline playbook and reference implementation in `data_pipeline_example/`.

**Notes:** User wants to adapt the existing `data_pipeline_example/` codebase directly. Universe is UNIVERSE_30 (30 Binance crypto/USDT pairs). Data fetched from Binance REST API (no API key required). Cache stored at `vgp/data/cache/` (gitignored). The 27 existing parquet files in `data_pipeline_example/cache/` serve as the test fixture.

---

## Feature Set Scope

### Feature groups

| Option | Description | Selected |
|--------|-------------|----------|
| OHLCV-derived only | Returns, log prices, rolling vol, high-low range, volume ratio | |
| OHLCV + technical indicators | OHLCV base + RSI, MACD, Bollinger, momentum | ✓ |
| Minimal starter set — expand later | 5-8 essential features in Phase 2 only | |

**User's choice:** OHLCV + technical indicators

### Specific indicators selected (multiSelect)

| Indicator Group | Selected |
|-----------------|----------|
| Momentum: returns (1d/5d/20d) + log prices | ✓ |
| Volatility: rolling std (5d/20d), ATR, Parkinson vol | ✓ |
| Oscillators: RSI (14), normalized close position (0-1) | ✓ |
| Volume: volume ratio (vol/20d avg), OBV-style signal | ✓ |

**User's choice:** All four groups

### Cross-asset features

| Option | Description | Selected |
|--------|-------------|----------|
| Per-asset only | Each asset sees only its own features | |
| Include BTC as reference asset | BTC return/vol added as extra feature for every asset | |
| Claude's discretion | Defer to Phase 3 | ✓ |

**User's choice:** Claude's discretion — deferred to Phase 3

---

## Train/Val/Test Split Design

### Split type

| Option | Description | Selected |
|--------|-------------|----------|
| Date-based cutoffs | Explicit date strings as config | ✓ |
| Ratio-based (70/10/20) | Fractions of total dates | |
| Both forms supported | Accept either, convert to dates internally | |

**User's choice:** Date-based cutoffs

### Validation split

| Option | Description | Selected |
|--------|-------------|----------|
| Train + test only (2-way) | Simplest; validation within train via CV is Phase 3/4 concern | |
| Train + val + test (3-way) | Evolution on train, hyperparams on val, final OOS on test | ✓ |
| Train + test with optional val window | WalkForwardSplitter supports both 2-way and 3-way | |

**User's choice:** 3-way split

### Default dates

| Option | Description | Selected |
|--------|-------------|----------|
| Train: 2021–2022, Val: 2023, Test: 2024–2025 | 2yr/1yr/2yr | |
| Train: 2021–2023, Val: 2024-H1, Test: 2024-H2 to 2025 | 3yr/6mo/18mo | ✓ |
| Claude's discretion | Defer exact dates | |

**User's choice:** Train 2021-01-01–2023-12-31, Val 2024-01-01–2024-06-30, Test 2024-07-01–2025-12-31

---

## NaN Handling & Lookback Strategy

### Lookback period NaNs

| Option | Description | Selected |
|--------|-------------|----------|
| Trim: drop first N rows (N = max lookback window) | Guaranteed zero NaN output | ✓ |
| Forward-fill then trim | Preserves more of the early date range | |
| Raise ValueError if NaN found | Fail-fast, caller responsible for enough history | |

**User's choice:** Trim first N rows (N=20, the max rolling window)

### Mid-sample gaps

| Option | Description | Selected |
|--------|-------------|----------|
| Forward-fill up to 3 days (rp_pca pattern) | ffill(limit=3); consistent with existing code | ✓ |
| Drop asset if any mid-sample gap | Zero tolerance | |
| Claude's discretion on gap tolerance | Use min_obs_fraction logic | |

**User's choice:** Forward-fill up to 3 days (same as data_pipeline_example/processor.py)

---

## Claude's Discretion

- Exact OBV-style signal formula
- Parkinson volatility formula choice
- ATR lookback window
- DataConfig defaults beyond dates
- CONTRIBUTING.md structure

## Deferred Ideas

- Cross-asset features (BTC return as global factor) — deferred to Phase 3
- On-chain metrics — unknown if present in Binance parquets; deferred
- Feature expansion beyond selected groups — future additive work
- Preprocessed feature cache (DATA-V2-02) — v2 scope
