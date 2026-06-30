# Phase 2: Data Pipeline - Context

**Gathered:** 2026-06-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Produce a validated multi-asset feature matrix from Binance OHLCV data (fetched via REST API and cached as per-asset parquets), structurally enforce the train/validation/test split before any evolution code is written, and establish the vgp/ package module layout with CONTRIBUTING.md.

Downstream GP primitives (Phase 3) will operate on the feature array produced here. The OOS test split defined here must never be touched again until final reporting in Phase 5.

</domain>

<decisions>
## Implementation Decisions

### Data Schema & Layout
- **D-01:** BinanceFetcher pattern from `data_pipeline_example/` — one parquet file per asset (`{SYMBOL}_{interval}.parquet`, e.g., `BTCUSDT_1d.parquet`), cached in `vgp/data/cache/` (gitignored). DataLoader wraps BinanceFetcher to fetch full OHLCV for all universe assets.
- **D-02:** Universe: UNIVERSE_30 (30 crypto/USDT pairs on Binance), identical to `data_pipeline_example/universe.py`. Copy `universe.py` directly into `vgp/data/universe.py`.
- **D-03:** DataLoader returns full OHLCV as `dict[str, pd.DataFrame]` keyed by short ticker (not close-only). FeatureEngine needs OHLCV columns (open, high, low, close, volume) to compute ATR, Parkinson vol, volume ratio.
- **D-04:** Mid-sample price gaps: forward-fill up to 3 consecutive missing days (`ffill(limit=3)`) before feature computation. Same pattern as `data_pipeline_example/processor.py`. Assets missing more than 20% of total dates are dropped (min_obs_fraction=0.80).

### Feature Engine
- **D-05:** FeatureEngine output: float32 `[T × F × A]` numpy array where T=timesteps after lookback trim, F=number of features, A=30 assets. Features are identical across all assets (same F for every asset index).
- **D-06:** Feature groups — all four selected:
  - **Momentum**: 1-day returns, 5-day returns, 20-day returns, log close prices
  - **Volatility**: rolling std 5d, rolling std 20d, ATR (14), Parkinson volatility (14)
  - **Oscillators**: RSI (14), normalized close position 0–1 (close relative to rolling min/max)
  - **Volume**: volume ratio (vol / 20d rolling average volume), OBV-style directional signal
- **D-07:** Per-asset features only. Cross-asset features (e.g., BTC return as a global factor) deferred to Phase 3 when the GP primitive set design determines whether cross-asset inputs are needed.
- **D-08:** Lookback trim: drop the first N rows where N = max rolling window across all features (N=20 for rolling std 20d and returns 20d). Trim applied after all features are computed, guaranteeing zero NaN in output. Raises `ValueError` if NaN remains after trim as a sanity check.

### Train / Validation / Test Split
- **D-09:** WalkForwardSplitter uses date-based cutoffs (explicit date strings), not ratio-based fractions.
- **D-10:** Default split:
  - Train: 2021-01-01 – 2023-12-31
  - Validation: 2024-01-01 – 2024-06-30
  - Test (OOS holdout): 2024-07-01 – 2025-12-31
- **D-11:** Structural enforcement: WalkForwardSplitter raises `AssertionError` if `val_start <= train_end` or `test_start <= val_end`. Ordering failure is a hard error, not a warning.
- **D-12:** WalkForwardSplitter interface: `split(data, train_end, val_start, val_end, test_start)` returns `(train_data, val_data, test_data)` slices. Works on both DataFrames and `[T × F × A]` arrays by slicing the time axis using the DatetimeIndex.

### Package Layout
- **D-13:** `vgp/data/` module files: `universe.py` (UNIVERSE_30 + get_binance_symbols), `fetcher.py` (BinanceFetcher adapted from example), `feature_engine.py` (FeatureEngine), `splitter.py` (WalkForwardSplitter), `config.py` (DataConfig with date defaults).
- **D-14:** CONTRIBUTING.md covers: how to add a GP primitive (link to Phase 3 code when it exists), how to run an experiment (link to Phase 4 entry point), how to update the dependency lock file (`pip-compile`). Place at repo root.

### Claude's Discretion
- Exact OBV-style signal formula (directional: cumsum of sign(return) × volume is acceptable)
- Parkinson volatility formula (standard: using log(high/low))
- ATR lookback window (14 is the standard; may use 14 for both ATR and Parkinson)
- DataConfig class structure beyond dates (cache_dir, interval, min_obs_fraction defaults)
- CONTRIBUTING.md prose style and section structure

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Reference Implementation (adapt, do not reimport)
- `data_pipeline_example/fetcher.py` — BinanceFetcher: REST pagination, Parquet cache pattern, CCXT fallback, `_parse_klines()`. Copy logic into `vgp/data/fetcher.py`; adapt to return full OHLCV dict rather than close-only DataFrame.
- `data_pipeline_example/processor.py` — ReturnProcessor: ffill(limit=3) gap handling, min_obs_fraction asset dropping, winsorization. Port the gap-handling logic into FeatureEngine; do NOT port return computation (FeatureEngine computes features, not just returns).
- `data_pipeline_example/universe.py` — UNIVERSE_30 list and get_binance_symbols(). Copy verbatim into `vgp/data/universe.py`.

### Phase Requirements
- `.planning/REQUIREMENTS.md` §Data Pipeline — DATA-01 through DATA-04: exact acceptance criteria for DataLoader, FeatureEngine, WalkForwardSplitter, and fixture validation.
- `.planning/REQUIREMENTS.md` §Community & Repository — COMM-01 (vgp sub-module importability), COMM-03 (CONTRIBUTING.md content).

### Cached Fixture Data
- `data_pipeline_example/cache/` — 27 per-asset parquet files covering most of UNIVERSE_30 for 2021–2025 at 1d interval. Use these as the Phase 2 test fixture (DATA-04) — no download required in CI.

### Architecture Constraints
- `vgp/__init__.py` — architecture invariants: `vgp.evolution` must NOT import vectorbt; `vgp.backtest` must NOT import deap. Data module has no such restrictions.
- `CLAUDE.md` §Critical Technical Constraints — numpy<2.3 pin, pandas 3.0 idioms (`.to_numpy()` not `.values`, explicit `.copy()`, no chained assignment).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `data_pipeline_example/fetcher.py`: Complete BinanceFetcher implementation — REST pagination, per-symbol Parquet cache, CCXT fallback. Port directly into `vgp/data/fetcher.py`. Only change needed: expose `fetch_ohlcv()` as the primary API (returns full OHLCV dict) rather than `fetch_all()` which returns close-only.
- `data_pipeline_example/universe.py`: UNIVERSE_30 list with `assert len() == 30` guard and `get_binance_symbols()`. Copy verbatim.
- `data_pipeline_example/cache/*.parquet`: 27 asset parquet files ready to use as test fixtures without network access.

### Established Patterns
- Gap handling: `ffill(limit=3)` then drop columns below `min_obs_fraction=0.80` — validated in rp_pca research, carry this into FeatureEngine.
- Parquet cache: `{cache_dir}/{SYMBOL}_{interval}.parquet`, read with range check before downloading. Prevents redundant API calls.
- pandas 3.0: all DataFrame code must use `.to_numpy()` (not `.values`), explicit `.copy()`, no chained assignment.

### Integration Points
- `vgp/data/__init__.py` stub exists — add public exports: `DataLoader`, `FeatureEngine`, `WalkForwardSplitter`, `UNIVERSE_30`.
- Phase 3 (GP Core) will consume the `[T × F × A]` array from FeatureEngine and the split boundaries from WalkForwardSplitter. The interface between phases is: numpy array + date indices out of Phase 2.
- Tests in `tests/` — Phase 2 adds `tests/test_data_pipeline.py` covering the full fixture test (DATA-04 success criterion).

</code_context>

<specifics>
## Specific Ideas

- "Build a simple pipeline first, then expand later" — start from the rp_pca/data pattern, adapt it, don't overengineer.
- The existing `data_pipeline_example/cache/` parquet files are the test fixture — use them directly for DATA-04 so the pipeline test runs in CI without network access.
- The user's existing research pipeline (`data_pipeline_example/`) is the direct ancestor of `vgp/data/`. Treat it as a reference implementation to port and extend, not a competitor design.

</specifics>

<deferred>
## Deferred Ideas

- Cross-asset features (BTC return as global factor) — deferred to Phase 3 when GP primitive set design clarifies whether cross-asset inputs are needed.
- On-chain metrics (if not present in Binance parquets) — STATE.md flags this as unknown; if not in OHLCV, defer to a future data expansion phase.
- Feature expansion beyond the 4 groups selected — Phase 2 establishes the pipeline; feature additions are additive and don't require rework.
- Caching layer for preprocessed feature matrices (DATA-V2-02 in REQUIREMENTS.md) — explicitly v2 scope.

</deferred>

---

*Phase: 02-data-pipeline*
*Context gathered: 2026-06-08*
