---
phase: 02-data-pipeline
reviewed: 2026-06-08T00:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - vgp/data/universe.py
  - vgp/data/config.py
  - vgp/data/fetcher.py
  - vgp/data/feature_engine.py
  - vgp/data/splitter.py
  - vgp/data/__init__.py
  - tests/test_data_pipeline.py
  - CONTRIBUTING.md
findings:
  critical: 3
  warning: 6
  info: 4
  total: 13
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-06-08T00:00:00Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

The data pipeline is well-structured overall. Architecture boundaries are clean (no vectorbt or deap imports anywhere in `vgp/data/`), pandas 3.0 idioms are followed correctly in most places, and the lookahead risk is low — all features use only `.pct_change()`, `.rolling()`, and `.diff()` on historical data with no look-ahead indexing.

Three critical issues require attention before this code runs in any research context:

1. **Path traversal**: the cache directory is constructed from caller-supplied input without sanitisation, enabling a directory traversal attack.
2. **Crash on empty OHLCV input**: `fit_transform()` accesses `per_asset_trimmed[0]` unconditionally; a call with an empty `ohlcv` dict (all assets dropped) raises `IndexError` rather than a clear error.
3. **OBV global std uses full history**: `obv_raw.std()` is a scalar computed over the entire time series, so the z-score normalisation at time `t` implicitly uses future values. This is a lookahead violation for any downstream live or walk-forward usage.

Six warnings cover missing retry logic for transient network errors, a floating-point comparison against `0.0` in the RSI guard that can miss near-zero denominators, an incorrect ticker-stripping assumption in `fetch_all`, an unguarded empty-DataFrame path in the splitter test, and two test gaps.

---

## Critical Issues

### CR-01: Path traversal via unsanitised `cache_dir` input

**File:** `vgp/data/fetcher.py:257`
**Issue:** `cache_path` is built by joining `self.cache_dir` (caller-supplied) with a filename derived from the `symbol` parameter. `symbol` comes from `get_binance_symbols()` and is therefore safe for now, but `cache_dir` is passed in at construction time from `DataConfig.cache_dir` — or directly by any caller. A caller can supply `cache_dir=Path("/tmp/../../etc")` and the fetcher will happily read or overwrite files outside the intended cache directory. More concretely, `force_refresh=False` means the fetcher reads whatever Parquet is at the constructed path; a path-traversal value in `cache_dir` could cause it to load and deserialise arbitrary Parquet files.

```python
# fetcher.py line 257 — current
cache_path = self.cache_dir / f"{symbol}_{interval}.parquet"

# Fix: resolve and assert the final path stays inside cache_dir
cache_path = (self.cache_dir / f"{symbol}_{interval}.parquet").resolve()
assert str(cache_path).startswith(str(self.cache_dir.resolve())), (
    f"Resolved cache path {cache_path} escapes cache_dir {self.cache_dir}"
)
```

Additionally, validate `cache_dir` in `__init__` at construction time:

```python
def __init__(self, cache_dir: Path, ...) -> None:
    self.cache_dir = Path(cache_dir).resolve()
    self.cache_dir.mkdir(parents=True, exist_ok=True)
    ...
```

---

### CR-02: `IndexError` crash when all assets are dropped by `fit_transform`

**File:** `vgp/data/feature_engine.py:155`
**Issue:** After the observation-threshold filter (Step 3) and the lookback trim (Step 5), `per_asset_trimmed[0]` is accessed unconditionally to seed the intersection index. If every asset fails the `min_obs_fraction` check, `keep` is empty, `per_asset_trimmed` is an empty list, and line 155 raises an `IndexError` with no diagnostic context. The same crash occurs if every retained asset's trimmed DataFrame is empty (e.g., the date range is shorter than `lookback`).

```python
# feature_engine.py line 155 — current
common_idx = per_asset_trimmed[0].index   # IndexError if list is empty

# Fix: guard with an early ValueError
if not per_asset_trimmed:
    raise ValueError(
        f"FeatureEngine: no assets survived the min_obs_fraction={self.min_obs_fraction} "
        f"filter. Received {len(ohlcv)} assets; all were below the "
        f"{self.min_obs_fraction * 100:.0f}% observation threshold."
    )
```

---

### CR-03: OBV z-score is a lookahead (uses future std/mean)

**File:** `vgp/data/feature_engine.py:272-277`
**Issue:** The OBV signal is computed as a global z-score: `(obv_raw - obv_raw.mean()) / obv_raw.std()`. Both `.mean()` and `.std()` are scalars derived from the entire time series, meaning every historical value at time `t` is normalised using statistics that include data from `t+1` onward. This is a lookahead violation. In an offline research pipeline where the full history is always present the signal is stationary, but the moment this feature is used in any walk-forward or live context, the z-score shifts as future data accumulates, making historical and live feature values incomparable and inflating in-sample metrics.

The fix is to use an expanding or rolling window for both mean and std:

```python
# feature_engine.py lines 272-277 — current (lookahead)
obv_raw = (np.sign(close.pct_change(1)) * volume).cumsum()
obv_std = obv_raw.std()
if obv_std == 0.0 or np.isnan(obv_std):
    obv_signal = pd.Series(0.0, index=close.index)
else:
    obv_signal = (obv_raw - obv_raw.mean()) / obv_std

# Fix: use expanding window (consistent with the cumulative nature of OBV)
obv_raw = (np.sign(close.pct_change(1)) * volume).cumsum()
obv_mean = obv_raw.expanding(min_periods=1).mean()
obv_std_s = obv_raw.expanding(min_periods=2).std()
obv_signal = (obv_raw - obv_mean) / obv_std_s
# Guard: where std is zero or NaN, emit 0.0
obv_signal = obv_signal.where(obv_std_s.notna() & (obv_std_s != 0.0), other=0.0)
```

---

## Warnings

### WR-01: No retry logic for transient network errors in `_paginated_download`

**File:** `vgp/data/fetcher.py:308-309`
**Issue:** `requests.get(...).raise_for_status()` propagates any HTTP error (including 429 rate-limit, 500 server error, and transient timeouts) directly as an exception. The outer `fetch_ohlcv` loop catches all exceptions and logs a warning, which means a single rate-limit response permanently skips that symbol for the run. On a 30-symbol universe this can silently under-populate the feature matrix.

```python
# Fix: add exponential back-off retry around the request
import time

MAX_RETRIES = 5
for attempt in range(MAX_RETRIES):
    try:
        resp = requests.get(BINANCE_REST, params=params, timeout=30)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 2 ** attempt))
            logger.warning("Rate limited; sleeping %ds (attempt %d)", retry_after, attempt)
            time.sleep(retry_after)
            continue
        resp.raise_for_status()
        break
    except requests.exceptions.ConnectionError as exc:
        if attempt == MAX_RETRIES - 1:
            raise
        time.sleep(2 ** attempt)
```

---

### WR-02: Floating-point exact comparison in RSI denominator guard

**File:** `vgp/data/feature_engine.py:254`
**Issue:** `rsi_14.where(denom_rsi != 0.0, other=50.0)` uses exact equality to detect a zero denominator. In IEEE 754 arithmetic `up + dn` can be a very small positive float (e.g., 1e-15) when both deltas are sub-penny — the guard does not fire and `100 * up / denom_rsi` returns a near-zero or near-100 value that is technically valid but numerically unreliable. An epsilon guard is safer.

```python
# Fix
rsi_14 = rsi_14.where(denom_rsi > 1e-10, other=50.0)
```

The same pattern applies to `norm_close` (line 262) and `vol_ratio_20d` (line 269), though prices and volumes are unlikely to be sub-cent, so RSI is the highest-risk site.

---

### WR-03: Ticker stripping in `fetch_all` breaks for non-USDT pairs

**File:** `vgp/data/fetcher.py:124`
**Issue:** Ticker names are derived with `symbol.replace("USDT", "")`. If a symbol happens to contain the string "USDT" in its base ticker (e.g., a hypothetical "USTDT" → strips to "T"), or if someone initialises the fetcher with non-USDT quote (e.g., `get_binance_symbols(quote="BTC")`), the replacement produces wrong ticker names. The same pattern appears in `fetch_ohlcv` (line 93) and the CCXT block. The correct approach is to strip only a known suffix:

```python
# Fix
ticker = symbol.removesuffix(quote) if quote else symbol.replace("USDT", "")
# or more robustly, store quote on self and use it during strip
```

Since `DataConfig` hard-codes `"USDT"` and the universe only has known tickers, this is low probability in practice but a real bug waiting for the first non-USDT experiment.

---

### WR-04: `splitter` test with empty DataFrame succeeds vacuously

**File:** `tests/test_data_pipeline.py:58-63`
**Issue:** `test_splitter_ordering_assertion` passes `data=pd.DataFrame()` (a completely empty DataFrame with no index). The `split()` method receives this and, before the assertion fires, accesses `data.index` — an empty `RangeIndex`, not a `DatetimeIndex`. The assertion at line 81 (`pd.Timestamp(val_start) > pd.Timestamp(train_end)`) is evaluated first and does raise as expected, so the test passes. However, the fixture is misleading: if the assertion is ever removed or weakened, the boolean mask on a `RangeIndex` will not raise, it will silently return empty slices. The test should use a realistic DataFrame with a `DatetimeIndex` to be a true regression test.

```python
# Fix: use a realistic fixture
df = pd.DataFrame({"v": [1]}, index=pd.DatetimeIndex(["2023-01-01"]))
splitter.split(data=df, train_end="2024-01-01", val_start="2023-06-01", ...)
```

---

### WR-05: `fetch_all` and `fetch_ohlcv` do not share the CCXT fallback

**File:** `vgp/data/fetcher.py:103-160`
**Issue:** `fetch_ohlcv()` (the primary public API) has no CCXT fallback path at all — only `fetch_all()` (the deprecated close-only method) has the fallback logic. If a symbol fails the Binance REST request during `fetch_ohlcv()`, it is silently skipped. A research run that calls `fetch_ohlcv()` in a region where Binance is geo-blocked will get a partial OHLCV dict with no warning about fallback availability. Given that `fetch_ohlcv` is documented as "the primary public API", the CCXT fallback logic (lines 133-154) should be replicated or extracted into a shared helper called by both methods.

---

### WR-06: `per_asset_trimmed` alignment assumes sorted `DatetimeIndex`

**File:** `vgp/data/feature_engine.py:116-160`
**Issue:** `np.searchsorted` in `WalkForwardSplitter` (splitter.py line 116) assumes `dates` is sorted. `FeatureEngine.dates_` is assigned from `per_asset_trimmed[0].index` (line 189), which is sorted because `common_idx = common_idx.sort_values()` (line 158). However, `feature_engine.py` line 155-158 only calls `sort_values()` after building `common_idx` by intersecting the first asset's index with the rest. The individual `per_asset_trimmed` DataFrames are only sorted implicitly (because the raw OHLCV is sorted in `_fetch_symbol`). If the raw OHLCV for any asset arrives with an unsorted index (possible from a custom `ohlcv` dict passed directly to `fit_transform()`), `common_idx` will be unsorted until `sort_values()` is called. The real risk is that `feat_df.loc[common_idx]` on an unsorted `feat_df` will silently reorder rows. Add an explicit sort guard:

```python
# After Step 4, before trimming:
df_raw = df_raw.sort_index()   # already present — good
# After trim, before intersection:
per_asset_trimmed = [feat_df.sort_index() for feat_df in per_asset_trimmed]
```

This is already partially addressed by the `sort_index()` on `df_raw` (line 135), but the explicit guard on the trimmed list makes the invariant undeniable.

---

## Info

### IN-01: `DataConfig.cache_dir` is a relative path

**File:** `vgp/data/config.py:22`
**Issue:** The default `cache_dir = Path("vgp/data/cache")` is a relative path. Its resolution depends on the process working directory at runtime. If `DataConfig` is instantiated from a script not run from the project root (e.g., a Jupyter notebook in a subdirectory, or a `pytest` invocation with a non-standard rootdir), the cache is written to an unexpected location. Prefer an absolute path anchored to the package:

```python
from pathlib import Path
# At module level
_HERE = Path(__file__).parent
cache_dir: Path = field(default_factory=lambda: _HERE / "cache")
```

---

### IN-02: `test_vgp_submodule_imports` will fail until Phase 3-5 are complete

**File:** `tests/test_data_pipeline.py:145-153`
**Issue:** The test imports `vgp.gp`, `vgp.evolution`, `vgp.backtest`, and `vgp.analysis`, which do not yet exist. This test will always fail in the current state. It should be marked with `@pytest.mark.skip` or `xfail` until those phases land, or moved to an integration test suite.

```python
@pytest.mark.skip(reason="Phases 3-5 not yet implemented")
def test_vgp_submodule_imports():
    ...
```

---

### IN-03: No test for `fetch_ohlcv` CCXT gap (CR-01 consequence)

**File:** `tests/test_data_pipeline.py`
**Issue:** There is no test that exercises the CCXT fallback path, nor any test that verifies a partial OHLCV dict (some symbols missing) is handled gracefully by `FeatureEngine`. Adding a fixture with one asset's Parquet removed would cover the `min_obs_fraction` filter and the dropped-asset logging.

---

### IN-04: `SLEEP_BETWEEN_CALLS` is only applied between pages, not between symbols

**File:** `vgp/data/fetcher.py:318`
**Issue:** `time.sleep(SLEEP_BETWEEN_CALLS)` is only called between paginated requests for a single symbol (inside `_paginated_download`). When iterating across all 30 symbols, there is no inter-symbol sleep — the first request for symbol N+1 fires immediately after the last page of symbol N completes. Under normal conditions the 30-request burst is within Binance's 1200 request/minute weight limit, but during a `force_refresh` run (all 30 symbols, multiple pages each) the burst could approach the limit. A small sleep between symbols in `fetch_ohlcv` would be a safe guard:

```python
# In fetch_ohlcv, after appending to result:
result[ticker] = df
time.sleep(SLEEP_BETWEEN_CALLS)
```

---

_Reviewed: 2026-06-08_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
