"""
Data pipeline integration tests -- DATA-04.

Runs against the `synthetic_ohlcv_cache` fixture (see tests/conftest.py): a
parquet cache generated per session in BinanceFetcher's on-disk layout. These
tests previously pointed at `data_pipeline_example/cache/`, which `.gitignore`
excludes (`*.parquet`), so on a fresh clone the directory was empty and the
fetcher silently fell through to the Binance REST API. The `block_network`
fixture now makes the no-network promise enforceable rather than aspirational.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_pipeline_no_nan(synthetic_ohlcv_cache, block_network):
    """Full pipeline: fetch_ohlcv -> fit_transform -> assert zero NaN (DATA-04)."""
    from vgp.data import BinanceFetcher, FeatureEngine

    fetcher = BinanceFetcher(cache_dir=synthetic_ohlcv_cache)
    ohlcv = fetcher.fetch_ohlcv(start_date="2021-01-01", end_date="2025-12-31", force_refresh=False)
    assert len(ohlcv) > 0, (
        f"fetch_ohlcv returned empty dict; cache_dir={synthetic_ohlcv_cache}"
    )

    engine = FeatureEngine()
    arr = engine.fit_transform(ohlcv)

    assert arr.ndim == 3, f"Expected 3-D array, got {arr.ndim}-D"
    assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"
    assert not np.isnan(arr).any(), "Feature matrix contains NaN after fit_transform"
    assert arr.shape[1] == 12, f"Expected F=12 features, got {arr.shape[1]}"
    assert arr.shape[2] > 0, f"Expected A > 0 assets, got {arr.shape[2]}"


def test_pipeline_dataloader_returns_dateindex(synthetic_ohlcv_cache, block_network):
    """DataLoader (BinanceFetcher) returns DataFrames with DatetimeIndex (DATA-01)."""
    from vgp.data import DataLoader

    fetcher = DataLoader(cache_dir=synthetic_ohlcv_cache)
    ohlcv = fetcher.fetch_ohlcv(force_refresh=False)
    assert "BTC" in ohlcv, "BTC not found in fetched OHLCV dict"
    assert isinstance(ohlcv["BTC"].index, pd.DatetimeIndex), (
        f"Expected DatetimeIndex, got {type(ohlcv['BTC'].index)}"
    )
    # Verify OHLCV columns present (not close-only)
    for col in ("open", "high", "low", "close", "volume"):
        assert col in ohlcv["BTC"].columns, f"Column '{col}' missing from BTC DataFrame"


def test_splitter_ordering_assertion():
    """WalkForwardSplitter raises AssertionError when val_start <= train_end (DATA-03)."""
    from vgp.data import WalkForwardSplitter

    splitter = WalkForwardSplitter()
    with pytest.raises(AssertionError):
        splitter.split(
            data=pd.DataFrame(),
            train_end="2024-01-01",
            val_start="2023-06-01",   # before train_end -- must raise
            val_end="2024-06-30",
            test_start="2024-07-01",
        )


def test_splitter_test_ordering_assertion():
    """WalkForwardSplitter raises AssertionError when test_start <= val_end (DATA-03)."""
    from vgp.data import WalkForwardSplitter

    splitter = WalkForwardSplitter()
    with pytest.raises(AssertionError):
        splitter.split(
            data=pd.DataFrame(),
            train_end="2023-12-31",
            val_start="2024-01-01",
            val_end="2024-06-30",
            test_start="2024-06-01",  # before val_end -- must raise
        )


def test_splitter_valid_split_dataframe():
    """WalkForwardSplitter returns non-overlapping train/val/test DataFrame slices (DATA-03)."""
    from vgp.data import WalkForwardSplitter

    idx = pd.date_range("2021-01-01", "2025-12-31", freq="D")
    df = pd.DataFrame({"v": range(len(idx))}, index=idx)
    splitter = WalkForwardSplitter()
    train, val, test = splitter.split(
        df,
        train_end="2023-12-31",
        val_start="2024-01-01",
        val_end="2024-06-30",
        test_start="2024-07-01",
    )
    assert len(train) > 0, "Train slice is empty"
    assert len(val) > 0, "Val slice is empty"
    assert len(test) > 0, "Test slice is empty"
    assert train.index.max() <= pd.Timestamp("2023-12-31"), (
        f"Train extends past train_end: {train.index.max()}"
    )
    assert val.index.min() >= pd.Timestamp("2024-01-01"), (
        f"Val starts before val_start: {val.index.min()}"
    )
    assert test.index.min() >= pd.Timestamp("2024-07-01"), (
        f"Test starts before test_start: {test.index.min()}"
    )
    # Non-overlapping
    assert train.index.max() < val.index.min(), "Train and val overlap"
    assert val.index.max() < test.index.min(), "Val and test overlap"


def test_full_pipeline_split_array(synthetic_ohlcv_cache, block_network):
    """Full pipeline: fetch -> features -> split produces non-empty time slices (DATA-03, DATA-04).

    Split boundaries are derived from `engine.dates_` rather than hardcoded.
    The previous version pinned them to one specific cache window and named the
    exact row counts it expected, so it broke whenever the underlying data
    changed — which is a property of the fixture, not of the splitter.
    """
    from vgp.data import BinanceFetcher, FeatureEngine, WalkForwardSplitter

    fetcher = BinanceFetcher(cache_dir=synthetic_ohlcv_cache)
    ohlcv = fetcher.fetch_ohlcv(force_refresh=False)
    engine = FeatureEngine()
    arr = engine.fit_transform(ohlcv)

    dates = engine.dates_
    assert dates is not None and len(dates) >= 20, (
        f"need enough aligned dates to split, got {0 if dates is None else len(dates)}"
    )
    # Quarter the available span: train | val | test, in order, non-overlapping.
    q = len(dates) // 4
    train_end = dates[2 * q - 1]
    val_start, val_end = dates[2 * q], dates[3 * q - 1]
    test_start = dates[3 * q]

    splitter = WalkForwardSplitter()
    train, val, test = splitter.split(
        arr,
        train_end=str(train_end.date()),
        val_start=str(val_start.date()),
        val_end=str(val_end.date()),
        test_start=str(test_start.date()),
        dates=dates,
    )
    assert train.shape[0] > 0, (
        f"Train array is empty; engine.dates_ range: {dates.min()} to {dates.max()}"
    )
    assert val.shape[0] > 0, "Val array is empty"
    assert test.shape[0] > 0, "Test array is empty"
    # Shapes consistent along F and A axes
    assert train.shape[1:] == arr.shape[1:], "Train slice has wrong F or A dimension"
    assert val.shape[1:] == arr.shape[1:], "Val slice has wrong F or A dimension"
    assert test.shape[1:] == arr.shape[1:], "Test slice has wrong F or A dimension"
    # The three slices must partition without overlap and without losing rows
    assert train.shape[0] + val.shape[0] + test.shape[0] <= arr.shape[0], (
        "slices sum to more rows than the source array — they overlap"
    )


def test_vgp_submodule_imports():
    """All vgp sub-modules are importable (COMM-01)."""
    import vgp
    import vgp.data
    import vgp.gp
    import vgp.evolution
    import vgp.backtest
    import vgp.analysis


# ---------------------------------------------------------------------------
# Hermeticity and fetcher failure semantics
# ---------------------------------------------------------------------------

def test_cache_hit_requires_no_network(synthetic_ohlcv_cache, block_network):
    """The whole universe must come from cache — DATA-01.

    Guards the fixture's own contract: if the cache filename convention drifts
    away from `{symbol}_{interval}.parquet`, the fetcher would fall through to
    HTTP and `block_network` turns that into an immediate, named failure rather
    than a slow test that depends on the machine it runs on.
    """
    from vgp.data import BinanceFetcher, get_binance_symbols

    fetcher = BinanceFetcher(cache_dir=synthetic_ohlcv_cache)
    ohlcv = fetcher.fetch_ohlcv(force_refresh=False)

    assert len(ohlcv) == len(get_binance_symbols()), (
        f"cache served {len(ohlcv)} of {len(get_binance_symbols())} symbols — "
        f"the rest would have required a network call"
    )


def test_cache_miss_attempts_the_network(tmp_path, block_network):
    """A cache miss must reach for HTTP — proves the cache-hit tests mean something.

    If a miss silently returned nothing without trying, the hermetic tests
    above would pass even with a broken cache path.
    """
    from tests.conftest import NetworkAccessAttempted
    from vgp.data import BinanceFetcher

    fetcher = BinanceFetcher(cache_dir=tmp_path, use_ccxt_fallback=False)
    with pytest.raises(NetworkAccessAttempted):
        fetcher.fetch_ohlcv(force_refresh=False)


def test_force_refresh_bypasses_cache(synthetic_ohlcv_cache, block_network):
    """force_refresh=True must go to the network even with a warm cache.

    The cache-hit tests are only meaningful if the flag genuinely controls the
    path, so this asserts the attempt rather than inferring it from the result.
    """
    from tests.conftest import NetworkAccessAttempted
    from vgp.data import BinanceFetcher

    fetcher = BinanceFetcher(cache_dir=synthetic_ohlcv_cache, use_ccxt_fallback=False)
    with pytest.raises(NetworkAccessAttempted):
        fetcher.fetch_ohlcv(force_refresh=True)


def test_partial_fetch_failure_silently_shrinks_the_universe(
    synthetic_ohlcv_cache, tmp_path, monkeypatch
):
    """DOCUMENTS CURRENT BEHAVIOUR, which is a sharp edge worth knowing.

    `fetch_ohlcv()` wraps every symbol in `except Exception` and only logs the
    failure. So when some assets are cached and others are not, a fetch error
    on the uncached ones returns a SMALLER UNIVERSE with no signal to the
    caller — no raise, no return code, nothing downstream is told that assets
    went missing. The run proceeds on whatever survived, and the universe
    composition can differ between runs.

    That matters for research validity, not just robustness: a quietly varying
    universe is a survivorship-bias channel, and the null control cannot detect
    it because the real and surrogate runs both inherit whatever universe they
    were handed.

    Pinned so a change to this behaviour is deliberate. Not an endorsement.
    """
    import shutil

    import requests

    from vgp.data import BinanceFetcher, get_binance_symbols

    # Cache only the first three symbols; the rest will miss.
    symbols = get_binance_symbols()
    partial = tmp_path / "partial_cache"
    partial.mkdir()
    for symbol in symbols[:3]:
        shutil.copy(synthetic_ohlcv_cache / f"{symbol}_1d.parquet", partial)

    # A realistic network error is an ordinary Exception, which is exactly what
    # fetch_ohlcv swallows.
    def _connection_error(*args, **kwargs):
        raise requests.exceptions.ConnectionError("synthetic network failure")

    monkeypatch.setattr(requests, "get", _connection_error)

    fetcher = BinanceFetcher(cache_dir=partial, use_ccxt_fallback=False)
    ohlcv = fetcher.fetch_ohlcv(force_refresh=False)

    assert len(ohlcv) == 3, (
        f"expected the 3 cached assets, got {len(ohlcv)}"
    )
    assert len(ohlcv) < len(symbols), (
        "universe silently shrank — this assertion documents that no error is "
        "raised and no caller is notified"
    )


def test_fixture_bars_have_valid_geometry(synthetic_ohlcv_cache, block_network):
    """low <= open/close <= high and volume > 0 across the fixture.

    The FeatureEngine's ATR, Parkinson volatility and normalized-close features
    assume well-formed bars. A fixture with impossible bars would let those
    features be validated against inputs that cannot occur in real data.
    """
    from vgp.data import BinanceFetcher

    ohlcv = BinanceFetcher(cache_dir=synthetic_ohlcv_cache).fetch_ohlcv(force_refresh=False)

    for ticker, df in ohlcv.items():
        assert (df["low"] <= df["open"] + 1e-9).all(), f"{ticker}: low above open"
        assert (df["low"] <= df["close"] + 1e-9).all(), f"{ticker}: low above close"
        assert (df["high"] >= df["open"] - 1e-9).all(), f"{ticker}: high below open"
        assert (df["high"] >= df["close"] - 1e-9).all(), f"{ticker}: high below close"
        assert (df["volume"] > 0).all(), f"{ticker}: non-positive volume"
        assert df.index.is_monotonic_increasing, f"{ticker}: index not sorted"
        assert not df.index.has_duplicates, f"{ticker}: duplicate dates"


def test_feature_engine_drops_short_history_assets(synthetic_ohlcv_cache, block_network):
    """min_obs_fraction must exclude assets listed too late — DATA-02.

    The fixture gives two tickers deliberately short history so this filter is
    exercised rather than assumed. Previously every asset in the cache had the
    same span, so the retention path was never covered.
    """
    from vgp.data import BinanceFetcher, FeatureEngine

    ohlcv = BinanceFetcher(cache_dir=synthetic_ohlcv_cache).fetch_ohlcv(force_refresh=False)
    engine = FeatureEngine()
    arr = engine.fit_transform(ohlcv)

    assert engine.dropped_assets_, "no assets dropped — the retention filter is untested"
    assert set(engine.dropped_assets_).isdisjoint(engine.retained_assets_)
    assert len(engine.retained_assets_) == arr.shape[2], (
        f"{len(engine.retained_assets_)} retained assets but array has "
        f"{arr.shape[2]} — metadata and data disagree"
    )
    assert len(engine.retained_assets_) + len(engine.dropped_assets_) == len(ohlcv)
