"""
Data pipeline integration tests -- DATA-04.

Uses data_pipeline_example/cache/ parquet files as the test fixture.
No network access required: all tests run against pre-cached data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURE_CACHE = Path(__file__).parent.parent / "data_pipeline_example" / "cache"


def test_pipeline_no_nan():
    """Full pipeline: fetch_ohlcv -> fit_transform -> assert zero NaN (DATA-04)."""
    from vgp.data import BinanceFetcher, FeatureEngine

    fetcher = BinanceFetcher(cache_dir=FIXTURE_CACHE)
    ohlcv = fetcher.fetch_ohlcv(start_date="2021-01-01", end_date="2025-12-31", force_refresh=False)
    assert len(ohlcv) > 0, f"fetch_ohlcv returned empty dict; cache_dir={FIXTURE_CACHE}"

    engine = FeatureEngine()
    arr = engine.fit_transform(ohlcv)

    assert arr.ndim == 3, f"Expected 3-D array, got {arr.ndim}-D"
    assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"
    assert not np.isnan(arr).any(), "Feature matrix contains NaN after fit_transform"
    assert arr.shape[1] == 12, f"Expected F=12 features, got {arr.shape[1]}"
    assert arr.shape[2] > 0, f"Expected A > 0 assets, got {arr.shape[2]}"


def test_pipeline_dataloader_returns_dateindex():
    """DataLoader (BinanceFetcher) returns DataFrames with DatetimeIndex (DATA-01)."""
    from vgp.data import DataLoader

    fetcher = DataLoader(cache_dir=FIXTURE_CACHE)
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


def test_full_pipeline_split_array():
    """Full pipeline: fetch -> features -> split produces non-empty time slices (DATA-03, DATA-04).

    Uses date boundaries within the actual cache window (2024-05-01 to 2025-12-31):
      train_end="2025-01-31", val_start="2025-02-01", val_end="2025-06-30", test_start="2025-07-01"
    These produce: train=276 rows, val=150 rows, test=184 rows.
    """
    from vgp.data import BinanceFetcher, FeatureEngine, WalkForwardSplitter

    fetcher = BinanceFetcher(cache_dir=FIXTURE_CACHE)
    ohlcv = fetcher.fetch_ohlcv(force_refresh=False)
    engine = FeatureEngine()
    arr = engine.fit_transform(ohlcv)

    splitter = WalkForwardSplitter()
    train, val, test = splitter.split(
        arr,
        train_end="2025-01-31",
        val_start="2025-02-01",
        val_end="2025-06-30",
        test_start="2025-07-01",
        dates=engine.dates_,
    )
    assert train.shape[0] > 0, (
        f"Train array is empty; engine.dates_ range: {engine.dates_.min()} to {engine.dates_.max()}"
    )
    assert val.shape[0] > 0, "Val array is empty"
    assert test.shape[0] > 0, "Test array is empty"
    # Shapes consistent along F and A axes
    assert train.shape[1:] == arr.shape[1:], "Train slice has wrong F or A dimension"
    assert val.shape[1:] == arr.shape[1:], "Val slice has wrong F or A dimension"


def test_vgp_submodule_imports():
    """All vgp sub-modules are importable (COMM-01)."""
    import vgp
    import vgp.data
    import vgp.gp
    import vgp.evolution
    import vgp.backtest
    import vgp.analysis
