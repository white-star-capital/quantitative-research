"""Shared pytest fixtures.

DATA PIPELINE FIXTURES
----------------------
The data pipeline tests were written against parquet files in
`data_pipeline_example/cache/`, but `.gitignore` excludes `*.parquet`, so those
files were never committed. On any fresh clone the directory is empty,
`BinanceFetcher._fetch_symbol()` sees a cache miss and falls through to the
Binance REST API — despite the tests' docstring promising "no network access
required". They passed only on a machine that happened to have the cache
populated, and failed everywhere else (in a sandbox or CI, at the proxy).

The fix is to generate the cache as a test fixture instead of depending on
files that cannot be committed: `synthetic_ohlcv_cache` writes deterministic
parquet files in exactly the layout the fetcher expects, so the tests exercise
the real cache-read path with no network and no repository binaries.

`block_network` then makes the promise structural rather than aspirational: any
test that reaches for HTTP fails immediately with a clear message instead of
hanging, flaking, or quietly depending on someone's local cache.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Enough history for the 20-day rolling windows, the lookback trim, and
# walk-forward splits with room to spare.
_FIXTURE_START = "2021-01-01"
_FIXTURE_END = "2025-12-31"

# A couple of assets get deliberately short history so the fixture also
# exercises FeatureEngine's min_obs_fraction retention filter rather than
# handing it a uniformly perfect panel.
_SHORT_HISTORY_TICKERS = ("WLFI", "SYRUP")
_SHORT_HISTORY_START = "2025-06-01"


def _synthetic_ohlcv(
    n_rows: int,
    seed: int,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Deterministic OHLCV bars with realistic geometry.

    Prices follow a geometric random walk; open/high/low are placed around
    each close so that low <= min(open, close) and high >= max(open, close),
    which is what ATR and the normalized-close feature assume. Volume is
    strictly positive so the volume-ratio feature never divides by zero.
    """
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal(n_rows) * 0.02
    close = 100.0 * np.exp(np.cumsum(returns))

    # Open is the previous close nudged slightly; first open equals first close.
    open_ = np.empty(n_rows)
    open_[0] = close[0]
    open_[1:] = close[:-1] * (1.0 + rng.standard_normal(n_rows - 1) * 0.002)

    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    high = body_hi * (1.0 + np.abs(rng.standard_normal(n_rows)) * 0.01)
    low = body_lo * (1.0 - np.abs(rng.standard_normal(n_rows)) * 0.01)
    volume = np.abs(rng.standard_normal(n_rows)) * 1e6 + 1e5

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    df.index.name = "date"
    return df


@pytest.fixture(scope="session")
def synthetic_ohlcv_cache(tmp_path_factory) -> Path:
    """A parquet cache in BinanceFetcher's on-disk layout, built from scratch.

    Files are named `{symbol}_{interval}.parquet` (e.g. `BTCUSDT_1d.parquet`)
    to match `BinanceFetcher._fetch_symbol()`'s cache_path, so a fetcher
    pointed here gets cache hits for the whole universe and never calls out.

    Session-scoped: the panel is 30 assets x ~1800 rows and is read-only.
    """
    from vgp.data import get_binance_symbols

    cache_dir = tmp_path_factory.mktemp("ohlcv_cache")
    full_index = pd.date_range(_FIXTURE_START, _FIXTURE_END, freq="D")

    for i, symbol in enumerate(get_binance_symbols()):
        ticker = symbol.replace("USDT", "")
        if ticker in _SHORT_HISTORY_TICKERS:
            index = pd.date_range(_SHORT_HISTORY_START, _FIXTURE_END, freq="D")
        else:
            index = full_index
        df = _synthetic_ohlcv(len(index), seed=1000 + i, index=index)
        df.to_parquet(cache_dir / f"{symbol}_1d.parquet")

    return cache_dir


class NetworkAccessAttempted(BaseException):
    """Raised when a test that must be hermetic reaches for the network.

    Deliberately derived from BaseException, not Exception:
    `BinanceFetcher.fetch_ohlcv()` wraps each symbol in `except Exception` and
    only logs the failure, so an ordinary exception here would be swallowed and
    the test would see a quietly empty dict instead of the real cause. A
    BaseException propagates through that handler and names the problem.
    """


@pytest.fixture
def block_network(monkeypatch):
    """Make any outbound HTTP call fail loudly and unmissably.

    Applied to tests that claim to be hermetic. Without this, a missing cache
    file degrades into a network call: slow, dependent on the environment, and
    liable to pass for the wrong reason on a machine that happens to have the
    cache while failing in CI. That is exactly how the data pipeline tests came
    to depend on uncommitted parquet files without anyone noticing.
    """
    import requests

    def _blocked(*args, **kwargs):
        raise NetworkAccessAttempted(
            "This test attempted a network request. It must run entirely "
            "against the synthetic parquet cache — a request here means the "
            "fetcher missed the cache (check the {symbol}_{interval}.parquet "
            "naming in synthetic_ohlcv_cache) or the fixture was not used."
        )

    monkeypatch.setattr(requests, "get", _blocked)
    monkeypatch.setattr(requests, "request", _blocked)
    monkeypatch.setattr(requests.Session, "request", _blocked)
    monkeypatch.setattr(requests.Session, "get", _blocked, raising=False)
    return _blocked
