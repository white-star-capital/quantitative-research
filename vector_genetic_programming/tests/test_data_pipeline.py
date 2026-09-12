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
    assert len(ohlcv) > 0, f"fetch_ohlcv returned empty dict; cache_dir={synthetic_ohlcv_cache}"

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
    assert isinstance(
        ohlcv["BTC"].index, pd.DatetimeIndex
    ), f"Expected DatetimeIndex, got {type(ohlcv['BTC'].index)}"
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
            val_start="2023-06-01",  # before train_end -- must raise
            val_end="2024-06-30",
            test_start="2024-07-01",
            test_end="2024-12-31",
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
            test_end="2024-12-31",
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
        test_end="2024-12-31",
    )
    assert len(train) > 0, "Train slice is empty"
    assert len(val) > 0, "Val slice is empty"
    assert len(test) > 0, "Test slice is empty"
    assert train.index.max() <= pd.Timestamp(
        "2023-12-31"
    ), f"Train extends past train_end: {train.index.max()}"
    assert val.index.min() >= pd.Timestamp(
        "2024-01-01"
    ), f"Val starts before val_start: {val.index.min()}"
    assert test.index.min() >= pd.Timestamp(
        "2024-07-01"
    ), f"Test starts before test_start: {test.index.min()}"
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
    assert (
        dates is not None and len(dates) >= 20
    ), f"need enough aligned dates to split, got {0 if dates is None else len(dates)}"
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
        test_end=str(dates[-1].date()),
        dates=dates,
    )
    assert (
        train.shape[0] > 0
    ), f"Train array is empty; engine.dates_ range: {dates.min()} to {dates.max()}"
    assert val.shape[0] > 0, "Val array is empty"
    assert test.shape[0] > 0, "Test array is empty"
    # Shapes consistent along F and A axes
    assert train.shape[1:] == arr.shape[1:], "Train slice has wrong F or A dimension"
    assert val.shape[1:] == arr.shape[1:], "Val slice has wrong F or A dimension"
    assert test.shape[1:] == arr.shape[1:], "Test slice has wrong F or A dimension"
    # The three slices must partition without overlap and without losing rows
    assert (
        train.shape[0] + val.shape[0] + test.shape[0] <= arr.shape[0]
    ), "slices sum to more rows than the source array — they overlap"


def test_vgp_submodule_imports():
    """All vgp sub-modules are importable (COMM-01).

    Written as a loop over module names rather than a block of bare `import`
    statements: those read as unused imports to a linter, and deleting them —
    which is the obvious automated "fix" — would silently empty this test of
    its only assertion.
    """
    import importlib

    for name in (
        "vgp",
        "vgp.data",
        "vgp.gp",
        "vgp.evolution",
        "vgp.backtest",
        "vgp.analysis",
    ):
        assert importlib.import_module(name) is not None, f"{name} failed to import"


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


def test_partial_fetch_failure_raises_by_default(synthetic_ohlcv_cache, tmp_path, monkeypatch):
    """A partial fetch must raise, not quietly return a smaller universe.

    This is the behaviour that used to be silent: `fetch_ohlcv()` wrapped every
    symbol in `except Exception` and only logged, so a fetch error on some
    assets returned a SMALLER UNIVERSE with nothing downstream told. A run
    could proceed on 3 assets instead of 30 and report results normally, and
    neither the DSR nor the null control could detect it — every trial and
    every surrogate inherits whatever universe it was handed.
    """
    import shutil

    import requests

    from vgp.data import BinanceFetcher, FetchError, get_binance_symbols

    symbols = get_binance_symbols()
    partial = tmp_path / "partial_cache"
    partial.mkdir()
    for symbol in symbols[:3]:
        shutil.copy(synthetic_ohlcv_cache / f"{symbol}_1d.parquet", partial)

    def _connection_error(*args, **kwargs):
        raise requests.exceptions.ConnectionError("synthetic network failure")

    monkeypatch.setattr(requests, "get", _connection_error)

    fetcher = BinanceFetcher(cache_dir=partial, use_ccxt_fallback=False)
    with pytest.raises(FetchError, match="symbols failed"):
        fetcher.fetch_ohlcv(force_refresh=False)

    # The report survives the raise, so a caller can see exactly what was lost
    report = fetcher.last_fetch_report_
    assert report is not None, "last_fetch_report_ must be set even when raising"
    assert report.n_realized == 3
    assert report.n_failed == len(symbols) - 3
    assert not report.is_complete
    assert "ConnectionError" in dict(report.failed)[symbols[5]]


def test_allow_partial_permits_a_declared_smaller_universe(
    synthetic_ohlcv_cache, tmp_path, monkeypatch
):
    """allow_partial=True proceeds, but the shrink is recorded, not hidden."""
    import shutil

    import requests

    from vgp.data import BinanceFetcher, get_binance_symbols

    symbols = get_binance_symbols()
    partial = tmp_path / "partial_cache"
    partial.mkdir()
    for symbol in symbols[:5]:
        shutil.copy(synthetic_ohlcv_cache / f"{symbol}_1d.parquet", partial)

    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(
            requests.exceptions.ConnectionError("synthetic failure")
        ),
    )

    fetcher = BinanceFetcher(cache_dir=partial, use_ccxt_fallback=False)
    ohlcv = fetcher.fetch_ohlcv(force_refresh=False, allow_partial=True)

    assert len(ohlcv) == 5
    assert fetcher.last_fetch_report_.n_realized == 5
    assert fetcher.last_fetch_report_.n_failed == len(symbols) - 5


def test_min_assets_floor_raises_even_when_partial_allowed(
    synthetic_ohlcv_cache, tmp_path, monkeypatch
):
    """min_assets is a hard floor: below it the run aborts rather than reports."""
    import shutil

    import requests

    from vgp.data import BinanceFetcher, FetchError, get_binance_symbols

    symbols = get_binance_symbols()
    partial = tmp_path / "partial_cache"
    partial.mkdir()
    for symbol in symbols[:3]:
        shutil.copy(synthetic_ohlcv_cache / f"{symbol}_1d.parquet", partial)

    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(
            requests.exceptions.ConnectionError("synthetic failure")
        ),
    )

    fetcher = BinanceFetcher(cache_dir=partial, use_ccxt_fallback=False)
    with pytest.raises(FetchError, match="below the min_assets"):
        fetcher.fetch_ohlcv(force_refresh=False, allow_partial=True, min_assets=10)


def test_empty_universe_always_raises(tmp_path, monkeypatch):
    """Zero assets is never a valid result, whatever the tolerance."""
    import requests

    from vgp.data import BinanceFetcher, FetchError

    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(
            requests.exceptions.ConnectionError("synthetic failure")
        ),
    )

    fetcher = BinanceFetcher(cache_dir=tmp_path, use_ccxt_fallback=False)
    with pytest.raises(FetchError, match="zero assets"):
        fetcher.fetch_ohlcv(force_refresh=False, allow_partial=True, min_assets=None)


def test_symbol_returning_no_rows_counts_as_a_failure(tmp_path, monkeypatch):
    """Binance answers 200 with [] for a pair that does not exist.

    That must be a failure, not a success carrying an empty DataFrame — an
    empty frame in the result dict pushes the problem downstream into the
    FeatureEngine instead of surfacing it where it happened.
    """
    import requests

    from vgp.data import BinanceFetcher, FetchError

    class _EmptyOkResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return []  # a real Binance reply for an unlisted pair

    monkeypatch.setattr(requests, "get", lambda *a, **k: _EmptyOkResponse())

    fetcher = BinanceFetcher(cache_dir=tmp_path, symbols=["NOPEUSDT"], use_ccxt_fallback=False)
    with pytest.raises(FetchError) as exc:
        fetcher.fetch_ohlcv(force_refresh=True)

    # Zero realized assets trips the always-raise guard; the per-symbol reason
    # still records that the download came back empty rather than errored.
    reasons = dict(exc.value.report.failed)
    assert "NOPEUSDT" in reasons, f"expected NOPEUSDT in {reasons}"
    assert "no rows" in reasons["NOPEUSDT"], reasons["NOPEUSDT"]


def test_empty_cached_file_triggers_a_refetch(synthetic_ohlcv_cache, tmp_path, block_network):
    """A zero-row cached parquet must not be served as a cache hit.

    Serving it would hand downstream code an empty asset; falling through to a
    download is correct, and `block_network` proves that is what happens.
    """
    import shutil

    from tests.conftest import NetworkAccessAttempted
    from vgp.data import BinanceFetcher

    cache = tmp_path / "cache_with_empty"
    cache.mkdir()
    shutil.copy(synthetic_ohlcv_cache / "BTCUSDT_1d.parquet", cache)

    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    empty.index = pd.DatetimeIndex([], name="date")
    empty.to_parquet(cache / "BTCUSDT_1d.parquet")

    fetcher = BinanceFetcher(cache_dir=cache, symbols=["BTCUSDT"], use_ccxt_fallback=False)
    with pytest.raises(NetworkAccessAttempted):
        fetcher.fetch_ohlcv(force_refresh=False)


def test_complete_fetch_reports_no_failures(synthetic_ohlcv_cache, block_network):
    """The happy path: full universe, no raise, report marked complete."""
    from vgp.data import BinanceFetcher, get_binance_symbols

    fetcher = BinanceFetcher(cache_dir=synthetic_ohlcv_cache)
    ohlcv = fetcher.fetch_ohlcv(force_refresh=False)

    report = fetcher.last_fetch_report_
    assert report.is_complete
    assert report.n_failed == 0
    assert report.n_realized == len(get_binance_symbols()) == len(ohlcv)
    assert "30/30" in report.summary()


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


# ---------------------------------------------------------------------------
# DATA-05: every feature must be CAUSAL — computable from the past alone.
#
# This is the structural form of the no-lookahead invariant (CLAUDE.md #2). If
# a feature at index i depends only on bars <= i, then truncating the series
# after i cannot change it. Any use of a full-sample statistic — a global mean,
# std, min or max — breaks that immediately.
#
# It is what should have caught the real defect: obv_signal was z-scored with
# obv_raw.mean() and obv_raw.std() over the whole series, and since
# FeatureEngine.fit_transform() runs once on the full panel before any
# walk-forward split, those constants included every OOS period. A correlation
# proxy did not catch it; this does, exactly and for all twelve features at
# once.
# ---------------------------------------------------------------------------


def _one_asset(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.standard_normal(n) * 0.001),
            "high": close * (1 + np.abs(rng.standard_normal(n)) * 0.01),
            "low": close * (1 - np.abs(rng.standard_normal(n)) * 0.01),
            "close": close,
            "volume": np.abs(rng.standard_normal(n)) * 1e6 + 1e5,
        },
        index=idx,
    )


def test_every_feature_is_causal_under_truncation():
    """Truncating the series must not change any earlier feature value.

    Checks each of the 12 features independently so a failure names the
    culprit rather than just reporting that something leaks.
    """
    from vgp.data.feature_engine import _compute_features

    df = _one_asset()
    k = 300  # truncate here
    full = _compute_features(df)
    trunc = _compute_features(df.iloc[:k])

    assert list(full.columns) == list(trunc.columns)

    leaky = []
    for col in full.columns:
        a = full[col].iloc[:k].to_numpy(dtype=np.float64)
        b = trunc[col].to_numpy(dtype=np.float64)
        both_nan = np.isnan(a) & np.isnan(b)
        if not np.allclose(a[~both_nan], b[~both_nan], rtol=0, atol=0, equal_nan=True):
            worst = np.nanmax(np.abs(a[~both_nan] - b[~both_nan]))
            leaky.append(f"{col} (max abs diff {worst:.3e})")

    assert not leaky, (
        "these features change when future bars are removed, so they use "
        "future information: " + "; ".join(leaky)
    )


@pytest.mark.parametrize("k", [150, 250, 350])
def test_causality_holds_at_several_truncation_points(k):
    """Not just one cut — a leak could hide at a particular boundary."""
    from vgp.data.feature_engine import _compute_features

    df = _one_asset(seed=11)
    full = _compute_features(df)
    trunc = _compute_features(df.iloc[:k])

    for col in full.columns:
        a = full[col].iloc[:k].to_numpy(dtype=np.float64)
        b = trunc[col].to_numpy(dtype=np.float64)
        both_nan = np.isnan(a) & np.isnan(b)
        np.testing.assert_allclose(
            a[~both_nan],
            b[~both_nan],
            rtol=0,
            atol=0,
            err_msg=f"{col} is not causal at truncation k={k}",
        )


def test_appending_future_bars_does_not_change_past_features():
    """The same invariant from the other direction, which is how live use works.

    Tomorrow's bar arriving must not revise today's feature value. A global
    normalisation silently rewrites the entire history every time new data
    lands, so a backtest and a live system would disagree about the past.
    """
    from vgp.data.feature_engine import _compute_features

    df = _one_asset(n=500, seed=21)
    today = _compute_features(df.iloc[:400])
    tomorrow = _compute_features(df)

    for col in today.columns:
        a = today[col].to_numpy(dtype=np.float64)
        b = tomorrow[col].iloc[:400].to_numpy(dtype=np.float64)
        both_nan = np.isnan(a) & np.isnan(b)
        np.testing.assert_allclose(
            a[~both_nan],
            b[~both_nan],
            rtol=0,
            atol=0,
            err_msg=(
                f"{col} was revised by the arrival of future bars — a backtest "
                f"and a live system would disagree about the past"
            ),
        )


# ---------------------------------------------------------------------------
# The test slice must END where the window says it ends (DATA-03)
# ---------------------------------------------------------------------------


def test_split_respects_test_end_dataframe():
    """The test slice stops at test_end, not at the end of the panel.

    Regression. `split()` had no `test_end` parameter at all, so every test
    slice ran from `test_start` to the last row of the data. Walk-forward
    windows that recorded a 2-month OOS period were in fact scored on
    everything from their start date onward, which makes the OOS periods
    NESTED rather than disjoint: window 0's OOS contains every later window's.
    Any statistic taken across windows — a median OOS Sharpe, a count of
    profitable windows — is then an average over six overlapping views of
    largely the same period, not six independent observations.

    It was invisible in the results because nothing reported the realized OOS
    length; the only trace was `oos_min_trades`, scaled by T_test/T_train,
    falling 66 -> 5 across six windows that all claimed the same 2-month span.
    """
    from vgp.data import WalkForwardSplitter

    idx = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    df = pd.DataFrame({"v": range(len(idx))}, index=idx)

    train, val, test = WalkForwardSplitter().split(
        df,
        train_end="2024-06-30",
        val_start="2024-07-01",
        val_end="2024-08-31",
        test_start="2024-09-01",
        test_end="2024-10-31",
    )

    assert test.index.min() >= pd.Timestamp("2024-09-01"), "test slice starts too early"
    assert test.index.max() <= pd.Timestamp("2024-10-31"), (
        f"test slice runs to {test.index.max().date()}, past test_end 2024-10-31 — "
        "the OOS window is unbounded and overlaps every later window"
    )
    assert len(test) == 61, f"expected 61 days of OOS (Sep+Oct), got {len(test)}"


def test_split_respects_test_end_ndarray():
    """Same bound on the ndarray path, which is what the feature matrix uses."""
    from vgp.data import WalkForwardSplitter

    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    arr = np.arange(len(dates) * 3, dtype=np.float32).reshape(len(dates), 3)

    _train, _val, test = WalkForwardSplitter().split(
        arr,
        train_end="2024-06-30",
        val_start="2024-07-01",
        val_end="2024-08-31",
        test_start="2024-09-01",
        test_end="2024-10-31",
        dates=dates,
    )

    assert test.shape[0] == 61, (
        f"expected 61 OOS rows (Sep+Oct), got {test.shape[0]} — the ndarray path "
        "ignores test_end and runs to the end of the panel"
    )


def test_walk_forward_oos_windows_are_disjoint():
    """Consecutive windows must not score overlapping OOS periods (VAL-01).

    The end-to-end version of the two tests above: run the real window
    generator through the real splitter and assert the OOS slices tile the
    sample instead of nesting inside one another.
    """
    from vgp.analysis import generate_windows
    from vgp.data import WalkForwardSplitter

    idx = pd.date_range("2024-01-01", "2026-03-31", freq="D")
    df = pd.DataFrame({"v": range(len(idx))}, index=idx)
    splitter = WalkForwardSplitter()

    windows = generate_windows(
        "2024-01-01", "2026-03-31", train_months=9, val_months=2, oos_months=2, step_months=2
    )
    assert len(windows) >= 2, f"need at least 2 windows to test disjointness, got {len(windows)}"

    spans = []
    for w in windows:
        _t, _v, test = splitter.split(
            df,
            train_end=w.train_end,
            val_start=w.val_start,
            val_end=w.val_end,
            test_start=w.test_start,
            test_end=w.test_end,
        )
        if len(test) == 0:
            continue
        spans.append((w.window_id, test.index.min(), test.index.max()))

    for (id_a, _start_a, end_a), (id_b, start_b, _end_b) in zip(spans, spans[1:]):
        assert end_a < start_b, (
            f"window {id_a} OOS ends {end_a.date()} but window {id_b} OOS starts "
            f"{start_b.date()} — the OOS periods overlap, so statistics across "
            "windows are not independent observations"
        )
