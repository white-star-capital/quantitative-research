"""Tests for Taostats subnet CSV → wide price loader."""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rp_pca.backtest.engine import WalkForwardBacktest
from rp_pca.config import TAO_SUBNET_WIDE_PARQUET
from rp_pca.data.tao_subnet_loader import (
    load_subnet_candles_combined,
    load_subnet_candles_from_dir,
    load_tao_subnet_prices,
)


def _write_sn_csv(path: Path, netuid: int, rows: list[tuple[str, float]]) -> None:
    lines = ["cmc_id,symbol,timestamp,open,high,low,close,volume,market_cap"]
    for ts, close in rows:
        lines.append(f"0,SN{netuid},{ts},0,0,0,{close},0,0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_subnet_candles_from_dir_wide_merge(tmp_path: Path) -> None:
    d = tmp_path / "candles"
    d.mkdir()
    _write_sn_csv(
        d / "sn1_tao_daily_candles.csv",
        1,
        [
            ("2026-01-01T12:00:00+00:00", 1.0),
            ("2026-01-02T12:00:00+00:00", 1.1),
        ],
    )
    _write_sn_csv(
        d / "sn4_tao_daily_candles.csv",
        4,
        [
            ("2026-01-01T08:00:00+00:00", 2.0),
            ("2026-01-02T08:00:00+00:00", 2.2),
        ],
    )
    prices = load_subnet_candles_from_dir(d)
    assert list(prices.columns) == ["SN1", "SN4"]
    assert len(prices) == 2
    assert prices.loc[pd.Timestamp("2026-01-01"), "SN1"] == pytest.approx(1.0)
    assert prices.loc[pd.Timestamp("2026-01-02"), "SN4"] == pytest.approx(2.2)


def test_load_subnet_candles_from_dir_last_obs_per_day(tmp_path: Path) -> None:
    d = tmp_path / "candles"
    d.mkdir()
    _write_sn_csv(
        d / "sn2_tao_daily_candles.csv",
        2,
        [
            ("2026-01-01T08:00:00+00:00", 1.0),
            ("2026-01-01T20:00:00+00:00", 1.5),
        ],
    )
    prices = load_subnet_candles_from_dir(d)
    assert len(prices) == 1
    assert prices.loc[pd.Timestamp("2026-01-01"), "SN2"] == pytest.approx(1.5)


def test_load_subnet_candles_combined_long(tmp_path: Path) -> None:
    csv = tmp_path / "tao_all_subnets_daily_candles_combined.csv"
    csv.write_text(
        textwrap.dedent(
            """\
            cmc_id,symbol,timestamp,open,high,low,close,volume,market_cap
            0,SN1,2026-01-01T00:00:00+00:00,1,1,1,10,0,0
            0,SN1,2026-01-02T00:00:00+00:00,1,1,1,11,0,0
            0,SN3,2026-01-01T00:00:00+00:00,1,1,1,3,0,0
            0,SN3,2026-01-02T00:00:00+00:00,1,1,1,3.3,0,0
            """
        ),
        encoding="utf-8",
    )
    prices = load_subnet_candles_combined(csv)
    assert list(prices.columns) == ["SN1", "SN3"]
    assert prices.loc[pd.Timestamp("2026-01-02"), "SN1"] == pytest.approx(11.0)


def test_load_tao_subnet_prices_prefers_parquet(tmp_path: Path) -> None:
    d = tmp_path / "cache"
    d.mkdir()
    _write_sn_csv(
        d / "sn9_tao_daily_candles.csv",
        9,
        [
            ("2026-01-01T12:00:00+00:00", 9.0),
        ],
    )
    wide = pd.DataFrame(
        {"SN1": [1.0, 1.1], "SN2": [2.0, 2.1]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
    )
    wide.index.name = "date"
    wide.to_parquet(d / TAO_SUBNET_WIDE_PARQUET, index=True)

    prices = load_tao_subnet_prices(d)
    assert "SN9" not in prices.columns
    assert list(prices.columns) == ["SN1", "SN2"]
    assert len(prices) == 2


def test_load_tao_subnet_prices_falls_back_to_csv(tmp_path: Path) -> None:
    d = tmp_path / "only_csv"
    d.mkdir()
    _write_sn_csv(
        d / "sn5_tao_daily_candles.csv",
        5,
        [("2026-01-01T12:00:00+00:00", 5.0)],
    )
    prices = load_tao_subnet_prices(d)
    assert list(prices.columns) == ["SN5"]
    assert prices.loc[pd.Timestamp("2026-01-01"), "SN5"] == pytest.approx(5.0)


def test_walkforward_synthetic_tao_buy_hold_for_subnet_universe() -> None:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=80, freq="D")
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (80, 2)),
        index=idx,
        columns=["SN1", "SN2"],
    )
    bt = WalkForwardBacktest(
        returns,
        cov_window=20,
        mean_window=10,
        rebalance_days=5,
        n_components=2,
        gamma=1.0,
        use_ewma=False,
        cov_method="sample",
        buy_hold_benchmark="TAO",
    )
    res = bt.run(include_pca=False, include_benchmarks=True)
    assert "TAO" in res.return_series
    assert np.allclose(res.return_series["TAO"].values, 0.0)


def test_walkforward_omits_buy_hold_when_benchmark_none() -> None:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=80, freq="D")
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (80, 2)),
        index=idx,
        columns=["SN1", "SN2"],
    )
    bt = WalkForwardBacktest(
        returns,
        cov_window=20,
        mean_window=10,
        rebalance_days=5,
        n_components=2,
        gamma=1.0,
        use_ewma=False,
        cov_method="sample",
        buy_hold_benchmark=None,
    )
    res = bt.run(include_pca=False, include_benchmarks=True)
    assert "BTC" not in res.return_series
    assert "Equal-Weight" in res.return_series


def test_walkforward_adapts_windows_for_short_history() -> None:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2026-01-01", periods=40, freq="D")
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (40, 3)),
        index=idx,
        columns=["SN1", "SN2", "SN3"],
    )
    bt = WalkForwardBacktest(
        returns,
        cov_window=504,
        mean_window=252,
        rebalance_days=7,
        n_components=2,
        gamma=1.0,
        use_ewma=False,
        cov_method="sample",
        buy_hold_benchmark="TAO",
    )
    res = bt.run(include_pca=False, include_benchmarks=True)
    assert "RP-PCA Tangency" in res.return_series
    assert len(res.return_series["RP-PCA Tangency"]) > 0
