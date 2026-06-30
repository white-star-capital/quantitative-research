"""
Load TAO-denominated subnet daily candles from Taostats export CSVs.

Expected per-subnet files: ``sn{netuid}_tao_daily_candles.csv`` (see
``get_tao_stats_all_subnets.py``). Produces the same wide price frame contract
as ``BinanceFetcher.fetch_all``: DatetimeIndex named ``date``, one column per
asset (``SN1``, ``SN4``, …).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from rp_pca.config import TAO_SUBNET_WIDE_PARQUET

logger = logging.getLogger(__name__)

_SUBNET_FILE_RE = re.compile(r"^sn(\d+)_tao_daily_candles\.csv$", re.IGNORECASE)


def _to_naive_date_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Match Binance fetcher: timezone-naive calendar dates."""
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()


def _netuid_from_filename(name: str) -> Optional[int]:
    m = _SUBNET_FILE_RE.match(name)
    if m is None:
        return None
    return int(m.group(1))


def _normalise_to_date_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'timestamp' column")
    ts = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    out = df.assign(_d=ts.dt.normalize())
    return out


def _series_from_subnet_file(path: Path, price_col: str) -> pd.Series:
    df = pd.read_csv(path)
    if price_col not in df.columns:
        raise ValueError(f"{path.name}: missing column {price_col!r}")
    df = _normalise_to_date_index(df)
    # Last observation per calendar day (UTC)
    sub = df.sort_values("_d").groupby("_d", as_index=True)[price_col].last()
    sub = sub.astype(float)
    netuid = _netuid_from_filename(path.name)
    if netuid is None:
        raise ValueError(f"Unexpected subnet filename: {path.name}")
    sub.name = f"SN{netuid}"
    sub.index.name = "date"
    return sub


def load_subnet_candles_from_dir(
    directory: Path | str,
    *,
    price_col: str = "close",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Discover ``sn*_tao_daily_candles.csv`` under ``directory`` and merge to a
    wide price DataFrame.

    Parameters
    ----------
    directory
        Folder containing per-subnet CSV exports.
    price_col
        Column to use as the daily close (Taostats pipeline uses ``close``).
    start_date, end_date
        Optional ``YYYY-MM-DD`` bounds (inclusive) applied to the merged index.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    paths = sorted(p for p in root.glob("sn*_tao_daily_candles.csv") if p.is_file())
    if not paths:
        raise FileNotFoundError(
            f"No sn*_tao_daily_candles.csv files found under {root}"
        )

    series_list: list[pd.Series] = []
    for p in paths:
        if _netuid_from_filename(p.name) is None:
            continue
        try:
            series_list.append(_series_from_subnet_file(p, price_col))
        except Exception as exc:
            logger.warning("Skipping %s: %s", p.name, exc)

    if not series_list:
        raise ValueError(f"No valid subnet CSVs could be loaded from {root}")

    prices = pd.concat(series_list, axis=1)
    prices = prices.sort_index()
    prices.index = _to_naive_date_index(prices.index)
    prices.index.name = "date"

    if start_date is not None:
        prices = prices.loc[prices.index >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        prices = prices.loc[prices.index <= pd.Timestamp(end_date).normalize()]

    # Stable column order: SN1, SN2, …
    def _sn_key(c: str) -> tuple[int, str]:
        if c.startswith("SN") and c[2:].isdigit():
            return (int(c[2:]), c)
        return (10**9, c)

    prices = prices.reindex(columns=sorted(prices.columns, key=_sn_key))
    return prices


def load_tao_subnet_prices(
    directory: Path | str,
    *,
    price_col: str = "close",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load wide subnet closes: prefer ``tao_subnets_wide.parquet`` if present,
    otherwise merge ``sn*_tao_daily_candles.csv`` (same contract as
    ``load_subnet_candles_from_dir``).
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    pq = root / TAO_SUBNET_WIDE_PARQUET
    if pq.is_file():
        prices = pd.read_parquet(pq)
        if "date" in prices.columns and prices.index.name != "date":
            prices = prices.set_index("date")
        prices.index.name = "date"
        prices.index = _to_naive_date_index(pd.DatetimeIndex(prices.index))
        prices = prices.sort_index().astype(float)
    else:
        prices = load_subnet_candles_from_dir(
            root,
            price_col=price_col,
            start_date=None,
            end_date=None,
        )

    if start_date is not None:
        prices = prices.loc[prices.index >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        prices = prices.loc[prices.index <= pd.Timestamp(end_date).normalize()]

    def _sn_key(c: str) -> tuple[int, str]:
        if c.startswith("SN") and c[2:].isdigit():
            return (int(c[2:]), c)
        return (10**9, c)

    prices = prices.reindex(columns=sorted(prices.columns, key=_sn_key))
    prices.index.name = "date"
    return prices


def load_tao_subnet_market_caps(
    directory: Path | str,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load wide market-cap DataFrame from individual ``sn*_tao_daily_candles.csv``.

    Same shape/index contract as :func:`load_tao_subnet_prices`: DatetimeIndex
    named ``date``, columns ``SN1``, ``SN2``, …, values = market_cap.

    NOTE: The wide parquet does **not** contain market_cap, so this always
    reads the individual CSVs.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    paths = sorted(p for p in root.glob("sn*_tao_daily_candles.csv") if p.is_file())
    if not paths:
        raise FileNotFoundError(
            f"No sn*_tao_daily_candles.csv files found under {root}"
        )

    series_list: list[pd.Series] = []
    for p in paths:
        if _netuid_from_filename(p.name) is None:
            continue
        try:
            series_list.append(_series_from_subnet_file(p, price_col="market_cap"))
        except Exception as exc:
            logger.warning("Skipping market_cap for %s: %s", p.name, exc)

    if not series_list:
        raise ValueError(f"No valid market_cap data could be loaded from {root}")

    mcaps = pd.concat(series_list, axis=1)
    mcaps = mcaps.sort_index()
    mcaps.index = _to_naive_date_index(mcaps.index)
    mcaps.index.name = "date"

    if start_date is not None:
        mcaps = mcaps.loc[mcaps.index >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        mcaps = mcaps.loc[mcaps.index <= pd.Timestamp(end_date).normalize()]

    def _sn_key(c: str) -> tuple[int, str]:
        if c.startswith("SN") and c[2:].isdigit():
            return (int(c[2:]), c)
        return (10**9, c)

    mcaps = mcaps.reindex(columns=sorted(mcaps.columns, key=_sn_key))
    return mcaps


def load_subnet_candles_combined(
    path: Path | str,
    *,
    price_col: str = "close",
    symbol_col: str = "symbol",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a long combined CSV (e.g. ``tao_all_subnets_daily_candles_combined.csv``)
    and pivot to wide prices.

    Expects columns including ``timestamp``, ``symbol`` (e.g. ``SN4``), and
    ``price_col``. If ``netuid`` is present and ``symbol`` is missing, columns
    are named ``SN{netuid}``.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Not a file: {p}")

    df = pd.read_csv(p)
    if price_col not in df.columns:
        raise ValueError(f"Combined CSV missing column {price_col!r}")

    df = _normalise_to_date_index(df)

    if symbol_col in df.columns:
        id_series = df[symbol_col].astype(str)
    elif "netuid" in df.columns:
        id_series = "SN" + df["netuid"].astype(int).astype(str)
    else:
        raise ValueError(
            "Combined CSV needs a 'symbol' column (e.g. SN4) or 'netuid'"
        )

    df = df.assign(_asset=id_series)
    df = df.sort_values("_d")
    df = df.groupby(["_d", "_asset"], as_index=False)[price_col].last()
    prices = df.pivot(index="_d", columns="_asset", values=price_col).astype(float)
    prices.index.name = "date"
    prices.columns.name = None

    prices.index = _to_naive_date_index(prices.index)
    prices = prices.sort_index()

    if start_date is not None:
        prices = prices.loc[prices.index >= pd.Timestamp(start_date).normalize()]
    if end_date is not None:
        prices = prices.loc[prices.index <= pd.Timestamp(end_date).normalize()]

    return prices
