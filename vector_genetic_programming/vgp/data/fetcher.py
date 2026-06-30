"""
Data fetcher — downloads OHLCV data from Binance's public REST API.

No API key is required.  Data is cached as Parquet files to avoid
redundant downloads.

Fallback: Uses CCXT as backup if Binance REST API is unavailable.
"""
from __future__ import annotations

import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from .universe import UNIVERSE_30, get_binance_symbols

logger = logging.getLogger(__name__)

BINANCE_REST = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000          # Binance max rows per request
SLEEP_BETWEEN_CALLS = 0.12  # seconds — stay well under rate limits


class BinanceFetcher:
    """
    Download and cache daily OHLCV data for the 30-coin universe.

    Parameters
    ----------
    cache_dir : Path
        Directory for Parquet cache files.
    symbols : list[str] | None
        Binance trading pair symbols.  Defaults to UNIVERSE_30 vs USDT.
    use_ccxt_fallback : bool
        If REST API fails, attempt to fetch via CCXT. Useful for bypassing
        geographic restrictions or API rate limits.
    """

    def __init__(
        self,
        cache_dir: Path,
        symbols: Optional[list[str]] = None,
        use_ccxt_fallback: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.symbols = symbols or get_binance_symbols()
        self.use_ccxt_fallback = use_ccxt_fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        start_date: str = "2021-01-01",
        end_date: str = "2025-12-31",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch full OHLCV DataFrames keyed by ticker symbol.

        This is the primary public API for downstream feature engineering.
        Returns full OHLCV (open, high, low, close, volume) rather than
        close-only, enabling ATR, Parkinson vol, and volume ratio features.

        Parameters
        ----------
        start_date : str
            ISO date string, e.g. "2021-01-01".
        end_date : str
            ISO date string, e.g. "2025-12-31".
        interval : str
            Binance kline interval, e.g. "1d".
        force_refresh : bool
            If True, bypass cache and re-download from Binance REST API.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of short ticker (e.g. "BTC") to OHLCV DataFrame with
            DatetimeIndex named "date" and columns [open, high, low, close, volume].
        """
        result: dict[str, pd.DataFrame] = {}
        for symbol in tqdm(self.symbols, desc="Fetching OHLCV"):
            ticker = symbol.replace("USDT", "")
            try:
                df = self._fetch_symbol(
                    symbol, start_date, end_date, interval, force_refresh
                )
                result[ticker] = df
            except Exception as exc:
                logger.warning("Skipping %s — %s", symbol, exc)
        return result

    def fetch_all(
        self,
        start_date: str = "2021-01-01",
        end_date: str = "2025-12-31",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Convenience method: fetch close prices for all symbols.

        Returns a (T, N) DataFrame of close prices. For full OHLCV data,
        use fetch_ohlcv() instead (the primary API).

        Returns
        -------
        pd.DataFrame
            Columns are base tickers (BTC, ETH, …); index is dates.
        """
        frames: dict[str, pd.Series] = {}

        for symbol in tqdm(self.symbols, desc="Fetching close prices"):
            ticker = symbol.replace("USDT", "")
            try:
                df = self._fetch_symbol(
                    symbol, start_date, end_date, interval, force_refresh
                )
                frames[ticker] = df["close"]
            except Exception as exc:
                logger.warning("REST API failed for %s — %s", symbol, exc)

        if self.use_ccxt_fallback:
            missing = [s for s in self.symbols if s.replace("USDT", "") not in frames]
            if missing and frames:
                logger.info(
                    "REST API returned %d/%d symbols. Using CCXT to fill %d missing: %s",
                    len(frames), len(self.symbols), len(missing),
                    [s.replace("USDT", "") for s in missing],
                )
                ccxt_frames = self._fetch_all_ccxt(
                    start_date, end_date, interval, symbols_to_fetch=missing
                )
                frames.update(ccxt_frames)
                if ccxt_frames:
                    logger.info(
                        "CCXT filled %d additional symbols: %s",
                        len(ccxt_frames), list(ccxt_frames.keys()),
                    )
                else:
                    logger.warning("CCXT could not fill any of the missing symbols.")
            elif not frames:
                logger.info("REST API returned no data. Attempting full CCXT fallback...")
                frames = self._fetch_all_ccxt(start_date, end_date, interval)

        prices = pd.DataFrame(frames)
        prices.index = pd.to_datetime(prices.index)
        prices.index.name = "date"
        prices = prices.sort_index()
        return prices

    # ------------------------------------------------------------------
    # CCXT Fallback (for geographic restrictions)
    # ------------------------------------------------------------------

    def _fetch_all_ccxt(
        self,
        start_date: str,
        end_date: str,
        interval: str,
        symbols_to_fetch: Optional[list[str]] = None,
    ) -> dict[str, pd.Series]:
        """
        Fetch close-price series via CCXT for the given symbols.

        Parameters
        ----------
        symbols_to_fetch : list[str] | None
            Binance-style symbols to fetch (e.g. ["BTCUSDT", "ETHUSDT"]).
            Defaults to ``self.symbols`` when not provided (full-universe fallback).

        Returns
        -------
        dict mapping ticker -> pd.Series of close prices.
        """
        try:
            import ccxt
        except ImportError:
            logger.error("CCXT not installed. Cannot fallback to CCXT.")
            return {}

        target_symbols = symbols_to_fetch if symbols_to_fetch is not None else self.symbols
        frames: dict[str, pd.Series] = {}

        for exchange_name in ["kraken", "coinbasepro", "bybit", "okx"]:
            if not target_symbols:
                break
            still_missing = [
                s for s in target_symbols if s.replace("USDT", "") not in frames
            ]
            if not still_missing:
                break

            try:
                exchange = getattr(ccxt, exchange_name)()
                for symbol in still_missing:
                    ticker = symbol.replace("USDT", "")
                    for sym_fmt in [symbol, f"{ticker}/USDT"]:
                        try:
                            ohlcv = exchange.fetch_ohlcv(sym_fmt, timeframe="1d")
                            series = _parse_ccxt_ohlcv(ohlcv, start_date, end_date)
                            if len(series) > 0:
                                frames[ticker] = series
                                logger.debug(
                                    "CCXT %s: fetched %s as %s (%d rows)",
                                    exchange_name, ticker, sym_fmt, len(series),
                                )
                                break
                        except Exception as exc:
                            logger.debug(
                                "CCXT %s failed for %s (%s): %s",
                                exchange_name, ticker, sym_fmt, exc,
                            )
                            continue
            except Exception as exc:
                logger.debug("CCXT %s unavailable: %s", exchange_name, exc)
                continue

        return frames

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
        force_refresh: bool,
    ) -> pd.DataFrame:
        """Fetch a single symbol, using cached Parquet if available.

        Cache hit strategy: if the cached file exists and its data overlaps
        with the requested range, serve the available subset without a network
        call. This allows partial-coverage caches (e.g. assets listed after
        start_date) to be used without redundant downloads.

        Full-coverage hit (cache min <= start_date and cache max >= end_date):
            return the exact requested slice.
        Partial-coverage hit (any overlap with [start_date, end_date]):
            return all cached rows within the requested range — no download.
        Cache miss (no file, force_refresh, or zero overlap):
            download from Binance REST API and save to cache.
        """
        cache_path = self.cache_dir / f"{symbol}_{interval}.parquet"
        if cache_path.exists() and not force_refresh:
            df = pd.read_parquet(cache_path)
            cache_min = str(df.index.min().date())
            cache_max = str(df.index.max().date())

            # Full coverage: cache spans the entire requested range
            if cache_min <= start_date and cache_max >= end_date:
                mask = (df.index >= start_date) & (df.index <= end_date)
                return df.loc[mask]

            # Partial coverage: cache overlaps with requested range
            # Serve from cache to avoid unnecessary network calls
            if cache_min <= end_date and cache_max >= start_date:
                mask = (df.index >= start_date) & (df.index <= end_date)
                subset = df.loc[mask]
                if not subset.empty:
                    logger.debug(
                        "Partial cache hit for %s: cache %s–%s, requested %s–%s, "
                        "returning %d rows",
                        symbol, cache_min, cache_max, start_date, end_date,
                        len(subset),
                    )
                    return subset

        df = self._paginated_download(symbol, start_date, end_date, interval)
        df.to_parquet(cache_path)
        return df

    def _paginated_download(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> pd.DataFrame:
        """Download in chunks of MAX_LIMIT rows to handle long date ranges."""
        start_ms = _to_ms(start_date)
        end_ms = _to_ms(end_date, end_of_day=True)

        all_rows: list[list] = []
        cursor = start_ms

        while cursor < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": MAX_LIMIT,
            }
            resp = requests.get(BINANCE_REST, params=params, timeout=30)
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                break
            all_rows.extend(rows)
            # Advance cursor past the last returned candle
            cursor = rows[-1][0] + 1
            if len(rows) < MAX_LIMIT:
                break
            time.sleep(SLEEP_BETWEEN_CALLS)

        return self._parse_klines(all_rows)

    def _parse_klines(self, klines: list) -> pd.DataFrame:
        """Parse raw Binance kline rows into an OHLCV DataFrame with DatetimeIndex."""
        if not klines:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "n_trades",
            "taker_buy_base", "taker_buy_quote", "_",
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("open_time")
        df.index = df.index.normalize().tz_localize(None)  # date only
        df.index.name = "date"
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df[["open", "high", "low", "close", "volume"]]


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _to_ms(date_str: str, end_of_day: bool = False) -> int:
    """Convert 'YYYY-MM-DD' to millisecond Unix timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    return int(dt.timestamp() * 1000)


def _parse_ccxt_ohlcv(
    ohlcv: list[list],
    start_date: str,
    end_date: str,
) -> pd.Series:
    """
    Parse CCXT OHLCV format [timestamp, o, h, l, c, v] into a Series.
    Filter by date range and return close prices only.
    """
    if not ohlcv:
        return pd.Series(dtype=float)

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df.index = df.index.normalize().tz_localize(None)
    df.index.name = "date"

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    mask = (df.index >= start_dt) & (df.index <= end_dt)

    return df.loc[mask, "close"].astype(float)
