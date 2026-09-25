"""
Data fetcher — downloads OHLCV data from Binance's public REST API.

No API key is required.  Data is cached as Parquet files to avoid
redundant downloads.

Fallback: Uses CCXT as backup if Binance REST API is unavailable.

FAILURE POLICY
--------------
A missing asset is not a warning, it is a change to the experiment. This
module used to wrap every symbol in `except Exception` and merely log it, so a
partial fetch failure returned a smaller universe with nothing downstream told
that assets had gone missing — a run could proceed on 3 assets instead of 30
and report results normally. A universe that varies silently between runs is a
survivorship-bias channel, and it is invisible to both the Deflated Sharpe
Ratio and the null control, because every trial and every surrogate inherits
whatever universe it was handed.

So `fetch_ohlcv()` now RAISES `FetchError` by default when any requested symbol
fails. A caller willing to proceed on fewer assets must say so explicitly with
`allow_partial=True`, and should set `min_assets` as a floor. Either way the
outcome is recorded in `FetchReport` (also on `self.last_fetch_report_`) so the
realized universe can be written alongside the results rather than inferred.

Two conditions always raise, regardless of `allow_partial`: an empty realized
universe, and a realized universe below `min_assets`. Neither can be a
legitimate research result.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from .universe import get_binance_symbols

logger = logging.getLogger(__name__)

BINANCE_REST = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000  # Binance max rows per request
SLEEP_BETWEEN_CALLS = 0.12  # seconds — stay well under rate limits


class FetchError(RuntimeError):
    """Raised when the fetcher cannot deliver the requested universe.

    Carries the `FetchReport` so a caller that catches it can still see
    exactly which symbols failed and why.
    """

    def __init__(self, message: str, report: FetchReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class FetchReport:
    """What a fetch actually delivered, versus what was asked for.

    The realized universe is a property of the run and belongs in the results,
    not in a log line that nobody reads. `UniverseRecord` in universe.py
    combines this with the FeatureEngine's retention decision to give the set
    of assets a run was actually computed on.
    """

    requested: tuple[str, ...]  # Binance symbols, e.g. BTCUSDT
    realized: tuple[str, ...]  # short tickers with usable data
    failed: tuple[tuple[str, str], ...]  # (symbol, reason) pairs
    start_date: str
    end_date: str
    interval: str

    @property
    def n_requested(self) -> int:
        return len(self.requested)

    @property
    def n_realized(self) -> int:
        return len(self.realized)

    @property
    def n_failed(self) -> int:
        return len(self.failed)

    @property
    def is_complete(self) -> bool:
        return self.n_failed == 0

    def to_dict(self) -> dict:
        return {
            "requested": list(self.requested),
            "realized": list(self.realized),
            "failed": {symbol: reason for symbol, reason in self.failed},
            "n_requested": self.n_requested,
            "n_realized": self.n_realized,
            "n_failed": self.n_failed,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "interval": self.interval,
        }

    def summary(self) -> str:
        lines = [
            f"Fetch: {self.n_realized}/{self.n_requested} symbols realized "
            f"({self.start_date} to {self.end_date}, {self.interval})"
        ]
        for symbol, reason in self.failed:
            lines.append(f"  FAILED {symbol}: {reason}")
        return "\n".join(lines)


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
        symbols: list[str] | None = None,
        use_ccxt_fallback: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.symbols = symbols or get_binance_symbols()
        self.use_ccxt_fallback = use_ccxt_fallback

        # Set by fetch_ohlcv()/fetch_all(); the realized universe of the last
        # call, for recording alongside results (see UniverseRecord).
        self.last_fetch_report_: FetchReport | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        start_date: str = "2021-01-01",
        end_date: str = "2025-12-31",
        interval: str = "1d",
        force_refresh: bool = False,
        allow_partial: bool = False,
        min_assets: int | None = None,
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
        allow_partial : bool
            Default False: any symbol that fails raises `FetchError`, because a
            silently shrunken universe changes the experiment (see module
            docstring). Set True only as a deliberate, recorded choice — the
            failures are then logged at WARNING and captured in the report.
        min_assets : int | None
            Floor on the realized universe. Falling below it raises
            `FetchError` even when `allow_partial=True`. An empty universe
            always raises.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of short ticker (e.g. "BTC") to OHLCV DataFrame with
            DatetimeIndex named "date" and columns [open, high, low, close, volume].

        Raises
        ------
        FetchError
            If any symbol failed and `allow_partial` is False; if the realized
            universe is empty; or if it is smaller than `min_assets`.

        Notes
        -----
        A symbol that returns zero rows counts as a FAILURE, not a success with
        an empty frame. Binance answers 200 with `[]` for a pair that does not
        exist, which would otherwise put an empty DataFrame into the result and
        push the problem downstream.
        """
        result: dict[str, pd.DataFrame] = {}
        failed: list[tuple[str, str]] = []

        for symbol in tqdm(self.symbols, desc="Fetching OHLCV"):
            ticker = symbol.replace("USDT", "")
            try:
                df = self._fetch_symbol(symbol, start_date, end_date, interval, force_refresh)
            except Exception as exc:
                failed.append((symbol, f"{type(exc).__name__}: {exc}"))
                continue

            if df is None or df.empty:
                failed.append((symbol, "no rows returned for the requested range"))
                continue
            result[ticker] = df

        report = FetchReport(
            requested=tuple(self.symbols),
            realized=tuple(result),
            failed=tuple(failed),
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
        self.last_fetch_report_ = report
        self._enforce_fetch_policy(report, allow_partial, min_assets)
        return result

    # ------------------------------------------------------------------
    # Failure policy
    # ------------------------------------------------------------------

    @staticmethod
    def _enforce_fetch_policy(
        report: FetchReport,
        allow_partial: bool,
        min_assets: int | None,
    ) -> None:
        """Raise or warn according to the declared tolerance.

        Kept separate so fetch_ohlcv() and fetch_all() cannot drift apart on
        what counts as an acceptable outcome.
        """
        if report.n_realized == 0:
            raise FetchError(
                "Fetched zero assets — an empty universe is never a valid "
                f"result.\n{report.summary()}",
                report,
            )

        if report.failed and not allow_partial:
            raise FetchError(
                f"{report.n_failed} of {report.n_requested} symbols failed. "
                "A partial universe changes the experiment, so this raises by "
                "default; pass allow_partial=True (with min_assets) to accept "
                f"it deliberately.\n{report.summary()}",
                report,
            )

        if min_assets is not None and report.n_realized < min_assets:
            raise FetchError(
                f"Realized universe has {report.n_realized} assets, below the "
                f"min_assets={min_assets} floor.\n{report.summary()}",
                report,
            )

        if report.failed:
            # allow_partial was set: proceed, but make the shrink impossible to
            # miss in the log as well as in the recorded report.
            logger.warning(
                "Proceeding on a PARTIAL universe: %d/%d symbols realized. " "Missing: %s",
                report.n_realized,
                report.n_requested,
                ", ".join(symbol for symbol, _ in report.failed),
            )
            for symbol, reason in report.failed:
                logger.warning("  %s failed: %s", symbol, reason)
        else:
            logger.info(
                "Fetched the complete universe: %d/%d symbols",
                report.n_realized,
                report.n_requested,
            )

    def fetch_all(
        self,
        start_date: str = "2021-01-01",
        end_date: str = "2025-12-31",
        interval: str = "1d",
        force_refresh: bool = False,
        allow_partial: bool = False,
        min_assets: int | None = None,
    ) -> pd.DataFrame:
        """
        Convenience method: fetch close prices for all symbols.

        Returns a (T, N) DataFrame of close prices. For full OHLCV data,
        use fetch_ohlcv() instead (the primary API).

        Same failure policy as fetch_ohlcv(): raises `FetchError` unless
        `allow_partial=True`, after the CCXT fallback has had its chance.
        The realized universe lands on `self.last_fetch_report_`.

        Returns
        -------
        pd.DataFrame
            Columns are base tickers (BTC, ETH, …); index is dates.

        Raises
        ------
        FetchError
            If any symbol is still missing and `allow_partial` is False; if
            nothing was fetched; or if fewer than `min_assets` were realized.
        """
        frames: dict[str, pd.Series] = {}
        failed: dict[str, str] = {}

        for symbol in tqdm(self.symbols, desc="Fetching close prices"):
            ticker = symbol.replace("USDT", "")
            try:
                df = self._fetch_symbol(symbol, start_date, end_date, interval, force_refresh)
            except Exception as exc:
                failed[symbol] = f"{type(exc).__name__}: {exc}"
                logger.debug("REST API failed for %s — %s", symbol, exc)
                continue
            if df is None or df.empty:
                failed[symbol] = "no rows returned for the requested range"
                continue
            frames[ticker] = df["close"]

        if self.use_ccxt_fallback:
            missing = [s for s in self.symbols if s.replace("USDT", "") not in frames]
            if missing and frames:
                logger.info(
                    "REST API returned %d/%d symbols. Using CCXT to fill %d missing: %s",
                    len(frames),
                    len(self.symbols),
                    len(missing),
                    [s.replace("USDT", "") for s in missing],
                )
                ccxt_frames = self._fetch_all_ccxt(
                    start_date, end_date, interval, symbols_to_fetch=missing
                )
                frames.update(ccxt_frames)
                if ccxt_frames:
                    logger.info(
                        "CCXT filled %d additional symbols: %s",
                        len(ccxt_frames),
                        list(ccxt_frames.keys()),
                    )
                else:
                    logger.warning("CCXT could not fill any of the missing symbols.")
            elif not frames:
                logger.info("REST API returned no data. Attempting full CCXT fallback...")
                frames = self._fetch_all_ccxt(start_date, end_date, interval)

        # Anything CCXT filled is no longer a failure.
        for symbol in list(failed):
            if symbol.replace("USDT", "") in frames:
                del failed[symbol]

        report = FetchReport(
            requested=tuple(self.symbols),
            realized=tuple(frames),
            failed=tuple(sorted(failed.items())),
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
        self.last_fetch_report_ = report
        self._enforce_fetch_policy(report, allow_partial, min_assets)

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
        symbols_to_fetch: list[str] | None = None,
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
            still_missing = [s for s in target_symbols if s.replace("USDT", "") not in frames]
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
                                    exchange_name,
                                    ticker,
                                    sym_fmt,
                                    len(series),
                                )
                                break
                        except Exception as exc:
                            logger.debug(
                                "CCXT %s failed for %s (%s): %s",
                                exchange_name,
                                ticker,
                                sym_fmt,
                                exc,
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
                        symbol,
                        cache_min,
                        cache_max,
                        start_date,
                        end_date,
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
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(
            klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "n_trades",
                "taker_buy_base",
                "taker_buy_quote",
                "_",
            ],
        )
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
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
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

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df.index = df.index.normalize().tz_localize(None)
    df.index.name = "date"

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    mask = (df.index >= start_dt) & (df.index <= end_dt)

    return df.loc[mask, "close"].astype(float)
