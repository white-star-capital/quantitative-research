"""Hyperliquid public info API: perp candles for `rv_daily` (MODELS.md §3.2). Keyless.

Input kind (InputSpec params["kind"]): `rv_daily` (RV_FORECAST_IMPLEMENTATION_GUIDE.md VRP-A-8),
with the same params, output and completed-day cache as Deribit's (adapters/rv_daily.py). The coin
is the scope's underlying unless `params.coin` names one; `resolution` is in minutes and maps to
a candle interval ("5" → "5m", "60" → "1h").

`snapshot_at` comes from the venue, never the local clock: Hyperliquid responses carry no server
time, so each fetch also asks for the 1-minute candles from CLOCK_LOOKBACK_MS before `as_of` to
CLOCK_LEAD_MS after it and uses the newest one's open time `t`. The venue returns candles only up
to the one in progress, so `t` is at most a minute behind the venue's clock when it answered, even
when that is well after `as_of` (a run late in its pass, or a recording's later assets), and the
manifest's `max_input_age_seconds` is satisfiable. The window depends only on `as_of`, so a
recording replays it exactly. Whether the venue accepts a future `endTime` is UNVERIFIED (check
live with `make probe`): if that request fails or comes back empty, it is asked once more with
`endTime = as_of`, which dates the input by the 1-minute candle in progress at `as_of` (at most
60 s old). No 1-minute candle from either means `venue_unavailable`.

Hyperliquid serves only about the latest 5,000 candles per interval (UNVERIFIED; check live,
VRP-A-8); older days come back empty and the cache grows the history over time. The per-call cap
is UNVERIFIED too, so CANDLE_ROWS_PER_CALL is only an upper bound: a page cut short is split and
asked again (adapters/rv_daily.py). Any failure is `venue_unavailable` (D-8); there is no
fallback data (D-10).
"""

from __future__ import annotations

from datetime import UTC, datetime

from terminal_models.adapters.base import AdapterError, DataAdapter
from terminal_models.adapters.hyperliquid.client import HyperliquidClient
from terminal_models.adapters.rv_daily import DayCache, fetch_rv_daily
from terminal_models.core.base import InputData, InputSpec

COINS = {"BTC": "BTC", "ETH": "ETH", "SOL": "SOL", "HYPE": "HYPE"}
INTERVALS = {"1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m", "60": "1h", "120": "2h",
             "240": "4h", "480": "8h", "720": "12h"}  # fmt: skip
CANDLE_ROWS_PER_CALL = 5000  # upper bound per candleSnapshot; a lower venue cap is split around
CLOCK_LOOKBACK_MS = 5 * 60_000  # 1-minute candles that date the response: this far before as_of
CLOCK_LEAD_MS = 10 * 60_000  # ... to this far after it (the venue stops at the candle in progress)


class HyperliquidAdapter(DataAdapter):
    name = "hyperliquid"

    def __init__(
        self, client: HyperliquidClient | None = None, rv_cache: DayCache | None = None
    ) -> None:
        self.client = client or HyperliquidClient()
        self.rv_cache = rv_cache  # None: no completed-day cache (recording, replay, tests)

    def fetch(self, spec: InputSpec, as_of: datetime) -> InputData:
        underlying = spec.params.get("underlying")
        coin = spec.params.get("coin") or COINS.get(underlying)  # type: ignore[arg-type]
        if coin is None:
            raise AdapterError("bad_input_spec", f"{spec.name}: unknown underlying {underlying!r}")
        kind = spec.params.get("kind")
        if kind != "rv_daily":
            raise AdapterError("bad_input_spec", f"{spec.name}: unknown kind {kind!r}")
        resolution = str(spec.params.get("resolution"))
        interval = INTERVALS.get(resolution)
        if interval is None:
            raise AdapterError(
                "bad_input_spec", f"{spec.name}: unsupported resolution {resolution}"
            )
        try:
            return self._rv_daily(spec, str(coin), resolution, interval, as_of)
        except AdapterError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterError(
                "venue_unavailable", f"{spec.name}: unexpected response: {exc}"
            ) from exc

    def _rv_daily(
        self, spec: InputSpec, coin: str, resolution: str, interval: str, as_of: datetime
    ) -> InputData:
        now_ms = int(as_of.timestamp() * 1000)
        try:
            clock = self.client.candles(
                coin, "1m", now_ms - CLOCK_LOOKBACK_MS, now_ms + CLOCK_LEAD_MS
            )
        except AdapterError:
            clock = []
        if not clock:  # the venue refused or ignored the future endTime: ask up to as_of once
            clock = self.client.candles(coin, "1m", now_ms - CLOCK_LOOKBACK_MS, now_ms)
        if not clock:
            raise AdapterError("venue_unavailable", f"{spec.name}: no recent {coin} 1m candle")
        venue_time = datetime.fromtimestamp(max(k["t"] for k in clock) / 1000, UTC)

        def page(start_ms: int, end_ms: int):
            candles = self.client.candles(coin, interval, start_ms, end_ms - 1)
            times = [int(k["t"]) for k in candles]
            return times, [float(k["c"]) for k in candles], venue_time, candles

        data, _, raws = fetch_rv_daily(
            spec.name,
            venue="hyperliquid",
            instrument=coin,
            resolution=resolution,
            lookback_days=int(spec.params["lookback_days"]),
            min_coverage=float(spec.params["min_coverage"]),
            as_of=as_of,
            rows_per_call=CANDLE_ROWS_PER_CALL,
            fetch_page=page,
            cache=self.rv_cache,
        )
        return InputData(spec.name, venue_time, data, raw={"candleSnapshot": raws, "clock": clock})
