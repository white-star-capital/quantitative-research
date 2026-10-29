"""Deribit public API: option chains, spot index, perp candles, DVOL (MODELS.md §3.2). Keyless.

Input kinds (InputSpec params["kind"]): `chain`, `index`, `candles_1h`, `dvol`, for
parity_monitor `futures_book`, `funding` and `order_book` (PARITY_MONITOR_IMPLEMENTATION_GUIDE.md
§2.1), and for vrp_signal `rv_daily` and `dvol_history` (RV_FORECAST_IMPLEMENTATION_GUIDE.md
VRP-A-8, §12). Any failure after retries is an AdapterError with code `venue_unavailable` (D-8);
there is no fallback data (D-10).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from terminal_models.adapters.base import AdapterError, DataAdapter
from terminal_models.adapters.deribit.chains import (
    BOOKS,
    book_side,
    find_perpetual,
    iso_ms,
    normalize,
    normalize_futures,
)
from terminal_models.adapters.deribit.client import DeribitClient
from terminal_models.adapters.rv_daily import (
    MS_PER_DAY,
    DayCache,
    day_floor_ms,
    fetch_rv_daily,
    iso_day,
)
from terminal_models.core.base import InputData, InputSpec

# Rows per get_tradingview_chart_data call: an upper bound; a lower venue cap is split around
# (adapters/rv_daily.py). The real cap is UNVERIFIED (check live with `make probe`, VRP-A-8).
CHART_ROWS_PER_CALL = 5000
DVOL_MAX_PAGES = 100  # continuation pages per dvol_history fetch; more means a paging fault


class DeribitAdapter(DataAdapter):
    name = "deribit"

    def __init__(
        self, client: DeribitClient | None = None, rv_cache: DayCache | None = None
    ) -> None:
        self.client = client or DeribitClient()
        self.rv_cache = rv_cache  # None: no completed-day cache (recording, replay, tests)

    def fetch(self, spec: InputSpec, as_of: datetime) -> InputData:
        underlying = spec.params.get("underlying")
        book = BOOKS.get(underlying)  # type: ignore[arg-type]
        if book is None:
            raise AdapterError("bad_input_spec", f"{spec.name}: unknown underlying {underlying!r}")
        kind = spec.params.get("kind")
        try:
            if kind == "chain":
                return self._chain(spec, underlying, book)
            if kind == "index":
                return self._index(spec, book.index, "spot")
            if kind == "dvol":
                if book.dvol is None:
                    raise AdapterError("bad_input_spec", f"no DVOL index for {underlying}")
                return self._index(spec, book.dvol, "dvol", scale=0.01)  # DVOL is in vol points
            if kind == "candles_1h":
                return self._candles(
                    spec, self._perpetual(spec, book), as_of, int(spec.params["lookback_days"])
                )
            if kind == "futures_book":
                return self._futures_book(spec, book)
            if kind == "funding":
                return self._funding(spec, self._perpetual(spec, book))
            if kind == "order_book":
                return self._order_book(spec, str(spec.params["instrument_name"]))
            if kind == "rv_daily":
                instrument = spec.params.get("instrument") or self._perpetual(spec, book)
                return self._rv_daily(spec, str(instrument), as_of)
            if kind == "dvol_history":
                if book.dvol is None:
                    raise AdapterError("bad_input_spec", f"no DVOL index for {underlying}")
                return self._dvol_history(spec, book.currency, as_of)
        except AdapterError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterError(
                "venue_unavailable", f"{spec.name}: unexpected response: {exc}"
            ) from exc
        raise AdapterError("bad_input_spec", f"{spec.name}: unknown kind {kind!r}")

    def _perpetual(self, spec: InputSpec, book) -> str:
        perp = find_perpetual(book, self.client.futures(book.currency).result)
        if perp is None:
            raise AdapterError(
                "no_perpetual", f"{spec.name}: Deribit lists no {book.base} perpetual"
            )
        return perp

    def _futures_book(self, spec: InputSpec, book) -> InputData:
        meta = self.client.futures(book.currency)
        summary = self.client.futures_summary(book.currency)
        rows = normalize_futures(book, meta.result, summary.result)
        return InputData(
            spec.name,
            summary.server_time,
            {"rows": rows},
            raw={"get_instruments": meta.raw, "get_book_summary_by_currency": summary.raw},
        )

    def _funding(self, spec: InputSpec, perp: str) -> InputData:
        r = self.client.ticker(perp)
        data = {
            "instrument_name": perp,
            "current_funding": r.result.get("current_funding"),
            "funding_8h": r.result.get("funding_8h"),
        }
        return InputData(spec.name, r.server_time, data, raw=r.raw)

    def _order_book(self, spec: InputSpec, instrument: str) -> InputData:
        depth = int(spec.params.get("depth", 5))
        r = self.client.order_book(instrument, depth)
        # D-13: each book's own matching-engine timestamp, so leg time spread is measurable.
        stamp = r.result.get("timestamp")
        at = datetime.fromtimestamp(stamp / 1000, UTC) if isinstance(stamp, int) else r.server_time
        data = {
            "instrument_name": instrument,
            "bids": book_side(r.result.get("bids")),
            "asks": book_side(r.result.get("asks")),
        }
        return InputData(spec.name, at, data, raw=r.raw)

    def _chain(self, spec: InputSpec, underlying: str, book) -> InputData:
        meta = self.client.instruments(book.currency)
        summary = self.client.book_summary(book.currency)
        rows = normalize(underlying, meta.result, summary.result)
        if not rows:
            raise AdapterError("venue_unavailable", f"{spec.name}: no {underlying} options in book")
        return InputData(
            spec.name,
            summary.server_time,
            {"rows": rows},
            raw={"get_instruments": meta.raw, "get_book_summary_by_currency": summary.raw},
        )

    def _index(self, spec: InputSpec, index_name: str, key: str, scale: float = 1.0) -> InputData:
        r = self.client.index_price(index_name)
        price = float(r.result["index_price"])
        if price <= 0:
            raise AdapterError("venue_unavailable", f"{spec.name}: non-positive {index_name}")
        return InputData(spec.name, r.server_time, {key: price * scale}, raw=r.raw)

    def _candles(self, spec: InputSpec, perp: str, as_of: datetime, days: int) -> InputData:
        r = self.client.candles(perp, as_of - timedelta(days=days), as_of)
        if r.result.get("status") != "ok":
            raise AdapterError(
                "venue_unavailable", f"{spec.name}: candles status {r.result.get('status')}"
            )
        ticks, closes = r.result["ticks"], r.result["close"]
        return InputData(
            spec.name,
            r.server_time,
            {"t": [iso_ms(t) for t in ticks], "close": [float(c) for c in closes]},
            raw=r.raw,
        )

    def _rv_daily(self, spec: InputSpec, instrument: str, as_of: datetime) -> InputData:
        resolution = str(spec.params["resolution"])

        def page(start_ms: int, end_ms: int):
            # end_timestamp is inclusive at Deribit; the shared code drops bars outside the window.
            r = self.client.chart_data(instrument, resolution, start_ms, end_ms - 1)
            status = r.result.get("status")
            if status == "no_data":
                return [], [], r.server_time, r.raw
            if status != "ok":
                raise AdapterError("venue_unavailable", f"{spec.name}: candles status {status}")
            closes = [float(c) for c in r.result["close"]]
            return [int(t) for t in r.result["ticks"]], closes, r.server_time, r.raw

        data, server_time, raws = fetch_rv_daily(
            spec.name,
            venue="deribit",
            instrument=instrument,
            resolution=resolution,
            lookback_days=int(spec.params["lookback_days"]),
            min_coverage=float(spec.params["min_coverage"]),
            as_of=as_of,
            rows_per_call=CHART_ROWS_PER_CALL,
            fetch_page=page,
            cache=self.rv_cache,
        )
        if server_time is None:  # every day was cached: the server clock still dates the input
            server_time = self.client.time().server_time
        return InputData(spec.name, server_time, data, raw={"get_tradingview_chart_data": raws})

    def _dvol_history(self, spec: InputSpec, currency: str, as_of: datetime) -> InputData:
        """Daily DVOL closes, in decimal (Deribit quotes DVOL in vol points: 55.5 → 0.555).

        Daily candles (resolution 1D). Day d's value is the close of the last candle closing in
        (d, d + 1d], i.e. the latest DVOL known at d's close: the candle opening at d when they
        open at 00:00 UTC. Whether they do is UNVERIFIED (check live with `make probe`), so the
        keying doesn't assume it: a candle opening at `ts` is day ceil_day(ts), which is `ts`
        when aligned and never a candle still open at C. `close_offset_h` is the hour of day
        (UTC) the candles close at, 0 when aligned, None with no candle; candles at mixed hours
        are not daily candles and fail. Completed days only, gap-free, None where Deribit has no
        candle. Pages back from C via `continuation` (next end_timestamp)."""
        end = day_floor_ms(as_of)
        days = list(range(end - int(spec.params["lookback_days"]) * MS_PER_DAY, end, MS_PER_DAY))
        lo = days[0] - MS_PER_DAY  # day[0]'s candle opens after this
        closes: dict[int, float] = {}
        offsets: set[int] = set()
        raws, cursor, r = [], end - 1, None
        for _ in range(DVOL_MAX_PAGES):
            r = self.client.volatility_index_page(currency, lo + 1, cursor, "1D")
            raws.append(r.raw)
            for ts, *_ohl, close in r.result["data"]:
                day = int(ts) + -int(ts) % MS_PER_DAY  # ceil to the day: the candle closes in it
                if days[0] <= day < end and close is not None and float(close) > 0:
                    offsets.add(int(ts) % MS_PER_DAY)
                    closes[day] = float(close) * 0.01
            nxt = r.result.get("continuation")
            if nxt is None or nxt <= lo:
                break
            if nxt >= cursor:
                raise AdapterError("venue_unavailable", f"{spec.name}: continuation did not move")
            cursor = nxt
        else:
            raise AdapterError("venue_unavailable", f"{spec.name}: over {DVOL_MAX_PAGES} pages")
        if len(offsets) > 1:
            raise AdapterError("venue_unavailable", f"{spec.name}: DVOL candles at mixed hours")
        offset_h = offsets.pop() / 3_600_000 if offsets else None
        data = {
            "day": [iso_day(d) for d in days],
            "dvol": [closes.get(d) for d in days],
            "close_offset_h": offset_h,
        }
        return InputData(spec.name, r.server_time, data, raw={"get_volatility_index_data": raws})
