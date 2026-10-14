"""Deribit public API: option chains, spot index, perp candles, DVOL (MODELS.md §3.2). Keyless.

Input kinds (InputSpec params["kind"]): `chain`, `index`, `candles_1h`, `dvol`, and for
parity_monitor `futures_book`, `funding` and `order_book` (PARITY_MONITOR_IMPLEMENTATION_GUIDE.md
§2.1). Any failure after retries is an AdapterError with code `venue_unavailable` (D-8); there is
no fallback data (D-10).
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
from terminal_models.core.base import InputData, InputSpec


class DeribitAdapter(DataAdapter):
    name = "deribit"

    def __init__(self, client: DeribitClient | None = None) -> None:
        self.client = client or DeribitClient()

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
