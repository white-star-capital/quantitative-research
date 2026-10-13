"""Deribit public API: option chains, spot index, perp candles, DVOL (MODELS.md §3.2). Keyless.

Input kinds (InputSpec params["kind"]): `chain`, `index`, `candles_1h`, `dvol`. Any failure after
retries is an AdapterError with code `venue_unavailable` (D-8); there is no fallback data (D-10).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from terminal_models.adapters.base import AdapterError, DataAdapter
from terminal_models.adapters.deribit.chains import BOOKS, iso_ms, normalize
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
                return self._candles(spec, book.perp, as_of, int(spec.params["lookback_days"]))
        except AdapterError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterError(
                "venue_unavailable", f"{spec.name}: unexpected response: {exc}"
            ) from exc
        raise AdapterError("bad_input_spec", f"{spec.name}: unknown kind {kind!r}")

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
