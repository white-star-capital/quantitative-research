"""Deribit public JSON-RPC over HTTP (BS_PRICER_IMPLEMENTATION_GUIDE.md §2.1–2.2). Keyless.

D-1 timeouts/retries/backoff with jitter (via HttpSource), D-4 server timestamps. Responses
are cached briefly so SOL and HYPE share one USDC book-summary fetch per tick, and instrument
metadata (which changes daily) is fetched at most hourly.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from terminal_models.adapters.base import AdapterError
from terminal_models.adapters.http import HttpSource

BASE_URL = "https://www.deribit.com"
INSTRUMENTS_TTL_S = 3600
SUMMARY_TTL_S = 60


class Response:
    def __init__(self, result: Any, server_time: datetime, raw: dict[str, Any]) -> None:
        self.result = result
        self.server_time = server_time
        self.raw = raw


class DeribitClient:
    def __init__(self, source: HttpSource | None = None, clock=time.monotonic) -> None:
        self.source = source or HttpSource(
            BASE_URL, timeout_seconds=10.0, max_attempts=4, backoff_seconds=1.0, jitter=0.5
        )
        self._clock = clock
        self._cache: dict[tuple, tuple[float, Response]] = {}

    def call(self, method: str, ttl_s: float = 0, **params: Any) -> Response:
        key = (method, tuple(sorted(params.items())))
        hit = self._cache.get(key)
        if ttl_s and hit and self._clock() - hit[0] < ttl_s:
            return hit[1]
        body = self.source.request_json("GET", f"/api/v2/public/{method}", params=params)
        if not isinstance(body, dict) or "result" not in body:
            error = body.get("error") if isinstance(body, dict) else None
            raise AdapterError("venue_unavailable", f"deribit {method}: {error or 'no result'}")
        us_out = body.get("usOut")
        if not isinstance(us_out, int):
            raise AdapterError("venue_unavailable", f"deribit {method}: response has no usOut")
        response = Response(body["result"], datetime.fromtimestamp(us_out / 1e6, UTC), body)
        if ttl_s:
            self._cache[key] = (self._clock(), response)
        return response

    def instruments(self, currency: str) -> Response:
        return self.call(
            "get_instruments", INSTRUMENTS_TTL_S, currency=currency, kind="option", expired="false"
        )

    def book_summary(self, currency: str) -> Response:
        return self.call(
            "get_book_summary_by_currency", SUMMARY_TTL_S, currency=currency, kind="option"
        )

    def index_price(self, index_name: str) -> Response:
        return self.call("get_index_price", index_name=index_name)

    def candles(self, instrument: str, start: datetime, end: datetime) -> Response:
        return self.call(
            "get_tradingview_chart_data",
            instrument_name=instrument,
            start_timestamp=int(start.timestamp() * 1000),
            end_timestamp=int(end.timestamp() * 1000),
            resolution="60",
        )
