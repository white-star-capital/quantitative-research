"""Deribit public JSON-RPC over HTTP (BS_PRICER_IMPLEMENTATION_GUIDE.md §2.1–2.2). Keyless.

D-1 timeouts/retries/backoff with jitter (via HttpSource), D-4 server timestamps. Responses
are cached briefly so SOL and HYPE share one USDC book-summary fetch per tick, and instrument
metadata (which changes daily) is fetched at most hourly. An optional `capture` hook sees every
network response body exactly as received (used by tools/record_deribit.py).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from terminal_models.adapters.base import AdapterError
from terminal_models.adapters.http import HttpSource

BASE_URL = "https://www.deribit.com"
INSTRUMENTS_TTL_S = 3600
SUMMARY_TTL_S = 60

# (method, params, body text) for every network response.
Capture = Callable[[str, dict[str, Any], str], None]


class Response:
    def __init__(self, result: Any, server_time: datetime, raw: dict[str, Any]) -> None:
        self.result = result
        self.server_time = server_time
        self.raw = raw


def parse(method: str, text: str) -> Response:
    """JSON-RPC envelope → Response. Requires `result` and an integer `usOut` (D-4)."""
    try:
        body = json.loads(text)
    except ValueError as exc:
        raise AdapterError("venue_unavailable", f"deribit {method}: invalid JSON: {exc}") from exc
    if not isinstance(body, dict) or "result" not in body:
        error = body.get("error") if isinstance(body, dict) else None
        raise AdapterError("venue_unavailable", f"deribit {method}: {error or 'no result'}")
    us_out = body.get("usOut")
    if not isinstance(us_out, int):
        raise AdapterError("venue_unavailable", f"deribit {method}: response has no usOut")
    return Response(body["result"], datetime.fromtimestamp(us_out / 1e6, UTC), body)


class DeribitClient:
    def __init__(
        self, source: HttpSource | None = None, clock=time.monotonic, capture: Capture | None = None
    ) -> None:
        self.source = source or HttpSource(
            BASE_URL, timeout_seconds=10.0, max_attempts=4, backoff_seconds=1.0, jitter=0.5
        )
        self._clock = clock
        self._capture = capture
        self._cache: dict[tuple, tuple[float, Response]] = {}

    def call(self, method: str, ttl_s: float = 0, **params: Any) -> Response:
        key = (method, tuple(sorted(params.items())))
        hit = self._cache.get(key)
        if ttl_s and hit and self._clock() - hit[0] < ttl_s:
            return hit[1]
        text = self.source.request_text("GET", f"/api/v2/public/{method}", params=params)
        if self._capture:
            self._capture(method, params, text)
        response = parse(method, text)
        if ttl_s:
            self._cache[key] = (self._clock(), response)
        return response

    def instruments(self, currency: str) -> Response:
        return self.call(
            "get_instruments", INSTRUMENTS_TTL_S, currency=currency, kind="option", expired="false"
        )

    def futures(self, currency: str) -> Response:
        """Futures metadata, used to discover each asset's perpetual (never constructed, D-2)."""
        return self.call("get_instruments", INSTRUMENTS_TTL_S, currency=currency, kind="future")

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

    def volatility_index(self, currency: str, start: datetime, end: datetime) -> Response:
        return self.call(
            "get_volatility_index_data",
            currency=currency,
            start_timestamp=int(start.timestamp() * 1000),
            end_timestamp=int(end.timestamp() * 1000),
            resolution="3600",
        )
