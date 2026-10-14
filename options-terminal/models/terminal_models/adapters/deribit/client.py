"""Deribit public JSON-RPC over HTTP (BS_PRICER_IMPLEMENTATION_GUIDE.md §2.1–2.2). Keyless.

D-1 timeouts/retries/backoff with jitter (via HttpSource), D-4 server timestamps. Responses go
through the tick snapshot cache (snapshot_cache.py, D-11): SOL and HYPE share one USDC book
fetch, bs_pricer and parity_monitor one option book; instrument metadata (which changes daily)
is fetched at most hourly. An optional `capture` hook sees every network response body exactly as
received (used by tools/record_deribit.py).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from terminal_models.adapters.base import AdapterError
from terminal_models.adapters.deribit.snapshot_cache import (
    METADATA_REUSE_S,
    TICK_MAX_REUSE_S,
    SnapshotCache,
)
from terminal_models.adapters.http import HttpSource

BASE_URL = "https://www.deribit.com"

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
        self._capture = capture
        self.cache = SnapshotCache(clock)

    def call(self, method: str, ttl_s: float = 0, **params: Any) -> Response:
        key = SnapshotCache.key(method, params)
        if ttl_s and (hit := self.cache.get(key, ttl_s)) is not None:
            return hit
        text = self.source.request_text("GET", f"/api/v2/public/{method}", params=params)
        if self._capture:
            self._capture(method, params, text)
        response = parse(method, text)
        if ttl_s:
            self.cache.put(key, response)
        return response

    def instruments(self, currency: str) -> Response:
        return self.call(
            "get_instruments", METADATA_REUSE_S, currency=currency, kind="option", expired="false"
        )

    def futures(self, currency: str) -> Response:
        """Futures metadata, used to discover each asset's perpetual (never constructed, D-2)."""
        return self.call("get_instruments", METADATA_REUSE_S, currency=currency, kind="future")

    def book_summary(self, currency: str) -> Response:
        return self.call(
            "get_book_summary_by_currency", TICK_MAX_REUSE_S, currency=currency, kind="option"
        )

    def futures_summary(self, currency: str) -> Response:
        """Dated futures and perps: bid/ask/mark in one call (parity D-11)."""
        return self.call(
            "get_book_summary_by_currency", TICK_MAX_REUSE_S, currency=currency, kind="future"
        )

    def ticker(self, instrument: str) -> Response:
        return self.call("ticker", TICK_MAX_REUSE_S, instrument_name=instrument)

    def order_book(self, instrument: str, depth: int) -> Response:
        """Never cached: confirmations must be fetched back to back (D-13)."""
        return self.call("get_order_book", instrument_name=instrument, depth=depth)

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
