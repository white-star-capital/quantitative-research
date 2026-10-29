"""Hyperliquid public info API over HTTP: `POST /info` (keyless).

Timeouts/retries/backoff with jitter via HttpSource (M-7). Any failure is `venue_unavailable`
(D-8). An optional `capture` hook sees every response body exactly as received, with the request
type and its `req` fields (used by tools/record_deribit.py).
"""

from __future__ import annotations

import json
from typing import Any

from terminal_models.adapters.base import AdapterError
from terminal_models.adapters.deribit.client import Capture
from terminal_models.adapters.http import HttpSource

BASE_URL = "https://api.hyperliquid.xyz"


def parse_candles(method: str, text: str) -> list[dict[str, Any]]:
    """candleSnapshot body → list of candles; each needs open time `t` and close `c`."""
    try:
        body = json.loads(text)
    except ValueError as exc:
        raise AdapterError(
            "venue_unavailable", f"hyperliquid {method}: invalid JSON: {exc}"
        ) from exc
    if not isinstance(body, list) or not all(
        isinstance(k, dict) and isinstance(k.get("t"), int) and "c" in k for k in body
    ):
        raise AdapterError("venue_unavailable", f"hyperliquid {method}: not a candle list")
    return body


class HyperliquidClient:
    def __init__(self, source: HttpSource | None = None, capture: Capture | None = None) -> None:
        self.source = source or HttpSource(
            BASE_URL, timeout_seconds=10.0, max_attempts=4, backoff_seconds=1.0, jitter=0.5
        )
        self._capture = capture

    def call(self, method: str, **req: Any) -> list[dict[str, Any]]:
        try:
            text = self.source.request_text("POST", "/info", json={"type": method, "req": req})
        except AdapterError as exc:
            raise AdapterError("venue_unavailable", f"hyperliquid {method}: {exc.message}") from exc
        if self._capture:
            self._capture(method, req, text)
        return parse_candles(method, text)

    def candles(self, coin: str, interval: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
        """Candles opening in [start_ms, end_ms]."""
        return self.call(
            "candleSnapshot", coin=coin, interval=interval, startTime=start_ms, endTime=end_ms
        )
