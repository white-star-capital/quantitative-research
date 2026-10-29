"""Replay the Hyperliquid calls in a recording (tools/record_deribit.py) through the production
adapter code. Same recording layout and sha256 check as adapters/deribit/replay.py; requests
are matched on their exact `req` fields, time window included."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from terminal_models.adapters.base import AdapterError
from terminal_models.adapters.deribit.replay import load_bodies, request_key
from terminal_models.adapters.hyperliquid.client import HyperliquidClient, parse_candles


class RecordedHyperliquidClient(HyperliquidClient):
    def __init__(self, root: Path) -> None:
        super().__init__(source=None)
        self.root = root
        self.manifest, self._bodies = load_bodies(root)

    def call(self, method: str, **req: Any) -> list[dict[str, Any]]:
        body = self._bodies.get(request_key(method, req, exact=True))
        if body is None:
            raise AdapterError("venue_unavailable", f"{method} {req} is not in the recording")
        return parse_candles(method, body)
