"""Hyperliquid public info API: underlying spot/perp prices (MODELS.md §3.2). Keyless."""

from __future__ import annotations

from datetime import datetime

from terminal_models.adapters.base import DataAdapter
from terminal_models.adapters.http import HttpSource
from terminal_models.core.base import InputData, InputSpec

BASE_URL = "https://api.hyperliquid.xyz"


class HyperliquidAdapter(DataAdapter):
    name = "hyperliquid"

    def __init__(self, source: HttpSource | None = None) -> None:
        self.source = source or HttpSource(BASE_URL)

    def fetch(self, spec: InputSpec, as_of: datetime) -> InputData:
        # M1: POST /info with {"type": "allMids"} / {"type": "candleSnapshot", ...}.
        # snapshot_at must come from the response (candle close time), not the local clock.
        raise NotImplementedError("Hyperliquid adapter is implemented in M1")
