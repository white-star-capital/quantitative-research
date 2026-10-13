"""Deribit public API: option chains, mark IV, greeks, index price (MODELS.md §3.2). Keyless."""

from __future__ import annotations

from datetime import datetime

from terminal_models.adapters.base import DataAdapter
from terminal_models.adapters.http import HttpSource
from terminal_models.core.base import InputData, InputSpec

BASE_URL = "https://www.deribit.com"


class DeribitAdapter(DataAdapter):
    name = "deribit"

    def __init__(self, source: HttpSource | None = None) -> None:
        self.source = source or HttpSource(BASE_URL)

    def fetch(self, spec: InputSpec, as_of: datetime) -> InputData:
        # M1: GET /api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option
        # and /api/v2/public/get_index_price. snapshot_at from the response's timestamp fields.
        raise NotImplementedError("Deribit adapter is implemented in M1")
