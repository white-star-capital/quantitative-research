"""Deribit inputs for parity_monitor (guide §2.1, D-11..D-15): futures book joined to metadata,
funding from the discovered perp, order books with their own timestamps, tick cache reuse.
Canned JSON-RPC shapes that exercise the plumbing; they never reach a payload or fixture."""

from datetime import UTC, datetime

import httpx
import pytest

from terminal_models.adapters.deribit import DeribitAdapter
from terminal_models.adapters.deribit.client import DeribitClient
from terminal_models.adapters.deribit.snapshot_cache import TICK_MAX_REUSE_S
from terminal_models.adapters.http import HttpSource
from terminal_models.core.base import InputSpec

AS_OF = datetime(2026, 10, 14, 12, tzinfo=UTC)
US_OUT = int(AS_OF.timestamp() * 1e6)
BOOK_TS = int(AS_OF.timestamp() * 1000) - 1500  # matching-engine time, 1.5 s before usOut


def future(name, base, settle, period, expiry, size, kind):
    return {"instrument_name": name, "base_currency": base, "settlement_currency": settle,
            "settlement_period": period, "expiration_timestamp": expiry, "contract_size": size,
            "instrument_type": kind, "is_active": True, "kind": "future"}  # fmt: skip


FUTURES = {
    "BTC": [
        future("BTC-27NOV26", "BTC", "BTC", "month", 1_795_766_400_000, 10.0, "reversed"),
        future("BTC-PERPETUAL", "BTC", "BTC", "perpetual", 32_503_708_800_000, 10.0, "reversed"),
    ],
    "USDC": [
        future("SOL_USDC-PERPETUAL", "SOL", "USDC", "perpetual", 32_503_708_800_000, 1.0, "linear"),
        future("HYPE_USDC-PERPETUAL", "HYPE", "USDC", "perpetual", 32_503_708_800_000, 1.0,
               "linear"),
    ],
}  # fmt: skip
SUMMARIES = {
    "BTC": [
        {"instrument_name": "BTC-27NOV26", "bid_price": 61_000.0, "ask_price": 61_010.0,
         "mark_price": 61_005.0},
        {"instrument_name": "BTC-PERPETUAL", "bid_price": 60_100.0, "ask_price": 60_100.5,
         "mark_price": 60_100.2},
        {"instrument_name": "BTC-UNLISTED", "bid_price": 1.0, "ask_price": 2.0},  # no metadata
    ],
    "USDC": [
        {"instrument_name": "SOL_USDC-PERPETUAL", "bid_price": 150.1, "ask_price": 0,
         "mark_price": 150.2},
    ],
}  # fmt: skip


def make_client(calls, clock=None):
    def handler(request: httpx.Request):
        method = request.url.path.rsplit("/", 1)[-1]
        p = request.url.params
        calls.append(method)
        if method == "get_instruments":
            result = FUTURES[p["currency"]]
        elif method == "get_book_summary_by_currency":
            result = SUMMARIES[p["currency"]]
        elif method == "ticker":
            result = {"instrument_name": p["instrument_name"], "current_funding": 0.00001,
                      "funding_8h": 0.00012, "timestamp": BOOK_TS}  # fmt: skip
        elif method == "get_order_book":
            result = {"instrument_name": p["instrument_name"], "timestamp": BOOK_TS,
                      "bids": [[0.05, 3.0], [0.049, 0.0]], "asks": [[0.052, 1.5]]}  # fmt: skip
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"result": result, "usOut": US_OUT})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x.test")
    source = HttpSource("https://x.test", backoff_seconds=0, client=http)
    return DeribitClient(source, clock=clock) if clock else DeribitClient(source)


def fetch(adapter, kind, underlying="BTC", **params):
    spec = InputSpec(kind, "deribit", {"kind": kind, "underlying": underlying} | params)
    return adapter.fetch(spec, AS_OF)


def test_futures_book_joins_metadata_and_marks_perps():
    data = fetch(DeribitAdapter(make_client([])), "futures_book")
    rows = {r["instrument_name"]: r for r in data.data["rows"]}
    assert set(rows) == {"BTC-27NOV26", "BTC-PERPETUAL"}  # unlisted name dropped (D-15)
    assert rows["BTC-27NOV26"]["expiration_ms"] == 1_795_766_400_000
    assert rows["BTC-PERPETUAL"]["is_perpetual"] and rows["BTC-PERPETUAL"]["expiration_ms"] is None
    assert rows["BTC-27NOV26"]["settlement"] == "inverse"
    assert rows["BTC-27NOV26"]["contract_size"] == 10.0  # from metadata (D-3)
    assert data.snapshot_at == AS_OF


def test_usdc_futures_book_is_per_asset_and_zero_is_missing():
    rows = fetch(DeribitAdapter(make_client([])), "futures_book", "SOL").data["rows"]
    assert [r["instrument_name"] for r in rows] == ["SOL_USDC-PERPETUAL"]
    assert rows[0]["ask"] is None and rows[0]["settlement"] == "linear"


def test_funding_comes_from_the_discovered_perp():
    data = fetch(DeribitAdapter(make_client([])), "funding", "HYPE")
    assert data.data == {"instrument_name": "HYPE_USDC-PERPETUAL", "current_funding": 0.00001,
                         "funding_8h": 0.00012}  # fmt: skip


def test_order_book_keeps_its_own_timestamp_and_drops_empty_levels():
    data = fetch(DeribitAdapter(make_client([])), "order_book", instrument_name="BTC-X", depth=5)
    assert data.snapshot_at == datetime.fromtimestamp(BOOK_TS / 1000, UTC)
    assert data.data["bids"] == [[0.05, 3.0]] and data.data["asks"] == [[0.052, 1.5]]


def test_tick_cache_reuses_books_but_never_order_books():
    now = [0.0]
    calls: list[str] = []
    adapter = DeribitAdapter(make_client(calls, clock=lambda: now[0]))
    fetch(adapter, "futures_book")
    fetch(adapter, "futures_book", "BTC")
    assert calls.count("get_book_summary_by_currency") == 1
    now[0] = TICK_MAX_REUSE_S + 1  # older than the reuse limit: refetched
    fetch(adapter, "futures_book")
    assert calls.count("get_book_summary_by_currency") == 2
    assert calls.count("get_instruments") == 1  # metadata reused for an hour
    for _ in range(2):
        fetch(adapter, "order_book", instrument_name="BTC-X")
    assert calls.count("get_order_book") == 2


@pytest.mark.parametrize("kind", ["funding", "futures_book"])
def test_unknown_underlying_is_rejected(kind):
    with pytest.raises(Exception, match="unknown underlying"):
        fetch(DeribitAdapter(make_client([])), kind, "DOGE")
