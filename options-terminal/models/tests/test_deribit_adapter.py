"""Deribit adapter: D-2 join, D-3 contract size, D-4 server time, D-5 USD conversion, caching,
failure codes. Responses are canned JSON-RPC shapes that exercise conversion arithmetic; they
never reach a payload or fixture."""

from datetime import UTC, datetime

import httpx
import pytest

from terminal_models.adapters.base import AdapterError
from terminal_models.adapters.deribit import DeribitAdapter
from terminal_models.adapters.deribit.client import DeribitClient
from terminal_models.adapters.http import HttpSource
from terminal_models.core.base import InputSpec

AS_OF = datetime(2026, 10, 13, 17, tzinfo=UTC)
US_OUT = 1_791_910_800_000_000  # 2026-10-13T17:00:00Z in µs


def meta(name, base, settle, strike, opt, size):
    return {
        "instrument_name": name,
        "base_currency": base,
        "settlement_currency": settle,
        "strike": strike,
        "option_type": opt,
        "contract_size": size,
        "kind": "option",
        "expiration_timestamp": 1_793_088_000_000,
    }


def summary(name, under, mark, bid, ask, iv):
    return {
        "instrument_name": name,
        "underlying_price": under,
        "mark_price": mark,
        "bid_price": bid,
        "ask_price": ask,
        "mark_iv": iv,
        "open_interest": 10,
    }


ROUTES = {
    ("get_instruments", "ETH"): [meta("ETH-27OCT26-2600-C", "ETH", "ETH", 2600, "call", 1.0)],
    ("get_book_summary_by_currency", "ETH"): [
        summary("ETH-27OCT26-2600-C", 2000.0, 0.01, 0.0, 0.012, 55.0),
        summary("ETH-UNKNOWN", 2000.0, 0.01, 0.009, 0.012, 55.0),  # no metadata → skipped
    ],
    ("get_instruments", "USDC"): [
        meta("SOL_USDC-27OCT26-9d2-P", "SOL", "USDC", 9.2, "put", 10.0),
        meta("HYPE_USDC-27OCT26-40-C", "HYPE", "USDC", 40, "call", 1.0),
    ],
    ("get_book_summary_by_currency", "USDC"): [
        summary("SOL_USDC-27OCT26-9d2-P", 10.0, 0.5, 0.45, None, 80.0),
        summary("HYPE_USDC-27OCT26-40-C", 38.0, 1.2, 1.1, 1.3, 95.0),
    ],
}


def client(calls):
    def handler(request: httpx.Request):
        method = request.url.path.rsplit("/", 1)[-1]
        calls.append(method)
        currency = request.url.params.get("currency")
        if method == "get_index_price":
            return httpx.Response(200, json={"result": {"index_price": 2001.5}, "usOut": US_OUT})
        return httpx.Response(200, json={"result": ROUTES[(method, currency)], "usOut": US_OUT})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x.test")
    return DeribitClient(HttpSource("https://x.test", backoff_seconds=0, client=http))


def chain(adapter, u):
    return adapter.fetch(InputSpec("chain", "deribit", {"kind": "chain", "underlying": u}), AS_OF)


def test_inverse_book_converts_coin_prices_with_forward():
    data = chain(DeribitAdapter(client([])), "ETH")
    assert data.snapshot_at == AS_OF  # usOut, not local time (D-4)
    [row] = data.data["rows"]  # unknown instrument skipped (D-2)
    assert row["settlement"] == "inverse"
    assert row["mark_usd"] == pytest.approx(20.0)  # 0.01 ETH × 2000
    assert row["bid_usd"] is None  # zero bid is missing, not $0
    assert row["mark_iv"] == pytest.approx(0.55)
    assert row["expiry_ts"].endswith("Z") and "raw" not in row


def test_linear_books_share_one_usdc_fetch():
    calls = []
    adapter = DeribitAdapter(client(calls))
    [sol] = chain(adapter, "SOL").data["rows"]
    [hype] = chain(adapter, "HYPE").data["rows"]
    assert calls.count("get_book_summary_by_currency") == 1
    assert sol["strike"] == 9.2 and sol["contract_size"] == 10.0  # fractional strike via metadata
    assert sol["mark_usd"] == 0.5 and sol["ask_usd"] is None and sol["settlement"] == "linear"
    assert hype["instrument_name"] == "HYPE_USDC-27OCT26-40-C"


def test_index_and_unknown_inputs():
    adapter = DeribitAdapter(client([]))
    spot = adapter.fetch(
        InputSpec("index", "deribit", {"kind": "index", "underlying": "BTC"}), AS_OF
    )
    assert spot.data == {"spot": 2001.5}
    with pytest.raises(AdapterError) as exc:
        adapter.fetch(InputSpec("dvol", "deribit", {"kind": "dvol", "underlying": "SOL"}), AS_OF)
    assert exc.value.code == "bad_input_spec"


@pytest.mark.parametrize(
    "body",
    [{"error": {"code": 10000, "message": "boom"}}, {"result": []}],  # error / no usOut
)
def test_bad_responses_are_venue_unavailable(body):
    http = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)),
        base_url="https://x.test",
    )
    adapter = DeribitAdapter(
        DeribitClient(HttpSource("https://x.test", backoff_seconds=0, client=http))
    )
    with pytest.raises(AdapterError) as exc:
        adapter.fetch(InputSpec("index", "deribit", {"kind": "index", "underlying": "BTC"}), AS_OF)
    assert exc.value.code == "venue_unavailable"
