"""Recorder and replay: raw bodies kept byte-for-byte, replay goes through the production adapter,
perps discovered from metadata. Responses are canned JSON-RPC shapes that exercise the plumbing;
they never reach a payload or fixture."""

import gzip
import json
from datetime import UTC, datetime

import httpx
import pytest

from terminal_models.adapters.base import AdapterError
from terminal_models.adapters.deribit import DeribitAdapter
from terminal_models.adapters.deribit.chains import BOOKS, find_perpetual, is_inverse
from terminal_models.adapters.deribit.client import DeribitClient
from terminal_models.adapters.deribit.replay import RecordedClient, RecordingError
from terminal_models.adapters.http import HttpSource
from terminal_models.core.base import InputSpec
from terminal_models.tools.record_deribit import record

US_OUT = 1_791_910_800_123_456


def future(name, base, settle, period="perpetual", active=True):
    return {
        "instrument_name": name,
        "base_currency": base,
        "settlement_currency": settle,
        "settlement_period": period,
        "is_active": active,
        "kind": "future",
        "contract_size": 1.0,
        "expiration_timestamp": 32_503_708_800_000 if period == "perpetual" else 1_793_088_000_000,
    }


OPTIONS = {
    "ETH": [
        {
            "instrument_name": "ETH-27OCT26-2600-C",
            "base_currency": "ETH",
            "settlement_currency": "ETH",
            "instrument_type": "reversed",
            "strike": 2600,
            "option_type": "call",
            "contract_size": 1.0,
            "kind": "option",
            "expiration_timestamp": 1_793_088_000_000,
        }
    ]
}
FUTURES = {
    "ETH": [
        future("ETH-27OCT26", "ETH", "ETH", period="month"),
        future("ETH_USDC-PERPETUAL", "ETH", "USDC"),
        future("ETH-PERPETUAL", "ETH", "ETH"),
    ]
}
SUMMARIES = {
    "ETH": [
        {
            "instrument_name": "ETH-27OCT26-2600-C",
            "underlying_price": 2000.0,
            "mark_price": 0.01,
            "bid_price": 0.009,
            "ask_price": 0.012,
            "bid_iv": 54.1,
            "ask_iv": 56.3,
            "mark_iv": 55.0,
            "open_interest": 10,
        }
    ]
}


def envelope(result):
    # Odd spacing and key order on purpose: the recording must keep bytes, not re-serialize.
    return (
        '{"jsonrpc":"2.0",  "result": ' + json.dumps(result) + f', "usIn": 1, "usOut": {US_OUT}}}'
    )


def handler(request: httpx.Request, fail: set[str]):
    method = request.url.path.rsplit("/", 1)[-1]
    p = request.url.params
    if method in fail:
        return httpx.Response(200, text='{"jsonrpc":"2.0","result":{}}')  # no usOut
    if method == "get_instruments":
        table = FUTURES if p["kind"] == "future" else OPTIONS
        return httpx.Response(200, text=envelope(table[p["currency"]]))
    if method == "get_book_summary_by_currency":
        if p["kind"] == "future":
            rows = [
                {
                    "instrument_name": f["instrument_name"],
                    "bid_price": 2000.0,
                    "ask_price": 2000.5,
                    "mark_price": 2000.2,
                }
                for f in FUTURES[p["currency"]]
            ]
            return httpx.Response(200, text=envelope(rows))
        return httpx.Response(200, text=envelope(SUMMARIES[p["currency"]]))
    if method == "ticker":
        result = {
            "instrument_name": p["instrument_name"],
            "current_funding": 0.0,
            "funding_8h": 0.0001,
            "timestamp": US_OUT // 1000,
        }
        return httpx.Response(200, text=envelope(result))
    if method == "get_index_price":
        return httpx.Response(
            200, text=envelope({"index_price": 2001.5, "estimated_delivery_price": 2001.5})
        )
    if method == "get_tradingview_chart_data":
        assert p["instrument_name"] == "ETH-PERPETUAL"
        t0 = int(p["end_timestamp"]) - 3_600_000
        return httpx.Response(
            200, text=envelope({"status": "ok", "ticks": [t0], "close": [2000.25]})
        )
    if method == "get_volatility_index_data":
        return httpx.Response(
            200, text=envelope({"data": [[1, 55, 56, 54, 55.5]], "continuation": None})
        )
    return httpx.Response(404)


def factory(calls: list[str], fail: set[str] = frozenset()):
    def make(capture=None):
        def h(request):
            calls.append(request.url.path.rsplit("/", 1)[-1])
            return handler(request, fail)

        http = httpx.Client(transport=httpx.MockTransport(h), base_url="https://x.test")
        source = HttpSource("https://x.test", max_attempts=1, backoff_seconds=0, client=http)
        return DeribitClient(source, capture=capture)

    return make


def test_perp_discovery_never_constructs_names():
    assert find_perpetual(BOOKS["ETH"], FUTURES["ETH"]) == "ETH-PERPETUAL"
    usdc = [
        future("SOL_USDC-PERPETUAL", "SOL", "USDC"),
        future("SOL-X", "SOL", "SOL", active=False),
    ]
    assert find_perpetual(BOOKS["SOL"], usdc) == "SOL_USDC-PERPETUAL"
    # Only a differently-settled perp listed: use it, still a listed name.
    assert find_perpetual(BOOKS["ETH"], [future("ETH_USDC-PERPETUAL", "ETH", "USDC")]) == (
        "ETH_USDC-PERPETUAL"
    )
    assert find_perpetual(BOOKS["HYPE"], FUTURES["ETH"]) is None


def test_instrument_type_decides_inverse():
    book = BOOKS["BTC"]
    assert is_inverse({"instrument_type": "reversed", "settlement_currency": "USDC"}, book)
    assert not is_inverse({"instrument_type": "linear", "settlement_currency": "BTC"}, book)
    assert is_inverse({"settlement_currency": "BTC"}, book)  # fallback
    assert not is_inverse({"settlement_currency": "USDC"}, book)


def test_missing_perpetual_is_an_error():
    client = factory([])()
    client.futures = lambda currency: type("R", (), {"result": []})()
    spec = InputSpec(
        "candles", "deribit", {"kind": "candles_1h", "underlying": "ETH", "lookback_days": 1}
    )
    with pytest.raises(AdapterError) as err:
        DeribitAdapter(client).fetch(spec, datetime(2026, 10, 13, tzinfo=UTC))
    assert err.value.code == "no_perpetual"


def test_recording_keeps_bytes_and_replays_like_live(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_GIT_SHA", "a1b2c3d")
    out = tmp_path / "rec"
    assert record(out, ["ETH"], allow_partial=False, client_factory=factory([])) == 0

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["complete"] and manifest["assets"]["ETH"]["ok"]
    assert manifest["git_sha"] == "a1b2c3d"
    methods = [c["method"] for c in manifest["calls"]]
    assert "get_volatility_index_data" in methods
    for entry in manifest["calls"]:
        body = gzip.decompress((out / entry["body"]).read_bytes()).decode()
        assert body.startswith('{"jsonrpc":"2.0",  "result": ')  # verbatim, not re-serialized
        meta = json.loads((out / entry["meta"]).read_text())
        assert meta["url"].startswith("https://www.deribit.com/api/v2/public/" + entry["method"])
        assert meta["received_at_local"].endswith("Z")

    as_of = datetime.fromisoformat(manifest["started_at"])
    live, replayed = DeribitAdapter(factory([])()), DeribitAdapter(RecordedClient(out))
    for kind in ("chain", "index", "dvol"):
        spec = InputSpec(kind, "deribit", {"kind": kind, "underlying": "ETH"})
        a, b = live.fetch(spec, as_of), replayed.fetch(spec, as_of)
        assert (a.snapshot_at, a.data) == (b.snapshot_at, b.data)
    spec = InputSpec(
        "c", "deribit", {"kind": "candles_1h", "underlying": "ETH", "lookback_days": 31}
    )
    assert replayed.fetch(spec, as_of).data == live.fetch(spec, as_of).data

    with pytest.raises(AdapterError, match="not in the recording"):
        replayed.fetch(InputSpec("x", "deribit", {"kind": "chain", "underlying": "BTC"}), as_of)


def test_failed_call_means_no_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_GIT_SHA", "a1b2c3d")
    out = tmp_path / "rec"
    calls: list[str] = []
    assert record(out, ["ETH"], False, client_factory=factory(calls, {"get_index_price"})) == 1
    assert not (out / "manifest.json").exists()
    assert "ETH: venue_unavailable" in (out / "FAILED.txt").read_text()
    with pytest.raises(RecordingError):
        RecordedClient(out)


def test_tampered_recording_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_GIT_SHA", "a1b2c3d")
    out = tmp_path / "rec"
    assert record(out, ["ETH"], False, client_factory=factory([])) == 0
    entry = json.loads((out / "manifest.json").read_text())["calls"][0]
    body = gzip.decompress((out / entry["body"]).read_bytes()).replace(b"2.0", b"2.1")
    (out / entry["body"]).write_bytes(gzip.compress(body, mtime=0))
    with pytest.raises(RecordingError, match="sha256 mismatch"):
        RecordedClient(out)


def test_refuses_non_empty_output(tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    assert record(tmp_path, ["ETH"], False, client_factory=factory([])) == 2
    assert (tmp_path / "keep.txt").read_text() == "x"
