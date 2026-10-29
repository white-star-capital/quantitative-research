"""tools/probe_venues.py on mocked transports only: row caps, the 5m history search, Hyperliquid
depth and future endTime, DVOL open hours, the report and the verdict table.

TEST-ONLY structural mocks: venue-shaped bodies whose hidden caps, history starts and candle hours
are set by the test. Closes are a constant placeholder the probe never reads; nothing here reaches
a payload or fixture (D-10)."""

import json
from datetime import UTC, datetime

import httpx
import pytest

from terminal_models.adapters.deribit.client import DeribitClient
from terminal_models.adapters.http import HttpSource
from terminal_models.adapters.hyperliquid.client import HyperliquidClient
from terminal_models.adapters.rv_daily import MS_PER_DAY
from terminal_models.tools import probe_venues
from terminal_models.tools.probe_venues import SEARCH_CALLS, earliest_day, probe

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)
NOW_MS = int(NOW.timestamp() * 1000)
C_MS = NOW_MS - NOW_MS % MS_PER_DAY
FIVE_M_FROM = C_MS - 1000 * MS_PER_DAY  # TEST-ONLY: Deribit's 5m history starts here
HL_FROM = {"1h": NOW_MS - 4999 * 3_600_000, "5m": NOW_MS - 4999 * 300_000}  # TEST-ONLY depth
FUTURES = [{"instrument_name": "BTC-PERPETUAL", "base_currency": "BTC",
            "settlement_currency": "BTC", "settlement_period": "perpetual", "is_active": True,
            "kind": "future"}]  # fmt: skip


def grid(start: int, end: int, step: int, first: int = 0) -> list[int]:
    """Bar open times in [start, end] (inclusive), none before `first` or after NOW."""
    lo = max(start, first)
    return list(range(-(-lo // step) * step, min(end, NOW_MS) + 1, step))


def deribit(calls, cap=700, dvol_hours=None, search_status="ok"):
    """`search_status`: what a one-day 5m request (the history search) answers."""
    dvol_hours = dvol_hours or {"BTC": 0, "ETH": 8}

    def handler(request: httpx.Request):
        method = request.url.path.rsplit("/", 1)[-1]
        p = request.url.params
        calls.append((method, dict(p)))
        if method == "get_time":
            result = NOW_MS
        elif method == "get_instruments":
            result = FUTURES
        elif method == "get_tradingview_chart_data":
            step = int(p["resolution"]) * 60_000
            first = FIVE_M_FROM if p["resolution"] == "5" else 0
            ticks = grid(int(p["start_timestamp"]), int(p["end_timestamp"]), step, first)[-cap:]
            result = {"status": "ok" if ticks else "no_data", "ticks": ticks,
                      "close": [1.0] * len(ticks)}  # fmt: skip
            one_day = int(p["end_timestamp"]) + 1 - int(p["start_timestamp"]) == MS_PER_DAY
            if one_day and p["resolution"] == "5" and search_status != "ok":
                result = {"status": search_status}
        elif method == "get_volatility_index_data":
            off = dvol_hours[p["currency"]] * 3_600_000
            days = grid(int(p["start_timestamp"]) - off, int(p["end_timestamp"]) - off, MS_PER_DAY)
            result = {"data": [[d + off, 1, 1, 1, 1] for d in days], "continuation": None}
        else:
            raise AssertionError(method)
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "result": result, "usOut": NOW_MS * 1000}
        )

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x.test")
    return DeribitClient(
        HttpSource("https://x.test", max_attempts=1, backoff_seconds=0, client=http)
    )


def hyperliquid(calls, cap=5000, future_end="reject", from_=None):
    from_ = HL_FROM if from_ is None else from_

    def handler(request: httpx.Request):
        r = json.loads(request.content)["req"]
        calls.append(r)
        if r["endTime"] > NOW_MS and future_end == "reject":
            return httpx.Response(422, text="")
        step = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000}[r["interval"]]
        ticks = grid(r["startTime"], r["endTime"], step, from_.get(r["interval"], 0))[-cap:]
        return httpx.Response(200, json=[{"t": t, "c": "1"} for t in ticks])

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://hl.test")
    source = HttpSource("https://hl.test", max_attempts=1, backoff_seconds=0, client=http)
    return HyperliquidClient(source)


def by_question(report):
    return {v["question"]: v for v in report["verdicts"]}


def test_probe_measures_caps_history_alignment_and_future_end_time():
    d_calls, h_calls = [], []
    report = probe(deribit(d_calls), hyperliquid(h_calls))
    d, h, v = report["deribit"], report["hyperliquid"], report["dvol"]
    assert report["probed_at"] == "2026-10-29T12:00:00Z"  # Deribit's server time
    assert d["cap_5m"]["cap"] == 700 and d["cap_5m"]["kept"] == "latest"
    assert d["cap_1h"]["requested"] == 10_000 and d["cap_1h"]["cap"] == 700
    hist = d["history_5m"]
    assert hist["earliest"] == "2024-02-02" and hist["exact"]  # FIVE_M_FROM
    assert hist["days"] == 1000 and hist["calls"] <= SEARCH_CALLS
    assert len([m for m, _ in d_calls if m == "get_tradingview_chart_data"]) <= 2 + SEARCH_CALLS
    # Hyperliquid serves the latest 5,000 bars per call and no older ones.
    assert h["1h"]["cap"] == 5000 and h["1h"]["history"] == "starts_at_first"
    assert h["5m"]["older_rows"] == 0
    assert h["future_end_time"]["accepted"] is False
    assert v["BTC"]["open_hours_utc"] == [0.0] and v["ETH"]["open_hours_utc"] == [8.0]
    q = by_question(report)
    assert q["Deribit row cap 5m"]["answer"] == "700 rows, keeps the latest"
    assert "set 700" in q["Deribit row cap 5m"]["consequence"]
    assert q["Hyperliquid row cap 1h"]["consequence"] == "the assumed 5000 holds"
    assert "falls back" in q["Hyperliquid future endTime"]["consequence"]
    assert q["DVOL 1D alignment BTC"]["consequence"] == "keyed as is"
    assert "dvol_offset" in q["DVOL 1D alignment ETH"]["consequence"]
    assert all(r["endTime"] <= NOW_MS for r in h_calls if r["interval"] != "1m")


def test_probe_without_caps_and_with_deeper_history():
    report = probe(deribit([], cap=10**6), hyperliquid([], cap=3000, future_end="ok", from_={}))
    d, h = report["deribit"], report["hyperliquid"]
    assert d["cap_1h"]["kept"] == "all" and d["cap_1h"]["cap"] is None
    assert h["1h"]["cap"] == 3000 and h["1h"]["history"] == "pages_further"
    assert h["future_end_time"]["accepted"] is True
    q = by_question(report)
    no_cap = "no cap at or below 5000: the assumed 5000 holds"
    assert q["Deribit row cap 1h"]["consequence"] == no_cap
    assert q["Hyperliquid history 1h"]["consequence"] == "older days page in"


@pytest.mark.parametrize("first, exact", [(10, True), (0, False)])
def test_earliest_day_search(first, exact):
    calls = []

    def has(day):
        calls.append(day)
        return day >= first * MS_PER_DAY

    got = earliest_day(has, 0, 5000 * MS_PER_DAY)
    assert got["earliest"] == ("1970-01-11" if first else "1970-01-01") and got["exact"] is exact
    assert len(calls) == got["calls"] <= SEARCH_CALLS


def test_failed_venue_is_reported_and_main_writes_the_report(tmp_path, monkeypatch, capsys):
    down = httpx.MockTransport(lambda request: httpx.Response(503))
    http = httpx.Client(transport=down, base_url="https://hl.test")
    broken = HttpSource("https://hl.test", max_attempts=1, backoff_seconds=0, client=http)
    monkeypatch.setattr(probe_venues, "DeribitClient", lambda: deribit([]))
    monkeypatch.setattr(probe_venues, "HyperliquidClient", lambda: HyperliquidClient(broken))
    out = tmp_path / "validation" / "venue_probe.json"
    assert probe_venues.main(["--out", str(out)]) == 1
    report = json.loads(out.read_text())
    assert "error" in report["hyperliquid"] and "cap_5m" in report["deribit"]
    printed = capsys.readouterr().out
    assert "Hyperliquid candles" in printed and "DVOL 1D alignment ETH" in printed


def test_a_failed_history_search_keeps_the_measured_caps():
    report = probe_venues.probe(deribit([], search_status="error"), hyperliquid([]))
    d = report["deribit"]
    assert d["cap_5m"]["cap"] == 700 and d["cap_1h"]["cap"] == 700
    assert d["history_5m"] == {"error": "candles status error"}
    answers = {r["question"]: r["answer"] for r in report["verdicts"]}
    assert answers["Deribit row cap 5m"] == "700 rows, keeps the latest"
    assert answers["Deribit 5m history"] == "error: candles status error"
