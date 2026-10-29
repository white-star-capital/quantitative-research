"""`rv_daily` (Deribit, Hyperliquid) and `dvol_history` (Deribit) inputs (RV guide VRP-A-8, §12):
paging, completed days only, coverage nulls, the immutable completed-day cache, DVOL percent →
decimal, Hyperliquid request shape and venue time, error codes.

TEST-ONLY structural mocks: bodies have the venues' response shapes. Closes alternate on a fixed
log step so each return is ±STEP and RV_d = n · STEP² by construction; they exercise the
arithmetic and plumbing, never reach a payload or fixture (D-10)."""

import gzip
import json
import math
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from terminal_models.adapters.base import AdapterError
from terminal_models.adapters.deribit import CHART_ROWS_PER_CALL, DeribitAdapter
from terminal_models.adapters.deribit.client import DeribitClient
from terminal_models.adapters.deribit.replay import RecordedClient
from terminal_models.adapters.http import HttpSource
from terminal_models.adapters.hyperliquid import HyperliquidAdapter
from terminal_models.adapters.hyperliquid.client import HyperliquidClient
from terminal_models.adapters.hyperliquid.replay import RecordedHyperliquidClient
from terminal_models.adapters.rv_daily import MS_PER_DAY, DayCache, fetch_rv_daily
from terminal_models.core.base import InputSpec
from terminal_models.tools.record_deribit import Recorder

AS_OF = datetime(2026, 10, 28, 0, 30, tzinfo=UTC)  # C = 2026-10-28T00:00Z
C_MS = int(datetime(2026, 10, 28, tzinfo=UTC).timestamp() * 1000)
STEP = 0.001  # TEST-ONLY log step between consecutive closes


def close_at(t_ms: int, res_ms: int) -> float:
    return 100.0 * math.exp(STEP * ((t_ms // res_ms) % 2))  # TEST-ONLY


def bars(start: int, end: int, res_ms: int, now_ms: int, drop: set[int]) -> list[int]:
    """Bar open times on the grid in [start, end], up to the bar in progress at now_ms."""
    first = -(-start // res_ms) * res_ms
    return [t for t in range(first, min(end, now_ms) + 1, res_ms) if t - t % MS_PER_DAY not in drop]


# ---- Deribit ------------------------------------------------------------------------------------

FUTURES = [{"instrument_name": "BTC-PERPETUAL", "base_currency": "BTC",
            "settlement_currency": "BTC", "settlement_period": "perpetual", "is_active": True,
            "kind": "future"}]  # fmt: skip


def deribit(calls, server=AS_OF, drop=frozenset(), dvol=None, status="ok"):
    us_out = int(server.timestamp() * 1e6)
    now_ms = us_out // 1000

    def handler(request: httpx.Request):
        method = request.url.path.rsplit("/", 1)[-1]
        p = request.url.params
        calls.append((method, dict(p)))
        if method == "get_instruments":
            result = FUTURES
        elif method == "get_time":
            result = now_ms
        elif method == "get_tradingview_chart_data":
            res_ms = int(p["resolution"]) * 60_000
            ticks = bars(int(p["start_timestamp"]), int(p["end_timestamp"]), res_ms, now_ms, drop)
            result = {"status": status if ticks else "no_data", "ticks": ticks,
                      "close": [close_at(t, res_ms) for t in ticks]}  # fmt: skip
        elif method == "get_volatility_index_data":
            result = dvol(int(p["start_timestamp"]), int(p["end_timestamp"]))
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "result": result, "usOut": us_out})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x.test")
    return DeribitClient(
        HttpSource("https://x.test", max_attempts=1, backoff_seconds=0, client=http)
    )


def rv_spec(lookback=3, resolution="60", min_coverage=0.9, **extra):
    params = {"kind": "rv_daily", "underlying": "BTC", "resolution": resolution,
              "lookback_days": lookback, "min_coverage": min_coverage} | extra  # fmt: skip
    return InputSpec("rv_daily", "deribit", params)


def chart_calls(calls):
    return [p for m, p in calls if m == "get_tradingview_chart_data"]


def test_completed_days_only_with_rv_and_coverage():
    calls = []
    got = DeribitAdapter(deribit(calls, drop={C_MS - 2 * MS_PER_DAY})).fetch(rv_spec(), AS_OF)
    assert got.snapshot_at == AS_OF  # usOut of the last page (D-4)
    d = got.data
    assert d["instrument"] == "BTC-PERPETUAL"  # discovered from metadata, never constructed
    assert d["day"] == ["2026-10-25", "2026-10-26", "2026-10-27"]  # 10-28 is in progress
    assert d["n_bars"] == [24, 0, 23]  # returns per day; the dropped day has none
    assert d["rv"][0] == pytest.approx(24 * STEP**2)
    assert d["rv"][1] is None  # below min_coverage: null, never filled
    # The day after a gap drops the return reaching back across it (core daily_rv): 23 returns.
    assert d["rv"][2] == pytest.approx(23 * STEP**2)
    [page] = chart_calls(calls)
    assert int(page["start_timestamp"]) == C_MS - 4 * MS_PER_DAY  # one extra day for a prior close
    assert int(page["end_timestamp"]) == C_MS - 1  # end is inclusive at Deribit
    assert got.raw["get_tradingview_chart_data"][0]["usOut"]


def test_partial_day_is_null_and_explicit_instrument_is_used():
    calls = []
    adapter = DeribitAdapter(deribit(calls))
    got = adapter.fetch(rv_spec(min_coverage=1.01, instrument="BTC_USDC-PERPETUAL"), AS_OF)
    assert got.data["instrument"] == "BTC_USDC-PERPETUAL"
    assert got.data["rv"] == [None, None, None] and got.data["n_bars"] == [24, 24, 24]
    assert "get_instruments" not in [m for m, _ in calls]


@pytest.mark.parametrize("resolution, lookback, pages", [("60", 300, 2), ("5", 40, 3)])
def test_pages_are_whole_days_under_the_row_cap(resolution, lookback, pages):
    calls = []
    got = DeribitAdapter(deribit(calls)).fetch(rv_spec(lookback, resolution), AS_OF)
    windows = [(int(p["start_timestamp"]), int(p["end_timestamp"]) + 1) for p in chart_calls(calls)]
    assert len(windows) == pages
    assert windows[0][0] == C_MS - (lookback + 1) * MS_PER_DAY and windows[-1][1] == C_MS
    per_day = 1440 // int(resolution)
    for (start, end), nxt in zip(windows, windows[1:] + [(C_MS, None)], strict=True):
        assert start % MS_PER_DAY == 0 and end == nxt[0]  # contiguous, day-aligned
        assert (end - start) // MS_PER_DAY * per_day <= CHART_ROWS_PER_CALL
    assert len(got.data["day"]) == lookback and None not in got.data["rv"]


def test_cache_serves_completed_days_and_never_rewrites(tmp_path):
    cache = DayCache(tmp_path)
    first = DeribitAdapter(deribit([]), rv_cache=cache).fetch(rv_spec(), AS_OF)
    files = sorted(p.name for p in tmp_path.rglob("*.json"))
    assert files == ["2026-10-24.json", "2026-10-25.json", "2026-10-26.json", "2026-10-27.json"]
    assert {p.parent.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.json")} == {
        "deribit/BTC-PERPETUAL/60"
    }
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}

    calls = []  # same day, venue now drops a cached day: the cache answers, nothing is refetched
    later = AS_OF + timedelta(minutes=30)
    again = DeribitAdapter(deribit(calls, later, drop={C_MS - MS_PER_DAY}), rv_cache=cache)
    second = again.fetch(rv_spec(), later)
    assert chart_calls(calls) == [] and [m for m, _ in calls] == ["get_instruments", "get_time"]
    assert second.data == first.data and second.snapshot_at == later
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before

    calls = []  # next day: only the newly closed day is fetched
    tomorrow = AS_OF + timedelta(days=1)
    third = DeribitAdapter(deribit(calls, tomorrow), rv_cache=cache).fetch(rv_spec(), tomorrow)
    [page] = chart_calls(calls)
    assert int(page["start_timestamp"]) == C_MS
    assert third.data["day"][-1] == "2026-10-28"


def test_empty_and_short_days_are_not_cached_and_refetch(tmp_path):
    cache = DayCache(tmp_path)
    short = {C_MS - 2 * MS_PER_DAY}  # an outage or a transient no_data: the day comes back empty
    first = DeribitAdapter(deribit([], drop=short), rv_cache=cache).fetch(rv_spec(), AS_OF)
    assert first.data["rv"][1] is None
    assert "2026-10-26.json" not in [p.name for p in tmp_path.rglob("*.json")]

    calls = []  # the venue has the day again: only that day is fetched, and now cached
    later = AS_OF + timedelta(minutes=30)
    second = DeribitAdapter(deribit(calls, later), rv_cache=cache).fetch(rv_spec(), later)
    [page] = chart_calls(calls)
    assert int(page["start_timestamp"]) == C_MS - 2 * MS_PER_DAY
    assert int(page["end_timestamp"]) == C_MS - MS_PER_DAY - 1
    assert second.data["rv"] == pytest.approx([24 * STEP**2] * 3)
    assert "2026-10-26.json" in [p.name for p in tmp_path.rglob("*.json")]


def test_partial_and_truncated_days_are_not_cached(tmp_path):
    cache = DayCache(tmp_path)
    hour = 3_600_000

    def fetch(fetch_page):
        return fetch_rv_daily("rv", venue="v", instrument="i", resolution="60", lookback_days=3,
                              min_coverage=0.9, as_of=AS_OF, rows_per_call=5000,
                              fetch_page=fetch_page, cache=cache)  # fmt: skip

    def short_last_day(start, end):  # TEST-ONLY: the window's last day loses 3 bars
        times = [t for t in range(start, end, hour) if t < end - 3 * hour]
        return times, [close_at(t, hour) for t in times], AS_OF, None

    data, _, _ = fetch(short_last_day)
    assert data["n_bars"][-1] == 21 and data["rv"][-1] is None  # 21 < 0.9 · 24
    names = sorted(p.name for p in tmp_path.rglob("*.json"))
    assert names == ["2026-10-24.json", "2026-10-25.json", "2026-10-26.json"]  # not 10-27

    for f in tmp_path.rglob("*.json"):
        f.unlink()

    def truncated(start, end):  # TEST-ONLY: a lower row cap than assumed cuts the page at 40 rows
        times = list(range(start, end, hour))[:40]
        return times, [close_at(t, hour) for t in times], AS_OF, None

    data, _, _ = fetch(truncated)
    assert data["rv"][1:] == [None, None]
    assert [p.name for p in tmp_path.rglob("*.json")] == ["2026-10-24.json"]  # the one full day


def test_day_closing_inside_the_settle_window_is_not_cached(tmp_path):
    early = datetime(2026, 10, 28, 0, 1, tzinfo=UTC)
    DeribitAdapter(deribit([], early), rv_cache=DayCache(tmp_path)).fetch(rv_spec(), early)
    assert "2026-10-27.json" not in [p.name for p in tmp_path.rglob("*.json")]
    assert "2026-10-26.json" in [p.name for p in tmp_path.rglob("*.json")]


def test_bad_resolution_and_bad_status():
    with pytest.raises(AdapterError) as err:
        DeribitAdapter(deribit([])).fetch(rv_spec(resolution="7"), AS_OF)
    assert err.value.code == "bad_input_spec"
    with pytest.raises(AdapterError) as err:
        DeribitAdapter(deribit([], status="error")).fetch(rv_spec(), AS_OF)
    assert err.value.code == "venue_unavailable"


def dvol_pages(start, end):
    """TEST-ONLY: daily DVOL candles [ts, o, h, l, c] in vol points, newest page first, two days
    per page with Deribit's `continuation`; 2026-10-25 is missing."""
    days = [t for t in range(start - start % MS_PER_DAY, end + 1, MS_PER_DAY) if t >= start]
    days = [t for t in days if t != C_MS - 3 * MS_PER_DAY]
    page = days[-2:]
    cont = page[0] - 1 if len(days) > 2 else None
    return {"data": [[t, 50, 52, 49, 50 + (t - C_MS) // MS_PER_DAY] for t in page],
            "continuation": cont}  # fmt: skip


def test_dvol_history_pages_back_and_converts_percent():
    calls = []
    spec = InputSpec("dvol", "deribit", {"kind": "dvol_history", "underlying": "ETH",
                                         "lookback_days": 5})  # fmt: skip
    got = DeribitAdapter(deribit(calls, dvol=dvol_pages)).fetch(spec, AS_OF)
    assert got.data["day"] == ["2026-10-23", "2026-10-24", "2026-10-25", "2026-10-26", "2026-10-27"]
    assert got.data["dvol"] == pytest.approx([0.45, 0.46, None, 0.48, 0.49])  # points → decimal
    pages = [p for m, p in calls if m == "get_volatility_index_data"]
    assert [p["currency"] for p in pages] == ["ETH"] * 2 and pages[0]["resolution"] == "1D"
    assert int(pages[0]["end_timestamp"]) == C_MS - 1  # the in-progress day is never asked for
    assert int(pages[1]["end_timestamp"]) == C_MS - 2 * MS_PER_DAY - 1  # from `continuation`
    assert len(got.raw["get_volatility_index_data"]) == 2


def test_dvol_history_refuses_assets_without_dvol_and_unaligned_candles():
    spec = InputSpec("dvol", "deribit", {"kind": "dvol_history", "underlying": "SOL",
                                         "lookback_days": 5})  # fmt: skip
    with pytest.raises(AdapterError) as err:
        DeribitAdapter(deribit([])).fetch(spec, AS_OF)
    assert err.value.code == "bad_input_spec"
    spec = InputSpec("dvol", "deribit", {"kind": "dvol_history", "underlying": "BTC",
                                         "lookback_days": 5})  # fmt: skip
    hourly = lambda s, e: {"data": [[C_MS - 3_600_000, 1, 1, 1, 50]], "continuation": None}  # noqa: E731
    with pytest.raises(AdapterError) as err:
        DeribitAdapter(deribit([], dvol=hourly)).fetch(spec, AS_OF)
    assert err.value.code == "venue_unavailable"


# ---- Hyperliquid --------------------------------------------------------------------------------

INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000}


def hyperliquid(calls, now=AS_OF, status=200, body=None):
    now_ms = int(now.timestamp() * 1000)

    def handler(request: httpx.Request):
        assert request.method == "POST" and request.url.path == "/info"
        req = json.loads(request.content)
        calls.append(req)
        if body is not None or status != 200:
            return httpx.Response(status, text=body or "")
        r = req["req"]
        step = INTERVAL_MS[r["interval"]]
        ticks = bars(r["startTime"], r["endTime"], step, now_ms, set())
        candles = [{"t": t, "T": t + step - 1, "s": r["coin"], "i": r["interval"], "o": "1",
                    "c": str(close_at(t, step)), "h": "1", "l": "1", "v": "1", "n": 1}
                   for t in ticks]  # fmt: skip
        return httpx.Response(200, json=candles)

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://hl.test")
    return HttpSource("https://hl.test", max_attempts=1, backoff_seconds=0, client=http)


def hl_spec(underlying="HYPE", lookback=3, resolution="60"):
    params = {"kind": "rv_daily", "underlying": underlying, "resolution": resolution,
              "lookback_days": lookback, "min_coverage": 0.9}  # fmt: skip
    return InputSpec("rv_daily", "hyperliquid", params)


def test_hyperliquid_request_body_and_venue_time():
    calls = []
    got = HyperliquidAdapter(HyperliquidClient(hyperliquid(calls))).fetch(hl_spec(), AS_OF)
    clock, page = calls
    now_ms = int(AS_OF.timestamp() * 1000)
    assert clock == {"type": "candleSnapshot", "req": {"coin": "HYPE", "interval": "1m",
                     "startTime": now_ms - 300_000, "endTime": now_ms + 600_000}}  # fmt: skip
    assert page == {"type": "candleSnapshot", "req": {"coin": "HYPE", "interval": "1h",
                    "startTime": C_MS - 4 * MS_PER_DAY, "endTime": C_MS - 1}}  # fmt: skip
    # The newest 1m candle's open time: the venue's clock, at most a minute behind.
    assert got.snapshot_at == AS_OF
    assert got.data["venue"] == "hyperliquid" and got.data["instrument"] == "HYPE"
    assert got.data["day"][-1] == "2026-10-27"
    assert got.data["rv"] == pytest.approx([24 * STEP**2] * 3)


def test_hyperliquid_venue_time_is_when_the_venue_answered():
    # A late fetch (a run late in its pass, a recording's later assets): the request depends only
    # on as_of, but the venue answers up to its candle in progress, so the input isn't stale.
    late = AS_OF + timedelta(seconds=150)
    got = HyperliquidAdapter(HyperliquidClient(hyperliquid([], now=late))).fetch(hl_spec(), AS_OF)
    assert got.snapshot_at == AS_OF + timedelta(minutes=2)  # the 1m candle open at `late`


def test_hyperliquid_cache_and_paging(tmp_path):
    cache = DayCache(tmp_path)
    calls = []
    HyperliquidAdapter(HyperliquidClient(hyperliquid(calls)), rv_cache=cache).fetch(
        hl_spec("SOL", 40, "5"), AS_OF
    )
    pages = [c["req"] for c in calls if c["req"]["interval"] == "5m"]
    assert len(pages) == 3
    assert all((p["endTime"] + 1 - p["startTime"]) // 300_000 <= 5000 for p in pages)
    calls.clear()
    later = AS_OF + timedelta(minutes=10)
    HyperliquidAdapter(HyperliquidClient(hyperliquid(calls, later)), rv_cache=cache).fetch(
        hl_spec("SOL", 40, "5"), later
    )
    assert [c["req"]["interval"] for c in calls] == ["1m"]  # only the clock; days come from cache
    assert (tmp_path / "hyperliquid" / "SOL" / "5" / "2026-10-27.json").exists()


@pytest.mark.parametrize(
    "kwargs, code",
    [
        ({"status": 500}, "venue_unavailable"),
        ({"status": 422, "body": "bad"}, "venue_unavailable"),
        ({"body": '{"error": "x"}'}, "venue_unavailable"),
        ({"body": "[]"}, "venue_unavailable"),  # no 1m candle: no venue time
    ],
)
def test_hyperliquid_failures_are_venue_unavailable(kwargs, code):
    adapter = HyperliquidAdapter(HyperliquidClient(hyperliquid([], **kwargs)))
    with pytest.raises(AdapterError) as err:
        adapter.fetch(hl_spec(), AS_OF)
    assert err.value.code == code


def test_hyperliquid_bad_specs():
    adapter = HyperliquidAdapter(HyperliquidClient(hyperliquid([])))
    wrong_kind = InputSpec("x", "hyperliquid", {"kind": "chain", "underlying": "BTC"})
    for spec in (hl_spec("DOGE"), hl_spec(resolution="7"), wrong_kind):
        with pytest.raises(AdapterError) as err:
            adapter.fetch(spec, AS_OF)
        assert err.value.code == "bad_input_spec"


# ---- Recording and replay -----------------------------------------------------------------------


def test_recorded_pages_replay_exactly(tmp_path):
    rec = Recorder(tmp_path / "rec")
    live_d = DeribitAdapter(deribit([]))
    live_d.client._capture = rec.capture
    live_h = HyperliquidAdapter(HyperliquidClient(hyperliquid([]), capture=rec.capture_hyperliquid))
    d_spec, h_spec = rv_spec(300), hl_spec("HYPE", 40, "5")
    a, b = live_d.fetch(d_spec, AS_OF), live_h.fetch(h_spec, AS_OF)
    # A candles_1h call on the same method and resolution must not shadow the rv pages.
    live_d.client.candles("BTC-PERPETUAL", AS_OF - timedelta(days=31), AS_OF)
    (rec.out / "manifest.json").write_text(json.dumps({"calls": rec.entries}))
    metas = [json.loads((rec.out / e["meta"]).read_text()) for e in rec.entries]
    assert any(m["url"] == "https://api.hyperliquid.xyz/info" for m in metas)
    body = gzip.decompress((rec.out / rec.entries[-1]["body"]).read_bytes())
    assert body.startswith(b'{"jsonrpc"')

    ra = DeribitAdapter(RecordedClient(rec.out)).fetch(d_spec, AS_OF)
    rb = HyperliquidAdapter(RecordedHyperliquidClient(rec.out)).fetch(h_spec, AS_OF)
    assert (ra.snapshot_at, ra.data) == (a.snapshot_at, a.data)
    assert (rb.snapshot_at, rb.data) == (b.snapshot_at, b.data)
    with pytest.raises(AdapterError, match="not in the recording"):
        HyperliquidAdapter(RecordedHyperliquidClient(rec.out)).fetch(h_spec, AS_OF + timedelta(1))
