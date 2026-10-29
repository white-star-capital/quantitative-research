"""Probe the venue facts the vrp_signal adapters can't check offline (RV guide VRP-A-8).

    uv run python -m terminal_models.tools.probe_venues [--out <report.json>]    (or: make probe)

Read-only public calls through the production clients (HttpSource timeouts, retries, backoff),
at most about 30 of them:
- Deribit `get_tradingview_chart_data` on the BTC perpetual (discovered from metadata, D-2): the
  row cap at 5m and 1h, from one window of OVERSIZE × the assumed cap each (fewer rows than the
  window holds, all from one end of it, is the cap), and the earliest day with 5m candles, by a
  binary search over days of at most SEARCH_CALLS calls (it assumes history, once it starts, has
  no empty days);
- Hyperliquid `candleSnapshot` BTC 1h and 5m: the row cap the same way, then one window just
  before the earliest row returned (empty: the venue's history stops there; rows: the cap is per
  call and older days page in); and whether a 1m request with an `endTime` in the future works;
- Deribit `get_volatility_index_data` 1D for BTC and ETH: the hour (UTC) the candles open at.

Times are relative to Deribit's server time. Prints a verdict table and writes the report (default
library/rv_har/validation/venue_probe.json; the directory is created then). Nothing here changes a
run: the adapters work whatever the answers (split paging in adapters/rv_daily.py, the Hyperliquid
clock fallback, `dvol_history` keying); the report informs the 5-minute upgrade and the docs.
Exit 0 when every probe answered, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from terminal_models.adapters.base import AdapterError
from terminal_models.adapters.deribit import CHART_ROWS_PER_CALL
from terminal_models.adapters.deribit.chains import BOOKS, find_perpetual
from terminal_models.adapters.deribit.client import DeribitClient
from terminal_models.adapters.hyperliquid import CANDLE_ROWS_PER_CALL, CLOCK_LEAD_MS
from terminal_models.adapters.hyperliquid.client import HyperliquidClient
from terminal_models.adapters.rv_daily import MS_PER_DAY, day_floor_ms, iso_day

DEFAULT_OUT = Path(__file__).parents[1] / "library" / "rv_har" / "validation" / "venue_probe.json"
OVERSIZE = 2  # a cap probe asks for this many times the assumed cap
SEARCH_FROM = datetime(2016, 1, 1, tzinfo=UTC)  # before any Deribit perpetual
SEARCH_CALLS = 16  # 2^14 days > 40 years: the search ends well within it
HOUR_MS = 3_600_000


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def window_rows(times: list[int], start: int, end: int, step: int) -> dict[str, Any]:
    """Rows a [start, end) window returned, and which end of it they hold. `kept` is `all` when
    both ends are there (no cap hit; any shortfall is gaps), else `latest`/`earliest`, the end a
    capped venue keeps, and `cap` is the row count; `none` or `unclear` otherwise."""
    out: dict[str, Any] = {"requested": (end - start) // step, "returned": len(times)}
    if not times:
        return out | {"kept": "none", "cap": None}
    first, last = min(times), max(times)
    at_start, at_end = first - start < MS_PER_DAY, end - last <= MS_PER_DAY
    kept = {(True, True): "all", (False, True): "latest", (True, False): "earliest"}.get(
        (at_start, at_end), "unclear"
    )
    cap = len(times) if kept in ("latest", "earliest") else None
    return out | {"first": iso(first), "last": iso(last), "kept": kept, "cap": cap}


def deribit_ticks(client: DeribitClient, perp: str, res: str, start: int, end: int) -> list[int]:
    r = client.chart_data(perp, res, start, end - 1)  # end_timestamp is inclusive at Deribit
    status = r.result.get("status")
    if status not in ("ok", "no_data"):
        raise AdapterError("venue_unavailable", f"candles status {status}")
    return [int(t) for t in r.result.get("ticks") or []]


def deribit_cap(client: DeribitClient, perp: str, res: str, now_ms: int) -> dict[str, Any]:
    step = int(res) * 60_000
    end = day_floor_ms(datetime.fromtimestamp(now_ms / 1000, UTC))
    start = end - OVERSIZE * CHART_ROWS_PER_CALL * step
    return window_rows(deribit_ticks(client, perp, res, start, end), start, end, step)


def earliest_day(has: Callable[[int], bool], lo: int, hi: int) -> dict[str, Any]:
    """Binary search for the first day with data in [lo, hi]. `exact` is false when it is only
    known to be at or before `earliest` (data at `lo` already, or SEARCH_CALLS ran out)."""
    if not has(hi):
        return {"earliest": None, "days": 0, "exact": True, "calls": 1}
    first, last, calls = hi, hi, 2
    if has(lo):
        first = lo
    else:
        while first - lo > MS_PER_DAY and calls < SEARCH_CALLS:
            mid = lo + (first - lo) // MS_PER_DAY // 2 * MS_PER_DAY
            calls += 1
            lo, first = (lo, mid) if has(mid) else (mid, first)
    days = (last - first) // MS_PER_DAY + 1
    exact = first - lo == MS_PER_DAY
    return {"earliest": iso_day(first), "days": days, "exact": exact, "calls": calls}


def probe_deribit(client: DeribitClient, now_ms: int) -> dict[str, Any]:
    perp = find_perpetual(BOOKS["BTC"], client.futures("BTC").result)
    if perp is None:
        raise AdapterError("no_perpetual", "Deribit lists no BTC perpetual")
    # Each answer guarded on its own: a failed search call doesn't discard caps already measured.
    out: dict[str, Any] = {"instrument": perp}
    out["cap_5m"] = guarded(lambda: deribit_cap(client, perp, "5", now_ms))
    out["cap_1h"] = guarded(lambda: deribit_cap(client, perp, "60", now_ms))
    yesterday = day_floor_ms(datetime.fromtimestamp(now_ms / 1000, UTC)) - MS_PER_DAY
    has = lambda day: bool(deribit_ticks(client, perp, "5", day, day + MS_PER_DAY))  # noqa: E731
    out["history_5m"] = guarded(lambda: earliest_day(has, day_floor_ms(SEARCH_FROM), yesterday))
    return out


def probe_dvol(client: DeribitClient, now_ms: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for currency in ("BTC", "ETH"):
        r = client.volatility_index_page(currency, now_ms - 10 * MS_PER_DAY, now_ms, "1D")
        stamps = [int(row[0]) for row in r.result["data"]]
        hours = sorted({ts % MS_PER_DAY / HOUR_MS for ts in stamps})
        out[currency] = {"candles": len(stamps), "open_hours_utc": hours,
                         "continuation": r.result.get("continuation")}  # fmt: skip
    return out


def probe_hyperliquid(client: HyperliquidClient, now_ms: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for interval, step in (("1h", HOUR_MS), ("5m", 300_000)):
        start = now_ms - OVERSIZE * CANDLE_ROWS_PER_CALL * step
        times = [int(k["t"]) for k in client.candles("BTC", interval, start, now_ms)]
        rows = window_rows(times, start, now_ms + step, step)
        if times:  # a window of 100 bars just before the earliest row returned
            older = client.candles("BTC", interval, min(times) - 100 * step, min(times) - 1)
            rows["older_rows"] = len(older)
            rows["history"] = "pages_further" if older else "starts_at_first"
        out[interval] = rows
    try:
        clock = client.candles("BTC", "1m", now_ms - 5 * 60_000, now_ms + CLOCK_LEAD_MS)
        out["future_end_time"] = {"accepted": bool(clock), "candles": len(clock)}
        if clock:
            out["future_end_time"]["newest"] = iso(max(int(k["t"]) for k in clock))
    except AdapterError as exc:
        out["future_end_time"] = {"accepted": False, "error": exc.message}
    return out


def cap_verdict(rows: dict[str, Any], assumed: int) -> tuple[str, str]:
    if rows["cap"] is None:
        answer = f"{rows['returned']} of {rows['requested']} rows ({rows['kept']})"
        return answer, f"no cap at or below {assumed}: the assumed {assumed} holds"
    answer = f"{rows['cap']} rows, keeps the {rows['kept']}"
    if rows["cap"] >= assumed:
        return answer, f"the assumed {assumed} holds"
    return answer, f"pages are split around it (more calls, same coverage); set {rows['cap']}"


def verdicts(report: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[tuple[str, str, str]] = []
    d, h, v = report["deribit"], report["hyperliquid"], report["dvol"]
    if "error" in d:
        rows.append(("Deribit candles", f"error: {d['error']}", "rerun"))
    else:
        for res in ("5m", "1h"):
            c, q = d[f"cap_{res}"], f"Deribit row cap {res}"
            if "error" in c:
                rows.append((q, f"error: {c['error']}", "rerun"))
            else:
                rows.append((q, *cap_verdict(c, CHART_ROWS_PER_CALL)))
        s = d["history_5m"]
        if "error" in s:
            rows.append(("Deribit 5m history", f"error: {s['error']}", "rerun"))
        elif s["earliest"] is None:
            rows.append(("Deribit 5m history", "no 5m candle yesterday", "stay on 1h"))
        else:
            since = ("from " if s["exact"] else "at or before ") + s["earliest"]
            switch = "BTC/ETH stay on 1h in v1; a 5m switch needs this depth and validate_rv"
            rows.append(("Deribit 5m history", f"{since} ({s['days']} days)", switch))
    if "error" in h:
        rows.append(("Hyperliquid candles", f"error: {h['error']}", "rerun"))
    else:
        for interval in ("1h", "5m"):
            r = h[interval]
            rows.append((f"Hyperliquid row cap {interval}", *cap_verdict(r, CANDLE_ROWS_PER_CALL)))
            if "history" in r:
                deep = r["history"] == "pages_further"
                answer = "deeper than one call" if deep else f"starts {r['first']}"
                then = "older days page in" if deep else "the day cache grows it"
                rows.append((f"Hyperliquid history {interval}", answer, then))
        f = h["future_end_time"]
        if f["accepted"]:
            rows.append(("Hyperliquid future endTime", "accepted", "clock uses the lead window"))
        else:
            fallback = "clock falls back to endTime = as_of (at most 60 s old)"
            refused = f"refused ({f.get('error', 'empty')})"
            rows.append(("Hyperliquid future endTime", refused, fallback))
    if "error" in v:
        rows.append(("DVOL 1D alignment", f"error: {v['error']}", "rerun"))
    else:
        for currency, r in v.items():
            hours = r["open_hours_utc"]
            aligned = hours == [0.0]
            rows.append((f"DVOL 1D alignment {currency}",
                         "opens 00:00 UTC" if aligned else f"opens at hour(s) {hours} UTC",
                         "keyed as is" if aligned else "keyed to the day it closes in; "
                         "close_offset_h set (runs warn dvol_offset)"))  # fmt: skip
    return [{"question": q, "answer": a, "consequence": c} for q, a, c in rows]


def guarded(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return fn()
    except AdapterError as exc:
        return {"error": exc.message}
    except (KeyError, TypeError, ValueError) as exc:
        return {"error": f"unexpected response: {exc}"}


def probe(deribit: DeribitClient, hyperliquid: HyperliquidClient) -> dict[str, Any]:
    now = deribit.time().server_time
    now_ms = int(now.timestamp() * 1000)
    report: dict[str, Any] = {
        "probed_at": iso(now_ms),
        "deribit": guarded(lambda: probe_deribit(deribit, now_ms)),
        "hyperliquid": guarded(lambda: probe_hyperliquid(hyperliquid, now_ms)),
        "dvol": guarded(lambda: probe_dvol(deribit, now_ms)),
    }
    report["verdicts"] = verdicts(report)
    return report


def table(rows: list[dict[str, str]]) -> str:
    widths = [max(len(r[k]) for r in rows) for k in ("question", "answer")]
    return "\n".join(
        f"{r['question']:<{widths[0]}}  {r['answer']:<{widths[1]}}  {r['consequence']}"
        for r in rows
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m terminal_models.tools.probe_venues")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        report = probe(DeribitClient(), HyperliquidClient())
    except AdapterError as exc:
        print(f"Deribit server time unavailable: {exc.message}", file=sys.stderr)
        return 1
    print(table(report["verdicts"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    failed = any(a["answer"].startswith("error") for a in report["verdicts"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
