"""The `rv_daily` input kind, shared by the Deribit and Hyperliquid adapters (RV guide VRP-A-8).

Candles are paged from the venue and reduced to completed UTC days ([00:00, 24:00)) by
core/quant/rv.daily_rv: RV_d = Σ r² over log close-to-close returns, and `n_bars` counts the
day's returns. Output, gap-free over the lookback:

    {"venue", "instrument", "resolution",
     "day": ["YYYY-MM-DD", ...],   # the UTC date the day covers; its close is the next 00:00
     "rv": [float | None, ...],    # daily variance (not annualized); None below min_coverage
     "n_bars": [int, ...]}

`resolution` is the bar size in minutes as a string ("5", "60"), the same for every venue; it must
divide a day. The day in progress at `as_of` is never returned or cached.

Completed days are immutable, so their bars are cached per (venue, instrument, resolution, day)
under `SNAPSHOT_DIR/rv_daily/` and fetched once. A file is written once and never overwritten,
and only after the day's close is CACHE_SETTLE_S behind the venue's clock. Only days missing from
the cache are fetched. Pages are whole days, at most `rows_per_call` bars each.

`rows_per_call` is an upper bound, not a fact about the venue (VRP-A-8). A page that comes back
with some bars but some of its days below `min_coverage` (a venue that caps rows below
`rows_per_call` and keeps either end of the window, or a short day) is split into two halves of
whole days, and each half holding a short day is requested again, at most MAX_SPLIT_DEPTH times
deep. Splitting stops where it cannot help: a response with fewer bars than one covered day
(empty: history the venue doesn't serve; or a cap below one day), and a half that came back with
fewer bars than its parent and none more on a short day (a real shortfall, not a cap). Each day
keeps the bars of the one response that returned the most for it, so no bar is counted twice;
every response body is kept.

Only a covered day, one with at least `min_coverage` of its bars, is cached. An empty or short day
(an outage, a transient `no_data`, a page the venue cut short under a lower row cap than assumed,
or history it doesn't serve) stays uncached and is fetched again on the next run, so one bad
response never becomes permanent.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from terminal_models.adapters.base import AdapterError
from terminal_models.core.paths import MODELS_DIR
from terminal_models.core.quant.rv import daily_rv

MS_PER_DAY = 86_400_000
CACHE_SETTLE_S = 300  # a closed day's last bar may still be settling at the venue just after 00:00
MAX_SPLIT_DEPTH = 6  # halvings of a short page: covers a venue cap down to 1/64 of rows_per_call

# start_ms, end_ms (exclusive) -> (bar open times ms, closes, venue time, raw body)
FetchPage = Callable[[int, int], tuple[list[int], list[float], datetime, Any]]


def bars_per_day(spec_name: str, resolution: Any) -> int:
    minutes = int(resolution) if str(resolution).isdigit() else 0
    if minutes <= 0 or 1440 % minutes:
        raise AdapterError(
            "bad_input_spec", f"{spec_name}: resolution {resolution!r} must divide a day (minutes)"
        )
    return 1440 // minutes


def day_floor_ms(t: datetime) -> int:
    ms = int(t.timestamp() * 1000)
    return ms - ms % MS_PER_DAY


def iso_day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%d")


class DayCache:
    """Immutable bars per (venue, instrument, resolution, UTC day): `{"t": [ms], "close": []}`."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def default(cls) -> DayCache:
        """`SNAPSHOT_DIR/rv_daily/`, next to the snapshot store (runner/snapshots.py)."""
        return cls(Path(os.environ.get("SNAPSHOT_DIR", MODELS_DIR / ".snapshots")) / "rv_daily")

    def _path(self, venue: str, instrument: str, resolution: str, day_ms: int) -> Path:
        return self.root / venue / instrument / resolution / f"{iso_day(day_ms)}.json"

    def get(self, venue: str, instrument: str, resolution: str, day_ms: int):
        path = self._path(venue, instrument, resolution, day_ms)
        if not path.exists():
            return None
        doc = json.loads(path.read_text())
        return doc["t"], doc["close"]

    def put(self, venue, instrument, resolution, day_ms, t: list[int], close: list[float]) -> None:
        path = self._path(venue, instrument, resolution, day_ms)
        if path.exists():
            return  # completed days never change; the first write stands
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"t": t, "close": close}, allow_nan=False))
        tmp.replace(path)


def pages(days: list[int], days_per_page: int) -> list[tuple[int, int]]:
    """Contiguous runs of `days` (day-start ms, ascending), cut into [start, end) windows."""
    out: list[tuple[int, int]] = []
    for day in days:
        if out and out[-1][1] == day and (day - out[-1][0]) // MS_PER_DAY < days_per_page:
            out[-1] = (out[-1][0], day + MS_PER_DAY)
        else:
            out.append((day, day + MS_PER_DAY))
    return out


def fetch_rv_daily(
    spec_name: str,
    *,
    venue: str,
    instrument: str,
    resolution: str,
    lookback_days: int,
    min_coverage: float,
    as_of: datetime,
    rows_per_call: int,
    fetch_page: FetchPage,
    cache: DayCache | None,
) -> tuple[dict[str, Any], datetime | None, list[Any]]:
    """(data, venue time of the last page fetched or None if every day was cached, raw bodies)."""
    per_day = bars_per_day(spec_name, resolution)
    covered = min_coverage * per_day  # bars a day needs before it is cached
    end = day_floor_ms(as_of)  # C: the close of the last completed day
    # One day more than the lookback: the first returned day's first return needs a prior close.
    days = list(range(end - (lookback_days + 1) * MS_PER_DAY, end, MS_PER_DAY))
    bars: dict[int, tuple[list[int], list[float]]] = {}
    for day in days:
        hit = cache.get(venue, instrument, resolution, day) if cache else None
        if hit is not None:
            bars[day] = hit
    missing = [d for d in days if d not in bars]
    venue_time, raws = None, []

    def fetch(
        start: int, stop: int, depth: int, parent: tuple[dict[int, int], int] | None = None
    ) -> dict[int, tuple[list[int], list[float], Any]]:
        """Day → (bar times, closes, venue time of the response) over [start, stop). `parent` is
        the enclosing response's (bars per day, total bars), None for a page."""
        nonlocal venue_time
        times, closes, venue_time, raw = fetch_page(start, stop)
        raws.append(raw)
        got: dict[int, tuple[list[int], list[float], Any]] = {
            day: ([], [], venue_time) for day in range(start, stop, MS_PER_DAY)
        }
        for t, c in zip(times, closes, strict=True):
            if (day := t - t % MS_PER_DAY) in got:  # bars outside the window are dropped
                got[day][0].append(t)
                got[day][1].append(c)
        short = [day for day, (t, _, _) in got.items() if len(t) < covered]
        n = sum(len(t) for t, _, _ in got.values())
        # Fewer bars than one covered day: no window here can cover a day (an empty window is
        # history the venue doesn't serve; a non-empty one is a cap below a day, or sparse data).
        if n < covered or not short or depth >= MAX_SPLIT_DEPTH or stop - start <= MS_PER_DAY:
            return got
        # A half that added no bar to a short day and came back smaller than its parent was not
        # cut by a cap (a capped half returns as many rows as its parent): the shortfall is real.
        if parent and n < parent[1] and all(len(got[d][0]) <= parent[0][d] for d in short):
            return got
        mid = start + (stop - start) // MS_PER_DAY // 2 * MS_PER_DAY
        counts = {day: len(t) for day, (t, _, _) in got.items()}
        for lo, hi in ((start, mid), (mid, stop)):
            if any(lo <= day < hi for day in short):
                for day, sub in fetch(lo, hi, depth + 1, (counts, n)).items():
                    if len(sub[0]) > len(got[day][0]):
                        got[day] = sub
        return got

    for start, stop in pages(missing, max(1, rows_per_call // per_day)):
        for day, (t, c, page_time) in fetch(start, stop, 0).items():
            bars[day] = (t, c)
            settled_ms = page_time.timestamp() * 1000 - CACHE_SETTLE_S * 1000
            if cache and len(t) >= covered and day + MS_PER_DAY <= settled_ms:
                cache.put(venue, instrument, resolution, day, t, c)
    merged = sorted({t: c for day in days for t, c in zip(*bars[day], strict=True)}.items())
    if not merged:
        raise AdapterError(
            "venue_unavailable", f"{spec_name}: no {venue} {instrument} bars in the lookback"
        )
    found = daily_rv([t for t, _ in merged], [c for _, c in merged], end, per_day, min_coverage)
    # daily_rv keys days by their close; the first (extra) day only supplies a prior close.
    by_day = {close - MS_PER_DAY: (rv, n) for close, rv, n in zip(*found, strict=True)}
    out = days[1:]
    data = {
        "venue": venue,
        "instrument": instrument,
        "resolution": resolution,
        "day": [iso_day(d) for d in out],
        "rv": [by_day.get(d, (None, 0))[0] for d in out],
        "n_bars": [by_day.get(d, (None, 0))[1] for d in out],
    }
    return data, venue_time, raws
