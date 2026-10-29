"""Event calendar for the RV forecast (RV_FORECAST_IMPLEMENTATION_GUIDE.md §4, VRP-A-9).

Events come from the manifest's `params.events` (`{type, scope, ts, source}`) plus Deribit opex,
which is derived by rule: the last Friday of every month at 08:00 UTC, typed `opex_quarterly` in
March, June, September and December and `opex_monthly` otherwise (never both). Rule-derived opex is
GLOBAL with source `rule`. Days are UTC calendar days, identified by their close in Unix ms (the
00:00 UTC that ends them, guide §12.1).

A listed type is only as good as its list: the manifest's `params.events_listed_through` names,
per scheduled type, the last date through which its list is complete. Past that date a day with
no listed event may still be an event day, so a run that needs those days is flagged
(`incomplete_types`).
"""

from __future__ import annotations

import calendar as _cal
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np

EVENT_TYPES = (
    "fomc",
    "cpi",
    "opex_monthly",
    "opex_quarterly",
    "token_unlock",
    "halving",
    "custom",
)
RULE_TYPES = ("opex_monthly", "opex_quarterly")
QUARTERLY_MONTHS = (3, 6, 9, 12)
OPEX_HOUR_UTC = 8
GLOBAL = "GLOBAL"
RULE_SOURCE = "rule"
DAY_MS = 86_400_000


@dataclass(frozen=True)
class Event:
    ts: datetime  # UTC, aware
    type: str
    scope: str  # GLOBAL or an underlying
    source: str


def day_of(close_ms: int) -> date:
    """The UTC calendar date of the day that closes at `close_ms`."""
    return datetime.fromtimestamp((close_ms - DAY_MS) / 1000, UTC).date()


def weekdays(day_closes_ms: Sequence[int]) -> list[int]:
    """Monday = 0 … Sunday = 6, of each day (not of its close)."""
    return [day_of(c).weekday() for c in day_closes_ms]


def last_friday(year: int, month: int) -> date:
    last = date(year, month, _cal.monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - _cal.FRIDAY) % 7)


def deribit_opex(start: datetime, end: datetime) -> list[Event]:
    """Every Deribit monthly/quarterly expiry with start ≤ ts ≤ end, ascending."""
    out = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        d = last_friday(year, month)
        ts = datetime(d.year, d.month, d.day, OPEX_HOUR_UTC, tzinfo=UTC)
        if start <= ts <= end:
            kind = "opex_quarterly" if month in QUARTERLY_MONTHS else "opex_monthly"
            out.append(Event(ts, kind, GLOBAL, RULE_SOURCE))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def _parse_ts(value: str | datetime) -> datetime:
    when = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if when.tzinfo is None:
        raise ValueError(f"event ts {value!r} has no UTC offset")
    return when.astimezone(UTC)


def parse_events(entries: Iterable[Mapping]) -> list[Event]:
    """Manifest entries → events. Opex types are rejected: they are derived by rule."""
    out = []
    for e in entries:
        if e["type"] not in EVENT_TYPES:
            raise ValueError(f"unknown event type {e['type']!r}")
        if e["type"] in RULE_TYPES:
            raise ValueError(f"{e['type']} is derived by rule, not listed in the manifest")
        out.append(Event(_parse_ts(e["ts"]), e["type"], e["scope"], e["source"]))
    return out


def events_for(
    asset: str, manifest_events: Iterable[Mapping], start: datetime, end: datetime
) -> list[Event]:
    """In-scope events (GLOBAL or `asset`) with start ≤ ts ≤ end, manifest plus rule-derived
    opex, sorted by ts then type."""
    listed = [
        e
        for e in parse_events(manifest_events)
        if e.scope in (GLOBAL, asset) and start <= e.ts <= end
    ]
    return sorted(listed + deribit_opex(start, end), key=lambda e: (e.ts, e.type))


def incomplete_types(listed_through: Mapping[str, str | date], until: date) -> list[str]:
    """Types in `listed_through` whose list is complete only to a date before `until`, in
    EVENT_TYPES order. Types it doesn't name (rule-derived opex, ad hoc types) are not checked."""
    late = []
    for kind, through in listed_through.items():
        if kind not in EVENT_TYPES or kind in RULE_TYPES:
            raise ValueError(f"events_listed_through: {kind!r} is not a listed event type")
        when = through if isinstance(through, date) else date.fromisoformat(str(through))
        if when < until:
            late.append(kind)
    return [t for t in EVENT_TYPES if t in late]


def days_left(listed_through: Mapping[str, str | date], until: date) -> int | None:
    """Days from `until` to the earliest date in `listed_through`: negative exactly when
    `incomplete_types(listed_through, until)` is non-empty; None when it names no type.
    Same validation."""
    incomplete_types(listed_through, until)
    return min(
        (
            ((t if isinstance(t, date) else date.fromisoformat(str(t))) - until).days
            for t in listed_through.values()
        ),
        default=None,
    )


def window(event: Event, windows_by_type: Mapping[str, Sequence[int]]) -> tuple[date, date]:
    """First and last UTC date the event's window covers: D − before … D + after."""
    if event.type not in windows_by_type:
        raise ValueError(f"no event window configured for {event.type!r}")
    before, after = windows_by_type[event.type]
    if before < 0 or after < 0:
        raise ValueError(f"event window for {event.type!r} must be [before, after] ≥ 0")
    d = event.ts.date()
    return d - timedelta(days=before), d + timedelta(days=after)


def window_types(windows_by_type: Mapping[str, Sequence[int]]) -> list[str]:
    """Dummy column order: EVENT_TYPES order, configured types only. Stable across runs."""
    return [t for t in EVENT_TYPES if t in windows_by_type]


def window_dummies(
    day_closes_ms: Sequence[int],
    events: Iterable[Event],
    windows_by_type: Mapping[str, Sequence[int]],
) -> tuple[list[str], np.ndarray]:
    """Per-type 0/1 columns, one row per day: 1 where some event of that type covers the day.
    Types with no event in range are all-zero columns, so the matrix shape never depends on data."""
    types = window_types(windows_by_type)
    col = {t: i for i, t in enumerate(types)}
    days = [day_of(c) for c in day_closes_ms]
    out = np.zeros((len(days), len(types)))
    for e in events:
        first, last = window(e, windows_by_type)
        for i, d in enumerate(days):
            if first <= d <= last:
                out[i, col[e.type]] = 1.0
    return types, out
