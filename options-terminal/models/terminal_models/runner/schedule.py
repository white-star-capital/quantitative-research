"""Five-field cron matching (UTC) for `run-due`: minute hour day-of-month month day-of-week.
Supports `*`, `*/n`, `a-b`, `a-b/n` and comma lists. Day-of-week 0 or 7 is Sunday."""

from __future__ import annotations

from datetime import datetime

RANGES = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]


def _field(expr: str, lo: int, hi: int) -> set[int]:
    values: set[int] = set()
    for part in expr.split(","):
        base, _, step_s = part.partition("/")
        step = int(step_s) if step_s else 1
        if base == "*":
            start, end = lo, hi
        elif "-" in base:
            start, end = (int(x) for x in base.split("-"))
        else:
            start = end = int(base)
        if not (lo <= start <= end <= hi) or step < 1:
            raise ValueError(f"cron field '{part}' outside {lo}-{hi}")
        values.update(range(start, end + 1, step))
    return values


def parse(expr: str) -> list[set[int]]:
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(f"cron '{expr}' must have 5 fields")
    parsed = [_field(f, lo, hi) for f, (lo, hi) in zip(fields, RANGES, strict=True)]
    if 7 in parsed[4]:
        parsed[4].add(0)
    return parsed


def is_due(expr: str, at: datetime) -> bool:
    minute, hour, dom, month, dow = parse(expr)
    return (
        at.minute in minute
        and at.hour in hour
        and at.day in dom
        and at.month in month
        and (at.isoweekday() % 7) in dow
    )
