"""Event calendar (RV guide §4, VRP-A-9): Deribit opex rule, scope filtering, window dummies.
Event entries here are hand-made test calendar rows, not market data."""

from datetime import UTC, date, datetime

import numpy as np
import pytest

from terminal_models.core.quant.calendar import (
    DAY_MS,
    Event,
    day_of,
    days_left,
    deribit_opex,
    events_for,
    incomplete_types,
    last_friday,
    weekdays,
    window_dummies,
)


def utc(*args):
    return datetime(*args, tzinfo=UTC)


def close(d: date) -> int:
    """Unix-ms close (next 00:00 UTC) of UTC day d."""
    return int(utc(d.year, d.month, d.day).timestamp() * 1000) + DAY_MS


def test_last_friday_known_months():
    assert last_friday(2026, 10) == date(2026, 10, 30)
    assert last_friday(2026, 11) == date(2026, 11, 27)
    assert last_friday(2026, 12) == date(2026, 12, 25)  # Christmas; the rule does not move it
    assert last_friday(2027, 1) == date(2027, 1, 29)
    assert last_friday(2027, 4) == date(2027, 4, 30)  # month ends on a Friday


def test_deribit_opex_types_and_times():
    ev = deribit_opex(utc(2026, 10, 1), utc(2027, 3, 31))
    assert [(e.ts, e.type) for e in ev] == [
        (utc(2026, 10, 30, 8), "opex_monthly"),
        (utc(2026, 11, 27, 8), "opex_monthly"),
        (utc(2026, 12, 25, 8), "opex_quarterly"),  # quarterly only, never also monthly
        (utc(2027, 1, 29, 8), "opex_monthly"),
        (utc(2027, 2, 26, 8), "opex_monthly"),
        (utc(2027, 3, 26, 8), "opex_quarterly"),
    ]
    assert {(e.scope, e.source) for e in ev} == {("GLOBAL", "rule")}


def test_deribit_opex_bounds_are_inclusive_on_the_timestamp():
    assert len(deribit_opex(utc(2026, 10, 30, 8), utc(2026, 10, 30, 8))) == 1
    assert deribit_opex(utc(2026, 10, 30, 8, 1), utc(2026, 11, 27, 7)) == []


MANIFEST = [
    {"type": "fomc", "scope": "GLOBAL", "ts": "2026-12-09T19:00:00Z", "source": "test"},
    {"type": "token_unlock", "scope": "HYPE", "ts": "2026-11-29T00:00:00Z", "source": "test"},
    {"type": "custom", "scope": "BTC", "ts": "2027-02-01T00:00:00Z", "source": "test"},
]


def test_events_for_scope_and_order():
    ev = events_for("HYPE", MANIFEST, utc(2026, 11, 1), utc(2026, 12, 31))
    assert [(e.ts.date(), e.type, e.scope) for e in ev] == [
        (date(2026, 11, 27), "opex_monthly", "GLOBAL"),
        (date(2026, 11, 29), "token_unlock", "HYPE"),
        (date(2026, 12, 9), "fomc", "GLOBAL"),
        (date(2026, 12, 25), "opex_quarterly", "GLOBAL"),
    ]
    btc = events_for("BTC", MANIFEST, utc(2026, 11, 1), utc(2026, 12, 31))
    assert "token_unlock" not in {e.type for e in btc}


def test_events_for_rejects_bad_entries():
    opex = {"type": "opex_monthly", "scope": "GLOBAL", "ts": "2026-11-27T08:00:00Z", "source": "x"}
    with pytest.raises(ValueError, match="derived by rule"):
        events_for("BTC", [opex], utc(2026, 11, 1), utc(2026, 12, 1))
    with pytest.raises(ValueError, match="unknown event type"):
        events_for("BTC", [opex | {"type": "earnings"}], utc(2026, 11, 1), utc(2026, 12, 1))
    with pytest.raises(ValueError, match="UTC offset"):
        events_for("BTC", [opex | {"type": "cpi", "ts": "2026-11-12T13:30:00"}], utc(2026, 11, 1),
                   utc(2026, 12, 1))  # fmt: skip


def test_day_of_and_weekdays():
    assert day_of(close(date(2026, 10, 28))) == date(2026, 10, 28)
    assert weekdays([close(date(2026, 10, 26)), close(date(2026, 11, 1))]) == [0, 6]


def test_window_dummies_per_type_windows():
    days = [close(date(2026, 12, d)) for d in range(7, 13)]  # Dec 7..12
    events = [
        Event(utc(2026, 12, 9, 19), "fomc", "GLOBAL", "test"),
        Event(utc(2026, 12, 12, 1), "cpi", "GLOBAL", "test"),
    ]
    windows = {"cpi": [0, 0], "fomc": [1, 1], "halving": [3, 3]}
    types, m = window_dummies(days, events, windows)
    assert types == ["fomc", "cpi", "halving"]  # EVENT_TYPES order, configured types only
    assert m[:, 0].tolist() == [0, 1, 1, 1, 0, 0]  # Dec 8..10
    assert m[:, 1].tolist() == [0, 0, 0, 0, 0, 1]
    assert (m[:, 2] == 0).all()  # configured but absent: zero column, stable shape


def test_window_dummies_needs_a_window_per_event_type():
    with pytest.raises(ValueError, match="no event window"):
        window_dummies([close(date(2026, 12, 9))], [Event(utc(2026, 12, 9), "cpi", "GLOBAL", "t")],
                       {"fomc": [1, 1]})  # fmt: skip
    with pytest.raises(ValueError, match="before, after"):
        window_dummies([close(date(2026, 12, 9))], [Event(utc(2026, 12, 9), "cpi", "GLOBAL", "t")],
                       {"cpi": [-1, 1]})  # fmt: skip
    assert window_dummies([], [], {"cpi": [0, 0]})[1].shape == (0, 1)
    assert isinstance(window_dummies([], [], {})[1], np.ndarray)


def test_incomplete_types_names_lists_that_stop_too_early():
    through = {"cpi": "2025-10-31", "fomc": date(2026, 12, 31)}
    assert incomplete_types(through, date(2025, 10, 31)) == []
    assert incomplete_types(through, date(2026, 1, 15)) == ["cpi"]
    assert incomplete_types(through, date(2027, 1, 1)) == ["fomc", "cpi"]  # EVENT_TYPES order
    assert incomplete_types({}, date(2030, 1, 1)) == []
    for bad in ("opex_monthly", "earnings"):
        with pytest.raises(ValueError, match="not a listed event type"):
            incomplete_types({bad: "2026-01-01"}, date(2026, 1, 1))


def test_days_left_is_the_margin_of_incomplete_types():
    through = {"cpi": "2027-01-13", "fomc": date(2027, 12, 31)}
    assert days_left(through, date(2026, 12, 30)) == 14  # the earliest list decides
    assert days_left(through, date(2027, 1, 13)) == 0
    assert incomplete_types(through, date(2027, 1, 13)) == []
    assert days_left(through, date(2027, 1, 14)) == -1
    assert incomplete_types(through, date(2027, 1, 14)) == ["cpi"]
    with pytest.raises(ValueError, match="not a listed event type"):
        days_left({"opex_monthly": "2026-01-01"}, date(2026, 1, 1))
    assert days_left({}, date(2026, 1, 1)) is None  # no listed type, nothing runs out
    assert incomplete_types({}, date(2026, 1, 1)) == []
