"""Shared snapshot cache (PARITY_MONITOR_IMPLEMENTATION_GUIDE.md D-11, amendment A-3).

One DeribitClient serves every slot and asset in a `run-due` pass (runner/cli.py builds the
adapters once per pass), so the pass is the tick: identical calls inside it return the same
response, and bs_pricer, parity_monitor, SOL and HYPE see one option book. A response is
reused only while younger than its reuse limit, so a reused book never carries an old server
time into a 1-minute model (max_input_age 30 s).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

TICK_MAX_REUSE_S = 20.0  # books and tickers
METADATA_REUSE_S = 3600.0  # instrument listings change daily


class SnapshotCache:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._entries: dict[tuple, tuple[float, Any]] = {}

    @staticmethod
    def key(method: str, params: dict[str, Any]) -> tuple:
        return (method, tuple(sorted(params.items())))

    def get(self, key: tuple, max_reuse_s: float) -> Any | None:
        hit = self._entries.get(key)
        if hit and self._clock() - hit[0] < max_reuse_s:
            return hit[1]
        return None

    def put(self, key: tuple, value: Any) -> None:
        self._entries[key] = (self._clock(), value)
