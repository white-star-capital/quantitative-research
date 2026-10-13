"""TEST-ONLY synthetic inputs. Never used outside tests, never written to contracts/fixtures.

Prices are generated *from* a known smile with the Black-76 reference, so every expected value
in the tests has a closed form. They are not market data; golden tests on recorded Deribit
snapshots live in test_golden.py (guide §5.5).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from terminal_models.core.base import InputData
from terminal_models.core.quant.black76 import black76

AS_OF = datetime(2026, 10, 13, 17, tzinfo=UTC)
SPOT = 100.0
BASIS = 0.05
TENORS_DAYS = (3, 10, 31, 95)
STRIKES = [70 + 2.5 * i for i in range(29)]  # 70 … 140


def smile(k: float) -> float:
    return 0.6 + 0.3 * k * k - 0.1 * k


def expiry_at(days: int) -> datetime:
    return (AS_OF + timedelta(days=days)).replace(hour=8, minute=0, second=0)


def rows(tenors=TENORS_DAYS, strikes=STRIKES, spread=0.01, mark_scale=1.0) -> list[dict]:
    out = []
    for days in tenors:
        when = expiry_at(days)
        T = (when - AS_OF).total_seconds() / (365 * 86400)  # noqa: N806
        F = SPOT * math.exp(BASIS * T)  # noqa: N806
        for K in strikes:  # noqa: N806
            vol = smile(math.log(K / F))
            for opt in ("call", "put"):
                call = opt == "call"
                price = black76(F, K, T, vol, 0.0, call, SPOT)["price"]
                bid = black76(F, K, T, vol - spread, 0.0, call, SPOT)["price"]
                ask = black76(F, K, T, vol + spread, 0.0, call, SPOT)["price"]
                out.append(
                    {
                        "instrument_name": f"SYN-{when:%d%b%y}-{K:g}-{opt[0].upper()}",
                        "expiry_ts": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "strike": K,
                        "option_type": opt,
                        "forward": F,
                        "mark_iv": vol,
                        "mark_usd": price * mark_scale,
                        "bid_usd": bid if bid > 0 else None,
                        "ask_usd": ask,
                        "open_interest": 10.0,
                        "contract_size": 1.0,
                        "settlement": "linear",
                    }
                )
    return out


def closes(hours: int = 31 * 24 + 1, step: float = 0.004) -> list[float]:
    path = [SPOT]
    for i in range(hours - 1):
        path.append(path[-1] * math.exp(step if i % 2 else -step))
    return path


def inputs(chain_rows=None, age_s: int = 5, with_dvol: bool = True) -> dict[str, InputData]:
    at = AS_OF - timedelta(seconds=age_s)
    data = {
        "chain": InputData("chain", at, {"rows": chain_rows if chain_rows is not None else rows()}),
        "index": InputData("index", at, {"spot": SPOT}),
        "candles": InputData("candles", at, {"t": [], "close": closes()}),
    }
    if with_dvol:
        data["dvol"] = InputData("dvol", at, {"dvol": 0.62})
    return data
