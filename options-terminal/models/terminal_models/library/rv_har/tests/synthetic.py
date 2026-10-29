"""TEST-ONLY synthetic inputs for rv_har. Never used outside tests, never written to
contracts/fixtures (D-10).

Daily RV is a seeded AR(1) in log variance plus a known bump on Deribit opex days (the rule in
core/quant/calendar.py), so the fit has a real event effect to find. The chain is priced with
Black-76 from a flat-ish smile, as in the other implementations' synthetic chains. DVOL is a
multiple of the RV level with noise. None of it is market data.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np

from terminal_models.core.base import InputData
from terminal_models.core.quant.black76 import black76
from terminal_models.core.quant.calendar import deribit_opex

AS_OF = datetime(2026, 10, 14, 12, tzinfo=UTC)
LAST_DAY = (AS_OF - timedelta(days=1)).date()  # the last completed UTC day; C = AS_OF's 00:00
SPOT = {"BTC": 60_000.0, "ETH": 2_500.0, "SOL": 150.0, "HYPE": 40.0}
TENORS_DAYS = (3, 10, 31, 95)
MONEYNESS = [-0.3 + 0.05 * i for i in range(13)]  # ln(K/F)
VOL_LEVEL = 0.5  # annualized level the RV process reverts to
OPEX_BUMP = 0.8  # added to ln RV on each opex day
BASIS = 0.05


def days(n: int) -> list[str]:
    return [(LAST_DAY - timedelta(days=n - 1 - i)).isoformat() for i in range(n)]


def rv_series(n: int, seed: int = 7, phi: float = 0.9, sd: float = 0.35) -> list[float]:
    """Daily variance (not annualized) for the `n` days ending LAST_DAY."""
    rng = np.random.default_rng(seed)
    mu = math.log(VOL_LEVEL**2 / 365)
    names = days(n)
    first = datetime.fromisoformat(names[0]).replace(tzinfo=UTC)
    opex = {e.ts.date().isoformat() for e in deribit_opex(first, AS_OF)}
    x, out = 0.0, []
    for d in names:
        x = phi * x + sd * rng.standard_normal()
        out.append(math.exp(mu + x + (OPEX_BUMP if d in opex else 0.0)))
    return out


def dvol_series(rv: list[float | None], premium: float = 1.15, seed: int = 11) -> list[float]:
    """A 30-day-ish implied level: premium × trailing-30 RV, with noise; decimal."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(len(rv)):
        tail = [v for v in rv[max(0, i - 29) : i + 1] if v is not None]
        level = math.sqrt(365 * sum(tail) / len(tail)) if tail else VOL_LEVEL
        out.append(premium * level * math.exp(0.05 * rng.standard_normal()))
    return out


def expiry_at(days_out: int) -> datetime:
    return (AS_OF + timedelta(days=days_out)).replace(hour=8, minute=0, second=0)


def chain_rows(asset: str = "BTC", tenors=TENORS_DAYS, spread: float = 0.01) -> list[dict]:
    """The `chain` kind's rows (adapters/deribit/chains.py shape), USD prices."""
    S, rows = SPOT[asset], []  # noqa: N806
    for d in tenors:
        when = expiry_at(d)
        T = (when - AS_OF).total_seconds() / (365 * 86400)  # noqa: N806
        F = S * math.exp(BASIS * T)  # noqa: N806
        for m in MONEYNESS:
            K = F * math.exp(m)  # noqa: N806
            vol = 0.55 + 0.002 * d + 0.2 * m * m
            for opt in ("call", "put"):
                call = opt == "call"
                price = black76(F, K, T, vol, 0.0, call, S)["price"]
                bid = black76(F, K, T, vol - spread, 0.0, call, S)["price"]
                ask = black76(F, K, T, vol + spread, 0.0, call, S)["price"]
                rows.append(
                    {
                        "instrument_name": f"SYN-{when:%d%b%y}-{K:.4f}-{opt[0].upper()}",
                        "expiry_ts": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "strike": K,
                        "option_type": opt,
                        "forward": F,
                        "mark_iv": vol,
                        "mark_usd": price,
                        "bid_usd": bid if bid > 0 else None,
                        "ask_usd": ask,
                        "open_interest": 10.0,
                        "contract_size": 1.0,
                        "settlement": "linear",
                    }
                )
    return rows


def inputs(
    asset: str = "BTC",
    n_days: int = 600,
    rv: list[float | None] | None = None,
    chain: list[dict] | None = None,
    with_dvol: bool = True,
    lag_days: int = 0,
    age_s: int = 5,
) -> dict[str, InputData]:
    """`lag_days` drops the most recent completed days from the RV input (a lagging feed)."""
    at = AS_OF - timedelta(seconds=age_s)
    rv = rv if rv is not None else rv_series(n_days)
    names = days(len(rv))
    if lag_days:
        rv, names = rv[:-lag_days], names[:-lag_days]
    out = {
        "chain": InputData(
            "chain", at, {"rows": chain if chain is not None else chain_rows(asset)}
        ),
        "index": InputData("index", at, {"spot": SPOT[asset]}),
        "rv_daily": InputData(
            "rv_daily",
            at,
            {
                "venue": "synthetic",
                "instrument": f"{asset}-SYN",
                "resolution": "60",
                "day": names,
                "rv": rv,
                "n_bars": [24 if v is not None else 0 for v in rv],
            },
        ),
    }
    if with_dvol and asset in ("BTC", "ETH"):
        out["dvol_history"] = InputData("dvol_history", at, {"day": names, "dvol": dvol_series(rv)})
    return out
