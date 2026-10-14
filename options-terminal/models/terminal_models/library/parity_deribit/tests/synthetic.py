"""TEST-ONLY synthetic Deribit inputs for parity_deribit. Never used outside tests, never written
to contracts/fixtures (D-10).

Options are priced with Black-76 on each expiry's forward, so parity holds inside the bid/ask by
construction; `plant` pushes one strike's call up to create a known conversion edge. Rows have
exactly the shape the adapter produces (adapters/deribit/chains.py). The fee table is TEST-ONLY
arithmetic, not Deribit's schedule.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from terminal_models.core.base import InputData
from terminal_models.core.quant.black76 import black76

AS_OF = datetime(2026, 10, 14, 12, tzinfo=UTC)
TENORS_DAYS = (3, 10, 31, 95)
MONEYNESS = [-0.3 + 0.05 * i for i in range(13)]  # ln(K/F)
SPOT = {"BTC": 60_000.0, "ETH": 2_500.0, "SOL": 150.0, "HYPE": 40.0}
BASIS = 0.06
FUNDING_8H = 0.0001
TEST_FEES = {
    "schedule_version": "TEST-ONLY",
    "option_taker": 0.0003, "option_maker": 0.0002, "option_cap_pct_of_price": 0.125,
    "future_taker": 0.0005, "future_maker": 0.0001, "perp_taker": 0.0005, "perp_maker": 0.0001,
    "option_delivery": 0.00015, "future_delivery": 0.00025,
}  # fmt: skip


def inverse(asset: str) -> bool:
    return asset in ("BTC", "ETH")


def expiry_at(days: int) -> datetime:
    return (AS_OF + timedelta(days=days)).replace(hour=8, minute=0, second=0)


def ms(t: datetime) -> int:
    return int(t.timestamp() * 1000)


def forward(asset: str, days: int) -> float:
    T = (expiry_at(days) - AS_OF).total_seconds() / (365 * 86400)  # noqa: N806
    return SPOT[asset] * math.exp(BASIS * T)


def strike(asset: str, F: float, m: float) -> float:  # noqa: N803
    step = {"BTC": 500, "ETH": 25, "SOL": 1, "HYPE": 0.5}[asset]
    return round(F * math.exp(m) / step) * step


def chain_rows(asset: str, tenors=TENORS_DAYS, moneyness=MONEYNESS, spread=0.004) -> list[dict]:
    inv, rows = inverse(asset), []
    for days in tenors:
        when = expiry_at(days)
        T = (when - AS_OF).total_seconds() / (365 * 86400)  # noqa: N806
        F = forward(asset, days)  # noqa: N806
        for m in moneyness:
            K = strike(asset, F, m)  # noqa: N806
            vol = 0.6 + 0.2 * m * m
            for opt in ("call", "put"):
                usd = black76(F, K, T, vol, 0.0, opt == "call", SPOT[asset])["price"]
                half = spread * SPOT[asset] * 0.05 + 0.002 * usd
                to_native = 1 / F if inv else 1.0
                bid, ask = (usd - half) * to_native, (usd + half) * to_native
                name = f"{asset}{'' if inv else '_USDC'}-{when:%d%b%y}-{K:g}-{opt[0].upper()}"
                rows.append(
                    {
                        "instrument_name": name.upper(),
                        "expiry_ts": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "strike": float(K),
                        "option_type": opt,
                        "forward": F,
                        "mark_iv": vol,
                        "mark_usd": usd,
                        "bid_usd": bid / to_native if bid > 0 else None,
                        "ask_usd": ask / to_native,
                        "open_interest": 10.0,
                        "contract_size": 1.0,
                        "settlement": "inverse" if inv else "linear",
                        "bid_native": bid if bid > 0 else None,
                        "ask_native": ask,
                        "expiration_ms": ms(when),
                        "underlying_index": f"SYN.{asset}-{when:%d%b%y}".upper(),
                    }
                )
    return rows


def futures_rows(asset: str, dated_days=(10, 31, 95), perp=True, half_spread=0.0001) -> list:
    inv, rows = inverse(asset), []
    size = 10.0 if inv else 1.0
    for days in dated_days:
        F = forward(asset, days)  # noqa: N806
        rows.append(
            {
                "instrument_name": f"{asset}-{expiry_at(days):%d%b%y}".upper(),
                "is_perpetual": False,
                "expiration_ms": ms(expiry_at(days)),
                "bid": F * (1 - half_spread),
                "ask": F * (1 + half_spread),
                "mark": F,
                "contract_size": size,
                "settlement": "inverse" if inv else "linear",
                "settlement_currency": asset if inv else "USDC",
            }  # fmt: skip
        )
    if perp:
        S = SPOT[asset]  # noqa: N806
        rows.append(
            {
                "instrument_name": f"{asset}{'' if inv else '_USDC'}-PERPETUAL",
                "is_perpetual": True,
                "expiration_ms": None,
                "bid": S * (1 - half_spread),
                "ask": S * (1 + half_spread),
                "mark": S,
                "contract_size": size,
                "settlement": "inverse" if inv else "linear",
                "settlement_currency": asset if inv else "USDC",
            }  # fmt: skip
        )
    return rows


def plant(rows: list[dict], expiry_days: int, m: float, asset: str, bump: float) -> str:
    """Raise one call's bid/ask by `bump` × forward (native units) → a conversion edge there.
    Returns the call's instrument name."""
    F = forward(asset, expiry_days)  # noqa: N806
    K = strike(asset, F, m)  # noqa: N806
    when = ms(expiry_at(expiry_days))
    for r in rows:
        if r["expiration_ms"] == when and r["strike"] == K and r["option_type"] == "call":
            add = bump if inverse(asset) else bump * F
            r["bid_native"] += add
            r["ask_native"] += add
            return r["instrument_name"]
    raise AssertionError("no such strike")


def inputs(
    asset: str = "BTC",
    chain=None,
    futures=None,
    futures_age_s: float = 0.0,
    age_s: float = 2.0,
    funding_8h: float | None = FUNDING_8H,
) -> dict[str, InputData]:
    at = AS_OF - timedelta(seconds=age_s)
    return {
        "chain": InputData(
            "chain", at, {"rows": chain if chain is not None else chain_rows(asset)}
        ),
        "index": InputData("index", at, {"spot": SPOT[asset]}),
        "futures": InputData(
            "futures",
            at - timedelta(seconds=futures_age_s),
            {"rows": futures if futures is not None else futures_rows(asset)},
        ),
        "funding": InputData(
            "funding",
            at,
            {
                "instrument_name": "PERP",
                "current_funding": 0.0,
                "funding_8h": funding_8h,
            },  # fmt: skip
        ),
    }


def order_books(chain: list[dict], futures: list[dict], names: list[str], size: float = 5.0,
                spread_s: float = 0.2) -> dict[str, InputData]:  # fmt: skip
    """Order books whose top of book equals the summary prices (TEST-ONLY). Options get `size`
    contracts; futures are deep (inverse amounts are USD, linear amounts coin), so the option
    legs bind the executable size. Book times are 1 s old, `spread_s` apart."""
    tops = {r["instrument_name"]: (r["bid_native"], r["ask_native"], size) for r in chain}
    for f in futures:
        depth = 1e9 if f["settlement"] == "inverse" else 1e6
        tops[f["instrument_name"]] = (f["bid"], f["ask"], depth)
    out = {}
    for i, name in enumerate(names):
        bid, ask, amount = tops[name]
        out[f"order_book:{name}"] = InputData(
            f"order_book:{name}",
            AS_OF - timedelta(seconds=1 + i * spread_s),
            {
                "instrument_name": name,
                "bids": [[bid, amount]] if bid else [],
                "asks": [[ask, amount]],
            },  # fmt: skip
        )
    return out
