"""Put-call parity on Deribit (PARITY_MONITOR_IMPLEMENTATION_GUIDE.md §3).

Two formula sets, never mixed: linear (SOL/HYPE, USD per unit, hedged with 1 unit of a linear
future/perp) and inverse (BTC/ETH, coin per 1-coin option, hedged with an inverse future at USD
notional K). Every function returns a value and a reason; the value is None when a guard fires
(§3.7): a non-positive input, 1 − (c − p) ≤ 0 on an inverse book, or T below the minimum.

Fees (amendment A-4), per 1 unit of underlying in the book's native unit:
    option trade     min(rate × U, cap × price), U = 1 coin (inverse) or the index in USD (linear)
    hedge trade      rate × notional: rate × K / F coin (inverse), rate × F USD (linear)
    delivery         one option leg finishes in the money: option_delivery × U (no cap; the
                     option's value at expiry is unknown, so the uncapped fee is the bound);
                     a dated future hedge also pays future_delivery; a perp pays none.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

Result = tuple[float | None, str | None]

NON_POSITIVE = "non_positive_input"
INVERSE_DENOMINATOR = "inverse_denominator_non_positive"
TENOR_TOO_SHORT = "tenor_below_min"
INSUFFICIENT_STRIKES = "insufficient_strikes"
YEAR_HOURS = 365 * 24


def _bad(*xs: float | None) -> bool:
    return any(x is None or not math.isfinite(x) or x <= 0 for x in xs)


# --- §3.1 linear ------------------------------------------------------------------------------


def lin_band(
    K, c_bid, c_ask, p_bid, p_ask, DF=1.0
) -> tuple[float | None, float | None, str | None]:  # noqa: N803
    """(F_impl_bid, F_impl_ask): the forwards at which conversion / reversal are flat."""
    if _bad(K, c_bid, c_ask, p_bid, p_ask, DF):
        return None, None, NON_POSITIVE
    return K + (c_bid - p_ask) / DF, K + (c_ask - p_bid) / DF, None


def lin_conversion(K, c_bid, p_ask, F_ask, DF=1.0) -> Result:  # noqa: N803
    """Sell call, buy put, buy hedge at F_ask: USD locked at expiry per unit."""
    if _bad(K, c_bid, p_ask, F_ask, DF):
        return None, NON_POSITIVE
    return (c_bid - p_ask) / DF - (F_ask - K), None


def lin_reversal(K, c_ask, p_bid, F_bid, DF=1.0) -> Result:  # noqa: N803
    """Buy call, sell put, sell hedge at F_bid: USD locked at expiry per unit."""
    if _bad(K, c_ask, p_bid, F_bid, DF):
        return None, NON_POSITIVE
    return (F_bid - K) - (c_ask - p_bid) / DF, None


def lin_forward(K, c_mid, p_mid, DF=1.0) -> Result:  # noqa: N803
    if _bad(K, c_mid, p_mid, DF):
        return None, NON_POSITIVE
    return K + (c_mid - p_mid) / DF, None


# --- §3.2 inverse -----------------------------------------------------------------------------


def _inv_forward(K: float, diff: float) -> Result:  # noqa: N803
    denominator = 1 - diff
    if denominator <= 0:
        return None, INVERSE_DENOMINATOR
    return K / denominator, None


def inv_band(K, c_bid, c_ask, p_bid, p_ask) -> tuple[float | None, float | None, str | None]:  # noqa: N803
    if _bad(K, c_bid, c_ask, p_bid, p_ask):
        return None, None, NON_POSITIVE
    lo, why_lo = _inv_forward(K, c_bid - p_ask)
    hi, why_hi = _inv_forward(K, c_ask - p_bid)
    return lo, hi, why_lo or why_hi


def inv_conversion(K, c_bid, p_ask, F_ask) -> Result:  # noqa: N803
    """Coin locked at expiry per 1 coin of options, hedge long K USD of inverse future."""
    if _bad(K, c_bid, p_ask, F_ask):
        return None, NON_POSITIVE
    return (c_bid - p_ask) - 1 + K / F_ask, None


def inv_reversal(K, c_ask, p_bid, F_bid) -> Result:  # noqa: N803
    if _bad(K, c_ask, p_bid, F_bid):
        return None, NON_POSITIVE
    return 1 - K / F_bid - (c_ask - p_bid), None


def inv_forward(K, c_mid, p_mid) -> Result:  # noqa: N803
    if _bad(K, c_mid, p_mid):
        return None, NON_POSITIVE
    return _inv_forward(K, c_mid - p_mid)


# --- dispatch on settlement -------------------------------------------------------------------


def band(inverse: bool, K, c_bid, c_ask, p_bid, p_ask, DF=1.0):  # noqa: N803
    return (
        inv_band(K, c_bid, c_ask, p_bid, p_ask)
        if inverse
        else lin_band(K, c_bid, c_ask, p_bid, p_ask, DF)
    )


def conversion(inverse: bool, K, c_bid, p_ask, F_ask, DF=1.0) -> Result:  # noqa: N803
    return (
        inv_conversion(K, c_bid, p_ask, F_ask)
        if inverse
        else lin_conversion(K, c_bid, p_ask, F_ask, DF)
    )


def reversal(inverse: bool, K, c_ask, p_bid, F_bid, DF=1.0) -> Result:  # noqa: N803
    return (
        inv_reversal(K, c_ask, p_bid, F_bid)
        if inverse
        else lin_reversal(K, c_ask, p_bid, F_bid, DF)
    )


def forward(inverse: bool, K, c_mid, p_mid, DF=1.0) -> Result:  # noqa: N803
    return inv_forward(K, c_mid, p_mid) if inverse else lin_forward(K, c_mid, p_mid, DF)


# --- §3.5 inventory lock ----------------------------------------------------------------------


def _tenor_ok(T: float, min_tte_hours: float) -> bool:  # noqa: N803
    return math.isfinite(T) and T * YEAR_HOURS >= min_tte_hours and T > 0


def lin_inventory_rate(K, S, c_bid, p_ask, T, fees=0.0, min_tte_hours=0.0) -> Result:  # noqa: N803
    """Hold 1 unit, sell call, buy put: continuous USD rate locked. Fees in USD per unit."""
    if _bad(K, S, c_bid, p_ask):
        return None, NON_POSITIVE
    if not _tenor_ok(T, min_tte_hours):
        return None, TENOR_TOO_SHORT
    cost = S - c_bid + p_ask + fees
    if cost <= 0:
        return None, NON_POSITIVE
    return math.log(K / cost) / T, None


def inv_inventory_rate(K, S, c_bid, p_ask, T, fees=0.0, min_tte_hours=0.0) -> Result:  # noqa: N803
    """Hold 1 − (c − p) coin, sell call, buy put on 1 coin: exactly K USD at expiry.
    Fees in coin per coin of options."""
    if _bad(K, S, c_bid, p_ask):
        return None, NON_POSITIVE
    if not _tenor_ok(T, min_tte_hours):
        return None, TENOR_TOO_SHORT
    held = 1 - (c_bid - p_ask) + fees
    if held <= 0:
        return None, INVERSE_DENOMINATOR
    return math.log(K / (S * held)) / T, None


def inventory_rate(inverse: bool, K, S, c_bid, p_ask, T, fees=0.0, min_tte_hours=0.0) -> Result:  # noqa: N803
    fn = inv_inventory_rate if inverse else lin_inventory_rate
    return fn(K, S, c_bid, p_ask, T, fees, min_tte_hours)


# --- §3.3 fees --------------------------------------------------------------------------------


@dataclass(frozen=True)
class FeeRates:
    """One side (maker or taker) of the manifest fee table (§5 `fees:`)."""

    option: float
    option_cap: float  # fraction of the option price
    future: float
    perp: float
    option_delivery: float
    future_delivery: float

    @classmethod
    def from_table(cls, table: dict, side: str = "taker") -> FeeRates:
        return cls(
            option=float(table[f"option_{side}"]),
            option_cap=float(table["option_cap_pct_of_price"]),
            future=float(table[f"future_{side}"]),
            perp=float(table[f"perp_{side}"]),
            option_delivery=float(table["option_delivery"]),
            future_delivery=float(table["future_delivery"]),
        )


def option_fee(price: float, rate: float, cap: float, unit_value: float) -> float:
    """Per option, native unit. unit_value: 1 (inverse, coin) or the index in USD (linear)."""
    return min(rate * unit_value, cap * price)


def hedge_fee(inverse: bool, rate: float, K: float, F: float) -> float:  # noqa: N803
    """Hedge leg per 1 unit of options: inverse K USD notional in coin; linear 1 unit in USD."""
    return rate * K / F if inverse else rate * F


def structure_fees(
    inverse: bool,
    rates: FeeRates,
    K: float,  # noqa: N803
    call_px: float,
    put_px: float,
    hedge_px: float,
    index: float,
    hedge_is_perp: bool,
) -> float:
    """All legs of a conversion or reversal held to expiry, per unit, native unit."""
    unit_value = 1.0 if inverse else index
    trade = (
        option_fee(call_px, rates.option, rates.option_cap, unit_value)
        + option_fee(put_px, rates.option, rates.option_cap, unit_value)
        + hedge_fee(inverse, rates.perp if hedge_is_perp else rates.future, K, hedge_px)
    )
    delivery = rates.option_delivery * unit_value
    if not hedge_is_perp:
        delivery += hedge_fee(inverse, rates.future_delivery, K, hedge_px)
    return trade + delivery


def inventory_fees(
    inverse: bool, rates: FeeRates, call_px: float, put_px: float, index: float
) -> float:
    """Sell call + buy put (+ delivery on the leg that finishes in the money)."""
    unit_value = 1.0 if inverse else index
    return (
        option_fee(call_px, rates.option, rates.option_cap, unit_value)
        + option_fee(put_px, rates.option, rates.option_cap, unit_value)
        + rates.option_delivery * unit_value
    )


# --- §3.4 carry -------------------------------------------------------------------------------


def weighted_median(values: Iterable[tuple[float, float]]) -> float | None:
    """Lower weighted median of (value, weight) pairs; weights must be positive."""
    pairs = sorted(values)
    total = sum(w for _, w in pairs)
    if not pairs or total <= 0:
        return None
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= total / 2:
            return value
    return pairs[-1][0]


def implied_forward(strikes: list[tuple[float, float, float]], min_strikes: int) -> Result:
    """Expiry forward from per-strike (F_mid, F_bid, F_ask): median weighted by 1/band width.
    Callers pass only strikes inside the ATM band with two-sided calls and puts."""
    usable = [(f, 1 / (hi - lo)) for f, lo, hi in strikes if hi > lo]
    if len(usable) < min_strikes:
        return None, INSUFFICIENT_STRIKES
    return weighted_median(usable), None


def basis_ann(F: float | None, S: float, T: float) -> float | None:  # noqa: N803
    """Simple annualized basis (F/S − 1)/T."""
    if F is None or F <= 0 or S <= 0 or T <= 0:
        return None
    return (F / S - 1) / T


def bps(a: float | None, b: float | None) -> float | None:
    """(a/b − 1) × 10⁴."""
    if a is None or b is None or b <= 0:
        return None
    return (a / b - 1) * 1e4


def breakeven_funding(structure: str, net_usd: float, K: float, T: float) -> float | None:  # noqa: N803
    """Constant annualized perp funding (paid by longs) at which a perp-hedged edge is zero.
    Conversion is long the perp: profitable while funding < breakeven. Reversal is short the
    perp: profitable while funding > breakeven."""
    if K <= 0 or T <= 0:
        return None
    rate = net_usd / (K * T)
    return rate if structure == "conversion" else -rate


def hedge_contracts(
    inverse: bool,
    units: float,
    K: float,
    F: float,
    contract_size: float,  # noqa: N803
) -> tuple[int, float]:
    """(whole contracts, rounding residual in USD) to hedge `units` of options.
    Inverse: notional K USD per coin, contract_size in USD. Linear: contract_size in coin."""
    if contract_size <= 0:
        raise ValueError("non_positive_contract_size")
    exact = units * K / contract_size if inverse else units / contract_size
    contracts = round(exact)
    residual = abs(contracts - exact) * contract_size * (1.0 if inverse else F)
    return contracts, residual


def net_metrics(inverse: bool, gross: float, fees: float, K: float, S: float, T: float) -> dict:  # noqa: N803
    """Net edge per unit in the native unit, its USD value (inverse: coin × index, flagged
    coin_denominated_edge by the caller), bp of the K USD notional, and simple annualized.
    Fees are subtracted as paid; with usd_rate = 0 (Deribit's convention) that is exact."""
    net = gross - fees
    net_usd = net * S if inverse else net
    return {
        "fees": fees,
        "net_edge": net,
        "net_edge_usd": net_usd,
        "net_edge_bps_notional": net_usd / K * 1e4,
        "net_edge_ann": net_usd / K / T if T > 0 else None,
    }
