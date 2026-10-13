"""Black-76 pricing, greeks and implied vol (BS_PRICER_IMPLEMENTATION_GUIDE.md §3.4–3.5).

This is the normative reference: the Rust port in api/src/pricing/ is held to it by the vectors in
contracts/fixtures/pricing/ (API A-33). Change the math here first, regenerate the vectors, then
port. All values are USD per 1 unit of underlying.
"""

from __future__ import annotations

import math

SQRT2PI = math.sqrt(2 * math.pi)

IV_LO = 0.01
IV_HI = 5.0
IV_TOL = 1e-10
IV_MAXIT = 200


def N(x: float) -> float:  # noqa: N802 - standard notation
    return 0.5 * math.erfc(-x / math.sqrt(2))


def n(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT2PI


def black76(
    F: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    is_call: bool,
    S: float,  # noqa: N803
) -> dict[str, float]:
    """Price and greeks. With T <= 0 or sigma <= 0 only the discounted intrinsic `price` is set."""
    if not (F > 0 and K > 0 and S > 0):
        raise ValueError("non_positive_input")
    DF = math.exp(-r * T)  # noqa: N806
    if T <= 0 or sigma <= 0:
        intr = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return {"price": DF * intr}
    sd = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if is_call:
        price, delta, p_itm = DF * (F * N(d1) - K * N(d2)), DF * N(d1), N(d2)
    else:
        price, delta, p_itm = DF * (K * N(-d2) - F * N(-d1)), -DF * N(-d1), N(-d2)
    fs = F / S
    gamma_spot = DF * n(d1) / (F * sd) * fs * fs
    return {
        "price": price,
        "delta": delta,
        "delta_spot": delta * fs,
        "gamma_1pct": gamma_spot * S * 0.01,
        "vega_1pt": DF * F * n(d1) * math.sqrt(T) / 100,
        "theta_1d": (-DF * F * n(d1) * sigma / (2 * math.sqrt(T)) + r * price) / 365,
        "p_itm": p_itm,
    }


def implied_vol(
    F: float,  # noqa: N803
    K: float,  # noqa: N803
    T: float,  # noqa: N803
    price: float,
    r: float,
    is_call: bool,
    S: float,  # noqa: N803
    lo: float = IV_LO,
    hi: float = IV_HI,
    tol: float = IV_TOL,
    maxit: int = IV_MAXIT,
) -> tuple[float | None, str | None]:
    """Bisection. Returns (sigma, None) or (None, reason); never a clamped bound."""
    DF = math.exp(-r * T)  # noqa: N806
    intrinsic = DF * (max(F - K, 0.0) if is_call else max(K - F, 0.0))
    upper = DF * (F if is_call else K)
    if price <= intrinsic:
        return None, "below_intrinsic"
    if price >= upper:
        return None, "above_max_price"

    def f(s: float) -> float:
        return black76(F, K, T, s, r, is_call, S)["price"] - price

    a, b = lo, hi
    if f(a) > 0:
        return None, "below_min_vol"
    if f(b) < 0:
        return None, "above_max_vol"
    for _ in range(maxit):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) < 1e-12:
            return m, None
        a, b = (a, m) if fm > 0 else (m, b)
    return 0.5 * (a + b), None
