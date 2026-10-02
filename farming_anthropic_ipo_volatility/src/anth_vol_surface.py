"""
Strike ANTH vol surface from the 16 Sep 2026 snapshot.

Each listed expiry gives ATM IV, 25-delta risk reversal and 25-delta
butterfly, in vol points. That is three numbers — enough to pin a
quadratic in forward delta, map those deltas to strikes, and interpolate
total variance across the calendar.

Nothing here invents 10-delta wings or a fifth SVI parameter. Extrapolation
beyond 25d is flagged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from anth_ipo_book import (
    DAYS_PER_YEAR,
    SPOT,
    SURFACE_2026_09_16,
    Expiry,
    _norm_cdf,
    _norm_pdf,
    bs_price,
    years,
)


def norm_inv(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    x = 0.0
    for _ in range(30):
        pdf = _norm_pdf(x)
        if pdf < 1e-18:
            break
        x -= (_norm_cdf(x) - p) / pdf
    return x


def wings_from_rr_fly(atm: float, rr_pts: float, fly_pts: float) -> tuple[float, float]:
    """
    RR  = IV_25c − IV_25p     (vol points)
    fly = 0.5 (IV_25c + IV_25p) − ATM
    ⇒  IV_25c = ATM + fly + 0.5 RR
        IV_25p = ATM + fly − 0.5 RR
    """
    rr = rr_pts / 100.0
    fly = fly_pts / 100.0
    return atm + fly + 0.5 * rr, atm + fly - 0.5 * rr


def strike_from_call_delta(spot: float, iv: float, tau: float, call_delta: float) -> float:
    """K such that N(d1) = call_delta, r = 0, F = S."""
    if not 0.0 < call_delta < 1.0:
        raise ValueError("call_delta must be in (0, 1)")
    if tau <= 0 or iv <= 0 or spot <= 0:
        raise ValueError("spot, iv, tau must be positive")
    d1 = norm_inv(call_delta)
    # d1 = (ln(S/K) + 0.5 σ² τ) / (σ √τ)
    return spot * math.exp(-d1 * iv * math.sqrt(tau) + 0.5 * iv * iv * tau)


@dataclass(frozen=True)
class SliceQuote:
    expiry: Expiry
    iv_25c: float
    iv_25p: float
    k_25c: float
    k_25p: float
    k_atm: float

    @property
    def days(self) -> float:
        return self.expiry.days

    @property
    def tau(self) -> float:
        return years(self.expiry.days)

    @property
    def atm(self) -> float:
        return self.expiry.atm_iv


def build_slice(expiry: Expiry, spot: float = SPOT) -> SliceQuote:
    iv_c, iv_p = wings_from_rr_fly(expiry.atm_iv, expiry.rr_25d, expiry.fly_25d)
    if iv_c <= 0 or iv_p <= 0:
        raise ValueError(f"non-positive wing IV on {expiry.label}")
    tau = years(expiry.days)
    if tau == 0:
        raise ValueError("cannot build a slice at expiry")
    return SliceQuote(
        expiry=expiry,
        iv_25c=iv_c,
        iv_25p=iv_p,
        k_25c=strike_from_call_delta(spot, iv_c, tau, 0.25),
        k_25p=strike_from_call_delta(spot, iv_p, tau, 0.75),
        k_atm=spot,
    )


def build_surface(spot: float = SPOT) -> tuple[SliceQuote, ...]:
    return tuple(build_slice(e, spot) for e in SURFACE_2026_09_16)


def _smile_coeff(sl: SliceQuote) -> tuple[float, float]:
    """IV(k) = ATM + α k + β k² with k = ln(K/S). Pins ATM and both 25Δ strikes."""
    kc = math.log(sl.k_25c / sl.k_atm)
    kp = math.log(sl.k_25p / sl.k_atm)
    yc = sl.iv_25c - sl.atm
    yp = sl.iv_25p - sl.atm
    if abs(kc) < 1e-16 or abs(kp) < 1e-16 or abs(kc - kp) < 1e-16:
        return 0.0, 0.0
    beta = (yc / kc - yp / kp) / (kc - kp)
    alpha = yc / kc - beta * kc
    return alpha, beta


def iv_from_call_delta(sl: SliceQuote, call_delta: float) -> float:
    """
    Quadratic in call delta through (0.25, IV_25c), (0.50, ATM), (0.75, IV_25p).

    IV(Δ) = ATM + a (Δ − 0.5) + b (Δ − 0.5)²
    a = IV_25p − IV_25c   (= −RR)
    b = 16 × fly
    """
    x = call_delta - 0.5
    a = 2.0 * (sl.iv_25p - sl.iv_25c)  # −2 × RR
    fly = 0.5 * (sl.iv_25c + sl.iv_25p) - sl.atm
    b = 16.0 * fly
    return sl.atm + a * x + b * x * x


def call_delta_from_strike(spot: float, strike: float, iv: float, tau: float) -> float:
    if tau <= 0 or iv <= 0:
        return 1.0 if strike < spot else (0.0 if strike > spot else 0.5)
    d1 = (math.log(spot / strike) + 0.5 * iv * iv * tau) / (iv * math.sqrt(tau))
    return _norm_cdf(d1)


def iv_from_strike(sl: SliceQuote, strike: float, spot: float = SPOT, iters: int = 8) -> float:
    """Log-moneyness quadratic. ATM is exact at K = S. Wings are exact at the 25Δ strikes."""
    del iters
    if strike <= 0:
        raise ValueError("strike must be positive")
    alpha, beta = _smile_coeff(sl)
    k = math.log(strike / sl.k_atm)
    return max(sl.atm + alpha * k + beta * k * k, 0.01)


def in_quoted_band(sl: SliceQuote, strike: float) -> bool:
    lo, hi = (sl.k_25p, sl.k_25c) if sl.k_25p < sl.k_25c else (sl.k_25c, sl.k_25p)
    return lo <= strike <= hi


def slice_moneyness_band(sl: SliceQuote, spot: float = SPOT) -> tuple[float, float]:
    """25Δ put/call moneyness on one listed slice. Put is the low strike."""
    lo, hi = (sl.k_25p, sl.k_25c) if sl.k_25p < sl.k_25c else (sl.k_25c, sl.k_25p)
    return lo / spot, hi / spot


def _bracketing_slices(
    slices: tuple[SliceQuote, ...], days: float
) -> tuple[SliceQuote, SliceQuote, float]:
    """Return (left, right, t) with t in [0, 1] for calendar interpolation."""
    ordered = tuple(sorted(slices, key=lambda s: s.days))
    if days <= ordered[0].days:
        return ordered[0], ordered[0], 0.0
    if days >= ordered[-1].days:
        return ordered[-1], ordered[-1], 0.0
    for i in range(len(ordered) - 1):
        if ordered[i].days <= days <= ordered[i + 1].days:
            left, right = ordered[i], ordered[i + 1]
            if right.days == left.days:
                return left, right, 0.0
            t = (days - left.days) / (right.days - left.days)
            return left, right, t
    return ordered[-1], ordered[-1], 0.0


def quoted_moneyness_band(
    slices: tuple[SliceQuote, ...],
    days: float,
    spot: float = SPOT,
) -> tuple[float, float]:
    """
    25Δ moneyness band at `days`. Linear in calendar between the
    bracketing listed slices. A 2-day slice is ~0.97–1.03; Nov 21 is ~0.85–1.32.
    """
    if days <= 0:
        raise ValueError("days must be positive")
    left, right, t = _bracketing_slices(slices, days)
    lo_l, hi_l = slice_moneyness_band(left, spot)
    lo_r, hi_r = slice_moneyness_band(right, spot)
    return (1.0 - t) * lo_l + t * lo_r, (1.0 - t) * hi_l + t * hi_r


def total_var_from_strike(sl: SliceQuote, strike: float, spot: float = SPOT) -> float:
    return iv_from_strike(sl, strike, spot) ** 2 * sl.tau


def interpolate_iv(
    slices: tuple[SliceQuote, ...],
    days: float,
    strike: float,
    spot: float = SPOT,
) -> float:
    """
    Linear in total variance between the bracketing listed expiries,
    at fixed strike. Outside the calendar, clamp to the nearest slice.
    """
    if days <= 0:
        raise ValueError("days must be positive")
    ordered = tuple(sorted(slices, key=lambda s: s.days))
    if days <= ordered[0].days:
        return iv_from_strike(ordered[0], strike, spot)
    if days >= ordered[-1].days:
        return iv_from_strike(ordered[-1], strike, spot)
    left = right = ordered[0]
    for i in range(len(ordered) - 1):
        if ordered[i].days <= days <= ordered[i + 1].days:
            left, right = ordered[i], ordered[i + 1]
            break
    if right.days == left.days:
        return iv_from_strike(left, strike, spot)
    w_l = total_var_from_strike(left, strike, spot)
    w_r = total_var_from_strike(right, strike, spot)
    t = (days - left.days) / (right.days - left.days)
    w = (1.0 - t) * w_l + t * w_r
    tau = years(days)
    return math.sqrt(w / tau)


def _pchip_end_slope(h0: float, h1: float, delta0: float, delta1: float) -> float:
    """Fritsch–Carlson endpoint slope."""
    d = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1)
    if d * delta0 <= 0.0:
        return 0.0
    if delta0 * delta1 <= 0.0 and abs(d) > 3.0 * abs(delta0):
        return 3.0 * delta0
    return d


def _pchip_slopes(xs: list[float], ys: list[float]) -> list[float]:
    n = len(xs)
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    delta = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    d = [0.0] * n
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            d[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    if n == 2:
        d[0] = delta[0]
        d[1] = delta[0]
        return d
    d[0] = _pchip_end_slope(h[0], h[1], delta[0], delta[1])
    d[-1] = _pchip_end_slope(h[-1], h[-2], delta[-1], delta[-2])
    return d


def pchip_eval(xs: list[float], ys: list[float], x: float) -> float:
    """Shape-preserving cubic Hermite. Hits every knot. No IV spline, no blur."""
    if len(xs) < 2:
        raise ValueError("need at least two knots")
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    d = _pchip_slopes(xs, ys)
    i = 0
    for k in range(len(xs) - 1):
        if xs[k] <= x <= xs[k + 1]:
            i = k
            break
    h = xs[i + 1] - xs[i]
    t = (x - xs[i]) / h
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * ys[i] + h10 * h * d[i] + h01 * ys[i + 1] + h11 * h * d[i + 1]


def total_var_from_delta(sl: SliceQuote, call_delta: float) -> float:
    return iv_from_call_delta(sl, call_delta) ** 2 * sl.tau


def _ordered_slices(slices: tuple[SliceQuote, ...]) -> tuple[SliceQuote, ...]:
    return tuple(sorted(slices, key=lambda s: s.days))


def interpolate_iv_pchip(
    slices: tuple[SliceQuote, ...],
    days: float,
    strike: float,
    spot: float = SPOT,
) -> float:
    """
    PCHIP in total variance at fixed strike. Same listed ATMs as interpolate_iv.
    No Gaussian blur, no spline in IV.
    """
    if days <= 0:
        raise ValueError("days must be positive")
    ordered = _ordered_slices(slices)
    if days <= ordered[0].days:
        return iv_from_strike(ordered[0], strike, spot)
    if days >= ordered[-1].days:
        return iv_from_strike(ordered[-1], strike, spot)
    xs = [sl.days for sl in ordered]
    ys = [total_var_from_strike(sl, strike, spot) for sl in ordered]
    w = pchip_eval(xs, ys, days)
    return math.sqrt(max(w, 1e-16) / years(days))


def interpolate_iv_pchip_delta(
    slices: tuple[SliceQuote, ...],
    days: float,
    call_delta: float,
) -> float:
    """
    PCHIP in total variance at fixed call delta. Domain [0.25, 0.75] is a
    rectangle on every expiry — that is the smooth sheet, not a K/S scarf.
    """
    if days <= 0:
        raise ValueError("days must be positive")
    if not 0.0 < call_delta < 1.0:
        raise ValueError("call_delta must be in (0, 1)")
    ordered = _ordered_slices(slices)
    if days <= ordered[0].days:
        return iv_from_call_delta(ordered[0], call_delta)
    if days >= ordered[-1].days:
        return iv_from_call_delta(ordered[-1], call_delta)
    xs = [sl.days for sl in ordered]
    ys = [total_var_from_delta(sl, call_delta) for sl in ordered]
    w = pchip_eval(xs, ys, days)
    return math.sqrt(max(w, 1e-16) / years(days))


def calendar_violations(
    slices: tuple[SliceQuote, ...],
    moneyness: tuple[float, ...] = (0.85, 0.92, 1.0, 1.08, 1.15),
    spot: float = SPOT,
    tol: float = 1e-8,
) -> list[tuple[str, str, float, float]]:
    """
    Pairs of adjacent slices where total variance *falls* at a fixed
    strike. That is calendar arbitrage on the reconstructed surface.
    """
    ordered = tuple(sorted(slices, key=lambda s: s.days))
    hits: list[tuple[str, str, float, float]] = []
    for m in moneyness:
        k = spot * m
        prev_w = None
        prev_lab = ""
        for sl in ordered:
            w = total_var_from_strike(sl, k, spot)
            if prev_w is not None and w + tol < prev_w:
                hits.append((prev_lab, sl.expiry.label, m, prev_w - w))
            prev_w, prev_lab = w, sl.expiry.label
    return hits


def call_price_on_slice(sl: SliceQuote, strike: float, spot: float = SPOT) -> float:
    return bs_price(spot, strike, iv_from_strike(sl, strike, spot), sl.tau, "call")


def digital_and_density(
    sl: SliceQuote,
    strike: float,
    spot: float = SPOT,
    bump: float = 5.0,
) -> tuple[float, float]:
    """
    Breeden–Litzenberger on one slice.
    digital ≈ −dC/dK
    pdf     ≈ d²C/dK²
    """
    c_up = call_price_on_slice(sl, strike + bump, spot)
    c_mid = call_price_on_slice(sl, strike, spot)
    c_dn = call_price_on_slice(sl, strike - bump, spot)
    digital = -(c_up - c_dn) / (2.0 * bump)
    pdf = (c_up - 2.0 * c_mid + c_dn) / (bump * bump)
    return digital, pdf


def surface_grid(
    slices: tuple[SliceQuote, ...],
    days: list[float],
    moneyness: list[float],
    spot: float = SPOT,
) -> list[list[float]]:
    """rows = days, cols = moneyness, values = IV."""
    grid = []
    for d in days:
        grid.append([interpolate_iv(slices, d, spot * m, spot) for m in moneyness])
    return grid


def surface_grid_delta_pchip(
    slices: tuple[SliceQuote, ...],
    days: list[float],
    deltas: list[float],
) -> list[list[float]]:
    """rows = days, cols = call delta, values = IV. PCHIP in w(T)."""
    return [[interpolate_iv_pchip_delta(slices, d, delta) for delta in deltas] for d in days]
