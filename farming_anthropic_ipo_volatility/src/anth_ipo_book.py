"""
Selling the wait, buying the listing.

Kernel for the two-phase Anthropic IPO book across Hyperliquid io:ANTH
and Strike cash-settled ANTH options. Numbers and identities follow the
White Star Capital Liquid Fund note dated 16 September 2026.

Spot and strikes are HIP-3 price units (1 unit = $1B implied cap).
A mark of 2170 is $2.17T. Option multiplier is 1 USD per price unit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

DAYS_PER_YEAR = 365.0
SQRT_2PI = math.sqrt(2.0 * math.pi)

# Snapshot as of 16 Sep 2026
SPOT = 2170.0
PERP_NOTIONAL = 1_000_000.0
PHASE1_IV = 0.65
PHASE1_PUTS_PER_UNIT = 2.0
FUNDING_ANNUAL = 0.24
PERP_FEE_BPS = 4.5
STRIKE_TAKER_BPS = 40.0
STRIKE_MAKER_BPS = 10.0
STRIKE_SETTLE_BPS = 50.0  # of intrinsic


def years(days: float) -> float:
    if days < 0:
        raise ValueError("days must be non-negative")
    return days / DAYS_PER_YEAR


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def d1_d2(spot: float, strike: float, vol: float, tau: float) -> tuple[float, float]:
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if vol < 0 or tau < 0:
        raise ValueError("vol and tau must be non-negative")
    if tau == 0.0 or vol == 0.0:
        itm = 1.0 if spot > strike else (0.0 if spot < strike else 0.5)
        return (10.0 if itm == 1.0 else (-10.0 if itm == 0.0 else 0.0), 0.0)
    vs = vol * math.sqrt(tau)
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * tau) / vs
    return d1, d1 - vs


def bs_price(spot: float, strike: float, vol: float, tau: float, right: Literal["call", "put"]) -> float:
    if tau == 0.0:
        return max(spot - strike, 0.0) if right == "call" else max(strike - spot, 0.0)
    d1, d2 = d1_d2(spot, strike, vol, tau)
    if right == "call":
        return spot * _norm_cdf(d1) - strike * _norm_cdf(d2)
    return strike * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta(spot: float, strike: float, vol: float, tau: float, right: Literal["call", "put"]) -> float:
    if tau == 0.0:
        if right == "call":
            return 1.0 if spot > strike else (0.5 if spot == strike else 0.0)
        return -1.0 if spot < strike else (-0.5 if spot == strike else 0.0)
    d1, _ = d1_d2(spot, strike, vol, tau)
    return _norm_cdf(d1) if right == "call" else _norm_cdf(d1) - 1.0


def bs_vega(spot: float, strike: float, vol: float, tau: float) -> float:
    """dPrice / dσ, σ in decimal. One vol point = this / 100."""
    if tau == 0.0 or vol == 0.0:
        return 0.0
    d1, _ = d1_d2(spot, strike, vol, tau)
    return spot * _norm_pdf(d1) * math.sqrt(tau)


def bs_gamma(spot: float, strike: float, vol: float, tau: float) -> float:
    if tau == 0.0 or vol == 0.0 or spot <= 0:
        return 0.0
    d1, _ = d1_d2(spot, strike, vol, tau)
    return _norm_pdf(d1) / (spot * vol * math.sqrt(tau))


def bs_theta_calendar(spot: float, strike: float, vol: float, tau: float, right: Literal["call", "put"]) -> float:
    """dPrice / d(calendar day), r = 0. Negative for long options."""
    if tau == 0.0:
        return 0.0
    d1, d2 = d1_d2(spot, strike, vol, tau)
    time_term = -spot * _norm_pdf(d1) * vol / (2.0 * math.sqrt(tau))
    # r = 0 so the K r N(d2) term is 0
    return time_term / DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Surface + event extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Expiry:
    label: str
    days: float
    atm_iv: float
    rr_25d: float = 0.0
    fly_25d: float = 0.0


SURFACE_2026_09_16: tuple[Expiry, ...] = (
    Expiry("Sep 18", 1, 0.781, -1.1, 0.6),
    Expiry("Sep 20", 3, 0.661, -2.3, 0.7),
    Expiry("Sep 21", 4, 0.588, -2.4, 0.1),
    Expiry("Sep 23", 6, 0.658, -5.9, 0.0),
    Expiry("Sep 26", 9, 0.756, -3.0, 1.5),
    Expiry("Oct 17", 30, 0.733, 0.8, -2.5),
    Expiry("Oct 24", 37, 0.766, -0.1, -4.3),
    Expiry("Nov 7", 51, 0.783, 1.8, -2.9),
    Expiry("Nov 21", 65, 0.792, 0.3, -1.5),
)


def total_variance(iv: float, days: float) -> float:
    return iv * iv * years(days)


def event_move(iv: float, days: float, baseline: float) -> float:
    """Equivalent one-shot move: sqrt((σ_imp² − b²) T)."""
    extra = total_variance(iv, days) - total_variance(baseline, days)
    if extra < 0:
        return 0.0
    return math.sqrt(extra)


def forward_vol(iv_near: float, days_near: float, iv_far: float, days_far: float) -> float:
    if days_far <= days_near:
        raise ValueError("far expiry must be after near")
    dv = total_variance(iv_far, days_far) - total_variance(iv_near, days_near)
    dt = years(days_far - days_near)
    if dv < 0:
        return 0.0
    return math.sqrt(dv / dt)


def weekend_fwd_vol() -> float:
    """Sep 18 → Sep 20, the weekend strip on the snapshot."""
    return forward_vol(0.781, 1, 0.661, 3)


# ---------------------------------------------------------------------------
# Strike short-option margin
# ---------------------------------------------------------------------------

def strike_short_margin_unit(
    mark: float,
    spot: float,
    strike: float,
    shock: float,
    floor: float,
) -> float:
    """
    unit = mark + max(shock * spot − OTM, floor * strike)
    OTM = max(strike − spot, 0) for a short put; max(spot − strike, 0) for a short call.
    For ATM, OTM = 0.
    """
    otm = abs(spot - strike)
    return mark + max(shock * spot - otm, floor * strike)


def strike_short_put_margin(
    qty_short: float,
    spot: float,
    strike: float,
    vol: float,
    tau: float,
    shock: float,
    floor: float,
) -> float:
    mark = bs_price(spot, strike, vol, tau, "put")
    return abs(qty_short) * strike_short_margin_unit(mark, spot, strike, shock, floor)


# ---------------------------------------------------------------------------
# Phase one: short perp + 2× short ATM puts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseOneGreeks:
    n_perp: float
    n_puts: float
    put_price: float
    premium: float
    delta: float
    theta_per_day: float
    vega_per_vol_point: float
    dollar_gamma_per_pct2: float
    breakeven_daily: float
    breakeven_hourly: float
    funding_per_day: float
    margin_entry: float
    margin_maint: float


def phase_one_qty(notional: float = PERP_NOTIONAL, spot: float = SPOT) -> tuple[float, float]:
    """Short perp qty in price units, short put qty. Puts = 2 per unit of |perp|."""
    n_perp = -notional / spot
    n_puts = -PHASE1_PUTS_PER_UNIT * abs(n_perp)
    return n_perp, n_puts


def phase_one_greeks(
    spot: float = SPOT,
    notional: float = PERP_NOTIONAL,
    iv: float = PHASE1_IV,
    days: float = 7.0,
    funding_annual: float = FUNDING_ANNUAL,
) -> PhaseOneGreeks:
    tau = years(days)
    n_perp, n_puts = phase_one_qty(notional, spot)
    put = bs_price(spot, spot, iv, tau, "put")
    premium = -n_puts * put  # n_puts < 0, premium collected > 0
    # Dollar delta: perp PnL = n_perp * dS; option = n_puts * delta_bs * dS
    delta = n_perp + n_puts * bs_delta(spot, spot, iv, tau, "put")
    theta = n_puts * bs_theta_calendar(spot, spot, iv, tau, "put")
    vega_pt = n_puts * bs_vega(spot, spot, iv, tau) / 100.0
    # Dollar gamma per 1% move squared:
    # d²Pnl = 0.5 * (n_puts * bs_gamma) * (0.01 S)² * 2 / (0.01 S)² wait.
    # PnL ≈ 0.5 * Γ_qty * (ΔS)² with Γ_qty = n_puts * bs_gamma
    # ΔS = 0.01 S for a 1% move → 0.5 * Γ_qty * (0.01 S)²
    # Paper quotes "dollar gamma per 1% move²" as the coefficient on m²
    # where m is the move in percent: PnL_gamma = dollar_gamma * m²
    # with m=1 → 0.5 * n_puts * γ * (0.01 S)²
    gamma_qty = n_puts * bs_gamma(spot, spot, iv, tau)
    dollar_gamma = 0.5 * gamma_qty * (0.01 * spot) ** 2
    # Daily BE: theta + dollar_gamma * m² = 0 → m = sqrt(-theta / dollar_gamma)
    if dollar_gamma >= 0 or theta <= 0:
        be_daily = float("inf")
    else:
        be_daily = math.sqrt(-theta / dollar_gamma)
    be_hourly = be_daily / math.sqrt(24.0)
    funding = -n_perp * spot * funding_annual / DAYS_PER_YEAR  # short perp, longs pay → we receive
    # n_perp is negative; -n_perp * spot = notional
    entry = strike_short_put_margin(-n_puts, spot, spot, iv, tau, shock=0.20, floor=0.10)
    maint = strike_short_put_margin(-n_puts, spot, spot, iv, tau, shock=0.12, floor=0.10)
    return PhaseOneGreeks(
        n_perp=n_perp,
        n_puts=n_puts,
        put_price=put,
        premium=premium,
        delta=delta,
        theta_per_day=theta,
        vega_per_vol_point=vega_pt,
        dollar_gamma_per_pct2=dollar_gamma,
        breakeven_daily=be_daily,
        breakeven_hourly=be_hourly,
        funding_per_day=funding,
        margin_entry=entry,
        margin_maint=maint,
    )


def variance_premium(iv: float, realized: float) -> float:
    """σ_imp² − σ_real², the gap the short-vol book is paid."""
    return iv * iv - realized * realized


# ---------------------------------------------------------------------------
# Phase two: short perp + long Nov 21 ATM call + short Oct 17 ATM put
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseTwoLegs:
    n_perp: float
    n_call_far: float
    n_put_near: float
    fwd_vol: float
    delta: float
    vega_per_vol_point_far: float
    theta_per_day_far: float


def phase_two_book(
    spot: float = SPOT,
    notional: float = PERP_NOTIONAL,
    near: Expiry = SURFACE_2026_09_16[5],  # Oct 17
    far: Expiry = SURFACE_2026_09_16[8],  # Nov 21
) -> PhaseTwoLegs:
    n_perp = -notional / spot
    # one ATM call far and one ATM put near per unit of perp
    unit = notional / spot
    n_call = unit
    n_put = -unit
    d_call = bs_delta(spot, spot, far.atm_iv, years(far.days), "call")
    d_put = bs_delta(spot, spot, near.atm_iv, years(near.days), "put")
    delta = n_perp + n_call * d_call + n_put * d_put
    fwd = forward_vol(near.atm_iv, near.days, far.atm_iv, far.days)
    vega_far = n_call * bs_vega(spot, spot, far.atm_iv, years(far.days)) / 100.0
    theta_far = n_call * bs_theta_calendar(spot, spot, far.atm_iv, years(far.days), "call")
    return PhaseTwoLegs(n_perp, n_call, n_put, fwd, delta, vega_far, theta_far)


def apply_early_listing_rule(
    listing_days: float,
    near_days: float = 30.0,
) -> Literal["swap_near_to_far", "hold", "exit_after_far"]:
    """
    If listing is set before Oct 17, buy back the short near puts and add far calls.
    If listing is confirmed after Nov 21, exit phase two.
    """
    if listing_days < near_days:
        return "swap_near_to_far"
    if listing_days > 65.0:
        return "exit_after_far"
    return "hold"


# ---------------------------------------------------------------------------
# Venue isolation / Hyperliquid margin
# ---------------------------------------------------------------------------

def hyperliquid_margin(notional: float, leverage: float) -> float:
    if leverage <= 0:
        raise ValueError("leverage must be positive")
    return notional / leverage


def approx_liquidation_rally(leverage: float, mm_frac_of_im: float = 0.5) -> float:
    """
    Isolated short. Initial margin = 1/L. Maintenance = mm_frac * IM.
    Short is liquidated when the rally eats IM − MM:
    rally ≈ (IM − MM) / 1 = (1 − mm_frac) / L
    Paper: 6x → +8%, 3x → +23%, 2x → +38%, 1.5x → +54%.
    Those fit mm_frac ≈ 0.5 with a small buffer: (0.5)/L is 8.3%, 16.7%, 25%, 33%
    which does NOT match the paper.

    Re-read: "Assumes maintenance margin of half the initial margin at max leverage."
    6x IM = 16.7%, half = 8.3% MM, loss to MM on a short = IM − MM? 
    Actually isolated liquidation when equity = MM.
    Equity start = IM. Loss = rally * notional. Equity = IM - rally.
    Liquidated when IM - rally = MM → rally = IM - MM.
    If MM = 0.5 * IM_at_max (max=6x, IM_max=1/6, MM=1/12=8.33%) regardless of chosen L:
    At 6x: rally = 16.7% - 8.3% = 8.3%
    At 3x: IM=33.3%, rally = 33.3% - 8.3% = 25%  (paper says 23%)
    At 2x: 50% - 8.3% = 41.7% (paper 38%)
    At 1.5x: 66.7% - 8.3% = 58.3% (paper 54%)

    Close enough that we treat the paper table as the fixture and expose both.
    """
    im = 1.0 / leverage
    mm = 0.5 * (1.0 / 6.0)
    return im - mm


PAPER_LIQ_RALLY = {6.0: 0.08, 3.0: 0.23, 2.0: 0.38, 1.5: 0.54}


def phase_one_capital(notional: float = PERP_NOTIONAL, leverage: float = 2.0) -> float:
    """HL margin + Strike entry margin. Paper: ~$970k per $1M at 2x."""
    g = phase_one_greeks(notional=notional)
    return hyperliquid_margin(notional, leverage) + g.margin_entry


# ---------------------------------------------------------------------------
# Expected-return fixtures from the note (section 7)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioReturn:
    name: str
    listing: str
    total: float
    p5: float
    p95: float
    options: float
    hedge: float
    funding: float
    fees: float


PAPER_SCENARIOS: tuple[ScenarioReturn, ...] = (
    ScenarioReturn("in_window", "~Nov 9", 0.167, 0.11, 0.21, -0.164, 0.315, 0.036, -0.019),
    ScenarioReturn("slip", "past Nov 21", 0.004, -0.07, 0.06, 0.014, -0.040, 0.046, -0.016),
    ScenarioReturn("early", "~Oct 12, rule on", 0.044, 0.00, 0.07, -0.064, 0.096, 0.033, -0.020),
)

SCENARIO_WEIGHTS = {"in_window": 0.60, "slip": 0.30, "early": 0.10}


def weighted_expected_return(funding_on: bool = True) -> float:
    """
    Paper: 60/30/10 → +10.6% at 24% funding, +6.8% at 0% funding.
    When funding is off, subtract each scenario's funding component.
    """
    total = 0.0
    for s in PAPER_SCENARIOS:
        r = s.total if funding_on else s.total - s.funding
        total += SCENARIO_WEIGHTS[s.name] * r
    return total


def funding_over_days(days: float, annual: float = FUNDING_ANNUAL, notional: float = PERP_NOTIONAL) -> float:
    return notional * annual * days / DAYS_PER_YEAR


# SPCX realized event-move fixtures (50% baseline)
SPCX_EVENT = {
    "48h_listing": (1.88, 0.133),
    "jun1_15": (1.05, 0.181),
    "jun1_22": (1.30, 0.281),
    "may18_jun22": (1.11, 0.304),
}
