"""Surface interpolation (§3.2–3.3) and context metrics (§3.6) against closed-form answers.
Inputs are synthetic test geometry, not market data."""

import math
from datetime import UTC, datetime, timedelta

import pytest

from terminal_models.core.quant.metrics import (
    atm_iv,
    change_24h,
    realized_vol,
    skew_at_tenor,
    strike_for_delta,
)
from terminal_models.core.quant.smile import (
    EXACT_TENOR_YEARS,
    STRIKE_EXTRAPOLATED,
    TENOR_LONG,
    TENOR_SHORT,
    Expiry,
    Node,
    Surface,
    k_grid,
)

T0 = datetime(2026, 10, 13, 17, tzinfo=UTC)
S = 100.0


def node(k, iv, bid=None, ask=None, quality="good"):
    return Node(k=k, strike=S * math.exp(k), iv=iv, iv_bid=bid, iv_ask=ask, quality=quality)


def expiry(days, forward, nodes):
    return Expiry(T0 + timedelta(days=days), days / 365, forward, nodes)


def flat(days, forward, vol):
    return expiry(days, forward, [node(k, vol, vol - 0.01, vol + 0.01) for k in (-0.3, 0.0, 0.3)])


def test_smile_interpolates_total_variance_and_flags_wings():
    e = expiry(30, 101.0, [node(-0.2, 0.6), node(0.0, 0.5), node(0.2, 0.55)])
    mid = e.sigma(-0.1)
    assert mid.sigma == pytest.approx(math.sqrt(0.5 * 0.6**2 + 0.5 * 0.5**2))
    assert mid.flags == ()
    assert e.sigma(0.0).sigma == 0.5
    wing = e.sigma(0.5)
    assert (wing.sigma, wing.flags) == (0.55, (STRIKE_EXTRAPOLATED,))
    assert e.display_sigma(0.5) is None and e.display_sigma(0.2) == 0.55


def test_bid_ask_need_every_bracketing_node():
    e = expiry(
        30, 101.0, [node(-0.2, 0.6, 0.58, 0.62), node(0.0, 0.5, None, 0.52), node(0.2, 0.55)]
    )
    assert e.sigma(-0.1, "iv_bid") is None
    assert e.sigma(-0.2, "iv_bid").sigma == 0.58
    assert e.sigma(-0.1, "iv_ask").sigma == pytest.approx(math.sqrt(0.5 * 0.62**2 + 0.5 * 0.52**2))


def test_surface_across_expiries():
    surf = Surface(S, [flat(7, 100.5, 0.6), flat(30, 102.0, 0.5)])
    exact = surf.sigma(0.0, 7 / 365 + 0.5 * EXACT_TENOR_YEARS)
    assert (exact.sigma, exact.interpolation) == (0.6, "exact")
    t = 14 / 365
    w = 0.6**2 * 7 / 365 + (0.5**2 * 30 / 365 - 0.6**2 * 7 / 365) * (t - 7 / 365) / (23 / 365)
    mid = surf.sigma(0.0, t)
    assert mid.sigma == pytest.approx(math.sqrt(w / t))
    assert mid.interpolation == "interpolated" and len(mid.expiries) == 2
    short = surf.sigma(0.0, 1 / 365)
    assert (short.sigma, short.flags, short.interpolation) == (0.6, (TENOR_SHORT,), "extrapolated")
    long_ = surf.sigma(0.4, 60 / 365)
    assert long_.flags == (STRIKE_EXTRAPOLATED, TENOR_LONG)


def test_forward_curve_is_linear_in_log_forward():
    surf = Surface(S, [flat(10, 101.0, 0.5), flat(20, 103.0, 0.5)])
    f5, flags = surf.forward(5 / 365)
    assert f5 == pytest.approx(math.exp(0.5 * (math.log(100) + math.log(101))))
    assert flags == ()
    f15, _ = surf.forward(15 / 365)
    assert f15 == pytest.approx(math.sqrt(101 * 103))
    f30, flags = surf.forward(30 / 365)
    assert f30 == pytest.approx(103 * 103 / 101)
    assert flags == (TENOR_LONG,)


def test_k_grid_is_exact_decimals():
    grid = k_grid(-0.5, 0.5, 0.025)
    assert len(grid) == 41 and grid[0] == -0.5 and grid[20] == 0.0 and grid[1] == -0.475


def test_realized_vol_on_alternating_path():
    a = 0.01
    closes = [100.0]
    for i in range(30 * 24):
        closes.append(closes[-1] * math.exp(a if i % 2 else -a))
    assert realized_vol(closes, 30) == pytest.approx(a * math.sqrt(8760))
    assert realized_vol(closes, 31) is None  # not enough history
    assert change_24h(closes) == pytest.approx(closes[-1] / closes[-25] - 1)


def test_25_delta_strike_on_flat_smile():
    e = flat(30, 100.0, 0.5)
    sd = 0.5 * math.sqrt(30 / 365)
    z = 0.6744897501960817  # N^-1(0.75)
    assert strike_for_delta(e, 0.25, 0.0, True) == pytest.approx(z * sd + 0.5 * sd * sd, abs=1e-9)
    assert strike_for_delta(e, 0.25, 0.0, False) == pytest.approx(-z * sd + 0.5 * sd * sd, abs=1e-9)


def test_rr_bf_signs_and_flat_case():
    flat_surface = Surface(S, [flat(30, 100.0, 0.5)])
    rr, bf = skew_at_tenor(flat_surface, 30, 0.0)
    assert rr.value == pytest.approx(0.0, abs=1e-12) and bf.value == pytest.approx(0.0, abs=1e-12)
    # Put wing richer than call wing → negative risk reversal; both wings above ATM → positive fly.
    skewed = Surface(S, [expiry(30, 100.0, [node(-0.3, 0.7), node(0.0, 0.5), node(0.3, 0.6)])])
    rr, bf = skew_at_tenor(skewed, 30, 0.0)
    assert rr.value < 0 < bf.value
    rr, _ = skew_at_tenor(skewed, 60, 0.0)
    assert rr.flags == (TENOR_LONG,)


def test_atm_iv_constant_maturity():
    surf = Surface(S, [flat(7, 100.0, 0.6), flat(30, 100.0, 0.5)])
    assert atm_iv(surf, 30).value == 0.5
    one_day = atm_iv(surf, 1)
    assert (one_day.value, one_day.flags) == (0.6, (TENOR_SHORT,))
