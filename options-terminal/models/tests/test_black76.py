"""Guide §3.4–3.5 reference: parity, greeks vs finite differences, limits, IV solver."""

import math

import pytest

from terminal_models.core.quant.black76 import black76, implied_vol

F, S = 2486.10, 2473.59


@pytest.mark.parametrize("r", [0.0, 0.05])
@pytest.mark.parametrize("K", [1500.0, 2400.0, 2660.0, 4000.0])
@pytest.mark.parametrize("T", [1 / 365, 0.25, 2.0])
def test_put_call_parity(K, T, r):
    call = black76(F, K, T, 0.55, r, True, S)["price"]
    put = black76(F, K, T, 0.55, r, False, S)["price"]
    assert call - put == pytest.approx(math.exp(-r * T) * (F - K), abs=1e-9 * F)


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("K", [2000.0, 2660.0, 3200.0])
def test_greeks_match_finite_differences(K, is_call):
    T, sigma, r = 0.1, 0.55, 0.05
    g = black76(F, K, T, sigma, r, is_call, S)
    base = {"F": F, "K": K, "T": T, "sigma": sigma, "r": r, "is_call": is_call, "S": S}

    def price(**kw):
        return black76(**(base | kw))["price"]

    h = 1e-4
    assert g["delta"] == pytest.approx((price(F=F + h) - price(F=F - h)) / (2 * h), rel=1e-6)
    assert g["vega_1pt"] == pytest.approx(
        (price(sigma=sigma + h) - price(sigma=sigma - h)) / (2 * h) / 100, rel=1e-6
    )
    # Spot moves move the forward proportionally (basis held): dV/dS = delta * F/S.
    hs = 1e-3

    def at_spot(s):
        return black76(F * s / S, K, T, sigma, r, is_call, s)["price"]

    assert g["delta_spot"] == pytest.approx(
        (at_spot(S + hs) - at_spot(S - hs)) / (2 * hs), rel=1e-6
    )

    def d_spot(s):
        return black76(F * s / S, K, T, sigma, r, is_call, s)["delta_spot"]

    gamma_fd = (d_spot(S + hs) - d_spot(S - hs)) / (2 * hs)
    assert g["gamma_1pct"] == pytest.approx(gamma_fd * S * 0.01, rel=1e-5)
    # theta: one calendar day of time decay with the forward fixed.
    dt = 1e-5
    theta_fd = -(price(T=T + dt) - price(T=T - dt)) / (2 * dt) / 365
    assert g["theta_1d"] == pytest.approx(theta_fd, rel=1e-5)


def test_limits_return_discounted_intrinsic():
    assert black76(F, 2000.0, 0.0, 0.5, 0.05, True, S) == {"price": F - 2000.0}
    assert black76(F, 2000.0, 1.0, 0.0, 0.05, False, S) == {"price": 0.0}
    short = black76(F, 2000.0, 1e-9, 0.5, 0.0, True, S)["price"]
    assert short == pytest.approx(F - 2000.0, abs=1e-6)


def test_non_positive_inputs_raise():
    with pytest.raises(ValueError, match="non_positive_input"):
        black76(0.0, 100.0, 1.0, 0.5, 0.0, True, 1.0)


def test_guide_example_spot_vs_forward():
    """Guide §1: 29d ETH 2660C at 55%, 8% basis: $82.45 on spot vs $88.05 on the forward."""
    T = 29 / 365
    fwd = S * math.exp(0.08 * T)
    assert round(black76(S, 2660.0, T, 0.55, 0.0, True, S)["price"], 2) == 82.45
    assert round(black76(fwd, 2660.0, T, 0.55, 0.0, True, S)["price"], 2) == 88.05


@pytest.mark.parametrize("is_call", [True, False])
# Strikes within ~3 sd: far wings have near-zero vega, where the 1e-10 price tolerance
# admits a range of vols (the vectors record that behavior; this test checks recovery).
@pytest.mark.parametrize("K", [2200.0, 2486.1, 2800.0])
@pytest.mark.parametrize("sigma", [0.2, 0.55, 1.5])
def test_implied_vol_round_trip(K, sigma, is_call):
    T = 30 / 365
    price = black76(F, K, T, sigma, 0.0, is_call, S)["price"]
    iv, reason = implied_vol(F, K, T, price, 0.0, is_call, S)
    assert reason is None
    assert iv == pytest.approx(sigma, abs=1e-8)


@pytest.mark.parametrize(
    "K, is_call, price, reason",
    [
        (2000.0, True, 480.0, "below_intrinsic"),
        (2000.0, True, F, "above_max_price"),
        (3000.0, False, 3000.0, "above_max_price"),
        (2486.1, True, black76(F, 2486.1, 30 / 365, 0.005, 0.0, True, S)["price"], "below_min_vol"),
        (2486.1, True, black76(F, 2486.1, 30 / 365, 6.0, 0.0, True, S)["price"], "above_max_vol"),
    ],
)
def test_implied_vol_reasons_never_clamp(K, is_call, price, reason):
    assert implied_vol(F, K, 30 / 365, price, 0.0, is_call, S) == (None, reason)
