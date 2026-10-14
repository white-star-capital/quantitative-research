"""PARITY_MONITOR_IMPLEMENTATION_GUIDE.md §3: formulas, guards, fees, carry helpers, and the
payoff simulation that proves each conversion/reversal/inventory lock is exact (§3.2, §3.5)."""

import math
import random

import pytest

from terminal_models.core.quant import parity as P

RATES = P.FeeRates(  # TEST-ONLY numbers that exercise the arithmetic; not a fee schedule
    option=0.0003, option_cap=0.125, future=0.0005, perp=0.0005,
    option_delivery=0.00015, future_delivery=0.00025,
)  # fmt: skip


def call(S, K):  # noqa: N803
    return max(S - K, 0.0)


def put(S, K):  # noqa: N803
    return max(K - S, 0.0)


def draws(n=2000, seed=7):
    rng = random.Random(seed)
    for _ in range(n):
        F0 = rng.uniform(5, 150_000)  # noqa: N806
        K = F0 * math.exp(rng.uniform(-0.6, 0.6))  # noqa: N806
        S_T = F0 * math.exp(rng.uniform(-1.5, 1.5))  # noqa: N806
        yield rng, F0, K, S_T


def test_linear_locks_are_exact():
    for rng, F0, K, S_T in draws():  # noqa: N806
        DF = math.exp(-rng.uniform(0, 0.08) * rng.uniform(0.01, 2))  # noqa: N806
        c_b, p_a = rng.uniform(0.01, 0.3) * F0, rng.uniform(0.01, 0.3) * F0
        c_a, p_b = c_b * 1.02, p_a * 0.98
        F_a, F_b = F0 * 1.001, F0 * 0.999  # noqa: N806
        conv, _ = P.lin_conversion(K, c_b, p_a, F_a, DF)
        # premium carried to expiry at 1/DF; short call, long put, long future at F_a
        sim = (c_b - p_a) / DF - call(S_T, K) + put(S_T, K) + (S_T - F_a)
        assert conv == pytest.approx(sim, rel=1e-9, abs=1e-9 * F0)
        rev, _ = P.lin_reversal(K, c_a, p_b, F_b, DF)
        sim = -(c_a - p_b) / DF + call(S_T, K) - put(S_T, K) + (F_b - S_T)
        assert rev == pytest.approx(sim, rel=1e-9, abs=1e-9 * F0)


def test_inverse_locks_are_exact_in_coin():
    for rng, F0, K, S_T in draws(seed=11):  # noqa: N806
        c_b, p_a = rng.uniform(0.001, 0.3), rng.uniform(0.001, 0.3)
        c_a, p_b = c_b + 0.0005, max(p_a - 0.0005, 1e-6)
        F_a, F_b = F0 * 1.001, F0 * 0.999  # noqa: N806
        # coin options: payoff / S_T; long inverse future, notional K USD: K (1/F0 − 1/S_T)
        conv, _ = P.inv_conversion(K, c_b, p_a, F_a)
        sim = (c_b - p_a) + (-call(S_T, K) + put(S_T, K)) / S_T + K * (1 / F_a - 1 / S_T)
        assert conv == pytest.approx(sim, abs=1e-12)
        rev, _ = P.inv_reversal(K, c_a, p_b, F_b)
        sim = -(c_a - p_b) + (call(S_T, K) - put(S_T, K)) / S_T + K * (1 / S_T - 1 / F_b)
        assert rev == pytest.approx(sim, abs=1e-12)


def test_band_edges_are_where_the_trades_are_flat():
    lo, hi, _ = P.lin_band(100, 6.0, 6.2, 5.0, 5.3)
    assert P.lin_conversion(100, 6.0, 5.3, lo)[0] == pytest.approx(0)
    assert P.lin_reversal(100, 6.2, 5.0, hi)[0] == pytest.approx(0)
    lo, hi, _ = P.inv_band(60_000, 0.051, 0.053, 0.040, 0.042)
    assert P.inv_conversion(60_000, 0.051, 0.042, lo)[0] == pytest.approx(0, abs=1e-15)
    assert P.inv_reversal(60_000, 0.053, 0.040, hi)[0] == pytest.approx(0, abs=1e-15)
    assert lo < hi


def test_inventory_locks_k_usd():
    for rng, F0, K, S_T in draws(seed=3):  # noqa: N806
        S, T = F0, rng.uniform(0.01, 1.5)  # noqa: N806
        c_b, p_a = rng.uniform(0.01, 0.2) * S, rng.uniform(0.01, 0.2) * S
        r, _ = P.lin_inventory_rate(K, S, c_b, p_a, T)
        assert (S - c_b + p_a) * math.exp(r * T) == pytest.approx(K, rel=1e-9)
        assert S_T - call(S_T, K) + put(S_T, K) == pytest.approx(K)
        c, p = rng.uniform(0.001, 0.2), rng.uniform(0.001, 0.2)
        r, _ = P.inv_inventory_rate(K, S, c, p, T)
        held = 1 - (c - p)  # coin kept; the net premium tops it up to 1 coin
        coin_at_t = held + (c - p) + (-call(S_T, K) + put(S_T, K)) / S_T
        assert coin_at_t * S_T == pytest.approx(K, rel=1e-9)
        assert S * held * math.exp(r * T) == pytest.approx(K, rel=1e-9)


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        (P.lin_conversion(100, 0.0, 5, 101), P.NON_POSITIVE),
        (P.lin_reversal(100, 5, None, 101), P.NON_POSITIVE),
        (P.inv_conversion(100, 0.1, -0.01, 101), P.NON_POSITIVE),
        (P.inv_forward(100, 1.2, 0.1), P.INVERSE_DENOMINATOR),
        (P.lin_inventory_rate(100, 100, 5, 5, 0.5 / 8760, min_tte_hours=1), P.TENOR_TOO_SHORT),
        (P.inv_inventory_rate(100, 100, 1.5, 0.2, 0.1), P.INVERSE_DENOMINATOR),
        (P.lin_inventory_rate(100, 100, 5, 5, math.nan), P.TENOR_TOO_SHORT),
    ],
)
def test_guards_return_none_with_reason(result, reason):
    assert result == (None, reason)


def test_inverse_band_guard():
    assert P.inv_band(100, 1.1, 1.2, 0.05, 0.06) == (None, None, P.INVERSE_DENOMINATOR)


def test_weighted_median_ignores_an_outlier_strike():
    strikes = [(100.0, 99.9, 100.1), (100.2, 100.1, 100.3), (99.9, 99.8, 100.0),
               (130.0, 125.0, 135.0), (100.1, 100.0, 100.2)]  # fmt: skip
    assert P.implied_forward(strikes, 3) == (100.1, None)
    assert P.implied_forward(strikes[:2], 3) == (None, P.INSUFFICIENT_STRIKES)


def test_fees_follow_the_cap_and_settlement():
    # inverse: 0.0003 coin per option unless 12.5% of the price is lower
    assert P.option_fee(0.05, 0.0003, 0.125, 1.0) == pytest.approx(0.0003)
    assert P.option_fee(0.001, 0.0003, 0.125, 1.0) == pytest.approx(0.000125)
    # linear: rate × index in USD
    assert P.option_fee(10.0, 0.0003, 0.125, 150.0) == pytest.approx(0.045)
    assert P.hedge_fee(True, 0.0005, 60_000, 61_000) == pytest.approx(0.0005 * 60_000 / 61_000)
    assert P.hedge_fee(False, 0.0005, 150, 151) == pytest.approx(0.0005 * 151)
    locked = P.structure_fees(True, RATES, 60_000, 0.05, 0.04, 61_000, 60_500, False)
    perp = P.structure_fees(True, RATES, 60_000, 0.05, 0.04, 61_000, 60_500, True)
    assert locked - perp == pytest.approx(RATES.future_delivery * 60_000 / 61_000)
    assert P.FeeRates.from_table(
        {"option_taker": 1, "option_maker": 2, "option_cap_pct_of_price": 3, "future_taker": 4,
         "future_maker": 5, "perp_taker": 6, "perp_maker": 7, "option_delivery": 8,
         "future_delivery": 9}, "maker",
    ) == P.FeeRates(2, 3, 5, 7, 8, 9)  # fmt: skip


def test_breakeven_funding_signs():
    # conversion is long the perp: a positive edge survives funding below the breakeven
    assert P.breakeven_funding("conversion", 1.0, 100.0, 0.5) == pytest.approx(0.02)
    assert P.breakeven_funding("reversal", 1.0, 100.0, 0.5) == pytest.approx(-0.02)
    assert P.breakeven_funding("reversal", 1.0, 100.0, 0.0) is None


def test_hedge_contracts_and_residual():
    # 1 BTC of options at K = 60,005 with 10 USD contracts → 6,000 contracts (6,000.5 exact)
    n, residual = P.hedge_contracts(True, 1.0, 60_005, 61_000, 10)
    assert (n, residual) == (6000, pytest.approx(5.0))
    n, residual = P.hedge_contracts(False, 2.5, 150, 151, 1.0)
    assert n == 2 and residual == pytest.approx(0.5 * 151)


def test_basis_and_bps():
    assert P.basis_ann(102, 100, 0.5) == pytest.approx(0.04)
    assert P.basis_ann(None, 100, 0.5) is None
    assert P.bps(100.12, 100) == pytest.approx(12.0)
