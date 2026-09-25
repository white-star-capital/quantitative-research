"""Golden tests against the 16 Sep 2026 White Star note."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anth_ipo_book import (  # noqa: E402
    PAPER_LIQ_RALLY,
    PAPER_SCENARIOS,
    PERP_NOTIONAL,
    SPOT,
    apply_early_listing_rule,
    event_move,
    forward_vol,
    funding_over_days,
    hyperliquid_margin,
    phase_one_capital,
    phase_one_greeks,
    phase_one_qty,
    phase_two_book,
    total_variance,
    variance_premium,
    weighted_expected_return,
    weekend_fwd_vol,
)


def close(a, b, tol=5e-4):
    assert math.isclose(a, b, rel_tol=0.0, abs_tol=tol), f"{a!r} != {b!r} (tol={tol})"


class TestEventExtraction:
    def test_oct17_at_50_is_15_4_pct(self):
        close(event_move(0.733, 30, 0.50), 0.154, tol=8e-4)

    def test_nov21_table(self):
        close(event_move(0.792, 65, 0.45), 0.275, tol=8e-4)
        close(event_move(0.792, 65, 0.50), 0.259, tol=8e-4)
        close(event_move(0.792, 65, 0.60), 0.218, tol=8e-4)

    def test_oct17_45_and_60(self):
        close(event_move(0.733, 30, 0.45), 0.166, tol=8e-4)
        close(event_move(0.733, 30, 0.60), 0.121, tol=8e-4)

    def test_nov21_total_variance(self):
        close(total_variance(0.792, 65), 0.1117, tol=5e-5)


class TestForwardVol:
    def test_listing_window_oct17_nov21(self):
        # Note prints 83.8% using 0.1117 − 0.0442 over 35.1 days.
        close(forward_vol(0.733, 30, 0.792, 65), 0.838, tol=2e-3)

    def test_sep21_weekend_strip(self):
        close(forward_vol(0.661, 3, 0.588, 4), 0.263, tol=8e-3)

    def test_guards(self):
        with pytest.raises(ValueError):
            forward_vol(0.7, 30, 0.8, 10)


class TestPhaseOne:
    def test_two_puts_per_perp_unit(self):
        n_perp, n_puts = phase_one_qty()
        close(n_perp, -PERP_NOTIONAL / SPOT, tol=1e-9)
        close(n_puts, 2 * n_perp, tol=1e-9)  # both negative; |puts| = 2 |perp|
        close(abs(n_puts), 922, tol=0.5)

    def test_greeks_match_note(self):
        g = phase_one_greeks()
        close(g.premium, 71_800, tol=50)
        close(g.theta_per_day, 5_125, tol=2)
        close(g.vega_per_vol_point, -1_104, tol=2)
        close(g.dollar_gamma_per_pct2, -443, tol=1)
        close(g.breakeven_daily, 3.40, tol=0.02)
        close(g.breakeven_hourly, 0.69, tol=0.01)
        close(g.funding_per_day, 658, tol=2)
        close(g.margin_entry, 472_000, tol=500)
        close(g.margin_maint, 312_000, tol=500)

    def test_near_flat_delta(self):
        g = phase_one_greeks()
        assert abs(g.delta) < 20  # residual vs $1M / 2170 ≈ 461 unit delta

    def test_variance_gap(self):
        close(variance_premium(0.65, 0.53), 0.65**2 - 0.53**2)


class TestPhaseTwo:
    def test_fwd_vol_on_the_book(self):
        b = phase_two_book()
        close(b.fwd_vol, 0.838, tol=2e-3)
        close(b.n_call_far, PERP_NOTIONAL / SPOT)
        close(b.n_put_near, -PERP_NOTIONAL / SPOT)

    def test_timing_rule(self):
        assert apply_early_listing_rule(26) == "swap_near_to_far"  # Oct 12 ≈ day 26
        assert apply_early_listing_rule(54) == "hold"  # Nov 9
        assert apply_early_listing_rule(70) == "exit_after_far"


class TestCapital:
    def test_hl_margin_table(self):
        close(hyperliquid_margin(1_000_000, 6), 167_000, tol=500)
        close(hyperliquid_margin(1_000_000, 2), 500_000, tol=1)
        close(hyperliquid_margin(1_000_000, 1.5), 667_000, tol=500)

    def test_paper_liq_fixtures(self):
        close(PAPER_LIQ_RALLY[6.0], 0.08)
        close(PAPER_LIQ_RALLY[2.0], 0.38)

    def test_phase_one_capital_near_970k(self):
        close(phase_one_capital(leverage=2.0), 970_000, tol=3_000)


class TestExpectedReturns:
    def test_components_sum_to_total(self):
        for s in PAPER_SCENARIOS:
            close(s.options + s.hedge + s.funding + s.fees, s.total, tol=5e-3)

    def test_weighted_24pct_funding(self):
        close(weighted_expected_return(True), 0.106, tol=8e-4)

    def test_weighted_zero_funding(self):
        close(weighted_expected_return(False), 0.068, tol=1.5e-3)

    def test_funding_65_days(self):
        close(funding_over_days(65) / PERP_NOTIONAL, 0.043, tol=5e-4)
        close(funding_over_days(7) / PERP_NOTIONAL, 0.0046, tol=2e-4)
