"""Surface reconstruction from ATM + 25Δ RR + 25Δ fly."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anth_ipo_book import SPOT, SURFACE_2026_09_16  # noqa: E402
from anth_vol_surface import (  # noqa: E402
    build_slice,
    build_surface,
    calendar_violations,
    call_delta_from_strike,
    interpolate_iv,
    interpolate_iv_pchip,
    interpolate_iv_pchip_delta,
    iv_from_call_delta,
    iv_from_strike,
    norm_inv,
    quoted_moneyness_band,
    strike_from_call_delta,
    surface_grid,
    wings_from_rr_fly,
)


def close(a, b, tol=1e-6):
    assert math.isclose(a, b, rel_tol=0.0, abs_tol=tol), f"{a!r} != {b!r}"


class TestWings:
    def test_sep18_put_skew(self):
        c, p = wings_from_rr_fly(0.781, -1.1, 0.6)
        close(c, 0.7815, tol=1e-6)
        close(p, 0.7925, tol=1e-6)
        assert p > c  # put wing richer

    def test_oct17_cheap_wings(self):
        c, p = wings_from_rr_fly(0.733, 0.8, -2.5)
        close(c, 0.712, tol=1e-6)
        close(p, 0.704, tol=1e-6)
        assert c < 0.733 and p < 0.733

    def test_nov7_call_skew(self):
        c, p = wings_from_rr_fly(0.783, 1.8, -2.9)
        assert c > p  # calls richer past mid-October


class TestDeltaMap:
    def test_norm_inv_quartiles(self):
        close(norm_inv(0.5), 0.0, tol=1e-8)
        close(norm_inv(0.25), -0.67448975, tol=1e-5)
        close(norm_inv(0.75), 0.67448975, tol=1e-5)

    def test_atm_delta_roundtrip(self):
        sl = build_slice(SURFACE_2026_09_16[5])  # Oct 17
        k = strike_from_call_delta(SPOT, sl.atm, sl.tau, 0.50)
        # ATM call delta is slightly > 0.5 because d1 = 0.5 σ√τ
        d = call_delta_from_strike(SPOT, SPOT, sl.atm, sl.tau)
        assert 0.50 < d < 0.56
        close(iv_from_call_delta(sl, 0.50), sl.atm)
        close(iv_from_call_delta(sl, 0.25), sl.iv_25c)
        close(iv_from_call_delta(sl, 0.75), sl.iv_25p)


class TestSlice:
    def test_surface_has_nine_slices(self):
        s = build_surface()
        assert len(s) == 9
        assert s[0].expiry.label == "Sep 18"
        assert s[-1].expiry.label == "Nov 21"

    def test_quoted_points_repriced(self):
        sl = build_slice(SURFACE_2026_09_16[3])  # Sep 23, richest put RR
        close(iv_from_strike(sl, sl.k_atm), sl.atm, tol=5e-4)
        close(iv_from_strike(sl, sl.k_25c), sl.iv_25c, tol=8e-3)
        close(iv_from_strike(sl, sl.k_25p), sl.iv_25p, tol=8e-3)

    def test_short_dated_25d_strikes_are_tight(self):
        sl = build_slice(SURFACE_2026_09_16[0])
        assert sl.k_25p < sl.k_atm < sl.k_25c
        assert sl.k_25c / sl.k_atm - 1 < 0.08  # 1-day 25d is close to spot

    def test_sep23_25d_call_is_not_1_25(self):
        """Sep 23 25Δ call is ~1.06. The smile chart must not draw solid to 1.25."""
        sl = build_slice(SURFACE_2026_09_16[3])
        close(sl.k_25c / SPOT, 1.06, tol=0.01)
        close(sl.k_25p / SPOT, 0.95, tol=0.01)
        assert sl.k_25c / SPOT < 1.10


class TestInterpolation:
    def test_pins_listed_atm(self):
        s = build_surface()
        for sl in s:
            close(interpolate_iv(s, sl.days, SPOT), sl.atm, tol=8e-4)

    def test_between_oct17_and_oct24(self):
        s = build_surface()
        mid = interpolate_iv(s, 33.5, SPOT)
        lo = interpolate_iv(s, 30.0, SPOT)
        hi = interpolate_iv(s, 37.0, SPOT)
        assert min(lo, hi) - 1e-6 <= mid <= max(lo, hi) + 1e-6

    def test_surface_grid_atm_column_pins_listed(self):
        """ATM scatter at (1.0, sl.days, sl.atm) sits on the interpolate_iv mesh."""
        s = build_surface()
        days = [sl.days for sl in s]
        grid = surface_grid(s, days, [1.0], SPOT)
        for row, sl in zip(grid, s):
            close(row[0], sl.atm, tol=8e-4)


class TestPchip:
    def test_pchip_pins_listed_atm_at_spot(self):
        s = build_surface()
        for sl in s:
            close(interpolate_iv_pchip(s, sl.days, SPOT), sl.atm, tol=8e-4)

    def test_pchip_delta_pins_listed_atm(self):
        s = build_surface()
        for sl in s:
            close(interpolate_iv_pchip_delta(s, sl.days, 0.50), sl.atm, tol=8e-4)

    def test_weekend_trench_still_there(self):
        s = build_surface()
        iv4 = interpolate_iv_pchip_delta(s, 4.0, 0.50)
        close(iv4, 0.588, tol=8e-4)
        iv35 = interpolate_iv_pchip_delta(s, 3.5, 0.50)
        assert iv35 < 0.70

    def test_pchip_hits_listed_25d_wings(self):
        s = build_surface()
        sl = s[5]  # Oct 17
        close(interpolate_iv_pchip_delta(s, sl.days, 0.25), sl.iv_25c, tol=8e-4)
        close(interpolate_iv_pchip_delta(s, sl.days, 0.75), sl.iv_25p, tol=8e-4)


class TestQuotedBand:
    def test_two_day_25d_is_near_spot(self):
        s = build_surface()
        lo, hi = quoted_moneyness_band(s, 2.0)
        close(lo, 0.97, tol=0.02)
        close(hi, 1.03, tol=0.02)
        assert lo > 0.95
        assert 0.86 < lo  # 0.86 is off the 2d quoted band — that was the yellow wall

    def test_nov21_25d_is_wide(self):
        s = build_surface()
        lo, hi = quoted_moneyness_band(s, 65.0)
        close(lo, 0.85, tol=0.02)
        close(hi, 1.32, tol=0.02)

    def test_listed_band_matches_slice_strikes(self):
        s = build_surface()
        sl = s[-1]
        lo, hi = quoted_moneyness_band(s, sl.days)
        close(lo, sl.k_25p / SPOT, tol=1e-12)
        close(hi, sl.k_25c / SPOT, tol=1e-12)


class TestCalendar:
    def test_atm_total_var_mostly_rising_after_the_weekend(self):
        """
        The 1d → 4d strip is a weekend collapse: ATM IV falls from 78% to 59%.
        That *can* cut total variance. After Sep 23 the reconstructed ATM
        variance must not fall.
        """
        s = build_surface()
        hits = calendar_violations(s, moneyness=(1.0,))
        after_sep23 = [h for h in hits if h[0] not in {"Sep 18", "Sep 20", "Sep 21"}]
        assert after_sep23 == []

    def test_sep20_to_sep21_wing_hole_still_reported(self):
        s = build_surface()
        hits = calendar_violations(s)  # default 0.85 / 0.92 / 1.0 / 1.08 / 1.15
        assert any(h[0] == "Sep 20" and h[1] == "Sep 21" and h[2] != 1.0 for h in hits)
