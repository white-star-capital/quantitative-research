"""Market context and the equity band must describe THIS run, not a plausible one.

A context chart drawn on the wrong universe, or a band built from curves of
different lengths, is worse than no chart: it looks authoritative and is quietly
about something else.
"""

import numpy as np
import pandas as pd
import pytest

from vgp.analysis.context import effective_bets, equity_band, market_context


def _panel(n_days=200, assets=("A", "B", "C", "D"), rho=0.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    common = rng.normal(0, 0.01, n_days)
    out = {}
    for i, t in enumerate(assets):
        idio = rng.normal(0, 0.01, n_days)
        r = rho * common + (1 - rho) * idio
        close = pd.Series(100 * np.exp(np.cumsum(r)), index=dates)
        out[t] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1.0}
        )
    return out, dates, list(assets)


def test_effective_bets_spans_one_to_n():
    """Identical assets are one bet; independent assets are N."""
    n = 5
    assert effective_bets(np.ones((n, n))) == pytest.approx(1.0, abs=1e-9)
    assert effective_bets(np.eye(n)) == pytest.approx(float(n), abs=1e-9)


def test_correlated_assets_report_fewer_bets_than_assets():
    """The headline claim — 21 assets, 2.17 bets — must be reproducible in miniature."""
    ohlcv, dates, assets = _panel(rho=0.9, seed=1)
    ctx = market_context(ohlcv, assets, dates, window=60)
    vals = [v for v in ctx["breadth"] if v is not None]
    assert vals, "no breadth computed"
    assert max(vals) < len(assets), "highly correlated assets cannot be N independent bets"


def test_context_uses_only_the_assets_it_is_given():
    """The realised universe, not the requested one.

    Context drawn on assets the run discarded is context for a different run.
    """
    ohlcv, dates, assets = _panel()
    ctx = market_context(ohlcv, assets[:2], dates)
    assert ctx["n_assets"] == 2
    assert set(ctx["assets"]) == set(assets[:2])


def test_index_is_rebased_and_aligned_to_the_dates():
    ohlcv, dates, assets = _panel()
    ctx = market_context(ohlcv, assets, dates)
    assert len(ctx["index"]) == len(dates) == len(ctx["dates"])
    assert ctx["index"][0] == pytest.approx(1.0, abs=1e-9)
    assert all(np.isfinite(ctx["index"]))


def test_breadth_is_null_before_the_window_fills():
    """A number computed on fewer bars than the window is not that number.

    The boundary is `window`, not `window - 1`: returns come from a diff, so bar
    0 is NaN and the first block of `window` FINITE returns ends at index
    `window`. Reporting a correlation from a block containing that NaN would be
    a breadth figure computed on one bar fewer than it claims.
    """
    ohlcv, dates, assets = _panel()
    ctx = market_context(ohlcv, assets, dates, window=60)
    assert ctx["breadth"][:60] == [None] * 60
    assert ctx["breadth"][60] is not None


def _rows(n_bars=50, n_windows=2, seeds=2):
    rows = []
    for w in range(n_windows):
        for s in range(seeds):
            rows.append(
                {
                    "window_id": w,
                    "seed": s,
                    "test_start": f"2025-0{w + 1}-01",
                    "test_end": f"2025-0{w + 3}-01",
                    "oos_equity": list(np.linspace(1.0, 1.1 + 0.01 * s, n_bars)),
                }
            )
    return rows


def test_band_percentiles_are_ordered_everywhere():
    rows = _rows()
    nulls = [[list(np.linspace(1.0, 1.0 + 0.004 * r, 50)) for _ in range(2)] for r in range(20)]
    band = equity_band(rows, nulls, n_null=20)
    assert len(band["windows"]) == 2
    for win in band["windows"]:
        p05, p50, p95 = map(np.array, (win["p05"], win["p50"], win["p95"]))
        assert np.all(p05 <= p50 + 1e-9) and np.all(p50 <= p95 + 1e-9)
        assert len(win["dates"]) == len(win["observed"]) == len(p50)


def test_band_refuses_to_draw_from_too_few_curves():
    """Five curves cannot describe a 5th-to-95th percentile envelope."""
    band = equity_band(_rows(), [[list(np.linspace(1, 1.05, 50))] * 2 for _ in range(3)], n_null=3)
    assert band["windows"] == []


def test_rows_without_a_curve_are_skipped_not_zero_filled():
    """A failed evaluation has no equity curve; it must not become a flat line."""
    rows = _rows()
    rows[0]["oos_equity"] = []
    nulls = [[list(np.linspace(1.0, 1.01, 50)) for _ in range(2)] for _ in range(20)]
    band = equity_band(rows, nulls, n_null=20)
    w0 = next(w for w in band["windows"] if w["w"] == 0)
    assert w0["observed"][0] == pytest.approx(1.0, abs=1e-6), "median taken over the surviving seed"


def test_curves_of_different_lengths_are_trimmed_not_padded():
    """Padding would invent returns at the end of a short curve."""
    rows = _rows(n_bars=50)
    rows[1]["oos_equity"] = list(np.linspace(1.0, 1.2, 40))
    nulls = [[list(np.linspace(1.0, 1.01, 50)) for _ in range(2)] for _ in range(20)]
    band = equity_band(rows, nulls, n_null=20)
    w0 = next(w for w in band["windows"] if w["w"] == 0)
    assert len(w0["observed"]) == 40
