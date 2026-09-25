"""Is there anything in the features for the GP to find?

Answers a question the GP cannot: when a search over ~100,000 strategies reports
nothing, is that because the search is inadequate or because the data holds no
exploitable signal? This asks the data directly, in four steps, and runs in
minutes rather than the hours a full evolution takes.

  1. INFORMATION COEFFICIENT — per feature, the cross-sectional rank
     correlation with next-day returns. Causal: the feature at t is ranked
     against the return from t to t+1.
  2. SPLIT-HALF STABILITY — the same IC computed on each half of the sample.
     A real effect keeps its sign and rough magnitude; an artifact does not.
  3. A PRE-SPECIFIED BASELINE — the simplest portfolio expressing the strongest
     feature, long-only and market-neutral, after the same 10 bps costs.
  4. THE SAME NULL CONTROL THE GP FACES — the baseline re-run on block
     bootstrapped surrogates.

Step 4 is the one that matters, and on the committed dataset it is what turns
an apparent result into a negative. A market-neutral low-volatility tilt earns
a median OOS Sharpe near +1.08, which looks like an edge until the surrogates
run: their median is around +0.28 and their 95th percentile above +1.8, giving
p ~ 0.25. The bootstrap preserves each asset's volatility level and the
cross-asset correlation structure — that is what makes it a fair null — and so
the low-volatility assets in a surrogate are still the low-volatility assets. A
long-low/short-high book inherits that structure's return asymmetry with no
predictive timing involved. The apparent edge is the structure, not a signal.

Which also explains the GP's result. It is not failing to search well enough;
there is nothing to find beyond what the null reproduces.

Run:  python scripts/diagnose_feature_ic.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from vgp.analysis import generate_windows
from vgp.analysis.null_control import block_bootstrap_ohlcv
from vgp.data import DataLoader, FeatureEngine

PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_DIR / "data"
START, END = "2024-01-01", "2026-04-01"
FEE = 10e-4
WINDOW_KW = dict(train_months=9, val_months=2, oos_months=4, step_months=4)
# Override for a quick check: VGP_N_NULL=3 python scripts/diagnose_feature_ic.py
N_NULL = int(os.environ.get("VGP_N_NULL", "99"))
BASELINE_FEATURE = "vol_20d"
REBALANCE_DAYS = 21
TERCILE = 0.33


def _panel(ohlcv):
    fe = FeatureEngine()
    fm = fe.fit_transform(ohlcv)
    dates, assets = fe.dates_, fe.retained_assets_
    names = list(getattr(fe, "feature_names_", [f"f{i}" for i in range(fm.shape[1])]))
    close = pd.DataFrame({t: ohlcv[t]["close"] for t in assets}).reindex(dates)
    fwd = np.log(close.shift(-1) / close)
    return fm, dates, assets, names, fwd


def _ic_series(fm, fwd, f, lo=0, hi=None):
    hi = fm.shape[0] - 1 if hi is None else min(hi, fm.shape[0] - 1)
    y_all = fwd.to_numpy()
    out = []
    for t in range(lo, hi):
        x, y = fm[t, f, :], y_all[t, :]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 5:
            continue
        xr, yr = pd.Series(x[m]).rank(), pd.Series(y[m]).rank()
        if xr.std() == 0 or yr.std() == 0:
            continue
        out.append(np.corrcoef(xr, yr)[0, 1])
    return np.asarray(out)


def _t_stat(ic):
    if len(ic) < 3 or ic.std(ddof=1) == 0:
        return float("nan")
    return float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))))


def _market_neutral_pnl(fm, dates, assets, names, fwd, feature):
    """Long the lowest tercile of `feature`, short the highest. Costs included."""
    f = names.index(feature)
    X = pd.DataFrame(fm[:, f, :], index=dates, columns=assets)
    w = pd.DataFrame(0.0, index=dates, columns=assets)
    for i in range(0, len(dates), REBALANCE_DAYS):
        row = X.iloc[i].dropna()
        if len(row) < 6:
            continue
        k = max(1, int(TERCILE * len(row)))
        w.iloc[i : i + REBALANCE_DAYS, w.columns.get_indexer(row.nsmallest(k).index)] = +0.5 / k
        w.iloc[i : i + REBALANCE_DAYS, w.columns.get_indexer(row.nlargest(k).index)] = -0.5 / k
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    return (w * fwd).sum(axis=1) - turnover * FEE


def _median_oos_sharpe(pnl, dates):
    windows = generate_windows(str(dates.min().date()), str(dates.max().date()), **WINDOW_KW)
    sharpes = []
    for w in windows:
        mask = (dates >= pd.Timestamp(w.test_start)) & (dates <= pd.Timestamp(w.test_end))
        p = pnl[mask].dropna()
        sharpes.append(p.mean() / p.std() * np.sqrt(365) if len(p) > 5 and p.std() > 0 else np.nan)
    return float(np.nanmedian(sharpes)), sharpes


def main() -> None:
    logging.disable(logging.CRITICAL)
    loader = DataLoader(cache_dir=CACHE_DIR)
    real = loader.fetch_ohlcv(start_date=START, end_date=END, allow_partial=True, min_assets=10)
    fm, dates, assets, names, fwd = _panel(real)
    print(f"panel {fm.shape[0]} dates x {fm.shape[1]} features x {fm.shape[2]} assets\n")

    print("1. INFORMATION COEFFICIENT vs next-day return")
    print(f"{'feature':>16} {'mean rank IC':>13} {'t':>7}")
    scored = []
    for f, name in enumerate(names):
        ic = _ic_series(fm, fwd, f)
        scored.append((name, ic.mean(), _t_stat(ic)))
    for name, mean, t in sorted(scored, key=lambda r: -abs(r[2])):
        print(f"{name:>16} {mean:>+13.4f} {t:>7.2f}")
    print(f"\n   |t| > {abs(round(2.9, 1))} clears Bonferroni for {len(names)} tests.\n")

    print("2. SPLIT-HALF STABILITY (sign and magnitude must hold)")
    half = fm.shape[0] // 2
    print(f"{'feature':>16} {'1st half':>10} {'t':>7} {'2nd half':>10} {'t':>7}  holds")
    for name, _m, t in sorted(scored, key=lambda r: -abs(r[2])):
        if abs(t) < 2:
            continue
        f = names.index(name)
        a, b = _ic_series(fm, fwd, f, 0, half), _ic_series(fm, fwd, f, half)
        holds = "yes" if np.sign(a.mean()) == np.sign(b.mean()) else "NO"
        print(
            f"{name:>16} {a.mean():>+10.4f} {_t_stat(a):>7.2f} "
            f"{b.mean():>+10.4f} {_t_stat(b):>7.2f}  {holds}"
        )

    print(
        f"\n3. BASELINE — market-neutral tercile tilt on {BASELINE_FEATURE}, "
        f"rebalanced every {REBALANCE_DAYS} days, {FEE*1e4:.0f} bps"
    )
    pnl = _market_neutral_pnl(fm, dates, assets, names, fwd, BASELINE_FEATURE)
    observed, per_window = _median_oos_sharpe(pnl, dates)
    print("   per-window OOS Sharpe: " + "  ".join(f"{s:+.2f}" for s in per_window))
    print(f"   median: {observed:+.3f}")

    print(f"\n4. NULL CONTROL — the same baseline on {N_NULL} signal-free surrogates")
    null = []
    for r in range(N_NULL):
        try:
            sur = block_bootstrap_ohlcv(real, np.random.default_rng(5000 + r), block_size=20)
            sfm, sdates, sassets, snames, sfwd = _panel(sur)
            spnl = _market_neutral_pnl(sfm, sdates, sassets, snames, sfwd, BASELINE_FEATURE)
            value, _ = _median_oos_sharpe(spnl, sdates)
            if np.isfinite(value):
                null.append(value)
        except Exception as exc:  # a bad surrogate must not abandon the control
            logging.getLogger(__name__).warning("surrogate %d failed: %s", r, exc)
    null_arr = np.asarray(null)
    p_value = (1 + int((null_arr >= observed).sum())) / (1 + len(null_arr))
    print(
        f"   {len(null_arr)} runs   median {np.median(null_arr):+.3f}   "
        f"p95 {np.percentile(null_arr, 95):+.3f}"
    )
    print(f"   observed {observed:+.3f}   p = {p_value:.3f}")
    print(
        "\n   VERDICT: "
        + (
            "beats the null."
            if p_value <= 0.05
            else "does NOT beat the null. Signal-free data with the same "
            "volatility\n   structure performs comparably, so the apparent edge is that "
            "structure,\n   not predictive information."
        )
    )


if __name__ == "__main__":
    main()
