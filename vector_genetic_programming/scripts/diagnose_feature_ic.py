"""Is there anything in the features for the GP to find?

Answers a question the GP cannot: when a search over ~100,000 strategies reports
nothing, is that because the search is inadequate or because the data holds no
exploitable signal? This asks the data directly and runs in minutes rather than
the hours a full evolution takes.

  1. RAW IC, AND HOW MUCH OF IT THE NULL REPRODUCES — the cross-sectional rank
     correlation of each feature with next-day returns, beside the same number
     computed on signal-free surrogates.
  2. TIMING IC vs THE SURROGATE — the same IC after each feature is
     standardised within its own asset, scored against the surrogate
     distribution rather than against zero.
  3. A PRE-SPECIFIED BASELINE — the simplest portfolio expressing the strongest
     feature, market-neutral, after the same 10 bps costs.
  4. THE SAME NULL CONTROL THE GP FACES — the baseline re-run on surrogates.

WHY STEPS 1 AND 2 ARE SHAPED THIS WAY. An earlier version of this script ranked
features by raw IC and tested each against zero. Both choices are wrong, and
together they produced a confident finding that does not survive contact with
the null.

A raw cross-sectional IC on `vol_20d` answers "do high-volatility assets earn
less than low-volatility ones". That is a STATIC property of the assets, not
predictability, and a surrogate that preserves each asset's return distribution
reproduces it in full — measured on the committed dataset, the surrogate
reproduces 92-124% of every feature's raw IC. The five features that cleared a
Bonferroni threshold against zero cleared it on a quantity signal-free data
reproduces entirely. Split-half stability was not independent evidence either: a
static asset property is stable across halves by construction, so that check
could only ever agree.

Standardising each feature within its own asset removes the static level and
leaves the only component a strategy can trade — is this asset unusual FOR
ITSELF right now. Scored that way against surrogates, 11 of 12 features are
indistinguishable from signal-free data. The survivor is `ret_1d`, 1-day
cross-sectional reversal, which the raw ranking placed near the BOTTOM.

BLOCK SIZE IS NOT A FREE PARAMETER HERE. `block_bootstrap_ohlcv` severs serial
dependence only at block boundaries, so an effect shorter than the block
survives into the surrogate and the null cannot flag it. These are 1-day
horizon ICs, so the IC steps use `block_size=1`: it preserves each asset's
return distribution and the cross-sectional structure while destroying time
ordering, which is the right null for a 1-day signal. Step 4's portfolio
rebalances every REBALANCE_DAYS, so it keeps the 20-bar default.

AN IC THAT BEATS THE NULL IS NECESSARY, NOT SUFFICIENT. On this dataset
`ret_1d` reaches p = 0.004 on timing IC and still produces a gross Sharpe of
+0.285 against a null mean of -0.309 — within one standard deviation, p = 0.23.
Daily rebalancing turns the book over 1.34x, which is a 49% annual drag at 10
bps and takes the net Sharpe to -2.18. Real information, too small to trade at
this breadth. Report both numbers or neither.

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
# Surrogates for the IC steps. Bonferroni over 12 features needs p <= 0.0042,
# and an empirical p cannot resolve below 1/(1+N), so N must exceed 237.
N_NULL_IC = int(os.environ.get("VGP_N_NULL_IC", "250"))
IC_BLOCK = 1  # 1-day horizon: the block must be shorter than the effect
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


def _standardise_within_asset(fm):
    """Remove each asset's own level and scale, per feature.

    What remains is the timing component: not "is this asset volatile" but "is
    this asset volatile FOR ITSELF right now". The raw feature is dominated by
    the former, which no strategy can harvest and which any surrogate
    preserving the asset's return distribution reproduces exactly.
    """
    mu = np.nanmean(fm, axis=0, keepdims=True)
    sd = np.nanstd(fm, axis=0, keepdims=True)
    # `sd > 0` is not a safe degeneracy test. A feature that never moves has a
    # floating-point std around 1e-16 rather than exactly zero, and the matching
    # 1-ULP error in `fm - mu` then divides out to a clean +/-1 — a constant
    # column arrives looking like a unit-variance signal. Scale the floor to the
    # feature's own magnitude so a genuinely flat series becomes NaN.
    scale = np.maximum(np.abs(mu), 1.0)
    degenerate = sd <= np.finfo(np.float64).eps * scale * fm.shape[0]
    return (fm - mu) / np.where(degenerate, np.nan, sd)


def _mean_ic(fm, fwd, f):
    ic = _ic_series(fm, fwd, f)
    return float(ic.mean()) if len(ic) else float("nan")


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

    # Both IC steps need the same surrogate panels, so build them once.
    print(f"building {N_NULL_IC} surrogates (block_size={IC_BLOCK}) for the IC steps...")
    sur_panels = []
    for r in range(N_NULL_IC):
        try:
            sur = block_bootstrap_ohlcv(real, np.random.default_rng(7000 + r), block_size=IC_BLOCK)
            sur_panels.append(_panel(sur))
        except Exception as exc:  # a bad surrogate must not abandon the control
            logging.getLogger(__name__).warning("IC surrogate %d failed: %s", r, exc)
    n_sur = len(sur_panels)
    print(f"  {n_sur} usable\n")

    print("1. RAW IC vs next-day return, AND HOW MUCH THE NULL REPRODUCES")
    print("   A raw IC ranks assets by a static property as much as by any signal.")
    print("   'null' is the same statistic on signal-free data: if it matches the")
    print("   observed value, the IC carries no predictive information at all.")
    print(f"{'feature':>16} {'raw IC':>9} {'t vs 0':>8} {'null IC':>9} {'reproduced':>11}")
    raw_rows = []
    for f, name in enumerate(names):
        obs = _mean_ic(fm, fwd, f)
        nul = float(np.nanmean([_mean_ic(sfm, sfwd, f) for sfm, _d, _a, _n, sfwd in sur_panels]))
        raw_rows.append((name, obs, _t_stat(_ic_series(fm, fwd, f)), nul))
    for name, obs, t, nul in sorted(raw_rows, key=lambda r: -abs(r[2])):
        pct = f"{100 * nul / obs:>10.0f}%" if obs else f"{'n/a':>11}"
        print(f"{name:>16} {obs:>+9.4f} {t:>8.2f} {nul:>+9.4f} {pct}")
    print("\n   The t column is against ZERO and is NOT evidence. Read the last column,")
    print("   and only for features whose raw IC is large: the ratio is a ratio of two")
    print("   near-zero numbers further down and means nothing there.\n")

    print("2. TIMING IC — feature standardised within its own asset, scored vs the null")
    print("   This is the only component a strategy can harvest.")
    dm = _standardise_within_asset(fm)
    sur_dm = [(_standardise_within_asset(sfm), sfwd) for sfm, _d, _a, _n, sfwd in sur_panels]
    crit = 0.05 / len(names)
    header = f"{'feature':>16} {'timing IC':>10} {'null mean':>10} {'null sd':>9}"
    print(f"{header} {'z':>7} {'p':>8}  verdict")
    timing = []
    for f, name in enumerate(names):
        obs = _mean_ic(dm, fwd, f)
        arr = np.asarray([_mean_ic(sdm, sfwd, f) for sdm, sfwd in sur_dm], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 3 or not np.isfinite(obs):
            continue
        m, sd = arr.mean(), arr.std(ddof=1)
        z = (obs - m) / sd if sd > 0 else float("nan")
        # Two-sided empirical p against the surrogate spread, never against zero.
        k = int((np.abs(arr - m) >= abs(obs - m)).sum())
        pv = (1 + k) / (1 + len(arr))
        timing.append((name, obs, m, sd, z, pv))
    for name, obs, m, sd, z, pv in sorted(timing, key=lambda r: -abs(r[4])):
        verdict = "SIGNAL" if pv <= crit else ("marginal" if pv <= 0.05 else "nothing")
        print(f"{name:>16} {obs:>+10.4f} {m:>+10.4f} {sd:>9.4f} {z:>7.2f} {pv:>8.4f}  {verdict}")
    floor = 1 / (1 + n_sur) if n_sur else float("nan")
    print(f"\n   Bonferroni over {len(names)} features: p <= {crit:.4f}. Floor is {floor:.4f}.")
    if floor > crit:
        print("   WARNING: the floor exceeds the threshold — raise VGP_N_NULL_IC above 237.")
    signals = [r[0] for r in timing if r[5] <= crit]
    print(f"   Features with timing signal: {', '.join(signals) if signals else 'NONE'}")
    print("   A timing IC that beats the null still has to survive turnover and costs.\n")

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
