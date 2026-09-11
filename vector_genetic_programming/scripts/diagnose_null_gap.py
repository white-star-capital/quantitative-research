#!/usr/bin/env python3
"""Why does the GP reach a HIGHER in-sample Sharpe on signal-free surrogates?

ANSWER (2026-09-10): it did not, once the surrogate was correct. The gap was a
bug in `block_bootstrap_ohlcv`, not a property of the data.

Co-movement was shared only over the intersection of ALL fetched assets. On the
27-asset Binance panel EUL lists 2025-10-13, so that intersection was the final
171 days — after every training window — and every training-window bar fell
into the per-asset fallback branch. The surrogate therefore had essentially
independent assets exactly where the GP trains:

    real training window     mean pairwise corr 0.615   2.40 effective bets
    surrogate (before fix)   mean pairwise corr 0.001  19.96 effective bets

A cross-sectional 1/N book with twenty independent bets instead of two reaches
a far higher in-sample Sharpe, so the null was structurally inflated and
unbeatable in-sample (p = 1.000) while out-of-sample behaved sensibly.

With per-stratum shared draws the random-tree Sharpe dispersion gap at the
production block size fell from +1.10 (std) / +1.88 (max) to +0.11 / +0.11,
and at longer blocks the surrogate sits slightly BELOW real — the correct
direction for data with no signal.

The general lesson, and why this survived review: a surrogate can preserve the
UNCONDITIONAL correlation matrix while destroying it inside every window.
Correlation is time-varying, and the window is what the model trains on. Any
faithfulness check has to be window-local. See the regression tests in
tests/test_null_control.py.

WHAT THIS SCRIPT MEASURES
-------------------------
Kept as a standing diagnostic: it verifies the surrogate remains a fair opponent
after any change to the bootstrap. The maximum in-sample Sharpe a search of N
trials can reach scales with the DISPERSION of the Sharpe statistic across
candidate strategies, roughly sigma_SR x E[max of N draws]. So the question is
never "does the surrogate have alpha" (it does not) but "is the Sharpe statistic
more dispersed on the surrogate, and if so why".

  H1 SEARCH — evolution behaves differently on the two datasets rather than the
     data being more fittable. Discriminated by evaluating RANDOM trees, with no
     evolution at all. (Ruled out: the gap appeared with random trees.)

  H2 CROSS-SECTION — the surrogate has weaker factor structure, so a 1/N book
     gets more independent bets and a higher achievable Sharpe. (THIS WAS IT,
     though measured over the full sample it looked fine — see above.)

  H3 TEMPORAL — the block bootstrap destroys serial structure beyond the block,
     above all the persistence of volatility. Discriminated by a dose-response
     in block size: at block_size >= T a single circular block is just a
     ROTATION of the real series, with every temporal property intact. (Ruled
     out: the gap was LARGEST at the rotation, and there was no dose-response.)

  H4 COSTS x VOLATILITY — Sharpe is scale-invariant but a fixed 10bps fee is
     not, so a quieter window has a lower net ceiling. (Ruled out separately:
     the gap was identical at fee_bps=0.)

Read the output as: surrogate std/max should sit at or slightly below REAL. If
either exceeds it materially, the surrogate is an easier problem than the real
data and any p-value against it is not trustworthy.

Usage
-----
    python scripts/diagnose_null_gap.py
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data_pipeline_example/cache")
N_TREES = 400  # random trees evaluated per dataset
N_REPS = 3  # surrogate replicates per block size
BLOCK_SIZES = (1, 5, 20, 60, 100_000)  # last one exceeds T -> pure rotation
SEED = 20260910

logging.basicConfig(level=logging.ERROR)


def _panel():
    """Real train-window panel, matching window 0 of the committed run."""
    from vgp.data import DataLoader, FeatureEngine

    loader = DataLoader(cache_dir=CACHE_DIR)
    ohlcv = loader.fetch_ohlcv(
        start_date="2024-01-01",
        end_date="2026-04-01",
        allow_partial=True,
        min_assets=10,
    )
    fe = FeatureEngine()
    fm = fe.fit_transform(ohlcv)
    close = (
        pd.DataFrame({t: ohlcv[t]["close"] for t in fe.retained_assets_})
        .reindex(fe.dates_)
        .ffill(limit=3)
    )
    return ohlcv, fe, fm, close


def _train_slice(fe, fm, close, train_end="2025-04-30"):
    mask = fe.dates_ <= train_end
    return fm[mask], close.loc[mask]


def _random_trees(n: int):
    from deap import gp

    from vgp.gp.gp_types import build_pset, creator

    pset = build_pset()
    random.seed(SEED)
    np.random.seed(SEED)
    return [creator.Individual(gp.genHalfAndHalf(pset, min_=2, max_=5)) for _ in range(n)]


def _sharpe_distribution(trees, fm, close, label: str) -> dict:
    """Evaluate a fixed tree set and summarise the IS Sharpe distribution.

    This is the quantity that governs how high a search can reach: not the mean
    Sharpe (which is ~0 with no edge) but its DISPERSION and upper tail.
    """
    from vgp.backtest.runner import EVAL_OK, EvalConfig, evaluate_with_status

    cfg = EvalConfig(fee_bps=10.0, min_trades=50, close_prices=close)
    sharpes, trades = [], []
    for tree in trees:
        fitness, status, n_trades = evaluate_with_status(tree, fm, cfg)
        trades.append(n_trades)
        if status == EVAL_OK and np.isfinite(fitness[0]):
            sharpes.append(float(fitness[0]))

    arr = np.asarray(sharpes, dtype=np.float64)
    return {
        "label": label,
        "n_valid": int(arr.size),
        "frac_valid": arr.size / max(1, len(trees)),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "p95": float(np.percentile(arr, 95)) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "mean_trades": float(np.mean(trades)),
    }


def _timeseries_stats(ohlcv: dict, label: str) -> dict:
    """Properties that plausibly drive Sharpe dispersion."""
    tickers = sorted(ohlcv)
    common = None
    for t in tickers:
        idx = ohlcv[t].index
        common = idx if common is None else common.intersection(idx)

    rets = np.column_stack(
        [np.diff(np.log(ohlcv[t].loc[common, "close"].to_numpy(dtype=np.float64))) for t in tickers]
    )  # [T-1 x A]

    def acf(x, lag):
        if len(x) <= lag + 2:
            return float("nan")
        a, b = x[:-lag], x[lag:]
        return float(np.corrcoef(a, b)[0, 1])

    # Return autocorrelation, averaged across assets
    ret_acf1 = float(np.mean([acf(rets[:, i], 1) for i in range(rets.shape[1])]))
    ret_acf5 = float(np.mean([acf(rets[:, i], 5) for i in range(rets.shape[1])]))

    # Volatility clustering: autocorrelation of |returns| at lags inside and
    # well beyond the 20-bar block. The far lags are what a block bootstrap
    # cannot preserve.
    abs_acf = {
        lag: float(np.mean([acf(np.abs(rets[:, i]), lag) for i in range(rets.shape[1])]))
        for lag in (1, 5, 20, 40, 60)
    }

    # Persistence of PORTFOLIO-level realised volatility: 20d rolling std of the
    # equal-weight return, autocorrelated at 20 lags. High values mean long
    # turbulent regimes that a fitted strategy cannot sidestep.
    ew = rets.mean(axis=1)
    roll_vol = pd.Series(ew).rolling(20).std().dropna().to_numpy()
    vol_acf20 = acf(roll_vol, 20)
    vol_of_vol = float(roll_vol.std() / roll_vol.mean()) if roll_vol.mean() else float("nan")

    # Cross-sectional factor concentration — H2's discriminator
    corr = np.corrcoef(rets, rowvar=False)
    eigs = np.linalg.eigvalsh(corr)[::-1]
    pc1_share = float(eigs[0] / eigs.sum())

    return {
        "label": label,
        "ret_acf1": ret_acf1,
        "ret_acf5": ret_acf5,
        "absret_acf": abs_acf,
        "vol_acf20": vol_acf20,
        "vol_of_vol": vol_of_vol,
        "pc1_share": pc1_share,
        "mean_ret_std": float(rets.std(axis=0).mean()),
        "mean_kurtosis": float(pd.DataFrame(rets).kurt().mean()),
    }


def main() -> None:
    from vgp.analysis import block_bootstrap_ohlcv
    from vgp.analysis.null_control import _default_feature_builder

    print("Loading real panel...")
    ohlcv, fe, fm, close = _panel()
    fm_tr, close_tr = _train_slice(fe, fm, close)
    T = fm_tr.shape[0]
    print(f"  train panel: {fm_tr.shape} (T={T}, A={fm_tr.shape[2]})")

    trees = _random_trees(N_TREES)
    print(f"  {len(trees)} random trees (no evolution — isolates the DATA)\n")

    # ---------------- Sharpe dispersion, real vs surrogates ----------------
    rows = [_sharpe_distribution(trees, fm_tr, close_tr, "REAL")]
    ts_rows = [_timeseries_stats(ohlcv, "REAL")]

    for block in BLOCK_SIZES:
        shown = f"{block}" if block < T else f"{block} (>T: rotation)"
        for rep in range(N_REPS):
            rng = np.random.default_rng(SEED + 1000 * block + rep)
            sur = block_bootstrap_ohlcv(ohlcv, rng, block_size=block)
            s_fm, s_close, s_dates = _default_feature_builder(sur)
            mask = s_dates <= "2025-04-30"
            rows.append(
                _sharpe_distribution(trees, s_fm[mask], s_close.loc[mask], f"block={shown} r{rep}")
            )
            if rep == 0:
                ts_rows.append(_timeseries_stats(sur, f"block={shown}"))
        print(f"  block_size={shown}: {N_REPS} replicate(s) done")

    # ---------------- report ----------------
    print("\n" + "=" * 92)
    print("RANDOM-TREE IS SHARPE DISTRIBUTION  (no evolution; H1 discriminator)")
    print("=" * 92)
    print(
        f"{'dataset':<26}{'n_valid':>8}{'valid%':>8}{'mean':>8}{'std':>8}"
        f"{'p95':>8}{'max':>8}{'trades':>9}"
    )
    for r in rows:
        print(
            f"{r['label']:<26}{r['n_valid']:>8}{100*r['frac_valid']:>7.0f}%"
            f"{r['mean']:>8.3f}{r['std']:>8.3f}{r['p95']:>8.3f}"
            f"{r['max']:>8.3f}{r['mean_trades']:>9.0f}"
        )

    real = rows[0]
    print(f"\nReal std={real['std']:.3f} max={real['max']:.3f}")
    print("If surrogate std/max exceed real with RANDOM trees, the data is more")
    print("fittable and H1 (search dynamics) is not the explanation.")

    # Dose-response on block size
    print("\n" + "=" * 92)
    print("DOSE-RESPONSE IN BLOCK SIZE  (H3 discriminator)")
    print("=" * 92)
    print(f"{'block':<26}{'mean std':>10}{'mean max':>10}{'vs real std':>13}{'vs real max':>13}")
    for block in BLOCK_SIZES:
        shown = f"{block}" if block < T else f"{block} (>T: rotation)"
        grp = [r for r in rows if r["label"].startswith(f"block={shown} ")]
        if not grp:
            continue
        ms = float(np.mean([g["std"] for g in grp]))
        mx = float(np.mean([g["max"] for g in grp]))
        print(
            f"{shown:<26}{ms:>10.3f}{mx:>10.3f}"
            f"{ms - real['std']:>+13.3f}{mx - real['max']:>+13.3f}"
        )
    print("\nExpected after the correlation fix: gaps at or slightly below zero at")
    print("every block size. A gap that GROWS with block size (largest at the")
    print("rotation, where all temporal structure is intact) is the signature of a")
    print("cross-sectional problem, not a temporal one — that is how the")
    print("correlation bug was found.")

    # Time-series properties
    print("\n" + "=" * 92)
    print("TIME-SERIES PROPERTIES  (mechanism; H2 vs H3)")
    print("=" * 92)
    print(
        f"{'dataset':<26}{'ret_ac1':>9}{'|r|ac1':>8}{'|r|ac20':>9}{'|r|ac40':>9}"
        f"{'|r|ac60':>9}{'volac20':>9}{'vol/vol':>9}{'pc1%':>7}{'kurt':>7}"
    )
    for t in ts_rows:
        a = t["absret_acf"]
        print(
            f"{t['label']:<26}{t['ret_acf1']:>9.3f}{a[1]:>8.3f}{a[20]:>9.3f}"
            f"{a[40]:>9.3f}{a[60]:>9.3f}{t['vol_acf20']:>9.3f}"
            f"{t['vol_of_vol']:>9.3f}{100*t['pc1_share']:>6.0f}%{t['mean_kurtosis']:>7.1f}"
        )
    print("\nNOTE: pc1% here is measured over the FULL SAMPLE and was preserved even")
    print("when every training window was broken — which is exactly why the bug")
    print("survived. Full-sample statistics cannot validate a surrogate; the")
    print("window-local checks in tests/test_null_control.py are what can.")


if __name__ == "__main__":
    sys.exit(main())
