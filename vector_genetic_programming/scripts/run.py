#!/usr/bin/env python3
"""VGP global experiment runner.

Loads parquet data, builds feature matrix, runs walk-forward evolution
across all windows and seeds, prints results, and writes results/ artifacts.

Usage
-----
    make start
    python scripts/run.py

Output
------
    results/results.csv          — per (window, seed) IS/OOS Sharpe + DSR
    results/pareto_front.png     — Pareto scatter for best window
    results/equity_curves.png    — IS/OOS equity curves for top 3 individuals
    results/tree_graph.png       — GP tree for best individual overall
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config — edit these to scale up or point at different data
# ---------------------------------------------------------------------------

CACHE_DIR     = Path("data_pipeline_example/cache")
RESULTS_DIR   = Path("results")
SEEDS         = [0, 1, 2]
POP_SIZE      = 400       # individuals per generation (raise to 200+ for real runs)
N_GENERATIONS = 100      # generations per seed (raise to 50+ for real runs)
N_JOBS        = max(1, (os.cpu_count() or 2) - 1)
FEE_BPS       = 10.0
MIN_TRADES    = 50

# ---------------------------------------------------------------------------
# Logging — INFO for setup steps, suppressed during evolution (tqdm handles it)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
_setup_log = logging.getLogger("vgp.run")
_setup_log.setLevel(logging.INFO)
_setup_handler = logging.StreamHandler(sys.stdout)
_setup_handler.setFormatter(logging.Formatter("  %(message)s"))
_setup_log.addHandler(_setup_handler)
_setup_log.propagate = False


def _banner(text: str) -> None:
    width = 62
    print(f"\n{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}")


def _section(text: str) -> None:
    print(f"\n  ── {text}")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    _banner("1 / 5  Loading OHLCV data")
    print(f"  Cache: {CACHE_DIR.resolve()}")

    from vgp.data import DataLoader, FeatureEngine
    loader = DataLoader(cache_dir=CACHE_DIR)
    ohlcv = loader.fetch_ohlcv(start_date="2024-01-01", end_date="2026-04-01")

    # ------------------------------------------------------------------
    # 2. Build feature matrix
    # ------------------------------------------------------------------
    _banner("2 / 5  Building feature matrix")
    fe = FeatureEngine()
    fm = fe.fit_transform(ohlcv)

    import pandas as pd
    close_prices = pd.DataFrame(
        {ticker: ohlcv[ticker]["close"] for ticker in fe.retained_assets_}
    ).reindex(fe.dates_).ffill(limit=3)

    print(
        f"  Shape    {fm.shape}  (timesteps × features × assets)\n"
        f"  Assets   {len(fe.retained_assets_)}  →  {fe.retained_assets_}\n"
        f"  Dates    {fe.dates_.min().date()}  →  {fe.dates_.max().date()}"
    )

    # ------------------------------------------------------------------
    # 3. Generate walk-forward windows
    # ------------------------------------------------------------------
    _banner("3 / 5  Walk-forward windows")
    from vgp.analysis import generate_windows

    total_start = str(fe.dates_.min().date())
    total_end   = str(fe.dates_.max().date())
    windows = generate_windows(total_start, total_end)

    if not windows:
        print(f"  ERROR: no windows from {total_start} → {total_end}")
        print("  Need at least 17 months (12m train + 2m val + 3m OOS).")
        sys.exit(1)

    for w in windows:
        print(
            f"  W{w.window_id}  train → {w.train_end}"
            f"   OOS {w.test_start} → {w.test_end}"
        )

    # ------------------------------------------------------------------
    # 4. Run walk-forward evolution
    # ------------------------------------------------------------------
    _banner(f"4 / 5  Evolution  ({len(windows)} windows × {len(SEEDS)} seeds × {N_GENERATIONS} gen)")
    print(
        f"  pop={POP_SIZE}  jobs={N_JOBS}  fee={FEE_BPS}bps  min_trades={MIN_TRADES}\n"
    )

    from vgp.analysis.runner import WalkForwardRunner
    from vgp.backtest.runner import EvalConfig

    eval_cfg = EvalConfig(fee_bps=FEE_BPS, min_trades=MIN_TRADES)
    evo_kwargs = dict(
        pop_size=POP_SIZE,
        n_generations=N_GENERATIONS,
        n_jobs=N_JOBS,
        checkpoint_freq=999,
    )
    n_trials = len(SEEDS) * len(windows)
    runner   = WalkForwardRunner(dates=fe.dates_)
    all_results: list[dict] = []

    for w_idx, window in enumerate(windows):
        print(
            f"\n  Window {w_idx + 1}/{len(windows)}"
            f"  train→{window.train_end}"
            f"  OOS {window.test_start}→{window.test_end}"
        )

        window_results: list[dict] = []
        for seed in SEEDS:
            from vgp.evolution.config import EvolutionConfig
            from vgp.evolution.loop import run_evolution
            from vgp.data.splitter import WalkForwardSplitter
            from vgp.backtest.runner import EvalConfig as EC

            # Split feature matrix for this window
            splitter = WalkForwardSplitter()
            train_fm, _val_fm, test_fm = splitter.split(
                fm,
                train_end=window.train_end,
                val_start=window.val_start,
                val_end=window.val_end,
                test_start=window.test_start,
                dates=fe.dates_,
            )
            train_close = close_prices.loc[close_prices.index <= window.train_end].copy()
            test_close  = close_prices.loc[close_prices.index >= window.test_start].copy()
            train_eval_cfg = EC(fee_bps=FEE_BPS, min_trades=MIN_TRADES, close_prices=train_close)
            test_eval_cfg  = EC(fee_bps=FEE_BPS, min_trades=MIN_TRADES, close_prices=test_close)

            cfg = EvolutionConfig(seed=seed, **evo_kwargs)
            _pop, hof, _logbook = run_evolution(
                cfg, train_fm, train_eval_cfg,
                desc=f"  W{w_idx} seed{seed}",
            )

            # OOS evaluate
            from vgp.backtest.runner import evaluate
            oos_fitness = evaluate(hof[0], test_fm, test_eval_cfg) if hof else (-999.0, -999.0, 0)
            oos_sharpe  = float(oos_fitness[0])
            is_sharpe   = float(hof[0].fitness.values[0]) if hof else float("nan")

            # DSR
            from vgp.analysis.dsr import compute_dsr
            from vgp.analysis.runner import _get_is_returns
            try:
                is_returns = _get_is_returns(hof[0], train_fm, train_eval_cfg)
                dsr = compute_dsr(is_returns, sr_hat=is_sharpe, n_trials=n_trials)
            except Exception:
                dsr = 0.0

            result = {
                "window_id": window.window_id,
                "seed": seed,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_sharpe,
                "dsr": dsr,
                "n_nodes_best": len(hof[0]) if hof else 0,
            }
            window_results.append(result)
            all_results.append(result)

        # Per-window summary line
        from vgp.analysis import aggregate_seeds
        agg = aggregate_seeds(window_results)
        print(
            f"    → median OOS SR {agg['median_oos_sharpe']:+.3f}"
            f"  IQR {agg['iqr_oos_sharpe']:.3f}"
            f"  DSR {agg['median_dsr']:.3f}"
            f"  ({agg['n_seeds_positive_oos']}/{len(SEEDS)} seeds positive)"
        )

    # ------------------------------------------------------------------
    # 5. Save results + plots
    # ------------------------------------------------------------------
    _banner("5 / 5  Saving results & plots")

    from vgp.analysis import save_results_csv, aggregate_seeds
    csv_path = str(RESULTS_DIR / "results.csv")
    save_results_csv(all_results, csv_path)
    print(f"  results.csv  →  {csv_path}")

    # Best (window, seed) pair for plots
    best = max(all_results, key=lambda r: r["oos_sharpe"])
    best_window = windows[best["window_id"]]

    with tqdm(["pareto_front", "tree_graph", "equity_curves"], desc="  Plots", unit="plot", leave=True) as pbar:
        from vgp.analysis import plot_pareto_front, plot_equity_curves, plot_tree_graph
        from vgp.data.splitter import WalkForwardSplitter
        from vgp.evolution.config import EvolutionConfig
        from vgp.evolution.loop import run_evolution
        from vgp.backtest.runner import EvalConfig as EC

        splitter = WalkForwardSplitter()
        train_fm, _, _ = splitter.split(
            fm,
            train_end=best_window.train_end,
            val_start=best_window.val_start,
            val_end=best_window.val_end,
            test_start=best_window.test_start,
            dates=fe.dates_,
        )
        train_close = close_prices.loc[close_prices.index <= best_window.train_end].copy()
        train_cfg   = EC(fee_bps=FEE_BPS, min_trades=MIN_TRADES, close_prices=train_close)

        cfg = EvolutionConfig(seed=best["seed"], **evo_kwargs)
        _, hof, _ = run_evolution(cfg, train_fm, train_cfg, desc="  best re-run")

        pbar.update(0)
        plot_pareto_front(hof, str(RESULTS_DIR / "pareto_front.png"))
        pbar.update(1)

        if hof:
            plot_tree_graph(hof[0], str(RESULTS_DIR / "tree_graph.png"))
        pbar.update(1)

        full_cfg = EC(fee_bps=FEE_BPS, min_trades=MIN_TRADES, close_prices=close_prices)
        plot_equity_curves(
            list(hof[: min(3, len(hof))]), fm, full_cfg,
            best_window.train_end,
            str(RESULTS_DIR / "equity_curves.png"),
        )
        pbar.update(1)

    print(f"\n  Saved to {RESULTS_DIR}/\n")

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    _banner("Results")
    print(f"  {'Win':<4} {'Seed':<5} {'IS SR':>7} {'OOS SR':>8} {'DSR':>6} {'Nodes':>6}")
    print(f"  {'-'*4} {'-'*5} {'-'*7} {'-'*8} {'-'*6} {'-'*6}")
    for r in all_results:
        print(
            f"  {r['window_id']:<4} {r['seed']:<5}"
            f" {r['is_sharpe']:>+7.3f} {r['oos_sharpe']:>+8.3f}"
            f" {r['dsr']:>6.3f} {r['n_nodes_best']:>6}"
        )
    print(f"  {'-'*4} {'-'*5} {'-'*7} {'-'*8} {'-'*6} {'-'*6}")
    for window in windows:
        w_res = [r for r in all_results if r["window_id"] == window.window_id]
        agg = aggregate_seeds(w_res)
        print(
            f"  W{window.window_id} aggregate"
            f"  median OOS SR {agg['median_oos_sharpe']:+.3f}"
            f"  DSR {agg['median_dsr']:.3f}"
            f"  {agg['n_seeds_positive_oos']}/{len(SEEDS)} positive"
        )
    print()


if __name__ == "__main__":
    main()
