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
import math
import os
import re
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

# Null control (see vgp/analysis/null_control.py). DSR cannot detect bias shared
# by every trial — a lookahead, a reused training window, a survivor-biased
# universe — so the pipeline is also run on signal-free surrogates to check it
# reports nothing there. This is the falsification test for the headline result.
#
# Cost is N_NULL_RUNS full experiments, which is why the null runs use fewer
# seeds: the statistic is the best Sharpe the SEARCH finds, so the null only has
# to represent the same procedure, not the same compute budget.
# A p-value cannot resolve below 1/(1+N_NULL_RUNS); 20 runs buys p >= 0.048.
# Set N_NULL_RUNS = 0 to skip, and then do not describe the result as validated.
N_NULL_RUNS   = 20
NULL_SEEDS    = [0]
NULL_BLOCK    = 20        # bootstrap block length in bars

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


def _fmt(value, spec: str) -> str:
    """Format a metric, rendering NaN/inf as 'n/a' rather than a fake number."""
    m = re.match(r"^[+\-]?(\d+)", spec)
    width = int(m.group(1)) if m else 0
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float("nan")
    if not math.isfinite(v):
        return f"{'n/a':>{width}}" if width else "n/a"
    return f"{v:{spec}}"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    _banner("1 / 6  Loading OHLCV data")
    print(f"  Cache: {CACHE_DIR.resolve()}")

    from vgp.data import DataLoader, FeatureEngine
    loader = DataLoader(cache_dir=CACHE_DIR)
    ohlcv = loader.fetch_ohlcv(start_date="2024-01-01", end_date="2026-04-01")

    # ------------------------------------------------------------------
    # 2. Build feature matrix
    # ------------------------------------------------------------------
    _banner("2 / 6  Building feature matrix")
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
    _banner("3 / 6  Walk-forward windows")
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
    _banner(f"4 / 6  Evolution  ({len(windows)} windows × {len(SEEDS)} seeds × {N_GENERATIONS} gen)")
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
    runner   = WalkForwardRunner(dates=fe.dates_)
    all_results: list[dict] = []

    # run_window() owns the split, the OOS evaluate and the reporting invariants
    # (no worst-fitness sentinel written out as a Sharpe). Do not re-implement it
    # here — a second copy of that logic is how the sentinel leak survived.
    for w_idx, window in enumerate(windows):
        print(
            f"\n  Window {w_idx + 1}/{len(windows)}"
            f"  train\u2192{window.train_end}"
            f"  OOS {window.test_start}\u2192{window.test_end}"
        )

        window_results = runner.run_window(
            window=window,
            feature_matrix=fm,
            close_prices=close_prices,
            base_eval_config=eval_cfg,
            seeds=SEEDS,
            evo_config_kwargs=evo_kwargs,
        )
        all_results.extend(window_results)

        # Per-window summary line (DSR is still NaN here — it needs the full
        # trial set, so it is filled in by attach_dsr() after every window runs)
        from vgp.analysis import aggregate_seeds
        agg = aggregate_seeds(window_results)
        print(
            f"    \u2192 median OOS SR {_fmt(agg['median_oos_sharpe'], '+.3f')}"
            f"  IQR {_fmt(agg['iqr_oos_sharpe'], '.3f')}"
            f"  ({agg['n_seeds_positive_oos']}/{agg['n_seeds_valid_oos']} valid seeds positive,"
            f" {agg['n_seeds_valid_oos']}/{agg['n_seeds']} measurable)"
        )

    # DSR must be computed across the WHOLE trial set — the multiple-testing
    # correction scales with the cross-sectional spread of trial Sharpes, which
    # is not knowable one row at a time.
    from vgp.analysis import attach_dsr
    attach_dsr(all_results)

    # ------------------------------------------------------------------
    # 5. Null control — run the same pipeline on signal-free surrogates
    # ------------------------------------------------------------------
    _banner(f"5 / 6  Null control  ({N_NULL_RUNS} signal-free runs)")
    null_result = None
    if N_NULL_RUNS < 1:
        print("  SKIPPED (N_NULL_RUNS = 0).")
        print("  Without it, a high DSR cannot distinguish real signal from bias")
        print("  shared by every trial. Do not report this run as validated.")
    else:
        from vgp.analysis import run_null_control

        def _null_experiment(null_fm, null_close, null_dates):
            """One full experiment on surrogate data — same code path as above."""
            null_runner = WalkForwardRunner(dates=null_dates)
            null_windows = generate_windows(
                str(null_dates.min().date()), str(null_dates.max().date())
            )
            rows: list[dict] = []
            for nw in null_windows:
                rows += null_runner.run_window(
                    window=nw,
                    feature_matrix=null_fm,
                    close_prices=null_close,
                    base_eval_config=EvalConfig(fee_bps=FEE_BPS, min_trades=MIN_TRADES),
                    seeds=NULL_SEEDS,
                    evo_config_kwargs=evo_kwargs,
                )
            return rows

        print(
            f"  {N_NULL_RUNS} runs x {len(NULL_SEEDS)} seed(s), "
            f"block={NULL_BLOCK} bars — this is the expensive part\n"
        )
        null_result = run_null_control(
            ohlcv=ohlcv,
            experiment_fn=_null_experiment,
            observed_results=all_results,
            n_runs=N_NULL_RUNS,
            block_size=NULL_BLOCK,
            seed=1000,
        )
        print(null_result.summary())

    # ------------------------------------------------------------------
    # 6. Save results + plots
    # ------------------------------------------------------------------
    _banner("6 / 6  Saving results & plots")

    from vgp.analysis import save_results_csv, aggregate_seeds
    csv_path = str(RESULTS_DIR / "results.csv")
    save_results_csv(all_results, csv_path)
    print(f"  results.csv  →  {csv_path}")

    null_path = RESULTS_DIR / "null_control.txt"
    if null_result is not None:
        null_path.write_text(
            null_result.summary()
            + "\n\nnull best IS Sharpe per run:  "
            + ", ".join(f"{v:+.4f}" for v in null_result.null_best_is_sharpe)
            + "\nnull best OOS Sharpe per run: "
            + ", ".join(f"{v:+.4f}" for v in null_result.null_best_oos_sharpe)
            + "\n"
        )
    else:
        null_path.write_text(
            "Null control SKIPPED (N_NULL_RUNS = 0).\n\n"
            "No falsification test was run, so this experiment cannot "
            "distinguish signal from bias shared across all trials.\n"
        )
    print(f"  null_control.txt  →  {null_path}")

    # Best (window, seed) pair for plots. NaN OOS Sharpe means "not measured" —
    # max() over NaN is undefined, so rank only measurable rows and fall back to
    # the best IS row when no seed produced a valid OOS measurement.
    measured = [r for r in all_results if math.isfinite(r["oos_sharpe"])]
    if measured:
        best = max(measured, key=lambda r: r["oos_sharpe"])
    else:
        print("  WARNING: no seed produced a measurable OOS Sharpe — "
              "plotting the best IS row instead")
        is_measured = [r for r in all_results if math.isfinite(r["is_sharpe"])]
        best = max(is_measured, key=lambda r: r["is_sharpe"]) if is_measured else all_results[0]
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
    hdr = f"  {'Win':<4} {'Seed':<5} {'IS SR':>7} {'OOS SR':>8} {'DSR':>6} {'Nodes':>6}  OOS status"
    rule = f"  {'-'*4} {'-'*5} {'-'*7} {'-'*8} {'-'*6} {'-'*6}  {'-'*16}"
    print(hdr)
    print(rule)
    for r in all_results:
        status = r.get("oos_status", "?")
        if status != "ok":
            n_tr, min_tr = r.get("oos_n_trades", "?"), r.get("oos_min_trades", "?")
            status = f"{status} ({n_tr}/{min_tr} trades)"
        print(
            f"  {r['window_id']:<4} {r['seed']:<5}"
            f" {_fmt(r['is_sharpe'], '+7.3f')} {_fmt(r['oos_sharpe'], '+8.3f')}"
            f" {_fmt(r['dsr'], '6.3f')} {r['n_nodes_best']:>6}  {status}"
        )
    print(rule)
    for window in windows:
        w_res = [r for r in all_results if r["window_id"] == window.window_id]
        agg = aggregate_seeds(w_res)
        print(
            f"  W{window.window_id} aggregate"
            f"  median OOS SR {_fmt(agg['median_oos_sharpe'], '+.3f')}"
            f"  DSR {_fmt(agg['median_dsr'], '.3f')}"
            f"  {agg['n_seeds_positive_oos']}/{agg['n_seeds_valid_oos']} valid positive"
            f"  ({agg['n_seeds_valid_oos']}/{agg['n_seeds']} measurable)"
        )
    print("\n  NaN / n/a = not measured (see OOS status), not a bad result.")

    r0 = all_results[0]
    print(
        f"\n  DSR trial set: {r0.get('dsr_n_trials', '?')} trials "
        f"({r0.get('dsr_trial_source', '?')}, {r0.get('dsr_n_evaluations', '?')} "
        f"evaluations); bests-only would use {r0.get('dsr_n_trials_bests', '?')}."
    )
    best_dsr = max(
        (r["dsr"] for r in all_results if math.isfinite(r.get("dsr", float("nan")))),
        default=float("nan"),
    )
    best_dsr_bests = max(
        (r["dsr_bests_only"] for r in all_results
         if math.isfinite(r.get("dsr_bests_only", float("nan")))),
        default=float("nan"),
    )
    print(
        f"  Best DSR {_fmt(best_dsr, '.4f')} (all evaluations, conservative)"
        f"  vs {_fmt(best_dsr_bests, '.4f')} (reported bests, optimistic)."
    )
    if (math.isfinite(best_dsr) and math.isfinite(best_dsr_bests)
            and best_dsr < 0.95 <= best_dsr_bests):
        print("  These straddle 0.95: significance depends on how trials are counted.")

    if null_result is not None:
        print()
        print(null_result.summary())
    print()


if __name__ == "__main__":
    main()
