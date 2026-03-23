"""
Headless pipeline — run the full analysis without the Streamlit UI.

Usage (from the repo root):
    python rp_pca/scripts/run_pipeline.py

Or with custom dates:
    python rp_pca/scripts/run_pipeline.py --start 2022-01-01 --end 2024-12-31

Outputs
-------
Prints a metrics table to stdout and saves:
    rp_pca/results/in_sample_metrics.csv
    rp_pca/results/oos_returns.csv      # daily OOS returns per strategy
    rp_pca/results/oos_cumulative.csv   # wealth index per strategy
    rp_pca/results/oos_metrics.csv      # OOS performance metrics
    rp_pca/results/oos_rebalance_log.csv
    rp_pca/results/variance_table.csv
    rp_pca/results/factor_sharpe.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rp_pca.config import Config
from rp_pca.data.fetcher import BinanceFetcher
from rp_pca.data.processor import ReturnProcessor, equal_weighted_returns
from rp_pca.data.tao_subnet_loader import load_tao_subnet_prices, load_tao_subnet_market_caps
from rp_pca.models.rp_pca import RPPCA
from rp_pca.models.covariance import get_cov_estimator
from rp_pca.portfolio.construction import PortfolioConstructor
from rp_pca.portfolio.metrics import compute_metrics_table
from rp_pca.backtest.engine import WalkForwardBacktest
from rp_pca.robustness.bootstrap import bootstrap_sharpe_comparison
from rp_pca.robustness.regimes import classify_regimes, compute_regime_metrics
from rp_pca.robustness.fama_macbeth import compare_fama_macbeth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="RP-PCA crypto portfolio pipeline")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--gamma", type=float, default=None,
                   help="RP-PCA penalty γ. Default: auto (=T)")
    p.add_argument("--n-components", type=int, default=5)
    p.add_argument("--cov-window", type=int, default=252)
    p.add_argument("--mean-window", type=int, default=63)
    p.add_argument("--no-bootstrap", action="store_true")
    p.add_argument("--no-oos", action="store_true")
    p.add_argument(
        "--data-source",
        choices=["binance", "tao_subnets"],
        default="binance",
        help="Price source: Binance USDT pairs or Taostats subnet CSVs",
    )
    p.add_argument(
        "--tao-subnet-dir",
        default=None,
        help="Directory with sn*_tao_daily_candles.csv / tao_subnets_wide.parquet (default: rp_pca/data/cache/tao_subnets)",
    )
    p.add_argument(
        "--buy-hold-benchmark",
        default=None,
        metavar="TICKER",
        help="Buy-and-hold benchmark (default: BTC for binance, TAO for tao_subnets; flat zeros if TAO not a column)",
    )
    p.add_argument(
        "--no-buy-hold-benchmark",
        action="store_true",
        help="Omit single-asset buy-and-hold benchmark from OOS results",
    )
    p.add_argument(
        "--tao-min-obs-fraction",
        type=float,
        default=None,
        help="Override min_obs_fraction for TAO data (default: 0.50)",
    )
    p.add_argument("--no-fama-macbeth", action="store_true",
                   help="Skip Fama-MacBeth cross-sectional tests")
    p.add_argument("--no-regimes", action="store_true",
                   help="Skip episodic regime analysis")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.data.start_date = args.start
    cfg.data.end_date = args.end
    cfg.model.gamma = args.gamma
    cfg.model.n_components = args.n_components
    cfg.backtest.cov_window = args.cov_window
    cfg.backtest.mean_window = args.mean_window
    cfg.data.data_source = args.data_source
    if args.tao_subnet_dir is not None:
        cfg.data.tao_subnet_csv_dir = Path(args.tao_subnet_dir).expanduser().resolve()

    if args.no_buy_hold_benchmark:
        cfg.backtest.buy_hold_benchmark = None
    elif args.buy_hold_benchmark is not None:
        cfg.backtest.buy_hold_benchmark = args.buy_hold_benchmark
    elif args.data_source == "tao_subnets":
        cfg.backtest.buy_hold_benchmark = "TAO"
    else:
        cfg.backtest.buy_hold_benchmark = "BTC"

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # ─── 1. Data ──────────────────────────────────────────────────────
    logger.info("Loading data %s → %s (%s)", cfg.data.start_date, cfg.data.end_date, cfg.data.data_source)
    if cfg.data.data_source == "binance":
        fetcher = BinanceFetcher(cache_dir=cfg.data.cache_dir)
        prices = fetcher.fetch_all(cfg.data.start_date, cfg.data.end_date)
    else:
        prices = load_tao_subnet_prices(
            cfg.data.tao_subnet_csv_dir,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
        )

    # Use relaxed min_obs for TAO subnets
    min_obs = cfg.data.min_obs_fraction
    if cfg.data.data_source == "tao_subnets":
        min_obs = (
            args.tao_min_obs_fraction
            if args.tao_min_obs_fraction is not None
            else cfg.data.tao_min_obs_fraction
        )

    processor = ReturnProcessor(
        winsorize_lower=cfg.data.winsorize_lower,
        winsorize_upper=cfg.data.winsorize_upper,
        min_obs_fraction=min_obs,
    )
    returns = processor.fit_transform(prices)
    X = returns.values
    logger.info("Return matrix: %d × %d", *X.shape)

    # Load market caps for value-weighted benchmark (TAO only)
    vw_mcaps = None
    if cfg.data.data_source == "tao_subnets":
        try:
            vw_mcaps = load_tao_subnet_market_caps(
                cfg.data.tao_subnet_csv_dir,
                start_date=cfg.data.start_date,
                end_date=cfg.data.end_date,
            )
            logger.info("Loaded market caps: %d × %d", *vw_mcaps.shape)
        except Exception as e:
            logger.warning("Could not load TAO market caps: %s", e)

    # ─── 2. In-sample ─────────────────────────────────────────────────
    logger.info("Fitting PCA and RP-PCA (in-sample)…")
    estimator = get_cov_estimator(cfg.model.cov_method)
    Sigma, mu = estimator(X)

    pca = RPPCA(n_components=cfg.model.n_components, gamma=1.0)
    pca.fit(X, cov_matrix=Sigma, mean_vector=mu)

    rppca = RPPCA(n_components=cfg.model.n_components, gamma=cfg.model.gamma)
    rppca.fit(X, cov_matrix=Sigma, mean_vector=mu)

    # Variance tables
    vt = pd.concat(
        [pca.variance_table().add_prefix("PCA_"),
         rppca.variance_table().add_prefix("RPPCA_")],
        axis=1,
    )
    vt.to_csv(results_dir / "variance_table.csv")
    logger.info("Variance table:\n%s", vt.to_string())

    # Factor Sharpe
    fs = pd.concat(
        [pca.factor_sharpe().add_prefix("PCA_"),
         rppca.factor_sharpe().add_prefix("RPPCA_")],
        axis=1,
    )
    fs.to_csv(results_dir / "factor_sharpe.csv")
    logger.info("Factor Sharpe:\n%s", fs.to_string())

    # In-sample portfolio metrics
    F_pca = X @ pca.loadings_
    F_rp = X @ rppca.loadings_
    pc_pca = PortfolioConstructor(pca.loadings_, F_pca)
    pc_rp = PortfolioConstructor(rppca.loadings_, F_rp)

    is_rets = {
        "RP-PCA Tangency": pc_rp.tangency_returns(),
        "RP-PCA Min-Var":  pc_rp.min_var_returns(),
        "PCA Tangency":    pc_pca.tangency_returns(),
        "PCA Min-Var":     pc_pca.min_var_returns(),
        "Equal-Weight":    equal_weighted_returns(returns).values,
    }
    if "BTC" in returns.columns:
        is_rets["BTC"] = returns["BTC"].values

    is_metrics = compute_metrics_table(is_rets)
    is_metrics.to_csv(results_dir / "in_sample_metrics.csv")
    logger.info("\n=== IN-SAMPLE METRICS ===\n%s", is_metrics.to_string())

    # ─── 3. OOS Walk-forward ──────────────────────────────────────────
    if not args.no_oos:
        logger.info("Running walk-forward OOS backtest…")
        bt = WalkForwardBacktest(
            returns=returns,
            cov_window=cfg.backtest.cov_window,
            mean_window=cfg.backtest.mean_window,
            rebalance_days=cfg.backtest.rebalance_days,
            n_components=cfg.model.n_components,
            gamma=cfg.model.gamma,
            cov_method=cfg.model.cov_method,
            buy_hold_benchmark=cfg.backtest.buy_hold_benchmark,
            vw_benchmark_mcaps=vw_mcaps,
        )
        bt_results = bt.run(include_pca=True, include_benchmarks=True)
        oos_metrics = bt_results.metrics(
            risk_free_rate=cfg.portfolio.risk_free_rate,
            trading_days=cfg.portfolio.trading_days,
        )
        bt_results.save_to_csv(
            results_dir,
            prefix="oos",
            risk_free_rate=cfg.portfolio.risk_free_rate,
            trading_days=cfg.portfolio.trading_days,
        )
        logger.info("\n=== OOS METRICS ===\n%s", oos_metrics.to_string())

        # ─── 4. Bootstrap ─────────────────────────────────────────────
        if not args.no_bootstrap:
            rs = bt_results.return_series
            if "RP-PCA Tangency" in rs and "PCA Tangency" in rs:
                logger.info("Running circular block bootstrap (n=1000)…")
                common = rs["RP-PCA Tangency"].index.intersection(rs["PCA Tangency"].index)
                boot = bootstrap_sharpe_comparison(
                    rs["RP-PCA Tangency"].loc[common].values,
                    rs["PCA Tangency"].loc[common].values,
                    n_reps=1000,
                    verbose=True,
                )
                logger.info("\n%s", boot.summary())

        # ─── 5. Regime analysis ────────────────────────────────────────
        if not args.no_regimes:
            logger.info("Running episodic regime analysis…")
            regime_labels = classify_regimes(
                returns, config=cfg.regime, trading_days=cfg.portfolio.trading_days,
            )
            sharpe_dict, regime_df = compute_regime_metrics(
                bt_results.return_series,
                regime_labels,
                risk_free_rate=cfg.portfolio.risk_free_rate,
                trading_days=cfg.portfolio.trading_days,
            )
            regime_df.to_csv(results_dir / "regime_metrics.csv")
            logger.info("\n=== REGIME METRICS ===\n%s", regime_df.to_string())

    # ─── 6. Fama-MacBeth cross-sectional tests ─────────────────────
    if not args.no_fama_macbeth:
        logger.info("Running Fama-MacBeth cross-sectional tests…")
        F_rp_fm = X @ rppca.loadings_
        F_pca_fm = X @ pca.loadings_
        rp_fm, pca_fm = compare_fama_macbeth(
            X, F_rp_fm, F_pca_fm,
            shanken_correction=cfg.fama_macbeth.shanken_correction,
            split_frac=cfg.fama_macbeth.split_frac,
        )
        # Save results
        fm_summary = pd.concat(
            [rp_fm.summary_df().add_prefix("RPPCA_"),
             pca_fm.summary_df().add_prefix("PCA_")],
            axis=1,
        )
        fm_summary.to_csv(results_dir / "fama_macbeth_results.csv")
        logger.info("\n=== FAMA-MACBETH ===")
        logger.info("\n%s", rp_fm.summary())
        logger.info("\n%s", pca_fm.summary())

    logger.info("Done. Results saved to %s", results_dir)


if __name__ == "__main__":
    main()
