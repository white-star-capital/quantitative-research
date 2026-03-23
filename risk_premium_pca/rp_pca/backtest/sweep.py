"""
Full parameter sweep for the walk-forward backtest.

Iterates over a Cartesian product of all configured parameter grids and runs
one WalkForwardBacktest per combination.  Results are collected into a
flat DataFrame with one row per scenario (param columns + flattened metrics).

Usage
-----
    from rp_pca.backtest.sweep import run_parameter_sweep

    for progress, total, params, row in run_parameter_sweep(returns, config):
        print(f"{progress}/{total}  {params}")   # update UI progress bar

    # Or collect all at once:
    rows = list(run_parameter_sweep(returns, config))
    df = build_sweep_dataframe(rows)
"""
from __future__ import annotations

import itertools
import logging
from typing import Generator, Optional

import numpy as np
import pandas as pd

from .engine import WalkForwardBacktest
from ..config import Config

logger = logging.getLogger(__name__)

# Metrics to extract from each BacktestResults.metrics() DataFrame
METRICS_TO_CAPTURE = [
    "Ann. Sharpe",
    "Ann. Arith. Ret (%)",
    "Ann. Geo. Ret (%)",
    "Ann. Vol (%)",
    "Max DD (%)",
    "Sortino",
]

# Strategies to capture metrics for
STRATEGIES_TO_CAPTURE = [
    "RP-PCA Tangency",
    "RP-PCA Min-Var",
    "PCA Tangency",
    "PCA Min-Var",
    "alpha_concentrated_RP-PCA-Tangency",
    "alpha_concentrated_RP-PCA-Min-Var",
    "alpha_concentrated_PCA-Tangency",
    "alpha_concentrated_PCA-Min-Var",
]

# Column prefix map (strategy name → short prefix used in output columns)
_STRAT_PREFIX = {
    "RP-PCA Tangency": "rp_tan",
    "RP-PCA Min-Var":  "rp_mv",
    "PCA Tangency":    "pca_tan",
    "PCA Min-Var":     "pca_mv",
    "alpha_concentrated_RP-PCA-Tangency": "alpha_conc_rp_tan",
    "alpha_concentrated_RP-PCA-Min-Var":  "alpha_conc_rp_mv",
    "alpha_concentrated_PCA-Tangency":    "alpha_conc_pca_tan",
    "alpha_concentrated_PCA-Min-Var":     "alpha_conc_pca_mv",
}

# Metric label → output column suffix
_METRIC_SUFFIX = {
    "Ann. Sharpe":         "sharpe",
    "Ann. Arith. Ret (%)": "arith_ret",
    "Ann. Geo. Ret (%)":   "geo_ret",
    "Ann. Vol (%)":        "vol",
    "Max DD (%)":          "max_dd",
    "Sortino":             "sortino",
}

PARAM_COLS = [
    "cov_window",
    "mean_window",
    "rebalance_days",
    "n_components",
    "gamma",
    "cov_method",
]


def _metric_col(strategy: str, metric: str) -> str:
    """Return the flat column name for a (strategy, metric) pair."""
    prefix = _STRAT_PREFIX.get(strategy, strategy.lower().replace(" ", "_"))
    suffix = _METRIC_SUFFIX.get(metric, metric.lower().replace(" ", "_"))
    return f"{prefix}_{suffix}"


def metric_col(strategy: str, metric: str) -> str:
    """Public accessor: column name for a (strategy, metric) pair."""
    return _metric_col(strategy, metric)


def _build_param_grid(config: Config) -> list[dict]:
    """Build the full (filtered) parameter grid from Config."""
    bc = config.backtest
    mc = config.model

    # γ grid: config stores floats; we add None (= auto) as a distinct sentinel
    gamma_sweep_grid = mc.gamma_grid + [None]  # type: ignore[operator]

    combos = itertools.product(
        bc.cov_window_grid,
        bc.mean_window_grid,
        bc.rebalance_days_grid,
        mc.n_components_grid,
        gamma_sweep_grid,
        mc.cov_method_grid,
    )

    grid = []
    for cov_window, mean_window, rebalance_days, n_components, gamma, cov_method in combos:
        if mean_window > cov_window:
            continue  # mean window must be ≤ cov window
        grid.append({
            "cov_window":    cov_window,
            "mean_window":   mean_window,
            "rebalance_days": rebalance_days,
            "n_components":  n_components,
            "gamma":         gamma,
            "cov_method":    cov_method,
        })
    return grid


def run_parameter_sweep(
    returns: pd.DataFrame,
    config: Config,
    *,
    include_pca: bool = True,
    include_benchmarks: bool = False,
) -> Generator[tuple[int, int, dict, dict], None, None]:
    """
    Generator that runs a walk-forward backtest for every combination in the
    parameter grid derived from ``config``.

    Yields
    ------
    (completed, total, params, metrics_row)
        - ``completed``: 1-based index of the current scenario.
        - ``total``: total number of scenarios.
        - ``params``: dict of the parameter values used.
        - ``metrics_row``: flat dict with all captured metrics (or NaN on failure).

    Example
    -------
    ::

        rows = []
        for completed, total, params, row in run_parameter_sweep(returns, cfg):
            rows.append(row)

        df = pd.DataFrame(rows)
    """
    grid = _build_param_grid(config)
    total = len(grid)
    rf = config.portfolio.risk_free_rate
    td = config.portfolio.trading_days

    for i, params in enumerate(grid, start=1):
        row: dict = {**params}

        try:
            bt = WalkForwardBacktest(
                returns=returns,
                cov_window=params["cov_window"],
                mean_window=params["mean_window"],
                rebalance_days=params["rebalance_days"],
                n_components=params["n_components"],
                gamma=params["gamma"],
                cov_method=params["cov_method"],
                ewma_cov_halflife=config.model.ewma_cov_halflife,
                ewma_mean_halflife=config.model.ewma_mean_halflife,
                risk_free_rate=rf,
                trading_days=td,
                target_gross_leverage=config.backtest.target_gross_leverage,
                concentrated_top_n=config.backtest.concentrated_top_n,
                buy_hold_benchmark=config.backtest.buy_hold_benchmark,
            )
            result = bt.run(
                include_pca=include_pca,
                include_benchmarks=include_benchmarks,
            )
            metrics_df = result.metrics(risk_free_rate=rf, trading_days=td)

            for strat in STRATEGIES_TO_CAPTURE:
                for metric in METRICS_TO_CAPTURE:
                    col = _metric_col(strat, metric)
                    if strat in metrics_df.index and metric in metrics_df.columns:
                        row[col] = float(metrics_df.loc[strat, metric])
                    else:
                        row[col] = np.nan

        except Exception as exc:
            logger.warning("Sweep scenario %d failed (%s): %s", i, params, exc)
            for strat in STRATEGIES_TO_CAPTURE:
                for metric in METRICS_TO_CAPTURE:
                    row[_metric_col(strat, metric)] = np.nan

        yield i, total, params, row


def build_sweep_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Convert a list of metric-row dicts (from ``run_parameter_sweep``) to a DataFrame."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Put param columns first
    metric_cols = [c for c in df.columns if c not in PARAM_COLS]
    return df[PARAM_COLS + metric_cols]


def top_n_configs(
    sweep_df: pd.DataFrame,
    strategy: str,
    metric: str,
    n: int = 5,
    ascending: bool | None = None,
) -> pd.DataFrame:
    """
    Return the top-N rows for the chosen (strategy, metric) combination.

    Parameters
    ----------
    sweep_df : DataFrame
        Output of ``build_sweep_dataframe``.
    strategy : str
        One of ``STRATEGIES_TO_CAPTURE``.
    metric : str
        One of ``METRICS_TO_CAPTURE``.
    n : int
        Number of rows to return.
    ascending : bool or None
        Sort order.  If None, infers from the metric:
        ``True`` for "Max DD (%)" and "Ann. Vol (%)" (lower is better),
        ``False`` for all others (higher is better).
    """
    col = _metric_col(strategy, metric)
    if col not in sweep_df.columns:
        raise KeyError(f"Column '{col}' not found in sweep results.")

    if ascending is None:
        ascending = metric in {"Max DD (%)", "Ann. Vol (%)"}

    return (
        sweep_df.dropna(subset=[col])
        .sort_values(col, ascending=ascending)
        .head(n)
        .reset_index(drop=True)
    )
