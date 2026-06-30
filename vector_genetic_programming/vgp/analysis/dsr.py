"""Deflated Sharpe Ratio (DSR) implementation.

Reference: Bailey, D.H. & Lopez de Prado, M. (2014).
"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
Overfitting and Non-Normality." Journal of Portfolio Management, 40(5), 94-107.

Formula: DSR = Phi((SR_hat - E[SR_max]) / sqrt(Var(SR)))

where E[SR_max] is the expected maximum SR under H0 (Proposition 3)
and Var(SR) accounts for non-normality (skewness, excess kurtosis).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

logger = logging.getLogger(__name__)

# Euler-Mascheroni constant
_EULER_GAMMA = 0.5772156649


def compute_dsr(
    returns: np.ndarray,
    sr_hat: float,
    n_trials: int,
    periods_per_year: int = 252,
) -> float:
    """Deflated Sharpe Ratio — probability that SR_hat exceeds E[max SR under H0].

    Parameters
    ----------
    returns : np.ndarray
        Per-period IS portfolio returns, shape [T]. Use pf.returns().to_numpy().
    sr_hat : float
        Annualized IS Sharpe ratio (from logbook or pf.sharpe_ratio()).
    n_trials : int
        Number of independent trials (seeds x windows) — used to compute E[SR_max].
    periods_per_year : int
        252 for daily data.

    Returns
    -------
    float
        DSR in [0.0, 1.0]. Values above 0.95 indicate statistical significance.
        Returns 0.0 if returns array is flat (std == 0) — no evidence of skill.
    """
    if np.std(returns) == 0.0:
        logger.debug("compute_dsr: flat returns (std=0) — returning 0.0")
        return 0.0

    T = len(returns)
    if T < 2:
        logger.debug("compute_dsr: too few observations (%d) — returning 0.0", T)
        return 0.0

    n_trials = max(1, n_trials)

    # Expected maximum SR under H0 (Bailey & Lopez de Prado 2014, Proposition 3)
    expected_max_sr = (
        (1 - _EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
        + _EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))
    )

    # De-annualize: formula requires per-period SR (Bailey & Lopez de Prado 2014, eq. 5)
    # sr_hat from vectorbt is annualized (daily × sqrt(252)); undo the scaling.
    sr_hat_pp = sr_hat / np.sqrt(periods_per_year)

    # Return distribution moments
    ret_skew = float(skew(returns))
    # kurtosis(fisher=True) returns EXCESS kurtosis — matches DSR formula
    ret_kurt = float(kurtosis(returns, fisher=True))

    # Variance of SR estimate (non-normality adjustment, per-period units)
    sr_var = (
        1 + (0.5 * sr_hat_pp**2) - ret_skew * sr_hat_pp + ((ret_kurt / 4) * sr_hat_pp**2)
    ) / (T - 1)

    if sr_var <= 0.0:
        logger.debug("compute_dsr: non-positive sr_var (%f) — returning 0.0", sr_var)
        return 0.0

    dsr = float(norm.cdf((sr_hat_pp - expected_max_sr) / np.sqrt(sr_var)))
    return float(np.clip(dsr, 0.0, 1.0))


def aggregate_seeds(seed_results: list[dict]) -> dict:
    """Aggregate per-seed results for one window into summary statistics.

    Parameters
    ----------
    seed_results : list[dict]
        Each dict has at minimum: oos_sharpe, dsr.

    Returns
    -------
    dict
        Keys: median_oos_sharpe, iqr_oos_sharpe, median_dsr, n_seeds_positive_oos
    """
    if not seed_results:
        return {
            "median_oos_sharpe": float("nan"),
            "iqr_oos_sharpe": float("nan"),
            "median_dsr": float("nan"),
            "n_seeds_positive_oos": 0,
        }

    oos_sharpes = np.array([r["oos_sharpe"] for r in seed_results])
    dsrs = np.array([r["dsr"] for r in seed_results])

    q25 = float(np.percentile(oos_sharpes, 25))
    q75 = float(np.percentile(oos_sharpes, 75))

    return {
        "median_oos_sharpe": float(np.median(oos_sharpes)),
        "iqr_oos_sharpe": float(q75 - q25),
        "median_dsr": float(np.median(dsrs)),
        "n_seeds_positive_oos": int(np.sum(oos_sharpes > 0)),
    }


def save_results_csv(results: list[dict], path: str) -> None:
    """Save per-(window, seed) results to CSV.

    Parameters
    ----------
    results : list[dict]
        Each dict is one (window_id, seed) pair with keys:
        window_id, seed, train_end, test_start, test_end,
        is_sharpe, oos_sharpe, dsr, n_nodes_best
    path : str
        Output file path. Parent directory must exist.
    """
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    logger.info("Results saved to %s (%d rows)", path, len(df))
