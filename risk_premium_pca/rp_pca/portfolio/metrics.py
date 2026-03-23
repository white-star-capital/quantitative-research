"""
Performance metrics for portfolio return series.

All annualised figures assume a standard 252-trading-day year.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
) -> float:
    """
    Annualised Sharpe ratio.

    SR = (μ - rf) / σ  ×  √T

    Parameters
    ----------
    returns : ndarray of shape (T,)
        Per-period (e.g. daily) return series.
    risk_free_rate : float
        Annualised risk-free rate.
    trading_days : int
    """
    rf_daily = risk_free_rate / trading_days
    excess = returns - rf_daily
    if len(excess) == 0 or excess.std(ddof=1) == 0:
        return 0.0
    return (excess.mean() / excess.std(ddof=1)) * np.sqrt(trading_days)


def annualised_return(returns: np.ndarray, trading_days: int = 252) -> float:
    """Arithmetic annualised return (%)."""
    return float(returns.mean() * trading_days * 100)


def annualised_geometric_return(returns: np.ndarray, trading_days: int = 252) -> float:
    """
    Geometric annualised return assuming log returns.

    For simple returns: (1+r̄)^T - 1
    For log returns:    exp(r̄·T) - 1
    """
    # Assume log returns throughout (as used in ReturnProcessor)
    cumlog = returns.sum()
    cum_ret = np.exp(cumlog) - 1.0
    years = len(returns) / trading_days
    if years == 0:
        return 0.0
    return float(((1 + cum_ret) ** (1 / years) - 1) * 100)


def annualised_vol(returns: np.ndarray, trading_days: int = 252) -> float:
    """Annualised volatility (%)."""
    return float(returns.std(ddof=1) * np.sqrt(trading_days) * 100)


def max_drawdown(returns: np.ndarray) -> float:
    """
    Maximum drawdown (%) from the cumulative return series.
    """
    if len(returns) == 0:
        return 0.0
    cumulative = np.exp(np.cumsum(returns))  # wealth index (log returns)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - rolling_max) / rolling_max
    return float(drawdowns.min() * 100)


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
) -> float:
    """Sortino ratio (downside deviation in the denominator)."""
    rf_daily = risk_free_rate / trading_days
    excess = returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.inf
    downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(trading_days)
    return float(excess.mean() * trading_days / downside_std)


def calmar_ratio(returns: np.ndarray, trading_days: int = 252) -> float:
    """Calmar ratio = annualised return / |max drawdown|."""
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.inf
    return annualised_return(returns, trading_days) / mdd


# ---------------------------------------------------------------------------
# Unified metrics bundle
# ---------------------------------------------------------------------------

class PerformanceMetrics:
    """
    Compute and store all performance metrics for a return series.

    Parameters
    ----------
    returns : array-like
        Per-period return series.
    name : str
        Label for display.
    risk_free_rate : float
        Annualised risk-free rate.
    trading_days : int
    """

    def __init__(
        self,
        returns: Union[np.ndarray, pd.Series],
        name: str = "Portfolio",
        risk_free_rate: float = 0.05,
        trading_days: int = 252,
    ) -> None:
        self.name = name
        self.returns = np.asarray(returns, dtype=float)
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

        self._compute()

    def _compute(self) -> None:
        r = self.returns
        if len(r) == 0:
            self.sharpe = 0.0
            self.ann_arith_ret = 0.0
            self.ann_geo_ret = 0.0
            self.ann_vol = 0.0
            self.max_dd = 0.0
            self.sortino = 0.0
            self.calmar = 0.0
            self.n_obs = 0
            return
        td = self.trading_days
        rf = self.risk_free_rate
        self.sharpe = sharpe_ratio(r, rf, td)
        self.ann_arith_ret = annualised_return(r, td)
        self.ann_geo_ret = annualised_geometric_return(r, td)
        self.ann_vol = annualised_vol(r, td)
        self.max_dd = max_drawdown(r)
        self.sortino = sortino_ratio(r, rf, td)
        self.calmar = calmar_ratio(r, td)
        self.n_obs = len(r)

    def to_dict(self) -> dict:
        return {
            "Strategy": self.name,
            "Ann. Sharpe": round(self.sharpe, 3),
            "Ann. Arith. Ret (%)": round(self.ann_arith_ret, 2),
            "Ann. Geo. Ret (%)": round(self.ann_geo_ret, 2),
            "Ann. Vol (%)": round(self.ann_vol, 2),
            "Max DD (%)": round(self.max_dd, 2),
            "Sortino": round(self.sortino, 3),
            "N Obs": self.n_obs,
        }


def compute_metrics_table(
    return_series: dict[str, Union[np.ndarray, pd.Series]],
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Compute a metrics table for multiple strategies.

    Parameters
    ----------
    return_series : dict
        Mapping of strategy name → return array.

    Returns
    -------
    DataFrame with strategies as rows and metrics as columns.
    """
    rows = []
    for name, rets in return_series.items():
        pm = PerformanceMetrics(
            rets, name=name,
            risk_free_rate=risk_free_rate,
            trading_days=trading_days,
        )
        rows.append(pm.to_dict())
    df = pd.DataFrame(rows).set_index("Strategy")
    return df.sort_values("Ann. Sharpe", ascending=False)


def cumulative_returns(returns: np.ndarray) -> np.ndarray:
    """Cumulative log-return series (useful for plotting wealth index)."""
    return np.exp(np.cumsum(returns)) - 1.0


def rolling_sharpe(
    returns: np.ndarray,
    window: int = 63,
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
) -> np.ndarray:
    """Rolling annualised Sharpe ratio over a given window."""
    rf_daily = risk_free_rate / trading_days
    excess = returns - rf_daily
    result = np.full(len(returns), np.nan)
    for t in range(window, len(returns) + 1):
        e = excess[t - window: t]
        std = e.std(ddof=1)
        if std > 0:
            result[t - 1] = (e.mean() / std) * np.sqrt(trading_days)
    return result
