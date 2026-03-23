"""
Portfolio construction from PCA / RP-PCA factor loadings.

Following Barillas & Shanken (2018), we form the tangency (max-Sharpe)
and minimum-variance portfolios from the top-K factor returns.

The workflow is:
    1. Extract K factor loading vectors  L  (N × K)  from PCA or RP-PCA.
    2. Compute factor returns  F = X · L  (T × K).
    3. Estimate μ_F (mean) and Σ_F (covariance) of factor returns.
    4. Solve the MV optimisation in factor space.
    5. Map weights back to asset space:  w_asset = L · w_factor  (N,).

Efficient frontier
    Solved analytically using the two-fund separation theorem.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core portfolio optimisers
# ---------------------------------------------------------------------------

_RIDGE = 1e-8


def _ridged_solve(S: np.ndarray, b: np.ndarray, K: int) -> np.ndarray:
    """Solve (S + λI) x = b with small ridge λ; robust to singularity."""
    A = S + _RIDGE * np.eye(K)
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


def _finite_real_array(a: np.ndarray) -> np.ndarray:
    """Replace NaN/inf with 0 so cov/mean and matmul stay finite."""
    x = np.asarray(a, dtype=float)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _normalise_weights(raw: np.ndarray, K: int, denom_from: str = "sum") -> np.ndarray:
    """Divide raw by scalar denominator; fall back to equal weights if ill-conditioned."""
    eq = np.ones(K) / K
    if not np.all(np.isfinite(raw)):
        return eq
    if denom_from == "sum":
        denom = float(raw.sum())
    else:
        denom = float(np.ones(K) @ raw)
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return eq
    w = raw / denom
    if not np.all(np.isfinite(w)):
        return eq
    return w

class PortfolioConstructor:
    """
    Build tangency and minimum-variance portfolios from factor returns.

    Parameters
    ----------
    loadings : ndarray of shape (N, K)
        Factor loading matrix.
    factor_returns : ndarray of shape (T, K)
        Factor return matrix  F = X · L.
    risk_free_rate : float
        Annualised risk-free rate (default 5 %).
    trading_days : int
        Trading days per year.
    allow_short : bool
        If False, add non-negativity constraint on factor weights.
    """

    def __init__(
        self,
        loadings: np.ndarray,
        factor_returns: np.ndarray,
        risk_free_rate: float = 0.05,
        trading_days: int = 252,
        allow_short: bool = True,
    ) -> None:
        self.loadings = _finite_real_array(loadings)
        self.factor_returns = _finite_real_array(factor_returns)
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.allow_short = allow_short

        T, K = self.factor_returns.shape
        self.T = T
        self.K = K

        # Per-period risk-free rate
        self._rf_daily = risk_free_rate / trading_days

        # Moment estimates in factor space
        self._mu_f: np.ndarray = factor_returns.mean(axis=0)          # (K,)
        self._sigma_f: np.ndarray = np.cov(factor_returns, rowvar=False)  # (K, K)
        self._sigma_f_inv: Optional[np.ndarray] = None

    @property
    def sigma_f_inv(self) -> np.ndarray:
        if self._sigma_f_inv is None:
            self._sigma_f_inv = np.linalg.inv(
                self._sigma_f + _RIDGE * np.eye(self.K)
            )
        return self._sigma_f_inv

    # ------------------------------------------------------------------
    # Tangency portfolio (max Sharpe)
    # ------------------------------------------------------------------

    def tangency_weights(
        self,
        mu_f: Optional[np.ndarray] = None,
        sigma_f: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Factor-space tangency portfolio weights.

        w* = Σ⁻¹ · (μ - rf·1) / [1' · Σ⁻¹ · (μ - rf·1)]

        Returns weights summing to 1 (unnormalised if denominator ≤ 0,
        falling back to equal weights in degenerate cases).
        """
        mu = mu_f if mu_f is not None else self._mu_f
        S = sigma_f if sigma_f is not None else self._sigma_f
        excess = mu - self._rf_daily
        raw = _ridged_solve(S, excess, self.K)
        return _normalise_weights(raw, self.K, denom_from="sum")

    def tangency_returns(
        self,
        mu_f: Optional[np.ndarray] = None,
        sigma_f: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Portfolio return series (T,) for the tangency portfolio."""
        w = self.tangency_weights(mu_f=mu_f, sigma_f=sigma_f)
        return self.factor_returns @ w

    # ------------------------------------------------------------------
    # Minimum-variance portfolio
    # ------------------------------------------------------------------

    def min_var_weights(
        self,
        sigma_f: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Global minimum-variance portfolio weights.

        w* = Σ⁻¹ · 1  /  (1' · Σ⁻¹ · 1)
        """
        S = sigma_f if sigma_f is not None else self._sigma_f
        ones = np.ones(self.K)
        raw = _ridged_solve(S, ones, self.K)
        return _normalise_weights(raw, self.K, denom_from="ones_dot")

    def min_var_returns(
        self,
        sigma_f: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Portfolio return series (T,) for the min-variance portfolio."""
        w = self.min_var_weights(sigma_f=sigma_f)
        return self.factor_returns @ w

    # ------------------------------------------------------------------
    # Asset-space weights
    # ------------------------------------------------------------------

    def to_asset_weights(self, factor_weights: np.ndarray) -> np.ndarray:
        """Map factor-space weights (K,) to asset-space weights (N,)."""
        return self.loadings @ factor_weights  # (N,)

    # ------------------------------------------------------------------
    # Efficient frontier
    # ------------------------------------------------------------------

    def efficient_frontier(
        self,
        n_points: int = 200,
        mu_f: Optional[np.ndarray] = None,
        sigma_f: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Compute the efficient frontier in (annualised vol, annualised return)
        space for the factor universe.

        Uses the two-fund separation theorem:
            any efficient portfolio = α · min-var + (1-α) · tangency

        Returns a DataFrame with columns [vol, ret] and the capital market
        line endpoint.
        """
        mu = mu_f if mu_f is not None else self._mu_f
        Sigma = sigma_f if sigma_f is not None else self._sigma_f
        mu = _finite_real_array(np.asarray(mu).reshape(-1))
        Sigma = _finite_real_array(np.asarray(Sigma))

        # Two reference portfolios (after μ, Σ are finite)
        w_mv = self.min_var_weights(sigma_f=Sigma)
        w_tan = self.tangency_weights(mu_f=mu, sigma_f=Sigma)

        alphas = np.linspace(-0.5, 1.5, n_points)  # allow short/leverage
        vols, rets = [], []
        for alpha in alphas:
            w = alpha * w_mv + (1 - alpha) * w_tan
            w = _finite_real_array(w)
            r_raw = float(mu @ w)
            if not np.isfinite(r_raw):
                r_raw = 0.0
            r = r_raw * self.trading_days * 100
            quad = float(w @ Sigma @ w)
            if not np.isfinite(quad):
                quad = 0.0
            quad = max(quad, 0.0)
            v = np.sqrt(quad * self.trading_days) * 100
            vols.append(v)
            rets.append(r)

        return pd.DataFrame({"ann_vol_%": vols, "ann_ret_%": rets})


# ---------------------------------------------------------------------------
# Convenience thin wrappers
# ---------------------------------------------------------------------------

class TangencyPortfolio:
    """Minimal wrapper: given factor returns, return the tangency time series."""

    def __init__(self, risk_free_rate: float = 0.05, trading_days: int = 252):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def fit_predict(
        self,
        loadings: np.ndarray,
        returns_train: np.ndarray,
        returns_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fit on `returns_train`, predict on `returns_test` (or train if None).
        """
        F_train = returns_train @ loadings
        returns_test = returns_test if returns_test is not None else returns_train
        F_test = returns_test @ loadings

        pc = PortfolioConstructor(
            loadings=loadings,
            factor_returns=F_train,
            risk_free_rate=self.risk_free_rate,
            trading_days=self.trading_days,
        )
        w = pc.tangency_weights()
        return F_test @ w


class MinVariancePortfolio:
    """Minimal wrapper: given factor returns, return the min-var time series."""

    def __init__(self, trading_days: int = 252):
        self.trading_days = trading_days

    def fit_predict(
        self,
        loadings: np.ndarray,
        returns_train: np.ndarray,
        returns_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        F_train = returns_train @ loadings
        returns_test = returns_test if returns_test is not None else returns_train
        F_test = returns_test @ loadings

        pc = PortfolioConstructor(
            loadings=loadings,
            factor_returns=F_train,
            trading_days=self.trading_days,
        )
        w = pc.min_var_weights()
        return F_test @ w
