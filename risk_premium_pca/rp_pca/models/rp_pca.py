"""
Risk-Premium PCA (RP-PCA) — Lettau & Pelger (2020).

Reference
---------
Lettau, M., & Pelger, M. (2020). Estimating latent asset-pricing factors.
Journal of Finance, 75(2), 919-969.

Core idea
---------
Standard PCA maximises explained variance (trace of projected covariance).
RP-PCA augments the objective to also reward factors that capture high
mean returns.  It does so by adding a scaled outer product of the mean
vector to the covariance matrix before decomposition:

    M_rppca = Σ  +  γ · μ · μ'

where
    Σ  = (demeaned) sample covariance of returns  (N × N)
    μ  = sample mean return vector                (N,)
    γ  = penalty parameter (> 0)

Special cases
    γ = 0  →  standard centered PCA (ignores means)
    γ = 1  →  standard uncentered PCA (Σ + μμ' = second moment matrix)
    γ > 1  →  RP-PCA: progressively favours high-mean factors

Modular design (as described in the article)
--------------------------------------------
The implementation separates moment estimation from decomposition so that
out-of-sample analysis can use different window lengths and estimators for
the covariance (longer window / EWMA) vs. the mean (shorter window / EWMA).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .covariance import sample_cov, sample_mean

logger = logging.getLogger(__name__)


class RPPCA:
    """
    Risk-Premium PCA.

    Parameters
    ----------
    n_components : int
        Number of factors to extract.
    gamma : float or None
        Penalty parameter γ.
        None → auto-set to T (number of observations) at fit time,
        which makes the mean outer-product term comparable in scale to
        the covariance matrix (as suggested by Lettau & Pelger).
    """

    def __init__(
        self,
        n_components: int = 5,
        gamma: Optional[float] = None,
    ) -> None:
        self.n_components = n_components
        self.gamma = gamma

        # Fitted attributes
        self.loadings_: Optional[np.ndarray] = None    # (N, K)
        self.eigenvalues_: Optional[np.ndarray] = None  # (K,)
        self.factors_: Optional[np.ndarray] = None     # (T, K)
        self.mean_: Optional[np.ndarray] = None        # (N,)
        self.cov_: Optional[np.ndarray] = None         # (N, N)
        self.composite_: Optional[np.ndarray] = None   # (N, N)  M_rppca
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.cumulative_variance_ratio_: Optional[np.ndarray] = None
        self.gamma_used_: Optional[float] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        returns: np.ndarray,
        cov_matrix: Optional[np.ndarray] = None,
        mean_vector: Optional[np.ndarray] = None,
    ) -> "RPPCA":
        """
        Fit RP-PCA.

        Parameters
        ----------
        returns : ndarray of shape (T, N)
            Asset return matrix (used only for default moment estimation
            and factor return computation).
        cov_matrix : ndarray of shape (N, N), optional
            Pre-computed covariance matrix.  Allows separate estimation
            windows for the OOS procedure.  Defaults to sample covariance.
        mean_vector : ndarray of shape (N,), optional
            Pre-computed mean vector.  Allows separate estimation windows.
            Defaults to sample mean.
        """
        T, N = returns.shape

        # ----------------------------------------------------------
        # Step 1: resolve γ
        # ----------------------------------------------------------
        gamma = float(T) if self.gamma is None else self.gamma
        self.gamma_used_ = gamma

        # ----------------------------------------------------------
        # Step 2: estimate moments (or accept pre-computed values)
        # ----------------------------------------------------------
        Sigma = cov_matrix if cov_matrix is not None else sample_cov(returns)
        mu = mean_vector if mean_vector is not None else sample_mean(returns)

        self.cov_ = Sigma
        self.mean_ = mu

        # ----------------------------------------------------------
        # Step 3: form composite matrix
        # M = Σ  +  γ · μ · μ'
        # ----------------------------------------------------------
        M = Sigma + gamma * np.outer(mu, mu)
        self.composite_ = M

        # ----------------------------------------------------------
        # Step 4: eigendecomposition (symmetric → eigh)
        # ----------------------------------------------------------
        eigenvalues, eigenvectors = np.linalg.eigh(M)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.loadings_ = eigenvectors[:, : self.n_components]   # (N, K)
        self.eigenvalues_ = eigenvalues[: self.n_components]    # (K,)

        # Explained variance ratio relative to Σ (not M) — matches the article
        total_var = np.trace(Sigma)
        if total_var > 0:
            factor_var = np.array([
                self.loadings_[:, k] @ Sigma @ self.loadings_[:, k]
                for k in range(self.n_components)
            ])
            self.explained_variance_ratio_ = factor_var / total_var
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components)

        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        # ----------------------------------------------------------
        # Step 5: compute factor returns  F = X · L
        # ----------------------------------------------------------
        self.factors_ = returns @ self.loadings_   # (T, K)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transform(self, returns: np.ndarray) -> np.ndarray:
        """Project a new (T', N) return matrix onto fitted loadings."""
        if self.loadings_ is None:
            raise RuntimeError("Call fit() first.")
        return returns @ self.loadings_

    def fit_transform(
        self,
        returns: np.ndarray,
        cov_matrix: Optional[np.ndarray] = None,
        mean_vector: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.fit(returns, cov_matrix=cov_matrix, mean_vector=mean_vector).factors_

    # ------------------------------------------------------------------
    # Explained variance table
    # ------------------------------------------------------------------

    def variance_table(self) -> pd.DataFrame:
        """Return a DataFrame summarising explained variance per component."""
        k = len(self.eigenvalues_)
        return pd.DataFrame(
            {
                "eigenvalue": self.eigenvalues_,
                "var_explained_%": self.explained_variance_ratio_ * 100,
                "cumulative_%": self.cumulative_variance_ratio_ * 100,
            },
            index=[f"PC{i+1}" for i in range(k)],
        )

    # ------------------------------------------------------------------
    # Factor mean returns and Sharpe decomposition
    # ------------------------------------------------------------------

    def factor_sharpe(self, trading_days: int = 252) -> pd.DataFrame:
        """
        Annualised mean return, volatility, and Sharpe ratio per factor.
        """
        if self.factors_ is None:
            raise RuntimeError("Call fit() first.")
        F = self.factors_
        mu_f = F.mean(axis=0)
        sig_f = F.std(axis=0, ddof=1)
        sharpe = (mu_f / sig_f) * np.sqrt(trading_days)
        ann_ret = mu_f * trading_days
        ann_vol = sig_f * np.sqrt(trading_days)
        k = F.shape[1]
        return pd.DataFrame(
            {
                "ann_return_%": ann_ret * 100,
                "ann_vol_%": ann_vol * 100,
                "ann_sharpe": sharpe,
                "var_explained_%": self.explained_variance_ratio_ * 100,
            },
            index=[f"PC{i+1}" for i in range(k)],
        )


# ---------------------------------------------------------------------------
# Convenience: fit both PCA and RP-PCA and return side-by-side comparison
# ---------------------------------------------------------------------------

def compare_models(
    returns: np.ndarray,
    n_components: int = 5,
    gamma: Optional[float] = None,
) -> tuple["RPPCA", "RPPCA"]:
    """
    Fit both standard uncentered PCA (γ=1) and RP-PCA (γ=gamma or auto).

    Returns
    -------
    pca_model, rp_pca_model
    """
    from .pca import UncenteredPCA

    pca = RPPCA(n_components=n_components, gamma=1.0)
    pca.fit(returns)

    rppca = RPPCA(n_components=n_components, gamma=gamma)
    rppca.fit(returns)

    return pca, rppca
