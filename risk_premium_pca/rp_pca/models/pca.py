"""
Standard Uncentered PCA via SVD.

The article implements "uncentered PCA on the 30 cryptocurrency returns
using singular value decomposition (SVD)."  Uncentered means we decompose
the second-moment matrix  M = X'X / T  (not the demeaned covariance).

This is equivalent to:  M = Σ + μμ'
where Σ = demeaned covariance and μ = cross-sectional mean vector.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class UncenteredPCA:
    """
    Uncentered (second-moment) PCA.

    The decomposed matrix is:
        M = (1/T) · X'X  =  Σ + μμ'

    Parameters
    ----------
    n_components : int
        Number of factors to retain.
    """

    def __init__(self, n_components: int = 5) -> None:
        self.n_components = n_components

        # Fitted attributes
        self.loadings_: Optional[np.ndarray] = None   # (N, K)
        self.eigenvalues_: Optional[np.ndarray] = None  # (K,)
        self.factors_: Optional[np.ndarray] = None     # (T, K)
        self.explained_variance_ratio_: Optional[np.ndarray] = None  # (K,)
        self.cumulative_variance_ratio_: Optional[np.ndarray] = None  # (K,)

    def fit(self, returns: np.ndarray) -> "UncenteredPCA":
        """
        Fit uncentered PCA.

        Parameters
        ----------
        returns : ndarray of shape (T, N)

        Returns
        -------
        self
        """
        T, N = returns.shape

        # Second-moment matrix
        M = (returns.T @ returns) / T  # (N, N)

        # Eigendecomposition (symmetric matrix → use eigh for speed + stability)
        eigenvalues, eigenvectors = np.linalg.eigh(M)

        # Sort descending by eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.loadings_ = eigenvectors[:, : self.n_components]   # (N, K)
        self.eigenvalues_ = eigenvalues[: self.n_components]

        total = eigenvalues.sum()
        self.explained_variance_ratio_ = self.eigenvalues_ / total
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        # Factor returns: F = X · L  →  (T, K)
        self.factors_ = returns @ self.loadings_

        return self

    def transform(self, returns: np.ndarray) -> np.ndarray:
        """Project a new (T', N) return matrix onto the fitted loadings."""
        if self.loadings_ is None:
            raise RuntimeError("Call fit() first.")
        return returns @ self.loadings_

    def fit_transform(self, returns: np.ndarray) -> np.ndarray:
        return self.fit(returns).factors_

    # ------------------------------------------------------------------
    # Convenience: explained variance table
    # ------------------------------------------------------------------

    def variance_table(self) -> pd.DataFrame:
        """Return a DataFrame summarising explained variance per component."""
        k = len(self.eigenvalues_)
        return pd.DataFrame(
            {
                "eigenvalue": self.eigenvalues_,
                "var_explained": self.explained_variance_ratio_ * 100,
                "cumulative": self.cumulative_variance_ratio_ * 100,
            },
            index=[f"PC{i+1}" for i in range(k)],
        )
