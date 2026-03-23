"""
Covariance and mean estimation methods.

Three covariance estimators are provided:
  - sample_cov       : standard sample covariance
  - ewma_cov         : exponentially weighted moving average covariance
  - ledoit_wolf_cov  : Ledoit-Wolf analytical shrinkage

And two mean estimators:
  - sample_mean
  - ewma_mean
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


# ---------------------------------------------------------------------------
# Covariance estimators
# ---------------------------------------------------------------------------

def sample_cov(returns: np.ndarray) -> np.ndarray:
    """
    Standard sample covariance matrix.

    Parameters
    ----------
    returns : ndarray of shape (T, N)

    Returns
    -------
    cov : ndarray of shape (N, N)
    """
    return np.cov(returns, rowvar=False)


def ewma_cov(returns: np.ndarray, halflife: int = 60) -> np.ndarray:
    """
    Exponentially weighted moving average covariance matrix.

    Weights decay with the specified half-life (in periods).  The most
    recent observation receives the highest weight.

    Parameters
    ----------
    returns : ndarray of shape (T, N)
    halflife : int
        Half-life of the exponential decay in periods (e.g. trading days).

    Returns
    -------
    cov : ndarray of shape (N, N)
    """
    T, N = returns.shape
    lam = 0.5 ** (1.0 / halflife)

    # Build exponential weights (most recent = index T-1 has highest weight)
    t_idx = np.arange(T)
    weights = lam ** (T - 1 - t_idx)
    weights /= weights.sum()

    # Demeaned returns (use weighted mean)
    mu = (weights[:, None] * returns).sum(axis=0)  # (N,)
    demeaned = returns - mu[None, :]  # (T, N)

    # Weighted outer products
    cov = (weights[:, None, None] * demeaned[:, :, None] * demeaned[:, None, :]).sum(
        axis=0
    )
    return cov


def ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf analytical shrinkage estimator.

    Reference: Ledoit & Wolf (2004), "A well-conditioned estimator for
    large-dimensional covariance matrices."

    Parameters
    ----------
    returns : ndarray of shape (T, N)

    Returns
    -------
    cov : ndarray of shape (N, N)
    """
    lw = LedoitWolf()
    lw.fit(returns)
    return lw.covariance_


# ---------------------------------------------------------------------------
# Mean estimators
# ---------------------------------------------------------------------------

def sample_mean(returns: np.ndarray) -> np.ndarray:
    """Arithmetic sample mean over the T observations."""
    return returns.mean(axis=0)


def ewma_mean(returns: np.ndarray, halflife: int = 21) -> np.ndarray:
    """
    Exponentially weighted mean (EWMA).

    Parameters
    ----------
    returns : ndarray of shape (T, N)
    halflife : int
        Half-life of the decay.

    Returns
    -------
    mean : ndarray of shape (N,)
    """
    T = len(returns)
    lam = 0.5 ** (1.0 / halflife)
    weights = lam ** np.arange(T - 1, -1, -1)  # most recent has highest weight
    weights /= weights.sum()
    return (weights[:, None] * returns).sum(axis=0)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_cov_estimator(method: str, **kwargs):
    """
    Return a callable that computes (cov, mean) from a (T, N) returns array.

    Parameters
    ----------
    method : str
        One of "sample", "ewma", "ledoit_wolf".
    **kwargs
        Passed to the underlying estimator (e.g. halflife for EWMA).
    """
    cov_halflife = kwargs.get("ewma_cov_halflife", 60)
    mean_halflife = kwargs.get("ewma_mean_halflife", 21)

    if method == "sample":
        def estimator(X: np.ndarray):
            return sample_cov(X), sample_mean(X)

    elif method == "ewma":
        def estimator(X: np.ndarray):
            return ewma_cov(X, halflife=cov_halflife), ewma_mean(X, halflife=mean_halflife)

    elif method == "ledoit_wolf":
        def estimator(X: np.ndarray):
            return ledoit_wolf_cov(X), sample_mean(X)

    else:
        raise ValueError(f"Unknown cov method '{method}'. Use sample/ewma/ledoit_wolf.")

    return estimator
