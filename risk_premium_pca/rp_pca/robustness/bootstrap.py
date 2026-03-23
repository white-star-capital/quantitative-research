"""
Circular block bootstrap for stationary time series.

Reference
---------
Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
Journal of the American Statistical Association, 89(428), 1303-1313.

Politis, D. N., & White, H. (2004). Automatic block-length selection for
the dependent bootstrap. Econometric Reviews, 23(1), 53-70.

The circular block bootstrap (CBB) wraps the time series into a circle and
samples contiguous blocks of length `block_length`, preserving the
short-run dependence structure of the data.

The article uses:
    - Block length: 5 days (fixed) or auto-selected via PW (2004)
    - n_reps: 1,000 bootstrap resamples
    - Reports: probability RP-PCA Sharpe > PCA Sharpe

Results from the article:
    RP-PCA tangency outperforms PCA (Sharpe criterion):
        - In-sample tangency: 92.3 % probability
        - In-sample min-var:  96.8 % probability
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..portfolio.metrics import sharpe_ratio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Automatic block-length selection (Politis-White 2004 heuristic)
# ---------------------------------------------------------------------------

def politis_white_block_length(
    returns: np.ndarray,
    max_lag: Optional[int] = None,
    c: float = 2.0,
    K_n: Optional[int] = None,
) -> int:
    """
    Estimate optimal block length for the circular block bootstrap using the
    Politis-White (2004) heuristic.

    Parameters
    ----------
    returns : ndarray of shape (T,) or (T, N)
        Return series (univariate or multivariate — uses first PC).
    c, K_n : float, int
        Tuning constants (see PW 2004).  Defaults follow common practice.

    Returns
    -------
    b_opt : int  (at least 1)
    """
    if returns.ndim > 1:
        # Use first principal component as summary series
        _, _, Vt = np.linalg.svd(returns - returns.mean(axis=0), full_matrices=False)
        series = returns @ Vt[0]
    else:
        series = returns.copy()

    T = len(series)
    if max_lag is None:
        max_lag = int(np.ceil(np.sqrt(T)))
    if K_n is None:
        K_n = max(5, int(np.sqrt(np.log10(T))))

    series = series - series.mean()
    acf = np.array([
        np.dot(series[: T - k], series[k:]) / T
        for k in range(max_lag + 1)
    ])

    # G-hat (estimate of spectral density at 0)
    G = acf[0] + 2 * sum(
        (1 - abs(k) / K_n) * acf[k] for k in range(1, K_n + 1) if k <= max_lag
    )
    D = 2 * G ** 2

    # Optimal block length for CBB
    b_opt = int(np.ceil((c ** 2 * T * D) ** (1 / 3)))
    return max(1, min(b_opt, T // 4))


# ---------------------------------------------------------------------------
# Circular block bootstrap sampler
# ---------------------------------------------------------------------------

class CircularBlockBootstrap:
    """
    Circular block bootstrap for multivariate return series.

    The series is treated as circular (wrap-around), so every observation
    is equally likely to be the start of a block.

    Parameters
    ----------
    block_length : int or None
        Fixed block length.  None → auto-select via Politis-White (2004).
    n_reps : int
        Number of bootstrap resamples.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        block_length: Optional[int] = 5,
        n_reps: int = 1_000,
        seed: int = 42,
    ) -> None:
        self.block_length = block_length
        self.n_reps = n_reps
        self.rng = np.random.default_rng(seed)

    def sample(self, returns: np.ndarray) -> np.ndarray:
        """
        Draw one bootstrap resample of length T.

        Parameters
        ----------
        returns : ndarray of shape (T, N)

        Returns
        -------
        resampled : ndarray of shape (T, N)
        """
        T, N = returns.shape
        b = self.block_length or politis_white_block_length(returns)

        n_blocks = int(np.ceil(T / b))
        # Sample starting indices uniformly from [0, T) (circular)
        starts = self.rng.integers(0, T, size=n_blocks)

        rows = []
        for s in starts:
            idx = [(s + j) % T for j in range(b)]
            rows.append(returns[idx])
        resampled = np.vstack(rows)[:T]  # trim to T
        return resampled

    def bootstrap_statistic(
        self,
        returns: np.ndarray,
        statistic_fn: Callable[[np.ndarray], float],
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Apply `statistic_fn` to `n_reps` bootstrap resamples.

        Parameters
        ----------
        returns : ndarray of shape (T, N) or (T,)
        statistic_fn : callable
            Function (ndarray) → float.

        Returns
        -------
        distribution : ndarray of shape (n_reps,)
        """
        dist = np.empty(self.n_reps)
        iterator = range(self.n_reps)
        if verbose:
            iterator = tqdm(iterator, desc="Bootstrapping")
        for i in iterator:
            sample = self.sample(
                returns.reshape(-1, 1) if returns.ndim == 1 else returns
            )
            dist[i] = statistic_fn(sample.squeeze() if returns.ndim == 1 else sample)
        return dist

    def confidence_interval(
        self,
        distribution: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Return (lower, upper) bootstrap confidence interval."""
        lo = np.percentile(distribution, alpha / 2 * 100)
        hi = np.percentile(distribution, (1 - alpha / 2) * 100)
        return float(lo), float(hi)


# ---------------------------------------------------------------------------
# High-level: compare RP-PCA vs PCA Sharpe ratio via bootstrap
# ---------------------------------------------------------------------------

@dataclass
class BootstrapComparisonResult:
    """Result of a two-strategy Sharpe ratio bootstrap comparison."""
    strategy_a: str
    strategy_b: str
    sharpe_a_observed: float
    sharpe_b_observed: float
    diff_distribution: np.ndarray   # bootstrap distribution of (SR_a - SR_b)
    prob_a_gt_b: float              # P(SR_a > SR_b) from bootstrap
    ci_lower: float
    ci_upper: float
    n_reps: int
    block_length_used: int

    def summary(self) -> str:
        return (
            f"{self.strategy_a} vs {self.strategy_b}\n"
            f"  Observed Sharpe:  {self.strategy_a}={self.sharpe_a_observed:.3f}  "
            f"{self.strategy_b}={self.sharpe_b_observed:.3f}\n"
            f"  Diff (A - B):  mean={self.diff_distribution.mean():.3f}  "
            f"95% CI=[{self.ci_lower:.3f}, {self.ci_upper:.3f}]\n"
            f"  P(A > B) = {self.prob_a_gt_b:.1%}  (n_reps={self.n_reps})"
        )


def bootstrap_sharpe_comparison(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    strategy_a_name: str = "RP-PCA",
    strategy_b_name: str = "PCA",
    block_length: Optional[int] = 5,
    n_reps: int = 1_000,
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
    seed: int = 42,
    verbose: bool = True,
) -> BootstrapComparisonResult:
    """
    Bootstrap test: P(Sharpe_A > Sharpe_B).

    Both return series must have the same length T.

    Parameters
    ----------
    returns_a, returns_b : ndarray of shape (T,)
        Per-period return series for the two strategies.

    Returns
    -------
    BootstrapComparisonResult
    """
    assert len(returns_a) == len(returns_b), "Return series must have equal length."
    T = len(returns_a)

    # Stack into joint matrix for block-coherent resampling
    joint = np.column_stack([returns_a, returns_b])  # (T, 2)

    b = block_length or politis_white_block_length(joint)
    cbb = CircularBlockBootstrap(block_length=b, n_reps=n_reps, seed=seed)

    observed_sa = sharpe_ratio(returns_a, risk_free_rate, trading_days)
    observed_sb = sharpe_ratio(returns_b, risk_free_rate, trading_days)

    def diff_stat(sample: np.ndarray) -> float:
        sa = sharpe_ratio(sample[:, 0], risk_free_rate, trading_days)
        sb = sharpe_ratio(sample[:, 1], risk_free_rate, trading_days)
        return sa - sb

    iterator = tqdm(range(n_reps), desc="Bootstrap") if verbose else range(n_reps)
    diffs = np.empty(n_reps)
    for i in iterator:
        s = cbb.sample(joint)
        diffs[i] = diff_stat(s)

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    prob = float((diffs > 0).mean())

    return BootstrapComparisonResult(
        strategy_a=strategy_a_name,
        strategy_b=strategy_b_name,
        sharpe_a_observed=observed_sa,
        sharpe_b_observed=observed_sb,
        diff_distribution=diffs,
        prob_a_gt_b=prob,
        ci_lower=float(lo),
        ci_upper=float(hi),
        n_reps=n_reps,
        block_length_used=b,
    )


# ---------------------------------------------------------------------------
# Regime-level bootstrap (run on sub-samples)
# ---------------------------------------------------------------------------

def bootstrap_regime_comparison(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    dates: pd.DatetimeIndex,
    regimes: dict,
    **kwargs,
) -> dict[str, BootstrapComparisonResult]:
    """
    Run bootstrap comparison for each regime sub-sample.

    Parameters
    ----------
    returns_a, returns_b : ndarray of shape (T,)
    dates : DatetimeIndex aligned with returns
    regimes : dict from regimes.REGIMES

    Returns
    -------
    dict mapping regime_name → BootstrapComparisonResult
    """
    results = {}
    for regime_name, (start, end, _) in regimes.items():
        mask = (dates >= start) & (dates <= end)
        ra = returns_a[mask]
        rb = returns_b[mask]
        if len(ra) < 30:
            logger.warning("Skipping regime %s: only %d obs", regime_name, len(ra))
            continue
        results[regime_name] = bootstrap_sharpe_comparison(
            ra, rb, verbose=False, **kwargs
        )
    return results
