"""
Return computation and pre-processing pipeline.

Steps
-----
1. Forward-fill small gaps in price series (max 3 consecutive days).
2. Drop assets that are missing more than (1 - min_obs_fraction) of dates.
3. Compute log returns (or simple returns if log=False).
4. Winsorise at the specified percentile pair.
5. Impute remaining non-finite returns with 0 (missing cell = flat return) so PCA
   and portfolio code never see NaN/inf from residual gaps.
6. Drop any remaining all-NaN rows (e.g. the first row after pct_change).
7. Align to a common date index.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ReturnProcessor:
    """
    Transform raw price data into a clean return matrix.

    Parameters
    ----------
    winsorize_lower : float
        Lower percentile for winsorisation (e.g. 0.01 = 1st percentile).
    winsorize_upper : float
        Upper percentile (e.g. 0.99 = 99th percentile).
    min_obs_fraction : float
        Minimum fraction of non-null rows required to keep a column.
    use_log_returns : bool
        If True, compute log returns; otherwise simple (arithmetic) returns.
    max_fill_days : int
        Maximum number of consecutive missing days to forward-fill.
    """

    def __init__(
        self,
        winsorize_lower: float = 0.01,
        winsorize_upper: float = 0.99,
        min_obs_fraction: float = 0.80,
        use_log_returns: bool = True,
        max_fill_days: int = 3,
    ) -> None:
        self.winsorize_lower = winsorize_lower
        self.winsorize_upper = winsorize_upper
        self.min_obs_fraction = min_obs_fraction
        self.use_log_returns = use_log_returns
        self.max_fill_days = max_fill_days

        # Set after fit
        self.dropped_assets_: list[str] = []
        self.retained_assets_: list[str] = []

    def fit_transform(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Full pipeline: clean → compute returns → winsorise.

        Parameters
        ----------
        prices : DataFrame, shape (T, N)
            Closing prices; index = dates, columns = ticker symbols.

        Returns
        -------
        returns : DataFrame, shape (T-1, N)
            Clean, winsorised return matrix with aligned dates. Any residual
            NaN/inf after winsorisation is replaced with 0 so downstream linear
            algebra (covariance, RP-PCA, matmul) stays finite.
        """
        prices = prices.copy()
        prices = prices.sort_index()

        # 1. Forward-fill short gaps
        prices = prices.ffill(limit=self.max_fill_days)

        # 2. Drop assets with too many missing values
        min_obs = int(self.min_obs_fraction * len(prices))
        n_valid = prices.notna().sum()
        keep = n_valid[n_valid >= min_obs].index.tolist()
        dropped = [c for c in prices.columns if c not in keep]
        if dropped:
            logger.info("Dropping assets with insufficient data: %s", dropped)
        self.dropped_assets_ = dropped
        self.retained_assets_ = keep
        logger.info(
            "Asset filter: %d/%d assets retained (min_obs_fraction=%.2f, min_obs=%d rows)",
            len(keep), len(prices.columns), self.min_obs_fraction, min_obs,
        )
        prices = prices[keep]

        # 3. Compute returns
        if self.use_log_returns:
            returns = np.log(prices / prices.shift(1)).iloc[1:]
        else:
            returns = prices.pct_change().iloc[1:]

        # 4. Winsorise each column independently
        returns = _winsorise_df(returns, self.winsorize_lower, self.winsorize_upper)

        # 5. Finite returns: residual gaps (beyond ffill window) stay NaN; ±inf can
        #    arise from bad prices. Impute with 0 = flat return for that asset-day.
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 6. Drop any remaining all-NaN rows (should be rare after fillna)
        returns = returns.dropna(how="all")

        logger.info(
            "Return matrix shape: %d dates × %d assets  (dropped %d assets)",
            *returns.shape,
            len(dropped),
        )
        return returns


# ---------------------------------------------------------------------------
# Benchmark return constructors
# ---------------------------------------------------------------------------

def equal_weighted_returns(returns: pd.DataFrame) -> pd.Series:
    """Simple cross-sectional average return (equal weight)."""
    return returns.mean(axis=1).rename("EW_Market")


def value_weighted_returns(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    supply: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Market-cap-weighted portfolio returns.

    If supply (circulating supply) is not provided, prices are used as
    a proxy for relative market caps (scales uniformly, direction preserved).
    """
    # Use lagged prices (previous-day caps) to avoid look-ahead
    if supply is not None:
        mcap = (prices * supply).shift(1)
    else:
        mcap = prices.shift(1)

    common_cols = returns.columns.intersection(mcap.columns)
    mcap = mcap[common_cols].reindex(returns.index)
    rets = returns[common_cols]

    weights = mcap.div(mcap.sum(axis=1), axis=0)
    return (rets * weights).sum(axis=1).rename("VW_Market")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _winsorise_df(df: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """Winsorise each column at [lower, upper] quantiles."""
    def _clip(col: pd.Series) -> pd.Series:
        lo = col.quantile(lower)
        hi = col.quantile(upper)
        return col.clip(lo, hi)

    return df.apply(_clip)


def compute_rolling_returns(
    returns: pd.DataFrame, window: int, min_obs: Optional[int] = None
) -> pd.DataFrame:
    """Compute rolling cumulative returns over `window` days."""
    min_obs = min_obs or window // 2
    return returns.rolling(window, min_periods=min_obs).sum()


def annualise_return(r: float, trading_days: int = 252) -> float:
    """Annualise a per-period log return."""
    return r * trading_days


def annualise_vol(sigma: float, trading_days: int = 252) -> float:
    """Annualise a per-period volatility."""
    return sigma * np.sqrt(trading_days)
