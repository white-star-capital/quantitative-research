"""
Glosten-Harris (1988) bid-ask spread estimator and mid-price recovery.

Reference
---------
Glosten, L. R., & Harris, L. E. (1988). Estimating the components of the
bid/ask spread. Journal of Financial Economics, 21(1), 123-142.

The price-impact decomposition models transaction prices as:

    ΔP_t = (c₀/2) · ΔQ_t  +  z₁ · Q_t  +  ε_t

where
    P_t   = transaction price at trade t
    Q_t   ∈ {+1, -1}  =  buy / sell indicator (aggressor flag)
    ΔQ_t  = Q_t − Q_{t−1}
    c₀/2  = transient (order-processing + adverse-selection) component
    z₁    = permanent (information) component

Effective spread = c₀ + 2·z₁  (full round-trip cost, in price units).
Mid-price at trade t: m_t = P_t − (c₀/2) · Q_t.

Usage
-----
The full Glosten-Harris estimator requires trade-level data available from
Binance Vision (data.binance.vision).  For quick-start / OHLCV-only mode,
a Roll (1984) spread approximation is provided instead.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Full Glosten-Harris estimator (trade-level data)
# ---------------------------------------------------------------------------

class GlostenHarris:
    """
    Fit the Glosten-Harris model to a sequence of individual trades.

    Parameters
    ----------
    min_trades : int
        Minimum trades required for a reliable OLS estimate.
    """

    def __init__(self, min_trades: int = 500) -> None:
        self.min_trades = min_trades

        # Fitted parameters
        self.c0_: Optional[float] = None   # transient component (half-spread)
        self.z1_: Optional[float] = None   # permanent component
        self.spread_: Optional[float] = None  # effective spread = c0 + 2*z1
        self.r_squared_: Optional[float] = None

    def fit(self, prices: np.ndarray, directions: np.ndarray) -> "GlostenHarris":
        """
        Estimate c₀ and z₁ via OLS.

        Parameters
        ----------
        prices : ndarray of shape (T,)
            Sequence of transaction prices.
        directions : ndarray of shape (T,)
            Trade direction: +1 for buyer-initiated, -1 for seller-initiated.
            (corresponds to is_buyer_maker=False → +1; is_buyer_maker=True → -1
             in Binance Vision format).

        Returns
        -------
        self
        """
        prices = np.asarray(prices, dtype=float)
        directions = np.asarray(directions, dtype=float)
        assert len(prices) == len(directions), "prices and directions must match"

        if len(prices) < self.min_trades:
            raise ValueError(
                f"Need at least {self.min_trades} trades, got {len(prices)}."
            )

        # Construct regression variables (drop first observation)
        dP = np.diff(prices)              # ΔP_t
        Q = directions[1:]                # Q_t  (aligned with ΔP_t)
        dQ = np.diff(directions)          # ΔQ_t

        # OLS: ΔP_t = β₁·ΔQ_t + β₂·Q_t + ε_t
        # β₁ = c₀/2,  β₂ = z₁
        X = np.column_stack([dQ, Q])
        result = np.linalg.lstsq(X, dP, rcond=None)
        beta = result[0]

        self.c0_ = 2.0 * beta[0]   # c₀ = 2·β₁
        self.z1_ = beta[1]         # z₁ = β₂
        self.spread_ = self.c0_ + 2.0 * self.z1_

        # R²
        fitted = X @ beta
        ss_res = np.sum((dP - fitted) ** 2)
        ss_tot = np.sum((dP - dP.mean()) ** 2)
        self.r_squared_ = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return self

    def recover_midprices(
        self, prices: np.ndarray, directions: np.ndarray
    ) -> np.ndarray:
        """
        Recover mid-prices: m_t = P_t − (c₀/2) · Q_t.

        Must call fit() first.
        """
        if self.c0_ is None:
            raise RuntimeError("Call fit() before recover_midprices().")
        prices = np.asarray(prices, dtype=float)
        directions = np.asarray(directions, dtype=float)
        return prices - (self.c0_ / 2.0) * directions


# ---------------------------------------------------------------------------
# Roll (1984) estimator — OHLCV approximation
# ---------------------------------------------------------------------------

def roll_spread(close_prices: pd.Series) -> float:
    """
    Estimate the effective bid-ask spread using Roll (1984):

        Ŝ = 2 · √(−Cov(ΔP_t, ΔP_{t-1}))

    Returns 0 if the covariance is non-negative (no spread signal).

    Parameters
    ----------
    close_prices : Series
        Daily closing prices for a single asset.
    """
    dP = close_prices.diff().dropna().values
    if len(dP) < 10:
        return 0.0
    cov = np.cov(dP[:-1], dP[1:])[0, 1]
    return 2.0 * np.sqrt(max(-cov, 0.0))


def roll_spread_all(prices: pd.DataFrame) -> pd.Series:
    """Apply Roll estimator to all columns of a price DataFrame."""
    return prices.apply(roll_spread)


# ---------------------------------------------------------------------------
# Binance Vision trade-data loader (for full Glosten-Harris)
# ---------------------------------------------------------------------------

def load_binance_vision_trades(filepath: str) -> pd.DataFrame:
    """
    Load a Binance Vision daily trades CSV/ZIP file.

    Columns returned: trade_id, price, qty, time, direction
    where direction = +1 (buyer aggressor) or -1 (seller aggressor).

    Binance Vision CSV columns:
        trade_id, price, qty, quote_qty, time, is_buyer_maker, is_best_match
    Note: is_buyer_maker=True means the BUYER is the market maker → SELL-initiated.
    """
    df = pd.read_csv(
        filepath,
        names=[
            "trade_id", "price", "qty", "quote_qty",
            "time", "is_buyer_maker", "is_best_match",
        ],
        dtype={
            "trade_id": int,
            "price": float,
            "qty": float,
            "quote_qty": float,
            "time": int,
            "is_buyer_maker": bool,
            "is_best_match": bool,
        },
    )
    # Convert aggressor flag: is_buyer_maker=False → buyer aggressor → +1
    df["direction"] = np.where(df["is_buyer_maker"], -1, 1)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df[["trade_id", "price", "qty", "time", "direction"]]


def estimate_daily_spread(
    trades_df: pd.DataFrame,
    min_trades_per_day: int = 500,
) -> pd.DataFrame:
    """
    Fit Glosten-Harris for each trading day and return a DataFrame with
    daily spread estimates and mid-prices (last mid-price of the day).

    Parameters
    ----------
    trades_df : DataFrame with columns [price, time, direction]
    min_trades_per_day : int
        Skip days with fewer trades.
    """
    trades_df = trades_df.copy()
    trades_df["date"] = trades_df["time"].dt.normalize().dt.tz_localize(None)

    records = []
    gh = GlostenHarris(min_trades=min_trades_per_day)

    for date, group in trades_df.groupby("date"):
        group = group.sort_values("time")
        if len(group) < min_trades_per_day:
            continue
        try:
            gh.fit(group["price"].values, group["direction"].values)
            last_price = group["price"].iloc[-1]
            last_dir = group["direction"].iloc[-1]
            mid = gh.recover_midprices(
                np.array([last_price]), np.array([last_dir])
            )[0]
            records.append({
                "date": date,
                "spread": gh.spread_,
                "c0": gh.c0_,
                "z1": gh.z1_,
                "mid_price": mid,
                "r_squared": gh.r_squared_,
            })
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.debug("GH fit failed for %s: %s", date, exc)

    return pd.DataFrame(records).set_index("date")
