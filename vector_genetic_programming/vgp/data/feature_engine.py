"""
Feature engineering — converts the raw OHLCV dict into a float32 [T x F x A]
numpy array suitable for GP tree evaluation.

Steps
-----
1. Build aligned close prices DataFrame to determine which assets pass the
   min_obs_fraction threshold.
2. Forward-fill short gaps (max max_fill_days consecutive days).
3. Drop assets missing more than (1 - min_obs_fraction) of total dates.
4. For each retained asset, compute the 12-feature DataFrame.
5. Trim the first `lookback` rows from every per-asset DataFrame.
6. Stack into [T, F, A] float32 numpy array.
7. Raise ValueError if any NaN persists after the trim.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical feature order (F = 12).  This list is the single source of truth —
# the array's F-axis layout must match this exactly.
FEATURE_NAMES: list[str] = [
    "ret_1d",       # 1-day simple return
    "ret_5d",       # 5-day simple return
    "ret_20d",      # 20-day simple return
    "log_close",    # natural log of close
    "vol_5d",       # 5-day realised volatility (std of daily returns)
    "vol_20d",      # 20-day realised volatility
    "atr_14",       # simplified 14-day ATR (high - low range, rolling mean)
    "parkinson_14", # 14-day Parkinson volatility estimator
    "rsi_14",       # 14-period RSI
    "norm_close",   # close normalised over 20-day min/max range
    "vol_ratio_20d",# current volume / 20-day mean volume
    "obv_signal",   # OBV z-score
]


class FeatureEngine:
    """
    Transform a raw OHLCV dict into a float32 [T x F x A] numpy array.

    Parameters
    ----------
    min_obs_fraction : float
        Minimum fraction of non-null observations required to keep an asset.
        Default 0.80 (assets missing > 20 % of dates are dropped).
    max_fill_days : int
        Maximum number of consecutive missing days to forward-fill. Default 3.
    lookback : int
        Number of rows to trim from the start of each per-asset DataFrame after
        feature computation.  This removes NaN from rolling windows.  Default 20.
    """

    def __init__(
        self,
        min_obs_fraction: float = 0.80,
        max_fill_days: int = 3,
        lookback: int = 20,
    ) -> None:
        self.min_obs_fraction = min_obs_fraction
        self.max_fill_days = max_fill_days
        self.lookback = lookback

        # Populated by fit_transform()
        self.dropped_assets_: list[str] = []
        self.retained_assets_: list[str] = []
        self.feature_names_: list[str] = []
        self.dates_: pd.DatetimeIndex | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, ohlcv: dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Compute features from raw OHLCV data and return a float32 tensor.

        Parameters
        ----------
        ohlcv : dict[str, pd.DataFrame]
            Mapping of ticker -> OHLCV DataFrame with DatetimeIndex and
            columns [open, high, low, close, volume].  Produced by
            BinanceFetcher.fetch_ohlcv().

        Returns
        -------
        np.ndarray
            Shape [T, F, A], dtype float32.  T = trading days after lookback
            trim, F = 12 features, A = number of retained assets.

        Raises
        ------
        ValueError
            If any NaN remains in the output array after the lookback trim.
        """
        # ----------------------------------------------------------------
        # Step 1: Build aligned close prices to determine observation counts.
        # ----------------------------------------------------------------
        closes = pd.DataFrame(
            {ticker: df["close"] for ticker, df in ohlcv.items()}
        )
        closes = closes.sort_index()

        # Step 2: Forward-fill short gaps before counting valid observations.
        closes = closes.ffill(limit=self.max_fill_days)

        # Step 3: Drop assets below the observation threshold.
        min_obs = int(self.min_obs_fraction * len(closes))
        n_valid = closes.notna().sum()
        keep = n_valid[n_valid >= min_obs].index.tolist()
        dropped = [c for c in closes.columns if c not in keep]
        if dropped:
            logger.info("Dropping assets with insufficient data: %s", dropped)
        self.dropped_assets_ = dropped
        self.retained_assets_ = keep
        logger.info(
            "Asset filter: %d/%d assets retained (min_obs_fraction=%.2f, min_obs=%d rows)",
            len(keep),
            len(closes.columns),
            self.min_obs_fraction,
            min_obs,
        )

        # ----------------------------------------------------------------
        # Step 4: Compute per-asset feature DataFrames.
        # ----------------------------------------------------------------
        per_asset: list[pd.DataFrame] = []
        for ticker in keep:
            df_raw = ohlcv[ticker].copy()
            df_raw = df_raw.sort_index()
            # Apply the same ffill used during filtering so features are
            # computed on the gap-filled prices.
            df_raw = df_raw.ffill(limit=self.max_fill_days)

            feat = _compute_features(df_raw)
            per_asset.append(feat)

        # ----------------------------------------------------------------
        # Step 5: Lookback trim — drop first `lookback` rows from every
        # per-asset DataFrame to eliminate NaN introduced by rolling windows.
        # ----------------------------------------------------------------
        per_asset_trimmed = [feat.iloc[self.lookback :] for feat in per_asset]

        # ----------------------------------------------------------------
        # Step 5b: Align to a common date index (intersection).
        # Assets with different listing dates will have different start dates
        # even after the min_obs_fraction filter.  Taking the intersection
        # ensures every per-asset array has identical shape before stacking.
        # ----------------------------------------------------------------
        common_idx = per_asset_trimmed[0].index
        for feat_df in per_asset_trimmed[1:]:
            common_idx = common_idx.intersection(feat_df.index)
        common_idx = common_idx.sort_values()

        per_asset_trimmed = [feat_df.loc[common_idx] for feat_df in per_asset_trimmed]
        logger.info(
            "Common date index after alignment: %d rows (%s to %s)",
            len(common_idx),
            common_idx.min().date() if len(common_idx) > 0 else "N/A",
            common_idx.max().date() if len(common_idx) > 0 else "N/A",
        )

        # ----------------------------------------------------------------
        # Step 6: Stack into [T, F, A] float32 array.
        # ----------------------------------------------------------------
        arrays = [
            feat_df.to_numpy(dtype=np.float32) for feat_df in per_asset_trimmed
        ]
        result = np.stack(arrays, axis=2)  # shape: [T, F, A]

        # ----------------------------------------------------------------
        # Step 7: NaN guard (D-08 — raise, do not silently propagate).
        # ----------------------------------------------------------------
        if np.isnan(result).any():
            raise ValueError(
                "FeatureEngine output contains NaN after lookback trim. "
                "Check rolling windows and ffill logic."
            )

        # ----------------------------------------------------------------
        # Step 8: Store metadata.
        # ----------------------------------------------------------------
        self.feature_names_ = list(FEATURE_NAMES)
        self.dates_ = per_asset_trimmed[0].index

        logger.info(
            "FeatureEngine result: shape=%s, dtype=%s, assets=%s",
            result.shape,
            result.dtype,
            keep,
        )
        return result


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 12 features for a single asset's OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns [open, high, low, close, volume].

    Returns
    -------
    pd.DataFrame
        Columns in FEATURE_NAMES order, same DatetimeIndex as input.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # -- Momentum ---------------------------------------------------
    ret_1d = close.pct_change(1)
    ret_5d = close.pct_change(5)
    ret_20d = close.pct_change(20)
    log_close = np.log(close)

    # -- Volatility -------------------------------------------------
    vol_5d = close.pct_change(1).rolling(5).std()
    vol_20d = close.pct_change(1).rolling(20).std()

    # Simplified ATR: rolling 14-period mean of (high - low) range.
    atr_14 = (high - low).abs().rolling(14).mean()

    # Parkinson volatility: rolling 14-period mean of the estimator.
    # Formula: sqrt( log(H/L)^2 / (4 * ln(2)) )
    # Clamp high/low ratio to avoid log(0) on degenerate bars.
    hl_ratio = (high / low).clip(lower=1e-8)
    parkinson_14 = (
        (np.log(hl_ratio) ** 2 / (4.0 * np.log(2.0))) ** 0.5
    ).rolling(14).mean()

    # -- Oscillators ------------------------------------------------
    # RSI-14 using Wilder's simple-mean approximation (rolling mean of
    # up/dn moves).  Division-by-zero guard: RSI = 50 when up + dn == 0.
    delta = close.diff()
    up = delta.clip(lower=0.0).rolling(14).mean()
    dn = (-delta).clip(lower=0.0).rolling(14).mean()
    denom_rsi = up + dn
    rsi_14 = 100.0 * up / denom_rsi
    # Where denominator is zero set RSI to 50.
    rsi_14 = rsi_14.where(denom_rsi != 0.0, other=50.0)

    # Normalised close: position within the 20-day high-low channel.
    # Division-by-zero guard: set 0.5 when max == min (flat price).
    roll_min = close.rolling(20).min()
    roll_max = close.rolling(20).max()
    roll_range = roll_max - roll_min
    norm_close = (close - roll_min) / roll_range
    norm_close = norm_close.where(roll_range != 0.0, other=0.5)
    norm_close = norm_close.clip(0.0, 1.0)

    # -- Volume -----------------------------------------------------
    # Volume ratio: current / 20-day mean.  Guard: set 1.0 when mean == 0.
    vol_mean_20 = volume.rolling(20).mean()
    vol_ratio_20d = volume / vol_mean_20
    vol_ratio_20d = vol_ratio_20d.where(vol_mean_20 != 0.0, other=1.0)

    # OBV signal: sign(ret_1d) * volume, cumulated, then z-score normalised.
    obv_raw = (np.sign(close.pct_change(1)) * volume).cumsum()
    obv_std = obv_raw.std()
    if obv_std == 0.0 or np.isnan(obv_std):
        obv_signal = pd.Series(0.0, index=close.index)
    else:
        obv_signal = (obv_raw - obv_raw.mean()) / obv_std

    # -- Assemble in canonical order --------------------------------
    feat = pd.DataFrame(
        {
            "ret_1d": ret_1d,
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "log_close": log_close,
            "vol_5d": vol_5d,
            "vol_20d": vol_20d,
            "atr_14": atr_14,
            "parkinson_14": parkinson_14,
            "rsi_14": rsi_14,
            "norm_close": norm_close,
            "vol_ratio_20d": vol_ratio_20d,
            "obv_signal": obv_signal,
        },
        index=df.index,
    )
    return feat
