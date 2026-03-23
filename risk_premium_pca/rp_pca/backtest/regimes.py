"""
Crypto market regime definitions matching the article's episodic analysis.

The article evaluates portfolio performance across six distinct regimes:

    Regime            PCA Sharpe  RP-PCA Sharpe  Improvement
    ─────────────     ──────────  ─────────────  ───────────
    Bull Peak           3.197        3.789          +18.5%
    Contagion           1.464        1.564           +6.8%
    Crypto Winter       1.706        4.165          +144.1%
    ETF Rally           3.834        4.377          +14.2%
    Consolidation       2.234        1.443          -35.4%
    Post-Election Bull  3.014        4.050          +34.4%
"""
from __future__ import annotations

import pandas as pd

# Each regime is defined by (start_date, end_date, description)
REGIMES: dict[str, tuple[str, str, str]] = {
    "Bull Peak": (
        "2021-01-01", "2021-11-10",
        "Post-DeFi summer run-up to ATH (BTC ~$69k, Nov 2021)",
    ),
    "Contagion": (
        "2021-11-10", "2022-07-01",
        "LUNA/UST collapse, crypto winter onset",
    ),
    "Crypto Winter": (
        "2022-07-01", "2023-01-01",
        "FTX collapse and sustained bear market",
    ),
    "ETF Rally": (
        "2023-01-01", "2024-01-10",
        "Spot Bitcoin ETF anticipation rally",
    ),
    "Consolidation": (
        "2024-01-10", "2024-10-31",
        "Post-ETF approval consolidation and choppy market",
    ),
    "Post-Election Bull": (
        "2024-11-01", "2025-12-31",
        "US election result crypto bull run",
    ),
}


def get_regime_mask(
    index: pd.DatetimeIndex,
    regime_name: str,
) -> pd.Series:
    """
    Return a boolean mask for the given regime name.

    Parameters
    ----------
    index : DatetimeIndex
    regime_name : str
        Key in REGIMES dict.

    Returns
    -------
    mask : pd.Series of bool
    """
    if regime_name not in REGIMES:
        raise ValueError(
            f"Unknown regime '{regime_name}'. "
            f"Available: {list(REGIMES.keys())}"
        )
    start, end, _ = REGIMES[regime_name]
    mask = (index >= start) & (index <= end)
    return pd.Series(mask, index=index)


def split_by_regimes(
    returns: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Split a return DataFrame into regime sub-samples.

    Returns a dict mapping regime name → DataFrame slice.
    Missing regimes (no overlap) are excluded.
    """
    result: dict[str, pd.DataFrame] = {}
    for name, (start, end, _) in REGIMES.items():
        sub = returns.loc[start:end]
        if len(sub) >= 20:  # require at least 20 observations
            result[name] = sub
    return result
