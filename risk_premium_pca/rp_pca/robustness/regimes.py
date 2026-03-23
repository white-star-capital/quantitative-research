"""
Episodic regime classification and per-regime performance analysis.

Classifies each date into one of four market regimes based on the
equal-weight portfolio's rolling return and cross-asset correlation:

    * **bull**      — rolling annualised return > bull_threshold
    * **bear**      — rolling annualised return < bear_threshold
    * **contagion** — high cross-asset correlation AND negative return
    * **sideways**  — everything else

The paper (hexshapeshifter) evaluates RP-PCA vs PCA across these
regimes to test whether the RP-PCA advantage is stable.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..config import RegimeConfig
from ..portfolio.metrics import sharpe_ratio, annualised_vol, max_drawdown

logger = logging.getLogger(__name__)

# Regime colour palette
REGIME_COLORS = {
    "bull": "#00C49F",
    "bear": "#FF4444",
    "sideways": "#FFBB28",
    "contagion": "#8B00FF",
}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_regimes(
    returns: pd.DataFrame,
    config: Optional[RegimeConfig] = None,
    trading_days: int = 252,
) -> pd.Series:
    """
    Classify each date into a market regime.

    Parameters
    ----------
    returns : DataFrame (T, N)
        Asset return matrix (dates × assets).
    config : RegimeConfig
        Thresholds and lookback window.
    trading_days : int
        Days per year (for annualising rolling returns).

    Returns
    -------
    pd.Series
        DatetimeIndex → regime label string.
    """
    if config is None:
        config = RegimeConfig()

    lb = config.lookback

    # 1. Rolling annualised return of the equal-weight portfolio
    ew = returns.mean(axis=1)
    rolling_cum = ew.rolling(lb, min_periods=max(1, lb // 2)).mean() * trading_days

    # 2. Rolling mean pairwise correlation
    #    Use a random subset of pairs if N is large to keep cost manageable
    N = returns.shape[1]
    if N > 30:
        # Sample ~30 columns for correlation estimation
        rng = np.random.default_rng(42)
        subset_cols = rng.choice(returns.columns, size=30, replace=False)
        ret_sub = returns[subset_cols]
    else:
        ret_sub = returns

    def _rolling_mean_corr(row_idx: int) -> float:
        start = max(0, row_idx - lb)
        window = ret_sub.iloc[start:row_idx]
        if len(window) < max(5, lb // 4):
            return np.nan
        corr = window.corr()
        # Mean of upper-triangle (excluding diagonal)
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        vals = corr.values[mask]
        return float(np.nanmean(vals))

    # Vectorised rolling correlation is expensive; compute at spaced intervals
    # and interpolate for efficiency on large datasets
    step = max(1, lb // 4)
    sample_idx = list(range(lb, len(returns), step))
    corr_vals = pd.Series(
        [_rolling_mean_corr(i) for i in sample_idx],
        index=returns.index[sample_idx],
    )
    rolling_corr = corr_vals.reindex(returns.index).interpolate(method="linear")

    # 3. Classify
    labels = pd.Series("sideways", index=returns.index)

    bull_mask = rolling_cum > config.bull_threshold
    bear_mask = rolling_cum < config.bear_threshold
    contagion_mask = (
        (rolling_corr > config.contagion_corr_threshold)
        & (rolling_cum < 0)
    )

    # Priority: contagion > bear > bull > sideways
    labels[bull_mask] = "bull"
    labels[bear_mask] = "bear"
    labels[contagion_mask] = "contagion"

    # Fill initial NaN period as sideways
    labels = labels.fillna("sideways")

    counts = labels.value_counts()
    logger.info("Regime classification: %s", counts.to_dict())

    return labels


# ---------------------------------------------------------------------------
# Per-regime metrics
# ---------------------------------------------------------------------------

def compute_regime_metrics(
    return_series: dict[str, pd.Series],
    regime_labels: pd.Series,
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """
    Compute per-regime, per-strategy Sharpe ratios and summary metrics.

    Parameters
    ----------
    return_series : dict  strategy_name → pd.Series with DatetimeIndex
    regime_labels : pd.Series  DatetimeIndex → regime label

    Returns
    -------
    sharpe_dict : dict  regime_name → {strategy_name: sharpe}
        Ready for ``plot_regime_comparison()``.
    metrics_df : DataFrame
        MultiIndex (regime, strategy) with columns: Sharpe, Vol%, MaxDD%, N_obs.
    """
    regimes = sorted(regime_labels.unique())
    sharpe_dict: dict[str, dict[str, float]] = {}
    rows = []

    for regime in regimes:
        mask = regime_labels == regime
        dates_in_regime = regime_labels.index[mask]
        sharpe_dict[regime] = {}

        for strat_name, series in return_series.items():
            # Align: only dates present in both
            common = series.index.intersection(dates_in_regime)
            r = series.loc[common].values

            if len(r) < 5:
                sr = np.nan
                vol = np.nan
                mdd = np.nan
            else:
                sr = sharpe_ratio(r, risk_free_rate, trading_days)
                vol = annualised_vol(r, trading_days)
                mdd = max_drawdown(r)

            sharpe_dict[regime][strat_name] = sr
            rows.append({
                "regime": regime,
                "strategy": strat_name,
                "Ann. Sharpe": round(sr, 3) if not np.isnan(sr) else np.nan,
                "Ann. Vol (%)": round(vol, 2) if not np.isnan(vol) else np.nan,
                "Max DD (%)": round(mdd, 2) if not np.isnan(mdd) else np.nan,
                "N_obs": len(r),
            })

    metrics_df = pd.DataFrame(rows).set_index(["regime", "strategy"])
    return sharpe_dict, metrics_df


# ---------------------------------------------------------------------------
# Regime timeline visualisation
# ---------------------------------------------------------------------------

def plot_regime_timeline(
    regime_labels: pd.Series,
    title: str = "Market Regime Timeline",
) -> go.Figure:
    """
    Color-coded timeline showing market regimes as a filled bar chart.

    Each date gets a bar of height 1 coloured by its regime, creating a
    continuous colour-coded timeline strip.
    """
    # Map regimes to numeric codes for a stacked-bar approach
    unique_regimes = sorted(regime_labels.unique())
    dates = regime_labels.index

    fig = go.Figure()

    # Create one bar trace per regime so each gets its own legend entry
    for regime in unique_regimes:
        mask = regime_labels == regime
        color = REGIME_COLORS.get(regime, "#888888")
        # Bar height = 1 where this regime is active, 0 otherwise
        y_vals = mask.astype(int).values
        fig.add_trace(go.Bar(
            x=dates,
            y=y_vals,
            name=regime.capitalize(),
            marker_color=color,
            marker_line_width=0,
            width=86400000,  # 1 day in ms (ensures no gaps between bars)
            showlegend=True,
        ))

    fig.update_layout(
        title=title,
        barmode="stack",
        bargap=0,
        xaxis_title="Date",
        yaxis=dict(
            visible=False,
            range=[0, 1.05],
            fixedrange=True,
        ),
        template="plotly_dark",
        height=180,
        margin=dict(t=40, b=40, l=40, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    return fig
