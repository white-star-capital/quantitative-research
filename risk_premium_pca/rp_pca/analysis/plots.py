"""
Plotly-based visualisation helpers.

All functions return a plotly.graph_objects.Figure so callers can either
show() them or embed them in the Streamlit dashboard.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Colour palette (consistent across all charts)
# ---------------------------------------------------------------------------
COLORS = {
    "RP-PCA Tangency": "#00C49F",
    "RP-PCA Min-Var":  "#0088FE",
    "PCA Tangency":    "#FF8042",
    "PCA Min-Var":     "#FFBB28",
    "Equal-Weight":    "#A28CF6",
    "BTC":             "#F7931A",
    "Value-Weight":    "#E95D5D",
}
DEFAULT_COLOR = "#888888"


def _color(name: str) -> str:
    return COLORS.get(name, DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# 1. Efficient Frontier
# ---------------------------------------------------------------------------

def plot_efficient_frontier(
    frontiers: dict[str, pd.DataFrame],
    portfolios: Optional[dict[str, tuple[float, float]]] = None,
    title: str = "Efficient Frontier",
) -> go.Figure:
    """
    Plot efficient frontiers for multiple factor universes.

    Parameters
    ----------
    frontiers : dict mapping label → DataFrame with columns [ann_vol_%, ann_ret_%]
    portfolios : dict mapping label → (vol, ret) for individual portfolios
                 (e.g. tangency / min-var points).
    """
    fig = go.Figure()

    line_styles = ["solid", "dash", "dot", "dashdot"]
    for i, (label, df) in enumerate(frontiers.items()):
        fig.add_trace(go.Scatter(
            x=df["ann_vol_%"],
            y=df["ann_ret_%"],
            mode="lines",
            name=label,
            line=dict(dash=line_styles[i % len(line_styles)], width=2),
        ))

    if portfolios:
        for label, (vol, ret) in portfolios.items():
            fig.add_trace(go.Scatter(
                x=[vol], y=[ret],
                mode="markers+text",
                name=label,
                marker=dict(size=10, color=_color(label), symbol="star"),
                text=[label],
                textposition="top center",
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Annualised Volatility (%)",
        yaxis_title="Annualised Return (%)",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(x=0.01, y=0.99),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Cumulative Returns
# ---------------------------------------------------------------------------

def plot_cumulative_returns(
    return_series: dict[str, pd.Series],
    title: str = "Cumulative Portfolio Returns",
    log_scale: bool = False,
) -> go.Figure:
    """
    Plot cumulative wealth index (starting from 1) for each strategy.
    """
    fig = go.Figure()

    for name, ret in return_series.items():
        cum = np.exp(np.cumsum(ret.values))  # wealth index
        fig.add_trace(go.Scatter(
            x=ret.index,
            y=cum,
            mode="lines",
            name=name,
            line=dict(color=_color(name), width=2),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Wealth Index (start = 1)",
        yaxis_type="log" if log_scale else "linear",
        hovermode="x unified",
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Explained Variance
# ---------------------------------------------------------------------------

def plot_explained_variance(
    var_tables: dict[str, pd.DataFrame],
    title: str = "Cumulative Explained Variance by Component",
) -> go.Figure:
    """
    Bar + line chart of cumulative explained variance.

    Parameters
    ----------
    var_tables : dict label → DataFrame with 'cumulative_%' column and PC index
    """
    fig = go.Figure()

    bar_colors = ["#00C49F", "#FF8042", "#0088FE"]
    for i, (label, df) in enumerate(var_tables.items()):
        color = bar_colors[i % len(bar_colors)]
        fig.add_trace(go.Bar(
            x=df.index,
            y=df["var_explained_%"],
            name=f"{label} (incremental)",
            marker_color=color,
            opacity=0.6,
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["cumulative_%"],
            mode="lines+markers",
            name=f"{label} (cumulative)",
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Component",
        yaxis_title="Variance Explained (%)",
        barmode="group",
        template="plotly_dark",
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Factor Loadings Heatmap
# ---------------------------------------------------------------------------

def plot_factor_loadings(
    loadings: np.ndarray,
    asset_names: list[str],
    model_name: str = "RP-PCA",
    n_components: int = 5,
) -> go.Figure:
    """
    Heatmap of factor loading matrix  L  (N × K).
    """
    L = loadings[:, :n_components]
    component_labels = [f"PC{k+1}" for k in range(L.shape[1])]

    fig = go.Figure(data=go.Heatmap(
        z=L,
        x=component_labels,
        y=asset_names,
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="Loading"),
        hoverongaps=False,
    ))
    fig.update_layout(
        title=f"{model_name} Factor Loadings",
        xaxis_title="Component",
        yaxis_title="Asset",
        template="plotly_dark",
        height=max(400, 20 * len(asset_names)),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Factor Sharpe Bar Chart
# ---------------------------------------------------------------------------

def plot_factor_sharpe(
    factor_sharpe_tables: dict[str, pd.DataFrame],
    title: str = "Factor-Level Sharpe Ratios",
) -> go.Figure:
    """
    Grouped bar chart of annualised Sharpe ratio per factor.

    Parameters
    ----------
    factor_sharpe_tables : dict label → DataFrame with 'ann_sharpe' column
    """
    fig = go.Figure()
    bar_colors = ["#00C49F", "#FF8042", "#0088FE"]

    for i, (label, df) in enumerate(factor_sharpe_tables.items()):
        fig.add_trace(go.Bar(
            x=df.index,
            y=df["ann_sharpe"],
            name=label,
            marker_color=bar_colors[i % len(bar_colors)],
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="Component",
        yaxis_title="Annualised Sharpe Ratio",
        barmode="group",
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Bootstrap Distribution
# ---------------------------------------------------------------------------

def plot_bootstrap_distribution(
    diffs: np.ndarray,
    strategy_a: str = "RP-PCA",
    strategy_b: str = "PCA",
    ci_lower: float = 0.0,
    ci_upper: float = 0.0,
    observed_diff: Optional[float] = None,
) -> go.Figure:
    """
    Histogram of bootstrap Sharpe ratio differences (A - B).
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=diffs,
        nbinsx=60,
        name="Bootstrap distribution",
        marker_color="#4A90D9",
        opacity=0.7,
    ))

    # 95% CI bands
    fig.add_vrect(
        x0=ci_lower, x1=ci_upper,
        fillcolor="rgba(0,196,159,0.15)",
        line_width=0,
        annotation_text="95% CI",
        annotation_position="top left",
    )

    # Zero line
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="0")

    if observed_diff is not None:
        fig.add_vline(
            x=observed_diff,
            line_dash="dot",
            line_color="yellow",
            annotation_text="Observed diff",
        )

    prob = float((diffs > 0).mean())
    fig.update_layout(
        title=f"Bootstrap Sharpe Difference: {strategy_a} − {strategy_b}   "
              f"P(A>B) = {prob:.1%}",
        xaxis_title=f"Sharpe Difference ({strategy_a} − {strategy_b})",
        yaxis_title="Frequency",
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Regime Comparison Bar Chart
# ---------------------------------------------------------------------------

def plot_regime_comparison(
    regime_results: dict[str, dict[str, float]],
    title: str = "Sharpe Ratio by Market Regime",
) -> go.Figure:
    """
    Grouped bar chart of Sharpe ratios across regimes.

    Parameters
    ----------
    regime_results : dict  regime_name → {strategy_name: sharpe}
    """
    # Filter out regimes where ALL strategies have NaN Sharpe
    regimes = [
        r for r in regime_results
        if any(not np.isnan(v) for v in regime_results[r].values())
    ]
    if not regimes:
        fig = go.Figure()
        fig.update_layout(title="No regimes with sufficient data", template="plotly_dark")
        return fig

    strategies = list(next(iter(regime_results.values())).keys())

    fig = go.Figure()
    for strat in strategies:
        color = _color(strat)
        sharpes = [regime_results[r].get(strat, np.nan) for r in regimes]
        fig.add_trace(go.Bar(
            x=[r.capitalize() for r in regimes],
            y=sharpes,
            name=strat,
            marker_color=color,
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="Regime",
        yaxis_title="Annualised Sharpe Ratio",
        barmode="group",
        template="plotly_dark",
        height=420,
        xaxis_tickangle=0,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Rolling Sharpe
# ---------------------------------------------------------------------------

def plot_rolling_sharpe(
    return_series: dict[str, pd.Series],
    window: int = 63,
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
    title: str = "Rolling 63-Day Sharpe Ratio",
) -> go.Figure:
    """Time series of rolling Sharpe ratio for each strategy."""
    from ..portfolio.metrics import rolling_sharpe

    fig = go.Figure()
    for name, ret in return_series.items():
        rs = rolling_sharpe(ret.values, window=window,
                            risk_free_rate=risk_free_rate,
                            trading_days=trading_days)
        fig.add_trace(go.Scatter(
            x=ret.index,
            y=rs,
            mode="lines",
            name=name,
            line=dict(color=_color(name), width=1.5),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Rolling Sharpe",
        hovermode="x unified",
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# 9. Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    returns: pd.DataFrame,
    title: str = "Asset Return Correlation Matrix",
) -> go.Figure:
    """Heatmap of the pairwise return correlation matrix."""
    corr = returns.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        colorbar=dict(title="ρ"),
    ))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=max(500, 18 * len(corr)),
        width=max(500, 18 * len(corr)),
    )
    return fig


# ---------------------------------------------------------------------------
# 10. Fama-MacBeth Risk Premia Comparison
# ---------------------------------------------------------------------------

def plot_fama_macbeth_comparison(
    results: list,
    title: str = "Fama-MacBeth Risk Premia: RP-PCA vs PCA",
) -> go.Figure:
    """
    Grouped bar chart comparing risk premia (lambdas) with error bars
    across models.  One group per factor, one bar per model.

    Parameters
    ----------
    results : list of FamaMacBethResult
    """
    model_colors = ["#00C49F", "#FF8042", "#0088FE", "#FFBB28"]
    fig = go.Figure()

    for i, res in enumerate(results):
        factor_labels = [f"F{k+1}" for k in range(res.n_factors)]
        # Error bar = |lambda| / |t-stat| (= standard error)
        se = np.where(
            np.abs(res.risk_premia_tstat) > 1e-10,
            np.abs(res.risk_premia) / np.abs(res.risk_premia_tstat),
            0.0,
        )
        fig.add_trace(go.Bar(
            x=factor_labels,
            y=res.risk_premia,
            name=res.model_name,
            marker_color=model_colors[i % len(model_colors)],
            error_y=dict(type="data", array=1.96 * se, visible=True),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="Factor",
        yaxis_title="Risk Premium (λ)",
        barmode="group",
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# 11. Cross-Sectional R² Over Time
# ---------------------------------------------------------------------------

def plot_cross_sectional_r2(
    results: list,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Cross-Sectional R² Over Time",
) -> go.Figure:
    """
    Time series of cross-sectional R² for each model.

    Parameters
    ----------
    results : list of FamaMacBethResult
    dates : DatetimeIndex for the pass-2 period (optional)
    """
    model_colors = ["#00C49F", "#FF8042", "#0088FE"]
    fig = go.Figure()

    for i, res in enumerate(results):
        x = dates[-res.n_periods_pass2:] if dates is not None else list(range(res.n_periods_pass2))
        fig.add_trace(go.Scatter(
            x=x,
            y=res.cross_sectional_r2,
            mode="lines",
            name=f"{res.model_name} (mean={res.mean_r2:.3f})",
            line=dict(color=model_colors[i % len(model_colors)], width=1.5),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date" if dates is not None else "Period",
        yaxis_title="Cross-Sectional R²",
        hovermode="x unified",
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# 12. Factor Score Card (for Best Features tab)
# ---------------------------------------------------------------------------

def plot_factor_scorecard(
    factor_df: pd.DataFrame,
    selected: list[int],
    model_name: str = "RP-PCA",
    title: str = "Factor Selection Score Card",
) -> go.Figure:
    """
    Horizontal bar chart of factor-level composite scores with
    selected factors highlighted.

    Parameters
    ----------
    factor_df : DataFrame
        Must have columns: 'score', 'ann_sharpe', 'var_explained_%'.
        Index = PC1, PC2, …
    selected : list of int
        0-based indices of selected factors.
    """
    from plotly.subplots import make_subplots

    n = len(factor_df)
    labels = factor_df.index.tolist()
    scores = factor_df["score"].values
    sharpes = factor_df["ann_sharpe"].values
    var_pct = factor_df["var_explained_%"].values

    colors = ["#00C49F" if i in selected else "#555555" for i in range(n)]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Composite Score", "Ann. Sharpe", "Variance Explained (%)"],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    fig.add_trace(go.Bar(
        y=labels, x=scores, orientation="h",
        marker_color=colors, showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        y=labels, x=sharpes, orientation="h",
        marker_color=colors, showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        y=labels, x=var_pct, orientation="h",
        marker_color=colors, showlegend=False,
    ), row=1, col=3)

    fig.update_layout(
        title=f"{title} — {model_name}",
        template="plotly_dark",
        height=max(250, 45 * n),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    # Reverse y-axis so PC1 is at the top
    fig.update_yaxes(autorange="reversed")
    return fig
