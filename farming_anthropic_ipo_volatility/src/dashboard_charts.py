"""
Live Plotly figures for the Selling-the-wait dashboard.

Numbers come from `anth_ipo_book` / `anth_vol_surface`. Visual language
matches the research-note charts (paper ground, navy ink, magma surface).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from anth_ipo_book import (
    PAPER_SCENARIOS,
    SCENARIO_WEIGHTS,
    SPOT,
    SURFACE_2026_09_16,
    event_move,
    forward_vol,
    weighted_expected_return,
)
from anth_vol_surface import (
    SliceQuote,
    build_surface,
    calendar_violations,
    digital_and_density,
    interpolate_iv,
    interpolate_iv_pchip_delta,
    iv_from_strike,
    quoted_moneyness_band,
    surface_grid,
)

PAPER = "#F4F1EA"
INK = "#142033"
MUTED = "#5B6573"
RULE = "#D7DCE3"
NAVY = "#0B1F3A"
TEAL = "#2A6F7F"
GOLD = "#B8893A"
RED = "#9B3A32"
GREEN = "#2F6B4F"
WHITE = "#FFFFFF"
MAGMA = "magma"

SMILE_COLORS = {
    "Sep 23": RED,
    "Oct 17": NAVY,
    "Nov 7": TEAL,
    "Nov 21": GOLD,
}

REALIZED_ANTH = 0.53
REALIZED_WEEKDAY = 0.59
SPCX_JUN1_22_MOVE = 0.281

EVENT_TENORS: tuple[tuple[str, float, bool], ...] = (
    ("Oct 17", 30.0, False),
    ("Oct 31*", 45.0, True),
    ("Nov 7", 51.0, False),
    ("Nov 21", 65.0, False),
)


def apply_paper(fig: go.Figure, *, height: int = 420) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=PAPER,
        plot_bgcolor=WHITE,
        font=dict(color=INK, size=13, family="Inter, Source Sans 3, sans-serif"),
        title=dict(font=dict(size=16, color=INK, family="Inter, Source Sans 3, sans-serif")),
        margin=dict(l=64, r=28, t=56, b=52),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12, color=INK),
        ),
        hovermode="x unified",
        height=height,
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        color=MUTED,
        linecolor=RULE,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=MUTED, size=12),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#EEF0F3",
        zeroline=False,
        color=MUTED,
        linecolor=RULE,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=MUTED, size=12),
    )
    return fig


def _atm_by_days() -> dict[float, float]:
    return {float(e.days): e.atm_iv for e in SURFACE_2026_09_16}


def iv_at_days(days: float, strike: float, spot: float = SPOT) -> float:
    listed = _atm_by_days()
    if days in listed and abs(strike - spot) < 1e-9:
        return listed[days]
    slices = build_surface(spot)
    return interpolate_iv(slices, days, strike, spot)


def surface_quotes_table(spot: float = SPOT) -> pd.DataFrame:
    rows = []
    for sl in build_surface(spot):
        e = sl.expiry
        rows.append(
            {
                "Expiry": e.label,
                "Days": int(e.days),
                "ATM": e.atm_iv,
                "25Δ RR": e.rr_25d / 100.0,
                "25Δ fly": e.fly_25d / 100.0,
                "25Δ call IV": sl.iv_25c,
                "25Δ put IV": sl.iv_25p,
                "K put": sl.k_25p,
                "K call": sl.k_25c,
            }
        )
    return pd.DataFrame(rows)


def event_move_table(spot: float = SPOT) -> pd.DataFrame:
    rows = []
    for label, days, interpolated in EVENT_TENORS:
        iv = iv_at_days(days, spot, spot)
        rows.append(
            {
                "Expiry": label,
                "Days": int(days),
                "ATM IV": iv,
                "Interpolated": interpolated,
                "b = 45%": event_move(iv, days, 0.45),
                "b = 50%": event_move(iv, days, 0.50),
                "b = 60%": event_move(iv, days, 0.60),
            }
        )
    return pd.DataFrame(rows)


def calendar_violations_table(spot: float = SPOT) -> pd.DataFrame:
    hits = calendar_violations(build_surface(spot), spot=spot)
    if not hits:
        return pd.DataFrame(columns=["From", "To", "K/S", "Δw"])
    return pd.DataFrame(
        [{"From": a, "To": b, "K/S": m, "Δw": dw} for a, b, m, dw in hits]
    )


def term_structure_fig() -> go.Figure:
    days = [e.days for e in SURFACE_2026_09_16]
    ivs = [e.atm_iv for e in SURFACE_2026_09_16]
    labels = [e.label.split()[-1] for e in SURFACE_2026_09_16]
    hover = [
        f"{e.label}<br>ATM {e.atm_iv:.1%} · {int(e.days)}d" for e in SURFACE_2026_09_16
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=days,
            y=ivs,
            mode="lines+markers+text",
            line=dict(color=NAVY, width=2.2),
            marker=dict(size=8, color=NAVY),
            text=labels,
            textposition="top center",
            textfont=dict(size=11, color=INK),
            hovertext=hover,
            hoverinfo="text",
            name="ATM IV",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 65],
            y=[REALIZED_WEEKDAY, REALIZED_WEEKDAY],
            mode="lines",
            line=dict(color=TEAL, width=1.2, dash="dot"),
            name="ANTH weekday realized 59%",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 65],
            y=[REALIZED_ANTH, REALIZED_ANTH],
            mode="lines",
            line=dict(color=GOLD, width=1.2, dash="dash"),
            name="ANTH realized 53%",
            hoverinfo="skip",
        )
    )
    fig.update_yaxes(tickformat=".0%", range=[0.45, 0.90], title="ATM implied vol")
    fig.update_xaxes(title="Calendar days from 16 Sep 2026", range=[-1, 68])
    fig.update_layout(title="Strike ANTH ATM term structure · snapshot 16 Sep 2026")
    return apply_paper(fig)


def event_variance_fig(spot: float = SPOT) -> go.Figure:
    labels = [row[0] for row in EVENT_TENORS]
    series = {
        "baseline 45%": (MUTED, [event_move(iv_at_days(d, spot, spot), d, 0.45) for _, d, _ in EVENT_TENORS]),
        "baseline 50%": (NAVY, [event_move(iv_at_days(d, spot, spot), d, 0.50) for _, d, _ in EVENT_TENORS]),
        "baseline 60%": (TEAL, [event_move(iv_at_days(d, spot, spot), d, 0.60) for _, d, _ in EVENT_TENORS]),
    }
    fig = go.Figure()
    for name, (color, ys) in series.items():
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=ys,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2.2),
                marker=dict(size=8, color=color),
                hovertemplate="%{x}<br>%{y:.1%}<extra>" + name + "</extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[SPCX_JUN1_22_MOVE] * len(labels),
            mode="lines",
            line=dict(color=GOLD, width=1.3, dash="dash"),
            name="SPCX Jun 1–22 event 28.1%",
            hoverinfo="skip",
        )
    )
    fig.update_yaxes(tickformat=".0%", title="Equivalent event move", range=[0.11, 0.29])
    fig.update_xaxes(title="")
    fig.update_layout(title="Event variance extracted from Strike · sqrt((σ² − b²) T)")
    return apply_paper(fig)


def rr_fly_fig() -> go.Figure:
    days = [e.days for e in SURFACE_2026_09_16]
    rr = [e.rr_25d for e in SURFACE_2026_09_16]
    fly = [e.fly_25d for e in SURFACE_2026_09_16]
    hover_rr = [f"{e.label}: RR {e.rr_25d:+.1f} vol pts" for e in SURFACE_2026_09_16]
    hover_fly = [f"{e.label}: fly {e.fly_25d:+.1f} vol pts" for e in SURFACE_2026_09_16]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=days,
            y=rr,
            mode="lines+markers",
            name="25Δ RR (call − put)",
            line=dict(color=NAVY, width=2.2),
            marker=dict(size=8, color=NAVY),
            hovertext=hover_rr,
            hoverinfo="text",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=fly,
            mode="lines+markers",
            name="25Δ fly",
            line=dict(color=GOLD, width=2.2),
            marker=dict(size=8, color=GOLD),
            hovertext=hover_fly,
            hoverinfo="text",
        )
    )
    fig.add_hline(y=0.0, line=dict(color=RULE, width=1))
    fig.add_vline(
        x=30,
        line=dict(color="#C5CAD3", width=1, dash="dot"),
        annotation_text="RR flips to calls",
        annotation_position="top",
        annotation_font=dict(size=11, color=MUTED),
    )
    fig.update_yaxes(title="Vol points", dtick=1)
    fig.update_xaxes(title="Calendar days", range=[-1, 68])
    fig.update_layout(title="Skew and convexity across the listing window")
    return apply_paper(fig)


def smiles_fig(
    spot: float = SPOT,
    labels: Iterable[str] | None = None,
) -> go.Figure:
    wanted = tuple(labels) if labels is not None else tuple(SMILE_COLORS)
    by_label = {sl.expiry.label: sl for sl in build_surface(spot)}
    fig = go.Figure()
    fig.add_vline(x=1.0, line=dict(color="#C5CAD3", width=1))

    for label in wanted:
        sl = by_label[label]
        color = SMILE_COLORS.get(label, NAVY)
        lo, hi = sl.k_25p / spot, sl.k_25c / spot
        if lo > hi:
            lo, hi = hi, lo
        mny = np.linspace(lo, hi, 80)
        iv = np.array([iv_from_strike(sl, spot * m, spot) for m in mny])
        fig.add_trace(
            go.Scatter(
                x=mny,
                y=iv,
                mode="lines",
                name=f"{label} {int(sl.days)}d",
                line=dict(color=color, width=2.2),
                hovertemplate="K/S %{x:.3f}<br>IV %{y:.1%}<extra>" + label + "</extra>",
            )
        )
        pins_m = [sl.k_25p / spot, 1.0, sl.k_25c / spot]
        pins_iv = [sl.iv_25p, sl.atm, sl.iv_25c]
        fig.add_trace(
            go.Scatter(
                x=pins_m,
                y=pins_iv,
                mode="markers",
                marker=dict(size=8, color=color, line=dict(width=0.8, color=WHITE)),
                showlegend=False,
                hovertemplate="K/S %{x:.3f}<br>IV %{y:.1%}<extra>" + label + " pin</extra>",
            )
        )

    fig.update_xaxes(title="Strike / spot", range=[0.80, 1.36])
    fig.update_yaxes(title="Implied vol", tickformat=".1%", range=[0.55, 0.90])
    fig.update_layout(title="Reconstructed smiles · dots are the quoted ATM and 25Δ wings")
    return apply_paper(fig, height=460)


def heatmap_fig(spot: float = SPOT) -> go.Figure:
    slices = build_surface(spot)
    days = np.linspace(1.0, 65.0, 80)
    mny = np.linspace(0.84, 1.32, 96)
    z = np.array(surface_grid(slices, days.tolist(), mny.tolist(), spot), dtype=float)
    for i, d in enumerate(days):
        lo, hi = quoted_moneyness_band(slices, float(d), spot)
        z[i, (mny < lo) | (mny > hi)] = np.nan

    fig = go.Figure(
        go.Heatmap(
            x=mny,
            y=days,
            z=z,
            colorscale=MAGMA,
            colorbar=dict(
                title=dict(text="IV", font=dict(color=MUTED, size=11)),
                tickformat=".0%",
                tickfont=dict(color=MUTED, size=10),
            ),
            hovertemplate="K/S %{x:.3f}<br>%{y:.0f}d<br>IV %{z:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[1.0] * len(slices),
            y=[sl.days for sl in slices],
            mode="markers+text",
            marker=dict(size=8, color=WHITE, line=dict(width=0.8, color=INK)),
            text=[sl.expiry.label for sl in slices],
            textposition="middle right",
            textfont=dict(size=11, color=INK),
            hovertext=[f"{sl.expiry.label}: ATM {sl.atm:.1%}" for sl in slices],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_xaxes(title="Strike / spot", range=[0.84, 1.32])
    fig.update_yaxes(title="Calendar days from 16 Sep")
    fig.update_layout(title="Total-variance interpolant · Strike ANTH surface, 16 Sep 2026")
    return apply_paper(fig, height=520)


def mesh_ks_surface(
    spot: float = SPOT,
    n_days: int = 48,
    n_mny: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[SliceQuote, ...]]:
    slices = build_surface(spot)
    days = np.linspace(2.0, 65.0, n_days)
    u = np.linspace(0.0, 1.0, n_mny)
    x = np.empty((n_days, n_mny))
    y = np.empty((n_days, n_mny))
    z = np.empty((n_days, n_mny))
    for i, d in enumerate(days):
        lo, hi = quoted_moneyness_band(slices, float(d), spot)
        mny = lo + u * (hi - lo)
        x[i, :] = mny
        y[i, :] = d
        z[i, :] = [interpolate_iv(slices, float(d), spot * m, spot) for m in mny]
    return x, y, z, slices


def mesh_delta_surface(
    n_days: int = 80,
    n_delta: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[SliceQuote, ...]]:
    slices = build_surface()
    days = np.linspace(1.0, 65.0, n_days)
    deltas = np.linspace(0.25, 0.75, n_delta)
    z = np.array(
        [[interpolate_iv_pchip_delta(slices, float(d), float(delta)) for delta in deltas] for d in days]
    )
    x, y = np.meshgrid(deltas, days)
    return x, y, z, slices


def _surface_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    slices: tuple[SliceQuote, ...],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    atm_x: float,
) -> go.Figure:
    finite = z[np.isfinite(z)]
    fig = go.Figure(
        go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=MAGMA,
            showscale=True,
            cmin=float(finite.min()),
            cmax=float(finite.max()),
            colorbar=dict(
                title=dict(text="IV", font=dict(color=MUTED, size=11)),
                tickformat=".0%",
                tickfont=dict(color=MUTED, size=10),
                len=0.65,
            ),
            hovertemplate=xlabel
            + " %{x:.3f}<br>"
            + ylabel
            + " %{y:.1f}<br>IV %{z:.1%}<extra></extra>",
            contours=dict(z=dict(show=False)),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[atm_x] * len(slices),
            y=[sl.days for sl in slices],
            z=[sl.atm for sl in slices],
            mode="markers",
            marker=dict(size=4, color=WHITE, line=dict(width=1, color=INK)),
            name="listed ATM",
            hovertext=[f"{sl.expiry.label}: {sl.atm:.1%}" for sl in slices],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        paper_bgcolor=PAPER,
        font=dict(color=INK, size=12, family="Inter, Source Sans 3, sans-serif"),
        title=dict(text=title, font=dict(size=16, color=INK)),
        margin=dict(l=8, r=8, t=48, b=8),
        height=560,
        scene=dict(
            xaxis=dict(title=xlabel, backgroundcolor=PAPER, gridcolor=RULE, color=MUTED),
            yaxis=dict(title=ylabel, backgroundcolor=PAPER, gridcolor=RULE, color=MUTED),
            zaxis=dict(
                title="IV",
                backgroundcolor=PAPER,
                gridcolor=RULE,
                color=MUTED,
                tickformat=".0%",
            ),
            bgcolor=PAPER,
            camera=dict(eye=dict(x=1.55, y=-1.45, z=0.85)),
            aspectmode="manual",
            aspectratio=dict(x=1.05, y=1.15, z=0.72),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK)),
    )
    return fig


def ks_surface_3d_fig(spot: float = SPOT) -> go.Figure:
    x, y, z, slices = mesh_ks_surface(spot)
    return _surface_3d(
        x,
        y,
        z,
        slices,
        xlabel="K / S",
        ylabel="calendar days",
        title="Total-variance interpolant · Strike ANTH, 16 Sep 2026",
        atm_x=1.0,
    )


def delta_surface_3d_fig() -> go.Figure:
    x, y, z, slices = mesh_delta_surface()
    return _surface_3d(
        x,
        y,
        z,
        slices,
        xlabel="call delta",
        ylabel="calendar days",
        title="PCHIP total-variance interpolant · call delta × time",
        atm_x=0.5,
    )


def density_fig(spot: float = SPOT, label: str = "Nov 21") -> go.Figure:
    sl = next(s for s in build_surface(spot) if s.expiry.label == label)
    # $T implied cap axis matches the note (2170 → 2.17).
    caps = np.linspace(1.50, 3.00, 120)
    strikes = caps * 1000.0
    pdf = np.array([digital_and_density(sl, float(k), spot)[1] for k in strikes])
    fig = go.Figure(
        go.Scatter(
            x=caps,
            y=pdf,
            mode="lines",
            line=dict(color=NAVY, width=2.4),
            hovertemplate="Cap $%{x:.2f}T<br>pdf %{y:.5f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(
        x=spot / 1000.0,
        line=dict(color=GOLD, width=1.5, dash="dash"),
        annotation_text=f"spot {spot / 1000:.2f}T",
        annotation_font=dict(size=11, color=GOLD),
    )
    fig.update_xaxes(title="Strike ($T implied cap)", range=[1.48, 3.02])
    fig.update_yaxes(title="Risk-neutral density")
    fig.update_layout(
        title=f"{label} risk-neutral density · Breeden–Litzenberger on the reconstructed smile"
    )
    return apply_paper(fig)


def returns_fig(funding_on: bool = True) -> go.Figure:
    names = ["In window<br>~Nov 9", "Slip past<br>Nov 21", "Early ~Oct 12<br>rule applied"]
    totals = [s.total if funding_on else s.total - s.funding for s in PAPER_SCENARIOS]
    colors = [GREEN, GOLD, GREEN]
    weighted = weighted_expected_return(funding_on)
    fig = go.Figure(
        go.Bar(
            x=names,
            y=totals,
            marker_color=colors,
            text=[f"{v:+.1%}" for v in totals],
            textposition="outside",
            textfont=dict(color=INK, size=13),
            hovertemplate="%{x}<br>%{y:.1%}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_hline(
        y=weighted,
        line=dict(color=NAVY, width=1.5, dash="dash"),
        annotation_text=f"60/30/10 weighted {weighted:.1%}",
        annotation_position="top right",
        annotation_font=dict(size=12, color=NAVY),
    )
    funding_label = "24% funding" if funding_on else "0% funding"
    fig.update_yaxes(title="Return on $1M perp notional", tickformat=".0%", rangemode="tozero")
    fig.update_layout(title=f"Full-book simulation through Nov 21 · {funding_label}")
    return apply_paper(fig, height=440)


def attribution_fig(funding_on: bool = True) -> go.Figure:
    scenarios = ["In window", "Slip", "Early"]
    components = {
        "Options": RED,
        "Hedge": GREEN,
        "Funding": GOLD,
        "Fees": MUTED,
    }
    fig = go.Figure()
    for name, color in components.items():
        key = name.lower()
        ys = [getattr(s, key) for s in PAPER_SCENARIOS]
        if name == "Funding" and not funding_on:
            ys = [0.0] * len(ys)
        fig.add_trace(
            go.Bar(
                x=scenarios,
                y=ys,
                name=name,
                marker_color=color,
                hovertemplate="%{x} · " + name + "<br>%{y:.1%}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", title="Return attribution · percent of perp notional")
    fig.update_yaxes(title="", tickformat=".0%")
    fig.add_hline(y=0.0, line=dict(color=RULE, width=1))
    return apply_paper(fig, height=440)


def forward_vol_oct_nov() -> float:
    near = SURFACE_2026_09_16[5]
    far = SURFACE_2026_09_16[8]
    return forward_vol(near.atm_iv, near.days, far.atm_iv, far.days)


def weekend_fwd() -> float:
    a, b = SURFACE_2026_09_16[1], SURFACE_2026_09_16[2]
    return forward_vol(a.atm_iv, a.days, b.atm_iv, b.days)
