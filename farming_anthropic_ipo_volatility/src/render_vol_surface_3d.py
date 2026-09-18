"""
3D Strike ANTH vol surface: mesh interpolate_iv, not the nine raw quotes.

    python src/render_vol_surface_3d.py
    python src/render_vol_surface_3d.py --elev 24 --azim -58 --out charts/ipo/12_surface_3d.png

Z is decimal vol (0.73 = 73%). Scatter listed ATM at (1.0, sl.days, sl.atm)
in the same units — if those dots float, the interpolant is wrong.

Cells outside each tenor's 25Δ strikes are masked. A 2-day 25Δ put is
about K/S ≈ 0.97, not 0.86 — that yellow wall was the k-quadratic off
the quoted band. The grid is wide enough for Nov 21 (~0.85–1.32); the
mask, not a shared rectangle, sets the edge.

    surface may look like this; the front-left spike may not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anth_ipo_book import SPOT  # noqa: E402
from anth_vol_surface import (  # noqa: E402
    build_surface,
    interpolate_iv,
    quoted_moneyness_band,
    surface_grid,
)

PACK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PACK_ROOT / "charts" / "ipo" / "12_surface_3d.png"
DEFAULT_HEATMAP = PACK_ROOT / "charts" / "ipo" / "09_surface.png"

PAPER = "#F4F1EA"
INK = "#142033"
MUTED = "#5B6573"

N_DAYS = 48
N_MNY = 40
DAYS_LO, DAYS_HI = 2.0, 65.0
# Axis box covers the union of listed 25Δ bands (Nov 21 ≈ 0.85–1.32).
# The mesh itself is *not* this rectangle — see mesh_surface().
MNY_LO, MNY_HI = 0.84, 1.32


def mesh_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    """
    Rows = calendar days, cols = 25Δ put → 25Δ call at that tenor.

    Every cell is inside the quoted band. A shared rectangle 0.86–1.16
    is what made the yellow wall and the near-right cliff.
    """
    slices = build_surface()
    days = np.linspace(DAYS_LO, DAYS_HI, N_DAYS)
    u = np.linspace(0.0, 1.0, N_MNY)  # 0 = 25Δ put, 1 = 25Δ call
    x = np.empty((N_DAYS, N_MNY))
    y = np.empty((N_DAYS, N_MNY))
    z = np.empty((N_DAYS, N_MNY))
    for i, d in enumerate(days):
        lo, hi = quoted_moneyness_band(slices, float(d), SPOT)
        mny = lo + u * (hi - lo)
        x[i, :] = mny
        y[i, :] = d
        z[i, :] = [interpolate_iv(slices, float(d), SPOT * m, SPOT) for m in mny]
    return x, y, z, slices


def mask_quoted_band(
    z: np.ndarray,
    days: np.ndarray,
    mny: np.ndarray,
    slices: tuple,
    spot: float = SPOT,
) -> np.ndarray:
    """NaN cells outside the 25Δ band at that tenor. plot_surface skips them."""
    out = np.array(z, dtype=float, copy=True)
    for i, d in enumerate(days):
        lo, hi = quoted_moneyness_band(slices, float(d), spot)
        out[i, (mny < lo) | (mny > hi)] = np.nan
    return out


def _style_3d_axes(ax) -> None:
    ax.set_facecolor(PAPER)
    ax.computed_zorder = False
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(PAPER)
    ax.yaxis.pane.set_edgecolor(PAPER)
    ax.zaxis.pane.set_edgecolor(PAPER)


def render(elev: float = 24.0, azim: float = -58.0) -> plt.Figure:
    x, y, z, slices = mesh_surface()
    finite = z[np.isfinite(z)]
    vmin, vmax = float(finite.min()), float(finite.max())

    fig = plt.figure(figsize=(9.2, 6.8), facecolor=PAPER)
    ax = fig.add_subplot(111, projection="3d")
    _style_3d_axes(ax)

    surf = ax.plot_surface(
        x,
        y,
        z,
        cmap="magma",
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
        shade=True,
        vmin=vmin,
        vmax=vmax,
        zorder=0,
    )

    ax.scatter(
        [1.0] * len(slices),
        [sl.days for sl in slices],
        [sl.atm for sl in slices],
        color="white",
        edgecolors=INK,
        linewidths=0.8,
        s=42,
        depthshade=False,
        zorder=10,
    )

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(MNY_LO, MNY_HI)
    ax.set_ylim(1.0, DAYS_HI)
    ax.set_zlim(vmin, vmax)
    ax.set_xlabel("K / S", color=MUTED, labelpad=8)
    ax.set_ylabel("calendar days", color=MUTED, labelpad=8)
    ax.set_zlabel("IV", color=MUTED, labelpad=6)
    ax.zaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(
        "Total-variance interpolant · Strike ANTH, 16 Sep 2026",
        color=INK,
        pad=12,
        fontsize=12,
        fontweight="600",
    )
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, aspect=18)
    cbar.set_label("IV", color=MUTED)
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.04, top=0.92)
    return fig


def render_heatmap() -> plt.Figure:
    """2D pin-check of the same interpolant. ATM dots must sit on K/S = 1."""
    slices = build_surface()
    days = np.linspace(1.0, DAYS_HI, 80)
    mny = np.linspace(MNY_LO, MNY_HI, 96)
    z = np.array(surface_grid(slices, days.tolist(), mny.tolist(), SPOT), dtype=float)
    z = mask_quoted_band(z, days, mny, slices)
    finite = z[np.isfinite(z)]

    fig, ax = plt.subplots(figsize=(10.2, 6.2), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    mesh = ax.pcolormesh(
        mny,
        days,
        z,
        cmap="magma",
        shading="auto",
        vmin=float(finite.min()),
        vmax=float(finite.max()),
        rasterized=True,
    )
    ax.scatter(
        [1.0] * len(slices),
        [sl.days for sl in slices],
        color="white",
        edgecolors=INK,
        linewidths=0.6,
        s=22,
        zorder=5,
    )
    for sl in slices:
        ax.annotate(
            sl.expiry.label,
            (1.0, sl.days),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=8,
            color=INK,
            va="center",
        )
    ax.set_xlim(MNY_LO, MNY_HI)
    ax.set_ylim(0.0, DAYS_HI)
    ax.set_xlabel("Strike / spot", color=MUTED)
    ax.set_ylabel("Calendar days from 16 Sep", color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#D7DCE3")
    ax.set_title(
        "Total-variance interpolant · Strike ANTH surface, 16 Sep 2026",
        color=INK,
        fontsize=12,
        fontweight="600",
        pad=10,
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("IV (%)", color=MUTED)
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    cbar.ax.tick_params(colors=MUTED)
    fig.tight_layout()
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mesh interpolate_iv as a 3D vol surface.")
    p.add_argument("--elev", type=float, default=24.0)
    p.add_argument("--azim", type=float, default=-58.0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--heatmap", type=Path, default=DEFAULT_HEATMAP)
    p.add_argument("--no-heatmap", action="store_true")
    return p.parse_args(argv)


def _save(fig: plt.Figure, path: Path) -> Path:
    out = path if path.is_absolute() else PACK_ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    written = [_save(render(elev=args.elev, azim=args.azim), args.out)]
    if not args.no_heatmap:
        written.append(_save(render_heatmap(), args.heatmap))
    return written


if __name__ == "__main__":
    for path in main():
        print(path)
