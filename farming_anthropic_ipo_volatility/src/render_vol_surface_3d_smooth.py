"""
Smooth 3D Strike ANTH vol surface in call-delta × time.

    python src/render_vol_surface_3d_smooth.py

Writes charts/ipo/13_surface_3d_delta.png.

Δ ∈ [0.25, 0.75] is a rectangle on every expiry, so the mesh is a sheet,
not a K/S scarf. Calendar interpolation is PCHIP in total variance w(T).
Listed ATMs still pin. The weekend trench stays. No Gaussian blur, no
spline in IV, no 10Δ wings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anth_vol_surface import (  # noqa: E402
    build_surface,
    surface_grid_delta_pchip,
)

PACK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PACK_ROOT / "charts" / "ipo" / "13_surface_3d_delta.png"

PAPER = "#F4F1EA"
INK = "#142033"
MUTED = "#5B6573"

N_DAYS = 200
N_DELTA = 120
DAYS_LO, DAYS_HI = 1.0, 65.0
DELTA_LO, DELTA_HI = 0.25, 0.75


def mesh_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    slices = build_surface()
    days = np.linspace(DAYS_LO, DAYS_HI, N_DAYS)
    deltas = np.linspace(DELTA_LO, DELTA_HI, N_DELTA)
    z = np.array(surface_grid_delta_pchip(slices, days.tolist(), deltas.tolist()))
    x, y = np.meshgrid(deltas, days)
    return x, y, z, slices


def render(elev: float = 22.0, azim: float = -58.0) -> plt.Figure:
    x, y, z, slices = mesh_surface()
    vmin, vmax = float(z.min()), float(z.max())

    fig = plt.figure(figsize=(9.4, 6.9), facecolor=PAPER)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PAPER)
    ax.computed_zorder = False
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(PAPER)
    ax.yaxis.pane.set_edgecolor(PAPER)
    ax.zaxis.pane.set_edgecolor(PAPER)

    surf = ax.plot_surface(
        x,
        y,
        z,
        cmap="magma",
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        rstride=1,
        cstride=1,
        shade=True,
        vmin=vmin,
        vmax=vmax,
        zorder=0,
    )

    ax.scatter(
        [0.5] * len(slices),
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
    ax.set_xlim(DELTA_LO, DELTA_HI)
    ax.set_ylim(DAYS_LO, DAYS_HI)
    ax.set_zlim(vmin, vmax)
    ax.set_xlabel("call delta", color=MUTED, labelpad=8)
    ax.set_ylabel("calendar days", color=MUTED, labelpad=8)
    ax.set_zlabel("IV", color=MUTED, labelpad=6)
    ax.zaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(
        "PCHIP total-variance interpolant · call delta × time",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smooth PCHIP vol surface in call-delta × time.")
    p.add_argument("--elev", type=float, default=22.0)
    p.add_argument("--azim", type=float, default=-58.0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    fig = render(elev=args.elev, azim=args.azim)
    out = args.out if args.out.is_absolute() else PACK_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(main())
