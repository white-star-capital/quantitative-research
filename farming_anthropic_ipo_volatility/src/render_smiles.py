"""
Reconstructed smiles: solid only between the quoted 25Δ wings.

    python src/render_smiles.py

Writes charts/ipo/08_smiles.png.

Dots and the curve between them are the snapshot (ATM + 25Δ). Solid
lines past the outer dots are not — that is the k-quadratic leaving
the quoted band. Dash is omitted; the line stops at the wing dots.
Do not add a 10Δ point to complete the smile.
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
from anth_vol_surface import SliceQuote, build_surface, iv_from_strike  # noqa: E402

PACK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PACK_ROOT / "charts" / "ipo" / "08_smiles.png"

PAPER = "#F4F1EA"
INK = "#142033"
MUTED = "#5B6573"
RULE = "#D7DCE3"

# Four slices from the note. Colors match the existing figure.
SLICE_STYLES: tuple[tuple[str, str], ...] = (
    ("Sep 23", "#9B3A32"),
    ("Oct 17", "#0B1F3A"),
    ("Nov 7", "#2A6F7F"),
    ("Nov 21", "#B8893A"),
)

MNY_LO, MNY_HI = 0.80, 1.36
IV_LO, IV_HI = 0.55, 0.90
N_SOLID = 80


def _by_label() -> dict[str, SliceQuote]:
    return {sl.expiry.label: sl for sl in build_surface()}


def quoted_smile(sl: SliceQuote, n: int = N_SOLID) -> tuple[np.ndarray, np.ndarray]:
    """Moneyness and IV on the solid quoted segment: 25Δ put → 25Δ call."""
    lo, hi = (sl.k_25p / SPOT, sl.k_25c / SPOT)
    mny = np.linspace(lo, hi, n)
    iv = np.array([iv_from_strike(sl, SPOT * m) for m in mny])
    return mny, iv


def render() -> plt.Figure:
    slices = _by_label()
    fig, ax = plt.subplots(figsize=(10.2, 5.6), facecolor=PAPER)
    ax.set_facecolor("white")

    ax.axvline(1.0, color="#C5CAD3", lw=1.0, zorder=0)

    for label, color in SLICE_STYLES:
        sl = slices[label]
        mny, iv = quoted_smile(sl)
        days = int(sl.days)
        ax.plot(
            mny,
            iv,
            color=color,
            lw=1.8,
            solid_capstyle="round",
            label=f"{label} {days}d",
            zorder=2,
        )
        pins_m = [sl.k_25p / SPOT, 1.0, sl.k_25c / SPOT]
        pins_iv = [sl.iv_25p, sl.atm, sl.iv_25c]
        ax.scatter(
            pins_m,
            pins_iv,
            s=28,
            color=color,
            zorder=3,
            edgecolors="white",
            linewidths=0.6,
        )

    ax.set_xlim(MNY_LO, MNY_HI)
    ax.set_ylim(IV_LO, IV_HI)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    ax.set_xlabel("Strike / spot", color=MUTED)
    ax.set_ylabel("Implied vol", color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.set_title(
        "Reconstructed smiles · dots are the quoted ATM and 25Δ wings",
        color=INK,
        fontsize=12,
        fontweight="600",
        pad=10,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.55, 1.0),
        ncol=4,
        frameon=False,
        fontsize=9,
        labelcolor=INK,
    )
    fig.tight_layout()
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconstructed smiles, solid only inside 25Δ.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    fig = render()
    out = args.out if args.out.is_absolute() else PACK_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(main())
