"""Smile and surface interpolation (BS_PRICER_IMPLEMENTATION_GUIDE.md §3.2–3.3).

Sticky-strike, piecewise linear in total variance w = σ²T:
- within an expiry, linear in w between adjacent nodes in log-moneyness k = ln(K/F);
- across expiries, linear in w at fixed k;
- outside the data, flat σ at the nearest node/expiry, always flagged.

A `Surface` can be built from venue rows (the model) or from a stored run's payload (evaluation),
and both must give identical numbers; the Rust port in api/src/pricing/surface.rs mirrors this file.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from datetime import datetime

SECONDS_PER_YEAR = 365 * 86400
EXACT_TENOR_YEARS = 60 / SECONDS_PER_YEAR  # |T − T_i| < 60 s counts as that expiry

STRIKE_EXTRAPOLATED = "strike_extrapolated"
TENOR_SHORT = "tenor_extrapolated_short"
TENOR_LONG = "tenor_extrapolated_long"


def year_fraction(start: datetime, end: datetime) -> float:
    """Calendar time, ACT/365, 24/7."""
    return (end - start).total_seconds() / SECONDS_PER_YEAR


@dataclass(frozen=True)
class Node:
    """A listed OTM option whose mark IV anchors the smile (§3.1)."""

    k: float
    strike: float
    iv: float
    iv_bid: float | None
    iv_ask: float | None
    quality: str  # good | wide | one_sided | no_quote


@dataclass(frozen=True)
class SmilePoint:
    sigma: float
    flags: tuple[str, ...]
    nodes: tuple[Node, ...]  # the one or two nodes the value came from


@dataclass
class Expiry:
    expiry: datetime
    T: float  # noqa: N815 - years from valuation time
    forward: float
    nodes: list[Node] = field(default_factory=list)  # sorted by k, unique k

    def __post_init__(self) -> None:
        self.nodes = sorted(self.nodes, key=lambda nd: nd.k)
        self._ks = [nd.k for nd in self.nodes]

    @property
    def k_min(self) -> float:
        return self._ks[0]

    @property
    def k_max(self) -> float:
        return self._ks[-1]

    def bracket(self, k: float) -> tuple[tuple[Node, ...], float, bool]:
        """Nodes around k, the interpolation weight on the upper node, and whether k is off-grid."""
        if k <= self._ks[0]:
            return (self.nodes[0],), 0.0, k < self._ks[0]
        if k >= self._ks[-1]:
            return (self.nodes[-1],), 0.0, k > self._ks[-1]
        j = bisect.bisect_right(self._ks, k) - 1
        lo, hi = self.nodes[j], self.nodes[j + 1]
        return (lo, hi), (k - lo.k) / (hi.k - lo.k), False

    def sigma(self, k: float, attr: str = "iv") -> SmilePoint | None:
        """σ_i(k) from node `attr` (iv, iv_bid or iv_ask). None if a bracketing node lacks it."""
        nodes, weight, off_grid = self.bracket(k)
        vols = [getattr(nd, attr) for nd in nodes]
        if any(v is None for v in vols):
            return None
        flags = (STRIKE_EXTRAPOLATED,) if off_grid else ()
        if len(nodes) == 1:
            return SmilePoint(vols[0], flags, nodes)
        w_lo, w_hi = vols[0] ** 2 * self.T, vols[1] ** 2 * self.T
        w = w_lo + (w_hi - w_lo) * weight
        return SmilePoint(math.sqrt(w / self.T), flags, nodes)

    def display_sigma(self, k: float) -> float | None:
        """Display grids never extrapolate (§3.7): None outside [k_min, k_max]."""
        if k < self.k_min or k > self.k_max:
            return None
        point = self.sigma(k)
        return point.sigma if point else None


@dataclass(frozen=True)
class SurfacePoint:
    sigma: float
    flags: tuple[str, ...]
    expiries: tuple[Expiry, ...]  # bracketing expiries (one when exact or extrapolated)
    nodes: tuple[Node, ...]
    interpolation: str  # exact | interpolated | extrapolated


class Surface:
    """Expiries sorted by tenor, plus spot for the forward curve's T=0 anchor."""

    def __init__(self, spot: float, expiries: list[Expiry]) -> None:
        if not expiries:
            raise ValueError("surface needs at least one expiry")
        if spot <= 0:
            raise ValueError("non_positive_input")
        self.spot = spot
        self.expiries = sorted(expiries, key=lambda e: e.T)
        self._ts = [e.T for e in self.expiries]

    @property
    def t_last(self) -> float:
        return self._ts[-1]

    def forward(self, T: float) -> tuple[float, tuple[str, ...]]:  # noqa: N803
        """ln F linear in T through (0, ln S), (T_i, ln F_i); last slope extended beyond T_n."""
        pts = [(0.0, math.log(self.spot))] + [(e.T, math.log(e.forward)) for e in self.expiries]
        if T >= pts[-1][0]:
            (t0, f0), (t1, f1) = pts[-2], pts[-1]
            flags = (TENOR_LONG,) if T - t1 >= EXACT_TENOR_YEARS else ()
        else:
            j = bisect.bisect_right([p[0] for p in pts], T) - 1
            (t0, f0), (t1, f1) = pts[j], pts[j + 1]
            flags = ()
        return math.exp(f0 + (f1 - f0) * (T - t0) / (t1 - t0)), flags

    def _exact(self, T: float) -> Expiry | None:  # noqa: N803
        j = bisect.bisect_left(self._ts, T)
        for i in (j - 1, j):
            if 0 <= i < len(self._ts) and abs(self._ts[i] - T) < EXACT_TENOR_YEARS:
                return self.expiries[i]
        return None

    def sigma(self, k: float, T: float, attr: str = "iv") -> SurfacePoint | None:  # noqa: N803
        """σ(k, T) per §3.3. None only when a bid/ask lookup lacks a bracketing node quote."""
        exact = self._exact(T)
        if exact is not None:
            p = exact.sigma(k, attr)
            return p and SurfacePoint(
                p.sigma, p.flags, (exact,), p.nodes, _status(p.flags, "exact")
            )
        if T < self._ts[0]:
            first = self.expiries[0]
            p = first.sigma(k, attr)
            flags = (*p.flags, TENOR_SHORT) if p else ()
            return p and SurfacePoint(p.sigma, flags, (first,), p.nodes, "extrapolated")
        if T > self._ts[-1]:
            last = self.expiries[-1]
            p = last.sigma(k, attr)
            flags = (*p.flags, TENOR_LONG) if p else ()
            return p and SurfacePoint(p.sigma, flags, (last,), p.nodes, "extrapolated")
        j = bisect.bisect_right(self._ts, T) - 1
        a, b = self.expiries[j], self.expiries[j + 1]
        pa, pb = a.sigma(k, attr), b.sigma(k, attr)
        if pa is None or pb is None:
            return None
        w_a, w_b = pa.sigma**2 * a.T, pb.sigma**2 * b.T
        w = w_a + (w_b - w_a) * (T - a.T) / (b.T - a.T)
        flags = tuple(dict.fromkeys(pa.flags + pb.flags))
        return SurfacePoint(
            math.sqrt(w / T), flags, (a, b), pa.nodes + pb.nodes, _status(flags, "interpolated")
        )


def _status(flags: tuple[str, ...], default: str) -> str:
    return "extrapolated" if flags else default


def k_grid(kmin: float, kmax: float, step: float) -> list[float]:
    """Inclusive grid, rounded so values are exact decimals in JSON (-0.5, -0.475, …)."""
    count = round((kmax - kmin) / step)
    return [round(kmin + i * step, 10) for i in range(count + 1)]
