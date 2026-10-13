"""Market context metrics (BS_PRICER_IMPLEMENTATION_GUIDE.md §3.6). These replace Volmex inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass

from terminal_models.core.quant.black76 import N
from terminal_models.core.quant.smile import TENOR_LONG, TENOR_SHORT, Expiry, Surface

HOURS_PER_YEAR = 8760
DAY = 1 / 365


@dataclass(frozen=True)
class Metric:
    value: float | None
    flags: tuple[str, ...] = ()


def atm_iv(surface: Surface, tenor_days: float) -> Metric:
    """σ(k = 0, T = τ): constant maturity, identical method for every asset."""
    p = surface.sigma(0.0, tenor_days * DAY)
    return Metric(p.sigma, p.flags) if p else Metric(None)


def realized_vol(closes: list[float], window_days: int) -> float | None:
    """√(8760 · mean(r²)) over the last window×24 hourly log returns (zero-mean, 24/7)."""
    n = window_days * 24
    if len(closes) < n + 1 or any(c <= 0 for c in closes[-(n + 1) :]):
        return None
    tail = closes[-(n + 1) :]
    returns = [math.log(b / a) for a, b in zip(tail, tail[1:], strict=False)]
    return math.sqrt(HOURS_PER_YEAR * sum(r * r for r in returns) / n)


def change_24h(closes: list[float]) -> float | None:
    if len(closes) < 25 or closes[-25] <= 0:
        return None
    return closes[-1] / closes[-25] - 1


def _forward_delta(expiry: Expiry, k: float, r: float, is_call: bool) -> float:
    p = expiry.sigma(k)
    assert p is not None  # mark IVs exist on every node
    sd = p.sigma * math.sqrt(expiry.T)
    d1 = (-k + 0.5 * sd * sd) / sd
    df = math.exp(-r * expiry.T)
    return df * N(d1) if is_call else -df * N(-d1)


def strike_for_delta(expiry: Expiry, target: float, r: float, is_call: bool) -> float:
    """k where |forward delta| = target on this expiry's smile (bisection on k ∈ [−5, 5]).

    Call delta falls as k rises; |put delta| rises as k falls. Both are monotone for the smiles
    we build (flat wings), so bisection on the sign is safe.
    """
    lo, hi = -5.0, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        d = abs(_forward_delta(expiry, mid, r, is_call))
        need_higher_k = d > target if is_call else d < target
        lo, hi = (mid, hi) if need_higher_k else (lo, mid)
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Skew:
    """Per-expiry vols at ATM and the 25Δ strikes."""

    atm: float
    call25: float
    put25: float

    @property
    def rr(self) -> float:
        return self.call25 - self.put25

    @property
    def bf(self) -> float:
        return 0.5 * (self.call25 + self.put25) - self.atm


def expiry_skew(expiry: Expiry, r: float) -> Skew:
    def vol(k: float) -> float:
        p = expiry.sigma(k)
        assert p is not None
        return p.sigma

    return Skew(
        atm=vol(0.0),
        call25=vol(strike_for_delta(expiry, 0.25, r, True)),
        put25=vol(strike_for_delta(expiry, 0.25, r, False)),
    )


def skew_at_tenor(surface: Surface, tenor_days: float, r: float) -> tuple[Metric, Metric]:
    """RR25 and BF25 at a constant tenor: per bracketing expiry solve the 25Δ strikes on its smile,
    then interpolate each leg's total variance to the tenor (§3.6)."""
    T = tenor_days * DAY  # noqa: N806
    exps = surface.expiries
    if T <= exps[0].T:
        legs, flags = [(exps[0], 1.0)], (TENOR_SHORT,) if exps[0].T - T > 1e-9 else ()
    elif T >= exps[-1].T:
        legs, flags = [(exps[-1], 1.0)], (TENOR_LONG,) if T - exps[-1].T > 1e-9 else ()
    else:
        j = next(i for i in range(len(exps) - 1) if exps[i].T <= T < exps[i + 1].T)
        a, b = exps[j], exps[j + 1]
        wt = (T - a.T) / (b.T - a.T)
        legs, flags = [(a, 1 - wt), (b, wt)], ()

    def blend(pick) -> float:
        if len(legs) == 1:
            return pick(expiry_skew(legs[0][0], r))
        w = sum(weight * pick(expiry_skew(e, r)) ** 2 * e.T for e, weight in legs)
        return math.sqrt(w / T)

    call25 = blend(lambda s: s.call25)
    put25 = blend(lambda s: s.put25)
    atm = blend(lambda s: s.atm)
    return Metric(call25 - put25, flags), Metric(0.5 * (call25 + put25) - atm, flags)
