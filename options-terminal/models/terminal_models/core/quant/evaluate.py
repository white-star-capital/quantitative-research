"""Evaluation over a stored bs_pricer run (BS_PRICER_IMPLEMENTATION_GUIDE.md §6.2).

Normative reference for api/src/pricing/evaluate.rs (API A-33). Pure: everything comes from the
stored payload and the request; nothing is fetched. Errors raise EvaluateError(code) and map to
400 problems; unsolvable quote IVs are `None` plus a reason, never a clamped number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from terminal_models.core.payloads import PayloadError, get_section, records, scalars
from terminal_models.core.quant.black76 import black76, implied_vol
from terminal_models.core.quant.smile import (
    Expiry,
    Node,
    Surface,
    year_fraction,
)

DISTRIBUTION_POINTS = 200
STRIKE_RANGE = (0.2, 5.0)  # × F
PREMIUM_UNITS = ("usd_per_unit", "usd_total", "pct_spot", "coin_per_unit")


class EvaluateError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def parse_utc(value: str, name: str) -> datetime:
    if not value.endswith("Z") or "T" not in value:
        raise EvaluateError(f"invalid_{name}", f"{name} must be an ISO 8601 UTC time ending in Z")
    try:
        return datetime.fromisoformat(value).astimezone(UTC)
    except ValueError as exc:
        raise EvaluateError(f"invalid_{name}", f"{name} is not a valid timestamp") from exc


def utc(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- stored payload → surface ------------------------------------------------------------


@dataclass(frozen=True)
class StoredRun:
    run_id: str
    as_of: datetime
    underlying: str
    spot: float
    usd_rate: float
    surface: Surface


def surface_from_payload(
    run_id: str, as_of: str, scope: dict[str, str], payload: dict
) -> StoredRun:
    """Rebuild the model's surface from `expiries` + node rows of `chain` (bit-identical inputs)."""
    valuation = parse_utc(as_of, "as_of")
    try:
        ctx = scalars(get_section(payload, "context"))
        chain = records(get_section(payload, "chain"))
        expiry_rows = records(get_section(payload, "expiries"))
    except PayloadError as exc:
        raise EvaluateError("run_incompatible", f"stored run is missing a section: {exc}") from exc
    nodes: dict[str, list[Node]] = {}
    for row in chain:
        if not row["is_node"]:
            continue
        nodes.setdefault(row["expiry_ts"], []).append(
            Node(
                k=row["k"],
                strike=row["strike"],
                iv=row["iv_mark"],
                iv_bid=row["iv_bid"],
                iv_ask=row["iv_ask"],
                quality=row["node_quality"],
            )
        )
    expiries = []
    for row in expiry_rows:
        when = parse_utc(row["expiry_ts"], "expiry_ts")
        expiries.append(
            Expiry(when, year_fraction(valuation, when), row["forward"], nodes[row["expiry_ts"]])
        )
    return StoredRun(
        run_id=run_id,
        as_of=valuation,
        underlying=scope["underlying"],
        spot=ctx["spot"],
        usd_rate=ctx["usd_rate"],
        surface=Surface(ctx["spot"], expiries),
    )


# ---- request ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    option_type: str
    expiry: datetime
    strike: float
    size: float = 1.0
    premium: float | None = None
    premium_unit: str | None = None
    side: str | None = None
    spot_ref: float | None = None
    quoted_at: datetime | None = None


def _flag(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


FLAG_TEXT = {
    "strike_extrapolated": "Strike is outside the listed strikes; vol held flat at nearest node",
    "tenor_extrapolated_short": "Expiry is before the first listed expiry; vol held flat",
    "tenor_extrapolated_long": "Expiry is after the last listed expiry; vol and forward extended",
}


def evaluate(run: StoredRun, req: Request, config: dict[str, Any], run_stale: bool) -> dict:
    """§6.2 steps 1–6. `config` is the slot's `evaluation` block."""
    if req.option_type not in ("call", "put"):
        raise EvaluateError("invalid_option_type", "option_type must be call or put")
    is_call = req.option_type == "call"
    if not (req.size > 0 and math.isfinite(req.size)):
        raise EvaluateError("invalid_size", "size must be positive")
    if not (req.strike > 0 and math.isfinite(req.strike)):
        raise EvaluateError("strike_out_of_range", "strike must be positive")

    surface, S, r = run.surface, run.spot, run.usd_rate  # noqa: N806
    T = year_fraction(run.as_of, req.expiry)  # noqa: N806
    if T <= 0:
        raise EvaluateError("expiry_in_past", "expiry is at or before the run's valuation time")
    max_t = config["max_tenor_multiple"] * surface.t_last
    if T > max_t:
        raise EvaluateError(
            "tenor_out_of_range",
            f"expiry is beyond {config['max_tenor_multiple']}x the last listed expiry",
        )

    F, f_flags = surface.forward(T)  # noqa: N806
    lo, hi = STRIKE_RANGE
    if not (lo * F <= req.strike <= hi * F):
        raise EvaluateError(
            "strike_out_of_range",
            f"strike must be within [{lo}F, {hi}F] = [{lo * F:.6g}, {hi * F:.6g}]",
        )
    K = req.strike  # noqa: N806
    k = math.log(K / F)
    mark = surface.sigma(k, T)
    assert mark is not None  # mark IVs exist on every node
    bid = surface.sigma(k, T, "iv_bid")
    ask = surface.sigma(k, T, "iv_ask")
    sigma = mark.sigma

    flags = [_flag(c, FLAG_TEXT[c]) for c in dict.fromkeys(mark.flags + f_flags)]
    if any(nd.quality != "good" for nd in mark.nodes):
        flags.append(_flag("low_quality_nodes", "A bracketing node is not two-sided and tight"))
    same_day = [e for e in surface.expiries if e.expiry.date() == req.expiry.date()]
    if same_day and all(abs((e.expiry - req.expiry).total_seconds()) >= 60 for e in same_day):
        flags.append(
            _flag(
                "expiry_time_differs_from_venue",
                f"Venue expiry that day is {utc(same_day[0].expiry)}",
            )
        )
    if run_stale:
        flags.append(_flag("run_stale", "The run is older than the slot's max age"))

    g = black76(F, K, T, sigma, r, is_call, S)
    fair = g["price"]
    dte = T * 365
    side_sign = -1.0 if req.side == "sell" else 1.0
    greek_keys = ("delta", "delta_spot", "gamma_1pct", "vega_1pt", "theta_1d")
    screen_bid = black76(F, K, T, bid.sigma, r, is_call, S)["price"] if bid else None
    screen_ask = black76(F, K, T, ask.sigma, r, is_call, S)["price"] if ask else None

    data: dict[str, Any] = {
        "run_id": run.run_id,
        "as_of": utc(run.as_of),
        "valuation_time": utc(run.as_of),
        "underlying": run.underlying,
        "spot": S,
        "contract": {
            "option_type": req.option_type,
            "expiry": utc(req.expiry),
            "strike": K,
            "size": req.size,
            "T": T,
            "dte": dte,
        },
        "market": {
            "forward": F,
            "k": k,
            "iv_mark": sigma,
            "iv_bid": bid.sigma if bid else None,
            "iv_ask": ask.sigma if ask else None,
            "interpolation": mark.interpolation,
            "bracketing_expiries": [utc(e.expiry) for e in mark.expiries],
            "screen_bid_usd": screen_bid,
            "screen_ask_usd": screen_ask,
        },
        "fair": {
            "per_unit_usd": fair,
            "total_usd": fair * req.size,
            "pct_spot": fair / S,
            "greeks_per_unit": {key: g[key] for key in greek_keys} | {"p_itm": g["p_itm"]},
            "greeks_position": {key: g[key] * req.size * side_sign for key in greek_keys},
            "breakeven": K + fair if is_call else K - fair,
            "yield_ann": fair / S * 365 / dte,
        },
        "quote": None,
        "distribution": distribution(F, T, surface),
        "flags": flags,
    }

    if req.premium is not None:
        data["quote"] = _quote(
            req, run, config, F, K, T, sigma, r, is_call, fair, screen_bid, screen_ask, flags
        )
    return data


def _quote(
    req: Request,
    run: StoredRun,
    config,
    F,
    K,
    T,
    sigma,
    r,
    is_call,
    fair,  # noqa: N803
    screen_bid,
    screen_ask,
    flags,
) -> dict[str, Any]:
    S = run.spot  # noqa: N806
    if req.side not in ("sell", "buy"):
        raise EvaluateError("premium_requires_side", "premium needs side=sell or side=buy")
    if req.premium_unit not in PREMIUM_UNITS:
        raise EvaluateError(
            "premium_requires_unit",
            f"premium needs premium_unit, one of: {', '.join(PREMIUM_UNITS)}",
        )
    if not (req.premium > 0 and math.isfinite(req.premium)):
        raise EvaluateError("invalid_premium", "premium must be positive")
    if req.spot_ref is not None and not (req.spot_ref > 0 and math.isfinite(req.spot_ref)):
        raise EvaluateError("invalid_spot_ref", "spot_ref must be positive")

    if req.premium_unit == "usd_per_unit":
        prem = req.premium
    elif req.premium_unit == "usd_total":
        prem = req.premium / req.size
    elif req.premium_unit == "pct_spot":
        if req.spot_ref is None:
            raise EvaluateError("pct_spot_requires_spot_ref", "pct_spot premiums need spot_ref")
        prem = req.premium / 100 * req.spot_ref  # premium is in percent: 2.45 = 2.45%
    else:  # coin_per_unit
        prem = req.premium * S

    s_ref = req.spot_ref if req.spot_ref is not None else S
    f_ref = F * s_ref / S  # keeps the basis
    iv_quote, reason = implied_vol(f_ref, K, T, prem, r, is_call, s_ref)
    sell = req.side == "sell"
    edge = prem - fair if sell else fair - prem
    vol_edge = None
    if iv_quote is not None:
        vol_edge = (iv_quote - sigma) * 100 * (1 if sell else -1)

    if screen_bid is not None and screen_ask is not None:
        position = (
            "below_bid" if prem < screen_bid else "above_ask" if prem > screen_ask else "inside"
        )
        width = screen_ask - screen_bid
        frac = (prem - screen_bid) / width if width > 0 else None
    else:
        position, frac = "n/a", None

    max_age = config["quote_max_age_minutes"]
    if req.quoted_at is not None:
        age_min = (run.as_of - req.quoted_at).total_seconds() / 60
        if age_min > max_age:
            flags.append(
                _flag(
                    "quote_stale",
                    f"Quote is {age_min / 1440:.1f} days old relative to "
                    f"the run (limit {max_age} minutes)",
                )
            )
    if req.spot_ref is not None:
        moved = abs(S / req.spot_ref - 1) * 100
        if moved > config["spot_move_flag_pct"]:
            flags.append(_flag("spot_moved", f"Spot moved {moved:.2f}% from the quote's spot ref"))

    dte = T * 365
    return {
        "side": req.side,
        "premium_per_unit_usd": prem,
        "premium_total_usd": prem * req.size,
        "iv_quote": iv_quote,
        "iv_quote_status": "ok" if reason is None else reason,
        "pct_of_fair": prem / fair if fair > 0 else None,
        "edge_per_unit_usd": edge,
        "edge_total_usd": edge * req.size,
        "vol_edge_vp": vol_edge,
        "screen_position": position,
        "screen_position_frac": frac,
        "fair_at_spot_ref": black76(f_ref, K, T, sigma, r, is_call, s_ref)["price"],
        "yield_ann": prem / S * 365 / dte,
    }


def distribution(F: float, T: float, surface: Surface) -> dict[str, Any]:  # noqa: N803
    """Lognormal pdf at the ATM vol for T (ignores skew): μ = ln F − ½σ²T over F·e^(±3σ√T)."""
    atm = surface.sigma(0.0, T)
    assert atm is not None
    sd = atm.sigma * math.sqrt(T)
    mu = math.log(F) - 0.5 * sd * sd
    lo, hi = F * math.exp(-3 * sd), F * math.exp(3 * sd)
    step = (hi - lo) / (DISTRIBUTION_POINTS - 1)
    xs = [lo + i * step for i in range(DISTRIBUTION_POINTS)]
    pdf = [
        math.exp(-((math.log(x) - mu) ** 2) / (2 * sd * sd)) / (x * sd * math.sqrt(2 * math.pi))
        for x in xs
    ]
    return {"basis": "lognormal_atm", "sigma": atm.sigma, "x": xs, "pdf": pdf}
