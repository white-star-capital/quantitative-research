"""bs_pricer_black76: Black-76 fair values, greeks and vol surface from Deribit marks.

Methodology: BS_PRICER_IMPLEMENTATION_GUIDE.md §3. Payload: §4.3 (composite). Pure: everything
comes from ctx.inputs (MODELS M-5); findings go through ctx.warn.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from terminal_models.core.base import InputSpec, Model, RunContext, Scope
from terminal_models.core.payloads import axis, column, item, section, series, unit
from terminal_models.core.quant.black76 import black76
from terminal_models.core.quant.metrics import (
    atm_iv,
    change_24h,
    expiry_skew,
    realized_vol,
    skew_at_tenor,
)
from terminal_models.core.quant.smile import Surface, k_grid
from terminal_models.core.quant.surface import build_expiries, label, solve_iv

IV = unit("implied_vol", "decimal")
VOL_SPREAD = unit("vol_spread", "decimal")
USD = unit("price_usd")
COUNT = unit("count")
K = unit("moneyness_log")
TENOR = unit("tenor_days")


class BsPricerBlack76(Model):
    def __init__(self, manifest) -> None:
        self.manifest = manifest

    def required_inputs(self, scope: Scope) -> list[InputSpec]:
        u = scope["underlying"]
        days = self.manifest.params["candle_lookback_days"]
        specs = [
            InputSpec("chain", "deribit", {"kind": "chain", "underlying": u}),
            InputSpec("index", "deribit", {"kind": "index", "underlying": u}),
            InputSpec(
                "candles", "deribit", {"kind": "candles_1h", "underlying": u, "lookback_days": days}
            ),
        ]
        if u in ("BTC", "ETH"):
            specs.append(InputSpec("dvol", "deribit", {"kind": "dvol", "underlying": u}))
        return specs

    def run(self, ctx: RunContext) -> dict[str, Any]:
        p = ctx.params
        r = float(p["usd_rate"])
        spot = ctx.inputs["index"].data["spot"]
        rows = ctx.inputs["chain"].data["rows"]
        closes = ctx.inputs["candles"].data["close"]
        dvol = ctx.inputs["dvol"].data["dvol"] if "dvol" in ctx.inputs else None

        expiries, by_expiry = build_expiries(rows, ctx, spot, r)
        surface = Surface(spot, expiries)
        grid = k_grid(p["k_grid"]["min"], p["k_grid"]["max"], p["k_grid"]["step"])
        return {
            "sections": [
                context_section(surface, ctx, closes, dvol, r),
                term_structure_section(surface, r),
                smile_section(surface, grid),
                surface_section(surface, grid),
                expiries_section(surface, r),
                chain_section(surface, by_expiry, ctx, r),
            ]
        }


IMPLEMENTATION = BsPricerBlack76


# ---- sections (§4.3) ----------------------------------------------------------------------


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def _diff(a: float | None, b: float | None) -> float | None:
    return a - b if a is not None and b is not None else None


def context_section(surface: Surface, ctx: RunContext, closes, dvol, r: float) -> dict:
    p = ctx.params
    atm = {d: atm_iv(surface, d) for d in p["atm_tenors_days"]}
    rv = {d: realized_vol(closes, d) for d in p["rv_windows_days"]}
    for d, value in rv.items():
        if value is None:
            ctx.warn("rv_insufficient_history", f"fewer than {d * 24} hourly returns for RV {d}d")
    rr, bf = skew_at_tenor(surface, 30, r)
    term_ratio = _ratio(atm[1].value, atm[30].value)
    nodes = [n for e in surface.expiries for n in e.nodes]

    items = [
        item("spot", "Spot", surface.spot, USD),
        item("spot_change_24h", "Spot 24h", change_24h(closes), unit("ratio", "decimal")),
        *[item(f"atm_iv_{d}d", f"ATM IV {d}d", m.value, IV, m.flags) for d, m in atm.items()],
        *[item(f"rv_{d}d", f"RV {d}d", v, unit("realized_vol", "decimal")) for d, v in rv.items()],
        item("vrp_30d", "VRP 30d", _diff(atm[30].value, rv.get(30)), VOL_SPREAD),
        item("term_ratio", "Term ratio 1d/30d", term_ratio, unit("ratio")),
        item(
            "event_flag",
            "Event",
            None if term_ratio is None else term_ratio >= p["event_ratio_flag"],
            unit("other"),
        ),
        item("rr25_30d", "RR25 30d", rr.value, VOL_SPREAD, rr.flags),
        item("bf25_30d", "BF25 30d", bf.value, VOL_SPREAD, bf.flags),
    ]
    if dvol is not None:
        items.append(item("dvol", "DVOL", dvol, IV))
    items += [
        item("perp_last", "Perp last", closes[-1] if closes else None, USD),
        item("usd_rate", "USD rate", r, unit("rate", "decimal")),
        item("n_expiries", "Expiries", len(surface.expiries), COUNT),
        item("n_nodes", "Nodes", len(nodes), COUNT),
        item("n_nodes_good", "Good nodes", sum(n.quality == "good" for n in nodes), COUNT),
    ]
    return section("context", "Market context", "scalar_set", {"items": items})


def term_structure_section(surface: Surface, r: float) -> dict:
    skews = [expiry_skew(e, r) for e in surface.expiries]
    x = [e.T * 365 for e in surface.expiries]
    return section(
        "term_structure",
        "Term structure",
        "curve",
        {
            "x": axis("Tenor (days)", TENOR, x),
            "series": [
                series("atm_iv", "ATM", IV, [s.atm for s in skews]),
                series("iv_25d_put", "25Δ put", IV, [s.put25 for s in skews]),
                series("iv_25d_call", "25Δ call", IV, [s.call25 for s in skews]),
            ],
        },
    )


def smile_section(surface: Surface, grid: list[float]) -> dict:
    return section(
        "smile",
        "Smiles",
        "curve",
        {
            "x": axis("Log-moneyness ln(K/F)", K, grid),
            "series": [
                series(
                    f"exp_{e.expiry:%Y%m%d}",
                    label(e.expiry),
                    IV,
                    [e.display_sigma(k) for k in grid],
                )
                for e in surface.expiries
            ],
        },
    )


def surface_section(surface: Surface, grid: list[float]) -> dict:
    return section(
        "surface",
        "Vol surface",
        "surface",
        {
            "x": axis("Log-moneyness ln(K/F)", K, grid),
            "y": axis("Tenor (days)", TENOR, [e.T * 365 for e in surface.expiries]),
            "z": {
                "label": "Implied vol",
                "unit": IV,
                "values": [[e.display_sigma(k) for k in grid] for e in surface.expiries],
            },
        },
    )


def expiries_section(surface: Surface, r: float) -> dict:
    cols = [
        column("expiry_ts", "Expiry", "timestamp"),
        column("label", "Label", "string"),
        column("dte", "DTE", "number", TENOR),
        column("forward", "Forward", "number", USD),
        column("basis_ann", "Basis (ann.)", "number", unit("rate", "decimal")),
        column("atm_iv", "ATM IV", "number", IV),
        column("rr25", "RR25", "number", VOL_SPREAD),
        column("bf25", "BF25", "number", VOL_SPREAD),
        column("n_nodes", "Nodes", "number", COUNT),
        column("n_good", "Good", "number", COUNT),
        column("k_min", "k min", "number", K),
        column("k_max", "k max", "number", K),
        column("median_spread_vp", "Median spread", "number", VOL_SPREAD),
    ]
    rows = []
    for e in surface.expiries:
        s = expiry_skew(e, r)
        spreads = [
            n.iv_ask - n.iv_bid for n in e.nodes if n.iv_ask is not None and n.iv_bid is not None
        ]
        rows.append(
            [
                f"{e.expiry:%Y-%m-%dT%H:%M:%SZ}",
                label(e.expiry),
                e.T * 365,
                e.forward,
                math.log(e.forward / surface.spot) / e.T,
                s.atm,
                s.rr,
                s.bf,
                len(e.nodes),
                sum(n.quality == "good" for n in e.nodes),
                e.k_min,
                e.k_max,
                statistics.median(spreads) if spreads else None,
            ]
        )
    return section("expiries", "Expiries", "table", {"columns": cols, "rows": rows})


CHAIN_COLUMNS = [
    column("instrument_name", "Instrument", "string"),
    column("expiry_ts", "Expiry", "timestamp"),
    column("strike", "Strike", "number", unit("strike")),
    column("option_type", "Type", "string"),
    column("k", "k", "number", K),
    column("is_node", "Node", "boolean"),
    column("node_quality", "Quality", "string"),
    column("iv_mark", "IV mark", "number", IV),
    column("iv_bid", "IV bid", "number", IV),
    column("iv_ask", "IV ask", "number", IV),
    column("venue_bid_usd", "Bid", "number", USD),
    column("venue_ask_usd", "Ask", "number", USD),
    column("venue_mark_usd", "Mark", "number", USD),
    column("fair_usd", "Fair", "number", USD),
    column("fair_pct_spot", "% spot", "number", unit("ratio", "decimal")),
    column("delta", "Δ", "number", unit("delta", convention="forward")),
    column("delta_spot", "Δ spot", "number", unit("delta", convention="spot")),
    column("gamma_1pct", "Γ 1%", "number", unit("gamma", convention="delta_change_per_1pct_spot")),
    column("vega_1pt", "Vega", "number", unit("vega", convention="usd_per_1vol_point")),
    column("theta_1d", "Θ/day", "number", unit("theta", convention="usd_per_calendar_day")),
    column("p_itm", "P(ITM)", "number", unit("probability", "decimal")),
    column("breakeven", "Breakeven", "number", USD),
    column("yield_ann", "Yield (ann.)", "number", unit("yield_annualized", "decimal")),
    column("apy", "APY", "number", unit("yield_annualized", "decimal", "compounded")),
    column("open_interest", "OI", "number", COUNT),
    column("contract_size", "Contract size", "number", unit("other")),
]


def _apy(fair: float, spot: float, dte: float) -> float | None:
    """Compounded yield; secondary to the simple APR headline (§3.5). None if it overflows."""
    try:
        value = (1 + fair / spot) ** (365 / dte) - 1
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def chain_section(surface: Surface, by_expiry, ctx: RunContext, r: float) -> dict:
    lo, hi = ctx.params["chain_delta_range"]
    S = surface.spot  # noqa: N806
    rows = []
    for expiry_ts in sorted(by_expiry):
        expiry, group, nodes = by_expiry[expiry_ts]
        F, T = expiry.forward, expiry.T  # noqa: N806
        for row in sorted(group, key=lambda x: (x["strike"], x["option_type"])):
            call = row["option_type"] == "call"
            k = math.log(row["strike"] / F)
            node = nodes.get(row["instrument_name"])
            sigma = node.iv if node else expiry.sigma(k).sigma  # type: ignore[union-attr]
            g = black76(F, row["strike"], T, sigma, r, call, S)
            # Node rows are always kept: evaluation rebuilds the surface from them (guide §6.3).
            if node is None and not lo <= abs(g["delta"]) <= hi:
                continue
            fair, dte = g["price"], T * 365
            bid = node.iv_bid if node else solve_iv(row["bid_usd"], F, row["strike"], T, r, call, S)
            ask = node.iv_ask if node else solve_iv(row["ask_usd"], F, row["strike"], T, r, call, S)
            rows.append(
                [
                    row["instrument_name"],
                    expiry_ts,
                    row["strike"],
                    row["option_type"],
                    k,
                    node is not None,
                    node.quality if node else "interpolated",
                    sigma,
                    bid,
                    ask,
                    row["bid_usd"],
                    row["ask_usd"],
                    row["mark_usd"],
                    fair,
                    fair / S,
                    g["delta"],
                    g["delta_spot"],
                    g["gamma_1pct"],
                    g["vega_1pt"],
                    g["theta_1d"],
                    g["p_itm"],
                    row["strike"] + fair if call else row["strike"] - fair,
                    fair / S * 365 / dte,
                    _apy(fair, S, dte),
                    row["open_interest"],
                    row["contract_size"],
                ]
            )
    return section("chain", "Priced chain", "table", {"columns": CHAIN_COLUMNS, "rows": rows})
