"""Chain rows → smile nodes per expiry (BS_PRICER_IMPLEMENTATION_GUIDE.md §3.1).

Shared by every implementation that reads the Deribit `chain` kind, so ATM IV means the same thing
in each slot (RV_FORECAST_IMPLEMENTATION_GUIDE.md VRP-A-6). Reads `min_tte_hours`,
`min_nodes_per_expiry` and `max_node_spread_vp` from the caller's params.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime

from terminal_models.core.base import ModelError, RunContext
from terminal_models.core.quant.black76 import implied_vol
from terminal_models.core.quant.smile import Expiry, Node, year_fraction


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def label(expiry: datetime) -> str:
    return expiry.strftime("%d %b %y").upper()


def solve_iv(
    price: float | None,
    F: float,
    K_: float,
    T: float,
    r: float,
    call: bool,  # noqa: N803
    S: float,
) -> float | None:  # noqa: N803
    if price is None:
        return None
    iv, _ = implied_vol(F, K_, T, price, r, call, S)
    return iv


def build_expiries(rows: list[dict], ctx: RunContext, spot: float, r: float):
    p, u = ctx.params, ctx.scope["underlying"]
    min_t = p["min_tte_hours"] / 8760
    min_nodes = p["min_nodes_per_expiry"][u]
    max_spread = p["max_node_spread_vp"] / 100

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["expiry_ts"]].append(row)

    expiries: list[Expiry] = []
    by_expiry: dict[str, tuple[Expiry, list[dict], dict[str, Node]]] = {}
    for expiry_ts in sorted(grouped):
        when = parse_utc(expiry_ts)
        T = year_fraction(ctx.as_of, when)  # noqa: N806
        if T < min_t:
            continue
        group = grouped[expiry_ts]
        F = statistics.median(row["forward"] for row in group)  # noqa: N806
        nodes: dict[str, Node] = {}
        for row in group:
            call = row["option_type"] == "call"
            otm = row["strike"] >= F if call else row["strike"] < F
            if not otm or row["mark_iv"] is None:
                continue
            bid = solve_iv(row["bid_usd"], F, row["strike"], T, r, call, spot)
            ask = solve_iv(row["ask_usd"], F, row["strike"], T, r, call, spot)
            if bid is not None and ask is not None:
                quality = "good" if ask - bid <= max_spread else "wide"
            elif bid is not None or ask is not None:
                quality = "one_sided"
            else:
                quality = "no_quote"
            nodes[row["instrument_name"]] = Node(
                math.log(row["strike"] / F), row["strike"], row["mark_iv"], bid, ask, quality
            )
        if len(nodes) < min_nodes:
            ctx.warn(
                "expiry_dropped_insufficient_nodes",
                f"{label(when)}: {len(nodes)} nodes, need {min_nodes}",
            )
            continue
        expiry = Expiry(when, T, F, list(nodes.values()))
        expiries.append(expiry)
        by_expiry[expiry_ts] = (expiry, group, nodes)

    if not expiries:
        raise ModelError("insufficient_market_data", f"no {u} expiry has enough nodes")
    return expiries, by_expiry
