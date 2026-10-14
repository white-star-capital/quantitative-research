"""parity_deribit: put-call parity, implied carry, executable edges, inventory lock rate.

Methodology: PARITY_MONITOR_IMPLEMENTATION_GUIDE.md §2–3 (with its amendments). Payload: §4.2.
Pure: everything comes from ctx.inputs (MODELS M-5). Order books for the screened candidates
arrive as follow-up inputs (M-5a): `followup_inputs` runs the same screen `run` does.

Parity runs on executable bid/ask, never marks (§13). Inverse books (BTC/ETH) use the coin
formulas and the inverse futures; linear books (SOL/HYPE) the USD ones (core/quant/parity.py).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from terminal_models.core.base import InputData, InputSpec, Model, ModelError, RunContext, Scope
from terminal_models.core.payloads import axis, column, item, section, series, unit
from terminal_models.core.quant import parity as P

YEAR_S = 365 * 86400
LEG_TIME_SPREAD_S = 1.0
HEDGE_ROUNDING_FLAG = 0.005  # residual > 0.5% of notional
MAX_OPPORTUNITIES = 200
ORDER_BOOK_DEPTH = 5
FUNDING_PERIODS_PER_YEAR = 3 * 365  # funding_8h → annual (confirm field semantics in M1)
FEE_KEYS = (
    "option_taker", "option_maker", "option_cap_pct_of_price", "future_taker", "future_maker",
    "perp_taker", "perp_maker", "option_delivery", "future_delivery",
)  # fmt: skip

RATE = unit("rate", "decimal", "simple annualized")
RATE_CONT = unit("rate", "decimal", "continuous annualized")
BPS = unit("bps", "bps")
USD = unit("price_usd")
COUNT = unit("count")
TENOR = unit("tenor_days")
SECONDS = unit("other", "absolute", "seconds")
LOG_K = unit("moneyness_log")


def utc(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, UTC)


def book_name(instrument: str) -> str:
    return f"order_book:{instrument}"


def fee_rates(params: dict[str, Any], side: str = "taker") -> P.FeeRates | None:
    fees = params.get("fees") or {}
    if any(fees.get(k) is None for k in FEE_KEYS):
        return None
    return P.FeeRates.from_table(fees, side)


# ---- analysis shared by followup_inputs and run ------------------------------------------


@dataclass
class Pair:
    strike: float
    call: dict
    put: dict

    def px(self, leg: str, side: str) -> float | None:
        return (self.call if leg == "call" else self.put)[f"{side}_native"]


@dataclass
class Hedge:
    kind: str  # locked_future | perp_funding
    row: dict

    @property
    def is_perp(self) -> bool:
        return self.kind == "perp_funding"

    @property
    def mid(self) -> float:
        return (self.row["bid"] + self.row["ask"]) / 2


@dataclass
class ExpiryBook:
    expiry: datetime
    T: float  # noqa: N815
    DF: float  # noqa: N815  e^(−usd_rate·T)
    inverse: bool
    venue_forward: float
    venue_source: str | None
    pairs: list[Pair]
    hedge: Hedge | None
    rows: list[dict] = field(default_factory=list)  # all option rows of the expiry


@dataclass
class Candidate:
    book: ExpiryBook
    pair: Pair
    structure: str  # conversion | reversal
    legs: tuple[float, float, float]  # call, put, hedge prices used
    gross: float
    metrics: dict

    @property
    def instruments(self) -> list[str]:
        assert self.book.hedge is not None
        return [
            self.pair.call["instrument_name"],
            self.pair.put["instrument_name"],
            self.book.hedge.row["instrument_name"],
        ]


def _two_sided(row: dict) -> bool:
    return row["bid_native"] is not None and row["ask_native"] is not None


def expiry_books(
    chain: list[dict], futures: list[dict], as_of: datetime, params: dict
) -> list[ExpiryBook]:
    """Group option rows by expiry, pair two-sided calls and puts by strike, match the hedge:
    same-expiry dated future, else the perp (§2.2). Expiries under min_tte_hours are dropped."""
    min_t = params["min_tte_hours"] * 3600 / YEAR_S
    dated = {
        f["expiration_ms"]: f
        for f in futures
        if not f["is_perpetual"] and f["bid"] is not None and f["ask"] is not None
    }
    perp = next(
        (f for f in futures if f["is_perpetual"] and f["bid"] is not None and f["ask"] is not None),
        None,
    )
    by_expiry: dict[int, list[dict]] = defaultdict(list)
    for row in chain:
        by_expiry[row["expiration_ms"]].append(row)
    books = []
    for expiration_ms in sorted(by_expiry):
        rows = by_expiry[expiration_ms]
        expiry = ms_to_dt(expiration_ms)
        T = (expiry - as_of).total_seconds() / YEAR_S  # noqa: N806
        if T < min_t:
            continue
        legs: dict[float, dict[str, dict]] = defaultdict(dict)
        for row in rows:
            if _two_sided(row):
                legs[row["strike"]][row["option_type"]] = row
        pairs = [Pair(k, v["call"], v["put"]) for k, v in sorted(legs.items()) if len(v) == 2]
        future = dated.get(expiration_ms)
        hedge = (
            Hedge("locked_future", future)
            if future
            else Hedge("perp_funding", perp)
            if perp
            else None
        )
        forwards = sorted(r["forward"] for r in rows)
        books.append(
            ExpiryBook(
                expiry=expiry,
                T=T,
                DF=math.exp(-params["usd_rate"] * T),
                inverse=rows[0]["settlement"] == "inverse",
                venue_forward=forwards[len(forwards) // 2],
                venue_source=rows[0].get("underlying_index"),
                pairs=pairs,
                hedge=hedge,
                rows=rows,
            )
        )
    return books


def candidate(
    book: ExpiryBook, pair: Pair, structure: str, legs, rates: P.FeeRates, spot: float
) -> Candidate | None:
    assert book.hedge is not None
    c, p, f = legs
    if structure == "conversion":
        gross, _ = P.conversion(book.inverse, pair.strike, c, p, f, book.DF)
    else:
        gross, _ = P.reversal(book.inverse, pair.strike, c, p, f, book.DF)
    if gross is None:
        return None
    fees = P.structure_fees(book.inverse, rates, pair.strike, c, p, f, spot, book.hedge.is_perp)
    metrics = P.net_metrics(book.inverse, gross, fees, pair.strike, spot, book.T)
    return Candidate(book, pair, structure, legs, gross, metrics)


def screen(books: list[ExpiryBook], rates: P.FeeRates, spot: float) -> list[Candidate]:
    """Conversion and reversal on every hedged pair at book-summary prices, best first."""
    out = []
    for book in books:
        if book.hedge is None:
            continue
        h = book.hedge.row
        for pair in book.pairs:
            for structure, legs in (
                ("conversion", (pair.px("call", "bid"), pair.px("put", "ask"), h["ask"])),
                ("reversal", (pair.px("call", "ask"), pair.px("put", "bid"), h["bid"])),
            ):
                cand = candidate(book, pair, structure, legs, rates, spot)
                if cand is not None:
                    out.append(cand)
    out.sort(
        key=lambda c: (-(c.metrics["net_edge_ann"] or -math.inf), c.book.expiry, c.pair.strike)
    )
    return out


def worth_confirming(cands: list[Candidate], params: dict) -> list[Candidate]:
    threshold = params["min_net_edge_bps"]
    keep = [c for c in cands if c.metrics["net_edge_bps_notional"] > threshold]
    return keep[: params["max_confirmations"]]


# ---- order-book confirmation (D-13) -------------------------------------------------------


def best(levels: list[list[float]]) -> tuple[float, float] | None:
    return (levels[0][0], levels[0][1]) if levels else None


def confirm(
    cand: Candidate, books: dict[str, InputData], rates: P.FeeRates, spot: float
) -> tuple[Candidate, dict] | None:
    """Recompute from the three legs' order books. None when any book is missing or empty."""
    names = cand.instruments
    found = [books.get(book_name(n)) for n in names]
    if any(b is None for b in found):
        return None
    conv = cand.structure == "conversion"
    sides = ("bids", "asks", "asks") if conv else ("asks", "bids", "bids")
    tops = [best(b.data[s]) for b, s in zip(found, sides, strict=True)]  # type: ignore[union-attr]
    if any(t is None for t in tops):
        return None
    (c, c_amt), (p, p_amt), (f, f_amt) = tops  # type: ignore[misc]
    again = candidate(cand.book, cand.pair, cand.structure, (c, p, f), rates, spot)
    if again is None:
        return None
    K = cand.pair.strike  # noqa: N806
    hedge_units = f_amt / K if cand.book.inverse else f_amt  # inverse amounts are USD (M1)
    units = min(
        c_amt * cand.pair.call["contract_size"], p_amt * cand.pair.put["contract_size"], hedge_units
    )
    times = [b.snapshot_at for b in found]  # type: ignore[union-attr]
    info = {
        "exec_size_units": units,
        "leg_time_spread_s": (max(times) - min(times)).total_seconds(),
        "oldest": min(times),
    }
    return again, info


# ---- the model ----------------------------------------------------------------------------


class ParityDeribit(Model):
    def __init__(self, manifest) -> None:
        self.manifest = manifest

    def required_inputs(self, scope: Scope) -> list[InputSpec]:
        u = scope["underlying"]
        return [
            InputSpec("chain", "deribit", {"kind": "chain", "underlying": u}),
            InputSpec("index", "deribit", {"kind": "index", "underlying": u}),
            InputSpec("futures", "deribit", {"kind": "futures_book", "underlying": u}),
            InputSpec("funding", "deribit", {"kind": "funding", "underlying": u}),
        ]

    def followup_inputs(self, scope: Scope, inputs: dict[str, InputData]) -> list[InputSpec]:
        """Order books for every leg of the top screened candidates (D-13)."""
        params = self.manifest.params
        rates = fee_rates(params)
        if rates is None:
            return []  # run() fails with fees_unavailable
        as_of = max(d.snapshot_at for d in inputs.values())
        books = expiry_books(
            inputs["chain"].data["rows"], inputs["futures"].data["rows"], as_of, params
        )
        names: list[str] = []
        for cand in worth_confirming(screen(books, rates, inputs["index"].data["spot"]), params):
            names += [n for n in cand.instruments if n not in names]
        return [
            InputSpec(
                book_name(n),
                "deribit",
                {"kind": "order_book", "underlying": scope["underlying"],
                 "instrument_name": n, "depth": ORDER_BOOK_DEPTH},
                optional=True,
            )
            for n in names
        ]  # fmt: skip

    def run(self, ctx: RunContext) -> dict[str, Any]:
        p = ctx.params
        rates = fee_rates(p)
        if rates is None:
            raise ModelError("fees_unavailable", "params.fees has nulls; fill from the schedule")
        chain, futures = ctx.inputs["chain"], ctx.inputs["futures"]
        spot = ctx.inputs["index"].data["spot"]
        books = expiry_books(chain.data["rows"], futures.data["rows"], ctx.as_of, p)
        if not any(b.pairs for b in books):
            raise ModelError("insufficient_market_data", "no two-sided call/put pairs")
        for b in books:
            if b.hedge is None:
                ctx.warn("no_hedge_instrument", f"{utc(b.expiry)}: no two-sided future or perp")

        skew_s = abs((chain.snapshot_at - futures.snapshot_at).total_seconds())
        skewed = skew_s > p["max_snapshot_skew_s"]
        if skewed:
            ctx.warn(
                "snapshot_skew",
                f"option and futures books {skew_s:.1f}s apart (max {p['max_snapshot_skew_s']}s)",
            )
        funding = ctx.inputs["funding"].data
        funding_ann = (
            funding["funding_8h"] * FUNDING_PERIODS_PER_YEAR
            if funding.get("funding_8h") is not None
            else None
        )

        cands = screen(books, rates, spot)
        opps = self._opportunities(ctx, cands, rates, spot, skewed, chain, futures)
        exp_rows = [self._expiry(b, cands, rates, spot, p) for b in books]
        for row in exp_rows:
            if row["status"] == P.INSUFFICIENT_STRIKES:
                ctx.warn("insufficient_strikes", f"{row['expiry_ts']}: carry point omitted")

        inverse = books[0].inverse
        native = (
            unit("price", "absolute", f"{ctx.scope['underlying']} per 1-coin option")
            if inverse
            else USD
        )
        # Headline numbers count locked rows only: perp-hedged edges are not locked (guide §11),
        # so they are reported separately and never as the best edge.
        locked = [o for o in opps if o["hedge_type"] == "locked_future"]
        perps = [o for o in opps if o["hedge_type"] == "perp_funding"]

        def best(rows: list[dict]) -> float | None:
            return max((o["net_edge_ann"] for o in rows if o["net_edge_ann"] is not None),
                       default=None)  # fmt: skip

        context = [
            item("spot", "Spot index", spot, USD),
            item("funding_ann", "Perp funding (ann.)", funding_ann, RATE),
            item("n_expiries", "Expiries", len(books), COUNT),
            item("n_locked_expiries", "Expiries with a dated future",
                 sum(1 for b in books if b.hedge and not b.hedge.is_perp), COUNT),
            item("n_candidates", "Opportunities after fees", len(opps), COUNT),
            item("n_confirmed", "Confirmed, locked future",
                 sum(1 for o in locked if o["confirmed"]), COUNT),
            item("best_net_edge_ann", "Best net edge, locked future (ann.)", best(locked), RATE),
            item("n_confirmed_perp", "Confirmed, perp-hedged (unlocked)",
                 sum(1 for o in perps if o["confirmed"]), COUNT),
            item("best_perp_net_edge_ann", "Best net edge, perp-hedged (ann., unlocked)",
                 best(perps), RATE),
            item("snapshot_skew_s", "Option vs futures book skew", skew_s, SECONDS,
                 ("snapshot_skew",) if skewed else ()),
        ]  # fmt: skip
        return {
            "sections": [
                section("context", "Context", "scalar_set", {"items": context}),
                carry_section(exp_rows, funding_ann),
                expiries_section(exp_rows),
                bands_section(books, native),
                opportunities_section(opps, native),
                fees_section(p["fees"]),
            ]
        }

    # -- opportunities --------------------------------------------------------------------

    def _opportunities(self, ctx, cands, rates, spot, skewed, chain, futures) -> list[dict]:
        p = ctx.params
        threshold = p["min_net_edge_bps"]
        to_confirm = {id(c) for c in worth_confirming(cands, p)}
        book_inputs = {k: v for k, v in ctx.inputs.items() if k.startswith("order_book:")}
        summary_time = min(chain.snapshot_at, futures.snapshot_at)
        rows = []
        for cand in cands:
            info = None
            if id(cand) in to_confirm:
                confirmed = confirm(cand, book_inputs, rates, spot)
                if confirmed is None:
                    ctx.warn(
                        "confirmation_unavailable",
                        f"{utc(cand.book.expiry)} K={cand.pair.strike:g} {cand.structure}: "
                        "order book missing or empty",
                    )
                else:
                    cand, info = confirmed
            if cand.metrics["net_edge_bps_notional"] <= threshold:
                continue
            rows.append(self._row(cand, info, spot, skewed, summary_time, ctx.as_of))
        rows.sort(key=lambda r: (not r["confirmed"], -(r["net_edge_ann"] or -math.inf)))
        return rows[:MAX_OPPORTUNITIES]

    def _row(self, cand, info, spot, skewed, summary_time, as_of) -> dict:
        book, pair, hedge = cand.book, cand.pair, cand.book.hedge
        K = pair.strike  # noqa: N806
        units = info["exec_size_units"] if info else None
        contracts, residual = P.hedge_contracts(
            book.inverse, units or 1.0, K, cand.legs[2], hedge.row["contract_size"]
        )
        notional = (units or 1.0) * (K if book.inverse else cand.legs[2])
        flags = []
        if skewed and not info:
            flags.append("snapshot_skew")
        if info and info["leg_time_spread_s"] > LEG_TIME_SPREAD_S:
            flags.append("leg_time_spread")
        if book.inverse:
            flags.append("coin_denominated_edge")
        if residual > HEDGE_ROUNDING_FLAG * notional:
            flags.append("hedge_rounding")
        m = cand.metrics
        oldest = info["oldest"] if info else summary_time
        native = "coin" if book.inverse else "usd"
        return {
            "expiry_ts": utc(book.expiry),
            "strike": K,
            "structure": cand.structure,
            "hedge_type": hedge.kind,
            "hedge_instrument": hedge.row["instrument_name"],
            "call_px": cand.legs[0],
            "call_px_unit": native,
            "put_px": cand.legs[1],
            "put_px_unit": native,
            "hedge_px": cand.legs[2],
            "hedge_px_unit": "usd",
            "gross_edge": cand.gross,
            "fees": m["fees"],
            "net_edge": m["net_edge"],
            "net_edge_usd": m["net_edge_usd"],
            "net_edge_bps_notional": m["net_edge_bps_notional"],
            "net_edge_ann": m["net_edge_ann"],
            "breakeven_funding_ann": (
                P.breakeven_funding(cand.structure, m["net_edge_usd"], K, book.T)
                if hedge.is_perp
                else None
            ),
            "confirmed": info is not None,
            "exec_size_units": units,
            "hedge_contracts": contracts,
            "hedge_rounding_residual_usd": residual,
            "snapshot_age_s": max(0.0, (as_of - oldest).total_seconds()),
            "flags": ",".join(flags),
        }

    # -- expiries and carry ---------------------------------------------------------------

    def _expiry(self, book: ExpiryBook, cands, rates, spot, p) -> dict:
        DF = book.DF  # noqa: N806
        near = [
            pair
            for pair in book.pairs
            if abs(math.log(pair.strike / book.venue_forward)) <= p["atm_band"]
        ]
        strikes, locks = [], []
        for pair in near:
            c_mid = (pair.px("call", "bid") + pair.px("call", "ask")) / 2
            p_mid = (pair.px("put", "bid") + pair.px("put", "ask")) / 2
            mid, _ = P.forward(book.inverse, pair.strike, c_mid, p_mid, DF)
            lo, hi, _ = P.band(
                book.inverse, pair.strike, pair.px("call", "bid"), pair.px("call", "ask"),
                pair.px("put", "bid"), pair.px("put", "ask"), DF,
            )  # fmt: skip
            if mid is not None and lo is not None and hi is not None:
                strikes.append((mid, lo, hi))
            fees = P.inventory_fees(
                book.inverse, rates, pair.px("call", "bid"), pair.px("put", "ask"), spot
            )
            r, _ = P.inventory_rate(
                book.inverse, pair.strike, spot, pair.px("call", "bid"), pair.px("put", "ask"),
                book.T, fees, p["min_tte_hours"],
            )  # fmt: skip
            if r is not None:
                locks.append((r, pair.strike, fees))
        f_impl, why = P.implied_forward(strikes, p["min_strikes_for_forward"])
        weights = [1 / (hi - lo) for _, lo, hi in strikes if hi > lo]
        usable = [(lo, hi) for _, lo, hi in strikes if hi > lo]
        f_lo = P.weighted_median(list(zip([u[0] for u in usable], weights, strict=True)))
        f_hi = P.weighted_median(list(zip([u[1] for u in usable], weights, strict=True)))
        if f_impl is None:
            f_lo = f_hi = None
        hedge = book.hedge
        locked = hedge is not None and not hedge.is_perp
        best_lock = max(locks, default=None)
        own = [c for c in cands if c.book is book]
        best_c = max(own, key=lambda c: c.metrics["net_edge_usd"], default=None)
        breakeven = (
            P.breakeven_funding(best_c.structure, best_c.metrics["net_edge_usd"],
                                best_c.pair.strike, book.T)
            if hedge is not None and hedge.is_perp and best_c is not None
            else None
        )  # fmt: skip
        return {
            "expiry_ts": utc(book.expiry),
            "dte": book.T * 365,
            "status": why or "ok",
            "hedge_type": hedge.kind if hedge else None,
            "hedge_instrument": hedge.row["instrument_name"] if hedge else None,
            "venue_forward_source": book.venue_source,
            "venue_forward": book.venue_forward,
            "f_impl": f_impl,
            "f_impl_band_bps": P.bps(f_hi, f_lo) if f_impl is not None else None,
            "f_hedge_bid": hedge.row["bid"] if hedge else None,
            "f_hedge_ask": hedge.row["ask"] if hedge else None,
            "options_vs_future_bps": P.bps(f_impl, hedge.mid) if locked else None,
            "implied_basis_ann": P.basis_ann(f_impl, spot, book.T),
            "implied_basis_ann_bid": P.basis_ann(f_lo, spot, book.T),
            "implied_basis_ann_ask": P.basis_ann(f_hi, spot, book.T),
            "future_basis_ann": P.basis_ann(hedge.mid, spot, book.T) if locked else None,
            "breakeven_funding_ann": breakeven,
            "n_strikes_used": len(usable),
            "r_lock_best": best_lock[0] if best_lock else None,
            "r_lock_best_strike": best_lock[1] if best_lock else None,
            "r_lock_best_fees": best_lock[2] if best_lock else None,
        }


# ---- sections ------------------------------------------------------------------------------

EXPIRY_COLUMNS = [
    ("expiry_ts", "Expiry", "timestamp", None),
    ("dte", "DTE", "number", TENOR),
    ("status", "Status", "string", None),
    ("hedge_type", "Hedge type", "string", None),
    ("hedge_instrument", "Hedge", "string", None),
    ("venue_forward_source", "Venue forward source", "string", None),
    ("venue_forward", "Venue forward", "number", USD),
    ("f_impl", "Implied forward", "number", USD),
    ("f_impl_band_bps", "Parity band", "number", BPS),
    ("f_hedge_bid", "Hedge bid", "number", USD),
    ("f_hedge_ask", "Hedge ask", "number", USD),
    ("options_vs_future_bps", "Options vs future", "number", BPS),
    ("implied_basis_ann", "Implied basis", "number", RATE),
    ("implied_basis_ann_bid", "Implied basis bid", "number", RATE),
    ("implied_basis_ann_ask", "Implied basis ask", "number", RATE),
    ("future_basis_ann", "Future basis", "number", RATE),
    ("breakeven_funding_ann", "Breakeven funding", "number", RATE),
    ("n_strikes_used", "Strikes used", "number", COUNT),
    ("r_lock_best", "Best r_lock", "number", RATE_CONT),
    ("r_lock_best_strike", "r_lock strike", "number", USD),
    ("r_lock_best_fees", "r_lock option fees (native, per unit)", "number", unit("other")),
]


def _table(key: str, label: str, cols, rows: list[dict]) -> dict:
    columns = [column(k, lbl, t, u) for k, lbl, t, u in cols]
    return section(
        key, label, "table", {"columns": columns, "rows": [[r[k] for k, *_ in cols] for r in rows]}
    )


def expiries_section(rows: list[dict]) -> dict:
    return _table("expiries", "Expiries", EXPIRY_COLUMNS, rows)


def carry_section(rows: list[dict], funding_ann: float | None) -> dict:
    """x = tenor days of expiries with an implied forward; insufficient expiries omitted."""
    pts = [r for r in rows if r["f_impl"] is not None]
    if not pts:
        raise ModelError("insufficient_market_data", "no expiry has an implied forward")

    def col(key: str) -> list:
        return [r[key] for r in pts]

    return section(
        "carry_curve",
        "Carry curve",
        "curve",
        {
            "x": axis("Tenor", TENOR, col("dte")),
            "series": [
                series("implied_basis_ann", "Options-implied basis", RATE,
                       col("implied_basis_ann")),
                series("implied_basis_ann_bid", "Implied basis (bid)", RATE,
                       col("implied_basis_ann_bid")),
                series("implied_basis_ann_ask", "Implied basis (ask)", RATE,
                       col("implied_basis_ann_ask")),
                series("future_basis_ann", "Future basis", RATE, col("future_basis_ann")),
                series("funding_ann", "Perp funding", RATE, [funding_ann] * len(pts)),
                series("r_lock_best", "Best inventory r_lock", RATE_CONT, col("r_lock_best")),
            ],
        },
    )  # fmt: skip


def bands_section(books: list[ExpiryBook], native: dict) -> dict:
    cols = [
        ("expiry_ts", "Expiry", "timestamp", None),
        ("strike", "Strike", "number", USD),
        ("k", "ln(K/F)", "number", LOG_K),
        ("f_impl_bid", "f_impl bid", "number", USD),
        ("f_impl_ask", "f_impl ask", "number", USD),
        ("band_bid_bps", "Bid vs hedge mid", "number", BPS),
        ("band_ask_bps", "Ask vs hedge mid", "number", BPS),
        ("hedge_bid", "Hedge bid", "number", USD),
        ("hedge_ask", "Hedge ask", "number", USD),
        ("gross_conv", "Gross conversion", "number", native),
        ("gross_rev", "Gross reversal", "number", native),
    ]
    rows = []
    for b in books:
        h = b.hedge.row if b.hedge else None
        for pair in b.pairs:
            c_b, c_a = pair.px("call", "bid"), pair.px("call", "ask")
            p_b, p_a = pair.px("put", "bid"), pair.px("put", "ask")
            lo, hi, _ = P.band(b.inverse, pair.strike, c_b, c_a, p_b, p_a, b.DF)
            mid = b.hedge.mid if b.hedge else None
            rows.append(
                {
                    "expiry_ts": utc(b.expiry),
                    "strike": pair.strike,
                    "k": math.log(pair.strike / b.venue_forward),
                    "f_impl_bid": lo,
                    "f_impl_ask": hi,
                    "band_bid_bps": P.bps(lo, mid),
                    "band_ask_bps": P.bps(hi, mid),
                    "hedge_bid": h["bid"] if h else None,
                    "hedge_ask": h["ask"] if h else None,
                    "gross_conv": P.conversion(b.inverse, pair.strike, c_b, p_a, h["ask"], b.DF)[0]
                    if h
                    else None,
                    "gross_rev": P.reversal(b.inverse, pair.strike, c_a, p_b, h["bid"], b.DF)[0]
                    if h
                    else None,
                }
            )
    return _table("bands", "Parity bands", cols, rows)


def opportunities_section(rows: list[dict], native: dict) -> dict:
    cols = [
        ("expiry_ts", "Expiry", "timestamp", None),
        ("strike", "Strike", "number", USD),
        ("structure", "Structure", "string", None),
        ("hedge_type", "Hedge type", "string", None),
        ("hedge_instrument", "Hedge", "string", None),
        ("call_px", "Call", "number", native),
        ("call_px_unit", "Call unit", "string", None),
        ("put_px", "Put", "number", native),
        ("put_px_unit", "Put unit", "string", None),
        ("hedge_px", "Hedge", "number", USD),
        ("hedge_px_unit", "Hedge unit", "string", None),
        ("gross_edge", "Gross edge", "number", native),
        ("fees", "Fees", "number", native),
        ("net_edge", "Net edge", "number", native),
        ("net_edge_usd", "Net edge (USD)", "number", USD),
        ("net_edge_bps_notional", "Net edge (bp of notional)", "number", BPS),
        ("net_edge_ann", "Net edge (ann.)", "number", RATE),
        ("breakeven_funding_ann", "Breakeven funding", "number", RATE),
        ("confirmed", "Confirmed", "boolean", None),
        ("exec_size_units", "Size (units)", "number", COUNT),
        ("hedge_contracts", "Hedge contracts", "number", COUNT),
        ("hedge_rounding_residual_usd", "Hedge rounding residual", "number", USD),
        ("snapshot_age_s", "Snapshot age", "number", SECONDS),
        ("flags", "Flags", "string", None),
    ]
    return _table("opportunities", "Opportunities", cols, rows)


def fees_section(fees: dict) -> dict:
    """Amendment A-2: the fee table the run used, so the display can recompute maker/taker."""
    rate = unit("ratio", "decimal")
    cols = [
        ("side", "Side", "string", None),
        ("option", "Option", "number", rate),
        ("option_cap", "Option cap (of price)", "number", rate),
        ("future", "Future", "number", rate),
        ("perp", "Perp", "number", rate),
        ("option_delivery", "Option delivery", "number", rate),
        ("future_delivery", "Future delivery", "number", rate),
        ("schedule_version", "Schedule", "string", None),
    ]
    rows = [
        {
            "side": side,
            "option": fees[f"option_{side}"],
            "option_cap": fees["option_cap_pct_of_price"],
            "future": fees[f"future_{side}"],
            "perp": fees[f"perp_{side}"],
            "option_delivery": fees["option_delivery"],
            "future_delivery": fees["future_delivery"],
            "schedule_version": str(fees["schedule_version"]),
        }
        for side in ("taker", "maker")
    ]
    return _table("fees", "Fee table", cols, rows)


IMPLEMENTATION = ParityDeribit
