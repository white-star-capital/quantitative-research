"""Normalize Deribit option books to USD-per-unit rows (D-2, D-3, D-5).

Rows are joined on `instrument_name` from instrument metadata; names are never constructed
(D-2). `contract_size` always comes from metadata (D-3). Prices become USD per 1 unit (D-5):
inverse books multiply by the row's `underlying_price`; linear USDC books are taken as USD.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class Book:
    currency: str  # book summary / instruments currency
    base: str  # base_currency filter
    index: str  # spot index name (confirmed live 2026-10-13)
    dvol: str | None  # display-only DVOL index (confirmed live 2026-10-13)


BOOKS: dict[str, Book] = {
    "BTC": Book("BTC", "BTC", "btc_usd", "btcdvol_usdc"),
    "ETH": Book("ETH", "ETH", "eth_usd", "ethdvol_usdc"),
    "SOL": Book("USDC", "SOL", "sol_usdc", None),
    "HYPE": Book("USDC", "HYPE", "hype_usdc", None),
}


def find_perpetual(book: Book, futures: list[dict]) -> str | None:
    """The asset's perpetual, from futures metadata (D-2: never a constructed name).

    Candidates: `base_currency` = asset and `settlement_period` = perpetual, still active. Prefer
    one settling in the book's currency (BTC-PERPETUAL for the inverse BTC book,
    SOL_USDC-PERPETUAL for the USDC book); ties break on name for determinism.
    """
    perps = [
        f
        for f in futures
        if f.get("base_currency") == book.base
        and f.get("settlement_period") == "perpetual"
        and f.get("is_active", True)
    ]
    perps.sort(key=lambda f: (f.get("settlement_currency") != book.currency, f["instrument_name"]))
    return perps[0]["instrument_name"] if perps else None


def is_inverse(meta: dict, book: Book) -> bool:
    """`instrument_type` reversed/linear when present; else coin-settled means inverse."""
    kind = meta.get("instrument_type")
    if kind in ("reversed", "linear"):
        return kind == "reversed"
    return meta.get("settlement_currency") == book.base


def _positive(value: Any) -> float | None:
    """Venue 'no value' arrives as null or 0; both are missing, never a zero price."""
    if value is None:
        return None
    number = float(value)
    return number if number > 0 else None


def iso_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize(underlying: str, instruments: list[dict], summaries: list[dict]) -> list[dict]:
    """One row per option with both metadata and a book summary. Rows are plain JSON so the
    snapshot can be stored and replayed (M-9)."""
    book = BOOKS[underlying]
    meta = {
        i["instrument_name"]: i
        for i in instruments
        if i.get("kind", "option") == "option" and i.get("base_currency") == book.base
    }
    rows = []
    for s in summaries:
        m = meta.get(s.get("instrument_name"))
        if m is None:
            continue  # not this asset, or listing metadata missing: skip, never guess (D-2)
        forward = _positive(s.get("underlying_price"))
        if forward is None:
            continue
        inverse = is_inverse(m, book)
        to_usd = forward if inverse else 1.0

        def usd(field: str, to_usd: float = to_usd, s: dict = s) -> float | None:
            price = _positive(s.get(field))
            return price * to_usd if price is not None else None

        mark_iv = _positive(s.get("mark_iv"))
        oi = s.get("open_interest")
        rows.append(
            {
                "instrument_name": m["instrument_name"],
                "expiry_ts": iso_ms(m["expiration_timestamp"]),
                "strike": float(m["strike"]),
                "option_type": m["option_type"],
                "forward": forward,
                "mark_iv": mark_iv / 100 if mark_iv is not None else None,  # venue sends percent
                "mark_usd": usd("mark_price"),
                "bid_usd": usd("bid_price"),
                "ask_usd": usd("ask_price"),
                "open_interest": float(oi) if oi is not None else None,
                "contract_size": float(m["contract_size"]),
                "settlement": "inverse" if inverse else "linear",
                # Native book prices (coin for inverse, USDC for linear): parity works on these
                # (PARITY_MONITOR_IMPLEMENTATION_GUIDE.md §3), never on USD conversions.
                "bid_native": _positive(s.get("bid_price")),
                "ask_native": _positive(s.get("ask_price")),
                "expiration_ms": int(m["expiration_timestamp"]),
                "underlying_index": s.get("underlying_index"),
            }
        )
    rows.sort(key=lambda r: (r["expiry_ts"], r["strike"], r["option_type"]))
    return rows


def normalize_futures(book: Book, instruments: list[dict], summaries: list[dict]) -> list[dict]:
    """Dated futures and perps for one asset, joined to metadata on instrument_name (D-15).
    Prices are USD (inverse futures quote USD; linear USDC futures quote USDC = USD).
    `contract_size` from metadata: USD per contract for inverse, coin for linear (D-3)."""
    meta = {
        f["instrument_name"]: f
        for f in instruments
        if f.get("base_currency") == book.base and f.get("is_active", True)
    }
    rows = []
    for s in summaries:
        m = meta.get(s.get("instrument_name"))
        if m is None:
            continue
        perpetual = m.get("settlement_period") == "perpetual"
        rows.append(
            {
                "instrument_name": m["instrument_name"],
                "is_perpetual": perpetual,
                "expiration_ms": None if perpetual else int(m["expiration_timestamp"]),
                "bid": _positive(s.get("bid_price")),
                "ask": _positive(s.get("ask_price")),
                "mark": _positive(s.get("mark_price")),
                "contract_size": float(m["contract_size"]),
                "settlement": "inverse" if is_inverse(m, book) else "linear",
                "settlement_currency": m.get("settlement_currency"),
            }
        )
    rows.sort(key=lambda r: (r["is_perpetual"], r["expiration_ms"] or 0, r["instrument_name"]))
    return rows


def book_side(levels: Any) -> list[list[float]]:
    """Order-book levels as [[price, amount], ...], best first, positive values only."""
    out = []
    for level in levels or []:
        price, amount = float(level[0]), float(level[1])
        if price > 0 and amount > 0:
            out.append([price, amount])
    return out
