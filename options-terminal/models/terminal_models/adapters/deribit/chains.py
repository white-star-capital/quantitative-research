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
    index: str  # spot index name
    perp: str  # perpetual for hourly candles
    dvol: str | None  # display-only DVOL index


BOOKS: dict[str, Book] = {
    "BTC": Book("BTC", "BTC", "btc_usd", "BTC-PERPETUAL", "btcdvol_usdc"),
    "ETH": Book("ETH", "ETH", "eth_usd", "ETH-PERPETUAL", "ethdvol_usdc"),
    "SOL": Book("USDC", "SOL", "sol_usdc", "SOL_USDC-PERPETUAL", None),
    "HYPE": Book("USDC", "HYPE", "hype_usdc", "HYPE_USDC-PERPETUAL", None),
}


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
        inverse = m.get("settlement_currency") == book.base
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
            }
        )
    rows.sort(key=lambda r: (r["expiry_ts"], r["strike"], r["option_type"]))
    return rows
