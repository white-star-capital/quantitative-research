"""
Asset universe definition.

The article uses 30 cryptocurrencies quoted against USDT on Binance,
selected for availability over the full January 2021 – December 2025
sample period. These correspond to the most widely recognised coins
at the start of the sample.
"""
from __future__ import annotations

# Canonical short tickers
UNIVERSE_30: list[str] = [
    "BTC", "ETH", "BNB", "HYPE", "XRP", "PENDLE", "UNI", "JUP", "TAO",
    "LINK", "ZEC", "DOGE", "MORPHO", "AERO", "SOL", "AVAX", "POL", "WLFI",
    "WIF", "PEPE", "AAVE", "COMP", "FLUID", "SHIB", "SUSHI", "CRV",
    "SYRUP", "ENA", "ONDO", "EUL",
]

assert len(UNIVERSE_30) == 30, "Universe must contain exactly 30 assets."


def get_binance_symbols(quote: str = "USDT") -> list[str]:
    """Return Binance trading pair symbols, e.g. ['BTCUSDT', 'ETHUSDT', ...]."""
    return [f"{ticker}{quote}" for ticker in UNIVERSE_30]
