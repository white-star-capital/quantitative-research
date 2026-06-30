"""
Asset universe definition.

The article uses 30 cryptocurrencies quoted against USDT on Binance,
selected for availability over the full January 2021 – December 2025
sample period. These correspond to the most widely recognised coins
at the start of the sample.
"""

# Canonical short tickers
UNIVERSE_30: list[str] = [
    "BTC",
    "ETH",
    "BNB",
    "HYPE",
    "XRP",
    "PENDLE",
    "UNI",
    "JUP",
    "TAO",
    "LINK",
    "ZEC",
    "DOGE",
    "MORPHO",
    "AERO",
    "SOL",
    "AVAX",
    "POL",
    "WLFI",
    "WIF",
    "PEPE",
    "AAVE",
    "COMP",
    "FLUID",
    "SHIB",
    "SUSHI",
    "CRV",
    "SYRUP",
    "ENA",
    "ONDO",
    "EUL",
]

# Canonical short tickers
UNIVERSE_30_old: list[str] = [
    "BTC",   # Bitcoin
    "ETH",   # Ethereum
    "BNB",   # BNB
    "ADA",   # Cardano
    "XRP",   # Ripple
    "DOT",   # Polkadot
    "UNI",   # Uniswap
    "LTC",   # Litecoin
    "BCH",   # Bitcoin Cash
    "LINK",  # Chainlink
    "XLM",   # Stellar
    "DOGE",  # Dogecoin
    "EOS",   # EOS
    "TRX",   # TRON
    "SOL",   # Solana
    "AVAX",  # Avalanche
    "MATIC", # Polygon
    "ATOM",  # Cosmos
    "ALGO",  # Algorand
    "FIL",   # Filecoin
    "AAVE",  # Aave
    "COMP",  # Compound
    "MKR",   # Maker
    "SNX",   # Synthetix
    "SUSHI", # SushiSwap
    "CRV",   # Curve
    "SAND",  # The Sandbox
    "MANA",  # Decentraland
    "AXS",   # Axie Infinity
    "NEAR",  # NEAR Protocol
]

assert len(UNIVERSE_30) == 30, "Universe must contain exactly 30 assets."


def get_binance_symbols(quote: str = "USDT") -> list[str]:
    """Return Binance trading pair symbols, e.g. ['BTCUSDT', 'ETHUSDT', ...]."""
    return [f"{ticker}{quote}" for ticker in UNIVERSE_30]


# Approximate circulating market caps (USD) as of 2021-01-01 for
# value-weighted benchmark construction.  These are rough estimates used
# only as initial weights; the backtest uses live price × supply data.
APPROX_MCAP_2021: dict[str, float] = {
    "BTC":   700_000_000_000,
    "ETH":   140_000_000_000,
    "BNB":    7_000_000_000,
    "ADA":    9_000_000_000,
    "XRP":   20_000_000_000,
    "DOT":   12_000_000_000,
    "UNI":    6_000_000_000,
    "LTC":   11_000_000_000,
    "BCH":    8_000_000_000,
    "LINK":   9_000_000_000,
    "XLM":    7_000_000_000,
    "DOGE":   2_000_000_000,
    "EOS":    3_000_000_000,
    "TRX":    3_000_000_000,
    "SOL":    2_000_000_000,
    "AVAX":   1_000_000_000,
    "MATIC":    500_000_000,
    "ATOM":   3_000_000_000,
    "ALGO":   2_000_000_000,
    "FIL":    6_000_000_000,
    "AAVE":   2_000_000_000,
    "COMP":   2_000_000_000,
    "MKR":    2_000_000_000,
    "SNX":    1_000_000_000,
    "SUSHI":    500_000_000,
    "CRV":      500_000_000,
    "SAND":     300_000_000,
    "MANA":     300_000_000,
    "AXS":      200_000_000,
    "NEAR":     500_000_000,
}
