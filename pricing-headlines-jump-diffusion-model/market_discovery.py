"""
Dynamic market discovery for the jump-diffusion pricing engines.

Replaces the old hardcoded MARKET_SLUGS list. Each run queries Polymarket's
Gamma API for the most actively-traded geopolitical events (by 24-hour volume)
and returns their slugs, which the engines' existing
`PolymarketMonitorClient.fetch_market_data(slug)` consumes directly (it handles
event slugs via the /events/slug fallback).
"""

import asyncio
from typing import Dict, List

import httpx

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

# Polymarket "Geopolitics" tag.
GEOPOLITICS_TAG_ID = "100265"


async def discover_top_geopolitical_slugs(
    top_n: int = 10,
    tag_id: str = GEOPOLITICS_TAG_ID,
    order: str = "volume24hr",
) -> List[Dict]:
    """Discover the top geopolitical events on Polymarket by trading activity.

    Args:
        top_n: Number of events to return.
        tag_id: Gamma tag id to filter by (defaults to Geopolitics).
        order: Field to sort by, descending (e.g. "volume24hr", "volume").

    Returns:
        A list of dicts {slug, title, volume24hr, volume, num_markets}, ordered
        as returned by the API (highest activity first). Returns [] on error.
    """
    params = {
        "tag_id": tag_id,
        "closed": "false",
        "active": "true",
        "order": order,
        "ascending": "false",
        "limit": str(top_n),
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{GAMMA_API_BASE}/events", params=params)
            response.raise_for_status()
            events = response.json()
        except httpx.HTTPError as e:
            print(f"⚠️  Failed to discover geopolitical markets: {e}")
            return []

    discovered = []
    for event in events:
        slug = event.get("slug")
        if not slug:
            continue

        def _to_float(value) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        discovered.append({
            "slug": slug,
            "title": event.get("title", ""),
            "volume24hr": _to_float(event.get("volume24hr")),
            "volume": _to_float(event.get("volume")),
            "num_markets": len(event.get("markets", []) or []),
        })

    return discovered


def slugs_only(discovered: List[Dict]) -> List[str]:
    """Extract just the slug strings from discovery results."""
    return [item["slug"] for item in discovered if item.get("slug")]


if __name__ == "__main__":
    async def _smoke_test():
        print(f"Discovering top geopolitical markets (tag {GEOPOLITICS_TAG_ID}, by volume24hr)...")
        discovered = await discover_top_geopolitical_slugs(top_n=10)
        print(f"\nFound {len(discovered)} markets:\n")
        for i, m in enumerate(discovered, 1):
            print(f"{i:>2}. v24h=${m['volume24hr']:>12,.0f} | "
                  f"vtot=${m['volume']:>14,.0f} | "
                  f"mkts={m['num_markets']:>2} | {m['slug']}")

    asyncio.run(_smoke_test())
