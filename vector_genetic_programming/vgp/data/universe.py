"""
Asset universe definition.

The article uses 30 cryptocurrencies quoted against USDT on Binance,
selected for availability over the full January 2021 – December 2025
sample period. These correspond to the most widely recognised coins
at the start of the sample.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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

assert len(UNIVERSE_30) == 30, "Universe must contain exactly 30 assets."


def get_binance_symbols(quote: str = "USDT") -> list[str]:
    """Return Binance trading pair symbols, e.g. ['BTCUSDT', 'ETHUSDT', ...]."""
    return [f"{ticker}{quote}" for ticker in UNIVERSE_30]


@dataclass(frozen=True)
class UniverseRecord:
    """The assets a run was ACTUALLY computed on, and everything lost on the way.

    The declared universe (`UNIVERSE_30`) is an intention. Two independent
    stages narrow it before any strategy is evaluated:

    1. the fetch — a symbol can fail or return no rows (see `FetchReport`);
    2. `FeatureEngine.fit_transform()` — assets below `min_obs_fraction`
       are dropped, and the remainder is intersected to a common date index.

    Neither stage is visible in `results.csv` unless it is recorded, and the
    difference matters: results computed on 28 assets are not comparable with
    results computed on 12, and a universe that varies between runs is a
    survivorship-bias channel that neither the Deflated Sharpe Ratio nor the
    null control can detect — every trial and every surrogate inherits the same
    universe, so nothing in the statistics reveals that it changed.

    `fingerprint` gives each composition a short stable id, stamped onto every
    result row, so two runs can be compared for universe equality at a glance
    rather than by reading a log.
    """

    requested: tuple[str, ...]  # short tickers, as declared
    fetched: tuple[str, ...]  # survived the fetch
    fetch_failed: tuple[tuple[str, str], ...]  # (symbol, reason)
    dropped_by_features: tuple[str, ...]  # below min_obs_fraction
    retained: tuple[str, ...]  # actually used
    n_dates: int = 0  # rows in the aligned panel

    @classmethod
    def from_pipeline(cls, fetch_report, engine) -> UniverseRecord:
        """Build from a `FetchReport` and a fitted `FeatureEngine`.

        The engine is duck-typed (`retained_assets_`, `dropped_assets_`,
        `dates_`) so this module stays free of pandas/numpy imports and can be
        used by any caller that produces the same metadata.
        """
        dates = getattr(engine, "dates_", None)
        return cls(
            requested=tuple(s.replace("USDT", "") for s in getattr(fetch_report, "requested", ())),
            fetched=tuple(getattr(fetch_report, "realized", ())),
            fetch_failed=tuple(getattr(fetch_report, "failed", ())),
            dropped_by_features=tuple(getattr(engine, "dropped_assets_", ()) or ()),
            retained=tuple(getattr(engine, "retained_assets_", ()) or ()),
            n_dates=0 if dates is None else len(dates),
        )

    @property
    def n_requested(self) -> int:
        return len(self.requested)

    @property
    def n_retained(self) -> int:
        return len(self.retained)

    @property
    def is_complete(self) -> bool:
        """True when nothing was lost between the declared and used universe."""
        return not self.fetch_failed and not self.dropped_by_features

    @property
    def fingerprint(self) -> str:
        """Short stable id for this composition.

        Order-independent (the retained list is sorted first) so a reordering
        of the universe definition does not read as a different universe.
        Returns "empty" for no retained assets rather than hashing nothing.
        """
        if not self.retained:
            return "empty"
        joined = ",".join(sorted(self.retained))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "n_requested": self.n_requested,
            "n_fetched": len(self.fetched),
            "n_retained": self.n_retained,
            "n_dates": self.n_dates,
            "is_complete": self.is_complete,
            "requested": list(self.requested),
            "fetched": list(self.fetched),
            "retained": sorted(self.retained),
            "dropped_by_features": list(self.dropped_by_features),
            "fetch_failed": {symbol: reason for symbol, reason in self.fetch_failed},
        }

    def stamp_rows(self, rows: list[dict]) -> list[dict]:
        """Tag every result row with the universe it was computed on, in place.

        Two columns rather than the whole list: `n_assets` for the obvious
        sanity check, and `universe_fingerprint` to tie the row to the full
        composition in universe.json.
        """
        for row in rows:
            row["n_assets"] = self.n_retained
            row["universe_fingerprint"] = self.fingerprint
        return rows

    def summary(self) -> str:
        lines = [
            f"Universe: {self.n_retained}/{self.n_requested} assets retained "
            f"[{self.fingerprint}]" + (f", {self.n_dates} aligned dates" if self.n_dates else ""),
        ]
        if self.fetch_failed:
            lines.append(
                f"  lost to fetch ({len(self.fetch_failed)}): "
                + ", ".join(symbol for symbol, _ in self.fetch_failed)
            )
        if self.dropped_by_features:
            lines.append(
                f"  dropped below min_obs_fraction ({len(self.dropped_by_features)}): "
                + ", ".join(self.dropped_by_features)
            )
        if self.is_complete:
            lines.append("  complete — nothing lost between declared and used")
        else:
            lines.append(
                "  INCOMPLETE — results are not comparable with a run on a "
                "different universe; compare fingerprints, not asset counts"
            )
        return "\n".join(lines)
