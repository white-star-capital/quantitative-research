"""Data pipeline: DataLoader, FeatureEngine, WalkForwardSplitter."""

from .config import DataConfig
from .feature_engine import FeatureEngine
from .fetcher import BinanceFetcher, FetchError, FetchReport
from .splitter import WalkForwardSplitter
from .universe import UNIVERSE_30, UniverseRecord, get_binance_symbols

# DataLoader is an alias for BinanceFetcher. Phase 3 (GP Core) imports DataLoader;
# BinanceFetcher is the concrete implementation for Phase 2.
DataLoader = BinanceFetcher

__all__ = [
    "UNIVERSE_30",
    "UniverseRecord",
    "get_binance_symbols",
    "FetchError",
    "FetchReport",
    "DataLoader",
    "BinanceFetcher",
    "FeatureEngine",
    "WalkForwardSplitter",
    "DataConfig",
]
