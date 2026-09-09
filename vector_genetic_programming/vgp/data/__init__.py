"""Data pipeline: DataLoader, FeatureEngine, WalkForwardSplitter."""

from .universe import UNIVERSE_30, UniverseRecord, get_binance_symbols
from .fetcher import BinanceFetcher, FetchError, FetchReport
from .feature_engine import FeatureEngine
from .splitter import WalkForwardSplitter
from .config import DataConfig

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
