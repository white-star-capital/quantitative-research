"""Data pipeline: DataLoader, FeatureEngine, WalkForwardSplitter."""

from .universe import UNIVERSE_30, get_binance_symbols
from .fetcher import BinanceFetcher
from .feature_engine import FeatureEngine
from .splitter import WalkForwardSplitter
from .config import DataConfig

# DataLoader is an alias for BinanceFetcher. Phase 3 (GP Core) imports DataLoader;
# BinanceFetcher is the concrete implementation for Phase 2.
DataLoader = BinanceFetcher

__all__ = [
    "UNIVERSE_30",
    "get_binance_symbols",
    "DataLoader",
    "BinanceFetcher",
    "FeatureEngine",
    "WalkForwardSplitter",
    "DataConfig",
]
