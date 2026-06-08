"""Pipeline-wide configuration for data fetching, feature engineering, and splitting."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Pipeline-wide configuration for data fetching, feature engineering, and splitting."""

    # Date boundaries (D-10)
    start_date: str = "2021-01-01"
    train_end: str = "2023-12-31"
    val_start: str = "2024-01-01"
    val_end: str = "2024-06-30"
    test_start: str = "2024-07-01"
    end_date: str = "2025-12-31"

    # Fetcher params
    interval: str = "1d"
    cache_dir: Path = field(default_factory=lambda: Path("vgp/data/cache"))
    use_ccxt_fallback: bool = True

    # FeatureEngine params
    min_obs_fraction: float = 0.80
    max_fill_days: int = 3
    lookback: int = 20
