# Phase 2: Data Pipeline - Pattern Map

**Mapped:** 2026-06-08
**Files analyzed:** 8 new/modified files
**Analogs found:** 6 / 8

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `vgp/data/universe.py` | config/constant | transform | `data_pipeline_example/universe.py` | exact |
| `vgp/data/fetcher.py` | service | file-I/O + request-response | `data_pipeline_example/fetcher.py` | exact |
| `vgp/data/feature_engine.py` | service | transform + batch | `data_pipeline_example/processor.py` | role-match |
| `vgp/data/splitter.py` | utility | transform | `data_pipeline_example/processor.py` (slice pattern) | partial |
| `vgp/data/config.py` | config | — | `data_pipeline_example/fetcher.py` (constructor params) | partial |
| `vgp/data/__init__.py` | config | — | `data_pipeline_example/__init__.py` | exact |
| `tests/test_data_pipeline.py` | test | batch | `tests/test_smoke.py` | role-match |
| `CONTRIBUTING.md` | docs | — | none | no analog |

---

## Pattern Assignments

### `vgp/data/universe.py` (config/constant, transform)

**Analog:** `data_pipeline_example/universe.py`
**Decision:** D-02 — copy verbatim (same UNIVERSE_30 list).

**Full file pattern** (lines 1–84 of analog):
```python
"""
Asset universe definition.

The article uses 30 cryptocurrencies quoted against USDT on Binance,
selected for availability over the full January 2021 – December 2025
sample period.
"""

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
```

**Note:** Drop `UNIVERSE_30_old` and `APPROX_MCAP_2021` — those are rp_pca artifacts, not needed in vgp.

---

### `vgp/data/fetcher.py` (service, file-I/O + request-response)

**Analog:** `data_pipeline_example/fetcher.py`
**Decision:** D-01, D-03 — port directly; change primary public API from `fetch_all()` (close-only) to `fetch_ohlcv()` (full OHLCV dict).

**Imports pattern** (analog lines 1–24):
```python
from __future__ import annotations

import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from .universe import UNIVERSE_30, get_binance_symbols

logger = logging.getLogger(__name__)

BINANCE_REST = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000
SLEEP_BETWEEN_CALLS = 0.12
```

**Class init pattern** (analog lines 46–55):
```python
def __init__(
    self,
    cache_dir: Path,
    symbols: Optional[list[str]] = None,
    use_ccxt_fallback: bool = True,
) -> None:
    self.cache_dir = Path(cache_dir)
    self.cache_dir.mkdir(parents=True, exist_ok=True)
    self.symbols = symbols or get_binance_symbols()
    self.use_ccxt_fallback = use_ccxt_fallback
```

**Primary public API — fetch_ohlcv()** (analog lines 122–143):
This is the method to promote as the main API in `vgp/data/fetcher.py`. The analog already implements it returning `dict[str, pd.DataFrame]` keyed by ticker. Copy as-is; it is already the correct signature for D-03.
```python
def fetch_ohlcv(
    self,
    start_date: str = "2021-01-01",
    end_date: str = "2025-12-31",
    interval: str = "1d",
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for symbol in tqdm(self.symbols, desc="Fetching OHLCV"):
        ticker = symbol.replace("USDT", "")
        try:
            df = self._fetch_symbol(
                symbol, start_date, end_date, interval, force_refresh
            )
            result[ticker] = df
        except Exception as exc:
            logger.warning("Skipping %s — %s", symbol, exc)
    return result
```

**Parquet cache pattern** (analog lines 228–241):
```python
def _fetch_symbol(self, symbol, start_date, end_date, interval, force_refresh):
    cache_path = self.cache_dir / f"{symbol}_{interval}.parquet"
    if cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
        if (
            str(df.index.min().date()) <= start_date
            and str(df.index.max().date()) >= end_date
        ):
            mask = (df.index >= start_date) & (df.index <= end_date)
            return df.loc[mask]
    df = self._paginated_download(symbol, start_date, end_date, interval)
    df.to_parquet(cache_path)
    return df
```

**Pagination pattern** (analog lines 243–277):
Copy `_paginated_download()` and `_parse_klines()` verbatim — these handle Binance's 1000-row page limit and produce a correctly indexed OHLCV DataFrame.

**Error handling pattern** (analog lines 82–90):
```python
try:
    df = self._fetch_symbol(...)
    result[ticker] = df
except Exception as exc:
    logger.warning("Skipping %s — %s", symbol, exc)
```
Per-symbol exceptions are caught and logged; the loop continues. This is the correct pattern — do not let one failing asset abort the entire fetch.

**What to drop from the analog:** `fetch_all()` (close-only return) can be removed or kept as a convenience method but must NOT be the primary API. CCXT fallback (`_fetch_all_ccxt`) may be carried across unchanged.

---

### `vgp/data/feature_engine.py` (service, transform + batch)

**Analog:** `data_pipeline_example/processor.py`
**Decision:** D-04 through D-08 — port gap-handling and asset-drop logic from `ReturnProcessor`; replace return computation with the 4 feature groups.

**Imports pattern** (analog lines 1–23):
```python
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
```

**Gap handling + asset drop pattern** (analog lines 78–97):
```python
prices = prices.copy()
prices = prices.sort_index()

# 1. Forward-fill short gaps (max 3 consecutive days)
prices = prices.ffill(limit=self.max_fill_days)   # max_fill_days=3

# 2. Drop assets with too many missing values
min_obs = int(self.min_obs_fraction * len(prices))  # min_obs_fraction=0.80
n_valid = prices.notna().sum()
keep = n_valid[n_valid >= min_obs].index.tolist()
dropped = [c for c in prices.columns if c not in keep]
if dropped:
    logger.info("Dropping assets with insufficient data: %s", dropped)
prices = prices[keep]
```
Apply this at the start of `FeatureEngine.fit_transform()` before computing any features, using the `close` column from each asset's OHLCV DataFrame to determine observation counts.

**Core output pattern — [T × F × A] float32 array:**
No direct analog exists for the 3-D stacking. Build it as:
```python
# After computing per-asset feature DataFrames, stack along asset axis
arrays = [feature_df.to_numpy(dtype=np.float32) for feature_df in per_asset_features]
result = np.stack(arrays, axis=2)  # shape: [T, F, A]
```

**NaN sanity check pattern** (D-08 requirement — raise on residual NaN):
```python
if np.isnan(result).any():
    raise ValueError(
        "FeatureEngine output contains NaN after lookback trim. "
        "Check rolling windows and ffill logic."
    )
```

**pandas 3.0 idiom** (enforced throughout — CLAUDE.md constraint):
```python
# CORRECT
arr = df["close"].to_numpy()   # not .values
df2 = df.copy()                # explicit copy before mutation
df2.loc[mask, "col"] = value   # no chained assignment
```

**Constructor pattern** (model from analog `ReturnProcessor.__init__`):
```python
class FeatureEngine:
    def __init__(
        self,
        min_obs_fraction: float = 0.80,
        max_fill_days: int = 3,
        lookback: int = 20,        # max rolling window; rows trimmed from start
    ) -> None:
        self.min_obs_fraction = min_obs_fraction
        self.max_fill_days = max_fill_days
        self.lookback = lookback
        self.dropped_assets_: list[str] = []
        self.retained_assets_: list[str] = []
        self.feature_names_: list[str] = []
```

---

### `vgp/data/splitter.py` (utility, transform)

**Analog:** `data_pipeline_example/processor.py` — no direct analog for a splitter; use the date-masking pattern from `_fetch_symbol` in fetcher.py (analog lines 233–237) as the slice pattern.

**Structural enforcement pattern** (D-11):
```python
def split(
    self,
    data,           # pd.DataFrame or np.ndarray with DatetimeIndex on axis 0
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
) -> tuple:
    assert pd.Timestamp(val_start) > pd.Timestamp(train_end), (
        f"val_start ({val_start}) must be strictly after train_end ({train_end})"
    )
    assert pd.Timestamp(test_start) > pd.Timestamp(val_end), (
        f"test_start ({test_start}) must be strictly after val_end ({val_end})"
    )
    ...
```

**Date-mask slice pattern** (from analog `_fetch_symbol`, lines 233–237):
```python
mask = (df.index >= start_date) & (df.index <= end_date)
return df.loc[mask]
```
Apply this pattern three times for train/val/test slices. For `np.ndarray` input, convert the DatetimeIndex to integer positions via `np.searchsorted`.

**Imports pattern:**
```python
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
```

---

### `vgp/data/config.py` (config, —)

**Analog:** Constructor parameters of `data_pipeline_example/fetcher.py` (lines 46–55) and `data_pipeline_example/processor.py` (lines 43–51).
**Pattern:** Use a `dataclasses.dataclass` to consolidate all pipeline parameters.

**Core pattern:**
```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    # Date boundaries (D-10)
    start_date: str = "2021-01-01"
    train_end: str = "2023-12-31"
    val_start: str = "2024-01-01"
    val_end: str = "2024-06-30"
    test_start: str = "2024-07-01"
    end_date: str = "2025-12-31"

    # Fetcher params (from fetcher.py constructor)
    interval: str = "1d"
    cache_dir: Path = field(default_factory=lambda: Path("vgp/data/cache"))
    use_ccxt_fallback: bool = True

    # FeatureEngine params (from processor.py constructor)
    min_obs_fraction: float = 0.80
    max_fill_days: int = 3
    lookback: int = 20
```

---

### `vgp/data/__init__.py` (config, —)

**Analog:** `data_pipeline_example/__init__.py` (lines 1–20)

**Pattern** — mirror the example's `__all__` structure but for vgp exports (D-13):
```python
"""Data pipeline: DataLoader, FeatureEngine, WalkForwardSplitter."""

from .universe import UNIVERSE_30, get_binance_symbols
from .fetcher import BinanceFetcher
from .feature_engine import FeatureEngine
from .splitter import WalkForwardSplitter
from .config import DataConfig

__all__ = [
    "UNIVERSE_30",
    "get_binance_symbols",
    "BinanceFetcher",
    "FeatureEngine",
    "WalkForwardSplitter",
    "DataConfig",
]
```

---

### `tests/test_data_pipeline.py` (test, batch)

**Analog:** `tests/test_smoke.py`
**Decision:** DATA-04 — fixture test against `data_pipeline_example/cache/` parquets; no network required.

**Imports pattern** (analog lines 1–15):
```python
import numpy as np
import pandas as pd
import pytest
```

**Test function structure** (analog lines 18–33 — assert-based, single-concern functions):
```python
def test_feature_engine_no_nan():
    """FeatureEngine output must have zero NaN entries (DATA-04)."""
    from vgp.data import BinanceFetcher, FeatureEngine
    from pathlib import Path

    cache_dir = Path("data_pipeline_example/cache")
    fetcher = BinanceFetcher(cache_dir=cache_dir)
    ohlcv = fetcher.fetch_ohlcv(
        start_date="2021-01-01",
        end_date="2025-12-31",
        force_refresh=False,
    )
    engine = FeatureEngine()
    arr = engine.fit_transform(ohlcv)

    assert arr.ndim == 3, f"Expected 3-D array, got {arr.ndim}-D"
    assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"
    assert not np.isnan(arr).any(), "Feature matrix contains NaN"
```

**Assertion style** (analog lines 30–33 — include diagnostic message):
```python
assert result == 100.0, f"numba JIT result was {result}, expected 100.0"
# Always include f-string message with actual value in assertions
```

**Fixture-based test pattern for splitter:**
```python
def test_walk_forward_splitter_ordering():
    """WalkForwardSplitter raises AssertionError on inverted dates (D-11)."""
    from vgp.data import WalkForwardSplitter
    import pytest

    splitter = WalkForwardSplitter()
    with pytest.raises(AssertionError):
        splitter.split(
            data=pd.DataFrame(),
            train_end="2024-01-01",
            val_start="2023-06-01",  # before train_end — must raise
            val_end="2024-06-30",
            test_start="2024-07-01",
        )
```

---

## Shared Patterns

### pandas 3.0 Idioms
**Source:** `tests/test_smoke.py` lines 93–111 + `CLAUDE.md` §Critical Technical Constraints
**Apply to:** All files in `vgp/data/` that touch DataFrames
```python
# ALWAYS use .to_numpy(), never .values
arr = df["close"].to_numpy()

# ALWAYS explicit .copy() before mutation
prices = prices.copy()

# ALWAYS .loc[] for assignment, never chained indexing
df.loc[mask, "col"] = value
```

### Module-Level Logger Pattern
**Source:** `data_pipeline_example/fetcher.py` line 24, `data_pipeline_example/processor.py` line 23
**Apply to:** `fetcher.py`, `feature_engine.py`, `splitter.py`
```python
import logging
logger = logging.getLogger(__name__)
```

### `from __future__ import annotations`
**Source:** `data_pipeline_example/fetcher.py` line 10, `data_pipeline_example/processor.py` line 16
**Apply to:** All `vgp/data/` Python files — enables PEP 563 postponed evaluation for forward references in type hints.

### Error Handling — Per-Asset Try/Except
**Source:** `data_pipeline_example/fetcher.py` lines 82–90
**Apply to:** `vgp/data/fetcher.py` (any loop over symbols)
```python
try:
    df = self._fetch_symbol(symbol, ...)
    result[ticker] = df
except Exception as exc:
    logger.warning("Skipping %s — %s", symbol, exc)
```
Never let a single-asset failure abort the full fetch loop.

### Dropped-Asset Logging
**Source:** `data_pipeline_example/processor.py` lines 88–96
**Apply to:** `vgp/data/feature_engine.py`
```python
if dropped:
    logger.info("Dropping assets with insufficient data: %s", dropped)
self.dropped_assets_ = dropped
self.retained_assets_ = keep
logger.info(
    "Asset filter: %d/%d assets retained (min_obs_fraction=%.2f, min_obs=%d rows)",
    len(keep), len(prices.columns), self.min_obs_fraction, min_obs,
)
```

---

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `vgp/data/splitter.py` | utility | transform | No splitter class exists in any project file; date-mask slice pattern borrowed from `_fetch_symbol`. The `AssertionError` enforcement pattern (D-11) has no analog and must be written from spec. |
| `CONTRIBUTING.md` | docs | — | No existing contributor guide. Write from CONTEXT.md D-14 requirements: sections for adding a GP primitive, running an experiment, updating the lock file. |

---

## Metadata

**Analog search scope:** `data_pipeline_example/`, `vgp/`, `tests/`
**Files scanned:** 9 Python files (all project Python files)
**Pattern extraction date:** 2026-06-08
