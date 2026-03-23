"""
Global configuration for the RP-PCA crypto portfolio system.

All tuneable parameters live here — downstream modules import from this
module so that changes propagate everywhere automatically.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data" / "cache"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

# Default date range for Taostats pool-history exports (keep in sync with fetch script defaults)
TAO_SUBNET_EXPORT_START: str = "2025-01-01"
TAO_SUBNET_EXPORT_END: str = "2026-03-19"
# Wide daily close matrix written by the fetch script for fast loads
TAO_SUBNET_WIDE_PARQUET: str = "tao_subnets_wide.parquet"


def _resolve_default_tao_dir() -> Path:
    """
    Resolve the default TAO subnet data directory using a priority chain:

    1. ``TAO_SUBNET_DIR`` environment variable  (set this in your shell / .env)
    2. ``<package>/data/cache/tao_subnets``     (co-located data, default for dev)
    3. ``./tao_subnets``                         (relative to current working directory)
    4. ``~/tao_subnets``                         (home-directory fallback)

    The first existing, non-empty directory wins.  If none exist, returns the
    package-relative path so ``mkdir(parents=True, exist_ok=True)`` can create it.
    """
    candidates: list[Path] = []

    env = os.environ.get("TAO_SUBNET_DIR", "").strip()
    if env:
        candidates.append(Path(env).expanduser().resolve())

    candidates.append(DATA_DIR / "tao_subnets")
    candidates.append(Path.cwd() / "tao_subnets")
    candidates.append(Path.home() / "tao_subnets")

    for p in candidates:
        if p.is_dir() and (
            (p / TAO_SUBNET_WIDE_PARQUET).is_file()
            or any(p.glob("sn*_tao_daily_candles.csv"))
        ):
            return p

    # Nothing found — return the canonical package-relative path
    return DATA_DIR / "tao_subnets"


TAO_SUBNET_DATA_DIR: Path = _resolve_default_tao_dir()

for _d in (DATA_DIR, TAO_SUBNET_DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    # Estimation period used in the article
    start_date: str = "2021-01-01"
    end_date: str = "2025-12-31"

    # Binance OHLCV interval
    interval: str = "1d"

    # Return winsorization percentiles (1st / 99th)
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99

    # Local cache directory
    cache_dir: Path = DATA_DIR

    # Minimum non-null observations required to include an asset
    min_obs_fraction: float = 0.80

    # "binance" = 30-coin USDT OHLCV; "tao_subnets" = Taostats CSV exports
    data_source: Literal["binance", "tao_subnets"] = "binance"

    # Directory with ``sn*_tao_daily_candles.csv`` and/or ``tao_subnets_wide.parquet``
    tao_subnet_csv_dir: Path = field(default_factory=lambda: TAO_SUBNET_DATA_DIR)

    # Override min_obs_fraction for TAO subnets (subnets have different birth dates
    # so many have <80% coverage).  Applied when data_source == "tao_subnets".
    tao_min_obs_fraction: float = 0.50


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # Number of principal components to retain
    n_components: int = 5

    # RP-PCA penalty parameter γ.
    # None → auto-set to T (number of observations) at fit time.
    # γ = 0 → centered PCA (ignores mean returns)
    # γ = 1 → uncentered PCA
    # γ > 1 → favours factors with high mean AND high variance (RP-PCA)
    gamma: Optional[float] = None

    # Grid of γ values for hyperparameter search
    gamma_grid: List[float] = field(
        default_factory=lambda: [0.0, 1.0, 10.0, 50.0, 100.0, 200.0]
    )

    # Grid of n_components values for parameter sweep
    n_components_grid: List[int] = field(
        default_factory=lambda: [3, 5, 7, 10]
    )

    # Covariance methods to include in the parameter sweep
    cov_method_grid: List[str] = field(
        default_factory=lambda: ["sample", "ewma", "ledoit_wolf"]
    )

    # Covariance estimation method: "sample" | "ewma" | "ledoit_wolf"
    cov_method: str = "sample"

    # EWMA half-lives in trading days
    ewma_cov_halflife: int = 60
    ewma_mean_halflife: int = 21


# ---------------------------------------------------------------------------
# Portfolio configuration
# ---------------------------------------------------------------------------
@dataclass
class PortfolioConfig:
    # Allow short positions in factor weights
    allow_short: bool = True

    # Max single-factor weight for the tangency portfolio (0 → no constraint)
    max_weight: float = 0.0

    # Risk-free rate (annualised) for Sharpe computation
    risk_free_rate: float = 0.05

    # Trading days per year
    trading_days: int = 252


# ---------------------------------------------------------------------------
# Backtest configuration
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    # Covariance estimation window (calendar days)
    cov_window: int = 252

    # Mean estimation window — shorter because means are more time-varying
    mean_window: int = 63

    # Walk-forward step size (rebalancing frequency in calendar days)
    rebalance_days: int = 21

    # Grid search: covariance windows to try
    cov_window_grid: List[int] = field(
        default_factory=lambda: [63, 126, 252, 504]
    )

    # Grid search: mean windows to try
    mean_window_grid: List[int] = field(
        default_factory=lambda: [7, 14, 21, 42, 63, 126]
    )

    # Grid search: rebalance frequencies to try (trading days)
    rebalance_days_grid: List[int] = field(
        default_factory=lambda: [7, 14, 21, 28, 35, 42]
    )

    # Minimum observations before the backtest starts making predictions
    min_train_obs: int = 63

    # Use EWMA moments in the walk-forward (separating cov / mean windows)
    use_ewma: bool = True

    # Target gross leverage: sum(|w_asset|) at each rebalance will equal this.
    # 1.0 = 1x (long $1 + short $1 per $1 NAV — dollar-neutral long-short).
    # 2.0 = 2x gross, etc.
    target_gross_leverage: float = 1.0

    # Alpha concentrated strategies: keep only top N assets by |weight| per rebalance.
    # When None or 0, concentrated strategies are not computed.
    concentrated_top_n: Optional[int] = 5

    # Single-asset buy-and-hold benchmark column name (e.g. "BTC"); None = omit
    buy_hold_benchmark: Optional[str] = "BTC"


# ---------------------------------------------------------------------------
# Alpha Model 2 (directional long/short) configuration
# ---------------------------------------------------------------------------
@dataclass
class AlphaModel2Config:
    # Signal lookbacks
    factor_premia_lookback: int = 63
    residual_momentum_days: int = 20
    reversal_short_days: int = 5

    # SCORE component weights (sum to 1)
    weight_factor_momentum: float = 0.40
    weight_factor_reversal: float = 0.25
    weight_residual_momentum: float = 0.25
    weight_risk_adjustment: float = 0.10

    # Entry thresholds (composite SCORE 0-100)
    score_long_threshold: float = 75.0
    score_short_threshold: float = 25.0

    # Portfolio constraints
    gross_leverage: float = 2.0
    net_exposure_cap: float = 0.20
    max_single_long_pct: float = 0.05
    max_single_short_pct: float = 0.03

    # Stretched factor for reversal signal: |F_k| > z * std(F_k)
    reversal_factor_z_threshold: float = 2.0


# ---------------------------------------------------------------------------
# Bootstrap configuration
# ---------------------------------------------------------------------------
@dataclass
class BootstrapConfig:
    # Number of bootstrap replications
    n_reps: int = 1_000

    # Fixed block length in days (Politis-Romano 1994).
    # None → auto-select via Politis-White (2004) heuristic.
    block_length: Optional[int] = 5

    # Random seed for reproducibility
    seed: int = 42


# ---------------------------------------------------------------------------
# Regime analysis configuration
# ---------------------------------------------------------------------------
@dataclass
class RegimeConfig:
    # Rolling window (trading days) for regime classification
    lookback: int = 63

    # Annualised return thresholds for bull/bear classification
    bull_threshold: float = 0.20
    bear_threshold: float = -0.20

    # Rolling mean pairwise correlation threshold for contagion detection
    contagion_corr_threshold: float = 0.80


# ---------------------------------------------------------------------------
# Fama-MacBeth cross-sectional test configuration
# ---------------------------------------------------------------------------
@dataclass
class FamaMacBethConfig:
    # Apply Shanken (1992) errors-in-variables correction to t-stats
    shanken_correction: bool = True

    # Fraction of sample used for pass 1 (time-series betas).
    # Remaining fraction is used for pass 2 (cross-sectional regressions).
    # 0 → use full sample for both passes (in-sample, less conservative).
    split_frac: float = 0.5


# ---------------------------------------------------------------------------
# Convenience: bundle all configs
# ---------------------------------------------------------------------------
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    alpha_model_2: AlphaModel2Config = field(default_factory=AlphaModel2Config)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    fama_macbeth: FamaMacBethConfig = field(default_factory=FamaMacBethConfig)


# Module-level default instance — importable directly
DEFAULT_CONFIG = Config()
