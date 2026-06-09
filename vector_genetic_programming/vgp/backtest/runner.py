"""BacktestRunner and evaluate() — vectorbt integration for GP fitness evaluation.

ARCHITECTURE INVARIANT (D-15):
  This module must NOT import deap at module level or inside any function.
  The interface is: numpy signal array in -> fitness tuple out.
  tree_size = len(individual) works without deap (PrimitiveTree implements __len__).

Transaction costs are applied INSIDE evaluate() via the fees= parameter to
Portfolio.from_signals. They are NEVER applied post-hoc (CLAUDE.md constraint #3).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import vectorbt as vbt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Configuration for evaluate() — backtest parameters and trade constraints.

    All parameters have safe defaults matching the Phase 3 design decisions.
    Override fee_bps for sensitivity analysis in Phase 4/5.
    """

    # Transaction costs (D-10): 10 bps round-trip = 5 bps per side
    # Applied via fees=fee_per_side in Portfolio.from_signals — not post-hoc.
    fee_bps: float = 10.0

    # Trade filter (D-14): individuals below min_trades receive worst fitness.
    # "Trade" = sign change in signal: np.sum(np.abs(np.diff(signals, axis=0)) > 0)
    min_trades: int = 50

    # Portfolio parameters
    # freq is REQUIRED for sharpe_ratio() — omitting causes silent NaN (Pitfall 1).
    freq: str = "1D"
    init_cash: float = 10_000.0

    # Close prices for vectorbt (D-12).
    # Must be a pd.DataFrame with DatetimeIndex, shape [T x A].
    # Log-close (feature index 3) cannot be used here — vectorbt needs raw prices for PnL.
    close_prices: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    individual,
    feature_matrix: np.ndarray,
    config: EvalConfig,
) -> tuple[float, float, float]:
    """Evaluate a GP individual and return a three-objective fitness tuple.

    This function is the complete evaluate() contract for NSGA-II in Phase 4.
    It must be importable and callable without deap being present in this module.

    Parameters
    ----------
    individual : creator.Individual (DEAP PrimitiveTree — passed as opaque object)
        The GP tree to evaluate. len(individual) gives tree size without importing deap.
    feature_matrix : np.ndarray
        Shape [T x F x A], dtype float32. The train-split feature matrix from Phase 2.
        T = timesteps, F = 12 features (FEATURE_NAMES order), A = number of assets.
    config : EvalConfig
        Backtest configuration. config.close_prices must be set to a [T x A] DataFrame.

    Returns
    -------
    tuple[float, float, float]
        (sharpe_ratio, total_return, -tree_size)
        Worst fitness = (-np.inf, -np.inf, -tree_size) for:
        - fewer than config.min_trades sign changes (D-14)
        - NaN Sharpe or NaN total_return from vectorbt

    Notes
    -----
    D-15: No deap import in this file. len(individual) works because
          DEAP's PrimitiveTree implements __len__.
    D-16: No per-bar Python loops. The per-asset loop (range(A)) is over assets,
          not timesteps. TreeEvaluator.execute() is vectorized over [T].
    """
    # Import GP layer here (deferred — maintains architectural separation D-15)
    from vgp.gp.tree_evaluator import TreeEvaluator
    from vgp.gp.gp_types import build_pset

    tree_size = len(individual)  # PrimitiveTree.__len__ — no deap import needed
    worst_fitness = (-np.inf, -np.inf, float(-tree_size))

    # Validate input shape
    if feature_matrix.ndim != 3:
        raise ValueError(
            f"feature_matrix must be 3-D [T x F x A], got shape {feature_matrix.shape}"
        )
    T, F, A = feature_matrix.shape
    if F != 12:
        raise ValueError(
            f"Expected F=12 feature columns (FEATURE_NAMES), got {F}"
        )

    # Build evaluator (pset is stateless — safe to build per call; Phase 4 will cache)
    pset = build_pset()
    evaluator = TreeEvaluator(pset)

    # Execute tree on each asset independently — loop is over A assets, not T timesteps.
    # D-01: single tree applied to single-asset [T x F] slice for each asset.
    signals = np.zeros((T, A), dtype=np.float32)
    for a in range(A):
        signals[:, a] = evaluator.execute(individual, feature_matrix[:, :, a])

    # Trade filter (D-14): count sign changes summed across all assets.
    # "Trade" = sign change in the 3-state signal at any timestep for any asset.
    # np.abs(np.diff(...)) > 0 is True when signal changes (including 0->±1, ±1->∓1).
    sign_changes = int(np.sum(np.abs(np.diff(signals, axis=0)) > 0))
    if sign_changes < config.min_trades:
        logger.debug(
            "Individual (size=%d) has only %d sign changes — below min_trades=%d. "
            "Returning worst fitness.",
            tree_size, sign_changes, config.min_trades,
        )
        return worst_fitness

    # Convert 3-state signals to boolean long/short entry/exit matrices.
    # Using explicit separate arrays (not direction='both') to support
    # size_type='percent' with position reversals (Pitfall 2).
    long_entries   = signals > 0    # [T x A] bool: go long
    short_entries  = signals < 0    # [T x A] bool: go short
    long_exits     = signals <= 0   # [T x A] bool: exit long
    short_exits    = signals >= 0   # [T x A] bool: exit short

    # Transaction costs: 10 bps round-trip = 5 bps per side.
    # fee_bps is round-trip; divide by 2 for per-side, then by 10_000 for decimal.
    fee_per_side = (config.fee_bps / 2.0) / 10_000.0  # 10 bps -> 0.0005

    pf = vbt.Portfolio.from_signals(
        close=config.close_prices,       # [T x A] DataFrame, DatetimeIndex
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        size=1.0 / A,                    # equal weight: 1/N per asset (D-11)
        size_type="percent",
        upon_opposite_entry="close",     # close existing position before reversing (Pitfall 2)
        fees=fee_per_side,               # 5 bps per side (EVAL-02 — inside evaluate, not post-hoc)
        freq=config.freq,                # "1D" — REQUIRED for sharpe_ratio() (Pitfall 1)
        init_cash=config.init_cash,
        group_by=True,                   # aggregate to single portfolio-level metrics
        cash_sharing=True,
    )

    sharpe = float(pf.sharpe_ratio())
    total_ret = float(pf.total_return())

    # NaN guard: flat portfolio or all-rejected trades produce NaN metrics.
    # These individuals receive worst fitness — not excluded (D-14 must be rankable).
    if np.isnan(sharpe) or np.isnan(total_ret):
        logger.debug(
            "Individual (size=%d) produced NaN metrics (sharpe=%s, total_ret=%s). "
            "Returning worst fitness.",
            tree_size, sharpe, total_ret,
        )
        return worst_fitness

    return (sharpe, total_ret, float(-tree_size))


# ---------------------------------------------------------------------------
# BacktestRunner class — stateful wrapper for Phase 4 (multiprocessing friendly)
# ---------------------------------------------------------------------------

class BacktestRunner:
    """Stateful wrapper around evaluate() for use in Phase 4 evolution loop.

    Phase 4 registers runner.run as the evaluation function in the DEAP toolbox.
    By encapsulating pset and config here, workers only need to pickle the
    BacktestRunner instance (not the pset separately).

    Parameters
    ----------
    config : EvalConfig
        Backtest configuration (fee_bps, min_trades, freq, close_prices, etc.)
    feature_matrix : np.ndarray
        Shape [T x F x A], dtype float32. Train-split feature matrix.
    """

    def __init__(self, config: EvalConfig, feature_matrix: np.ndarray) -> None:
        self._config = config
        self._feature_matrix = feature_matrix

    def run(self, individual) -> tuple[float, float, float]:
        """Evaluate a single individual. Callable by multiprocessing.Pool.map."""
        return evaluate(individual, self._feature_matrix, self._config)
