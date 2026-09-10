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
# Evaluation status codes
# ---------------------------------------------------------------------------
# The worst-fitness tuple (-inf, -inf, -tree_size) is a RANKING SENTINEL for
# NSGA-II, not a measurement. It means "this individual is unusable", which is
# not the same as "this individual scored minus infinity". Reporting code must
# never write it out as a Sharpe ratio — use evaluate_with_status() and treat
# any status other than EVAL_OK as "not measured" (NaN).

EVAL_OK = "ok"
EVAL_BELOW_MIN_TRADES = "below_min_trades"
EVAL_NAN_METRICS = "nan_metrics"


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
# Signal and portfolio construction — ONE definition, shared by every caller
# ---------------------------------------------------------------------------

def compute_signals(individual, feature_matrix: np.ndarray) -> np.ndarray:
    """Execute a GP tree per asset and return the [T x A] float32 signal matrix.

    D-01: one tree applied to each asset's [T x F] slice. The loop is over
    ASSETS, not timesteps — TreeEvaluator.execute() is vectorized over T (D-16).
    D-15: the GP import is deferred so this module still holds no deap
    dependency at import time.
    """
    from vgp.gp.gp_types import build_pset  # noqa: PLC0415 — deferred (D-15)
    from vgp.gp.tree_evaluator import TreeEvaluator  # noqa: PLC0415

    if feature_matrix.ndim != 3:
        raise ValueError(
            f"feature_matrix must be 3-D [T x F x A], got shape {feature_matrix.shape}"
        )
    T, F, A = feature_matrix.shape
    if F != 12:
        raise ValueError(f"Expected F=12 feature columns (FEATURE_NAMES), got {F}")

    # pset is stateless — safe to build per call
    evaluator = TreeEvaluator(build_pset())
    signals = np.zeros((T, A), dtype=np.float32)
    for a in range(A):
        signals[:, a] = evaluator.execute(individual, feature_matrix[:, :, a])
    return signals


def count_trades(signals: np.ndarray) -> int:
    """Sign changes summed across assets (D-14's definition of a "trade")."""
    return int(np.sum(np.abs(np.diff(signals, axis=0)) > 0))


def build_portfolio(signals: np.ndarray, config: EvalConfig):
    """Build the vectorbt Portfolio for a signal matrix.

    THE SINGLE DEFINITION of how a signal becomes a portfolio. It used to be
    written twice — here and in vgp.analysis.runner._get_is_returns, which
    re-derived it to obtain per-period returns for the DSR. Two copies of this
    that must agree is exactly the shape of defect this project keeps finding:
    they did agree (verified — the Sharpe from these returns matches
    evaluate()'s to within the annualization factor), but nothing structural
    kept them in step, so a change to fees, sizing or reversal handling in one
    would have silently deflated the DSR of a different portfolio.

    Transaction costs are applied HERE via fees=, never post-hoc
    (CLAUDE.md constraint #3).
    """
    A = signals.shape[1]

    # Convert 3-state signals to boolean long/short entry/exit matrices.
    # Using explicit separate arrays (not direction='both') to support
    # size_type='percent' with position reversals (Pitfall 2).
    long_entries   = signals > 0    # [T x A] bool: go long
    short_entries  = signals < 0    # [T x A] bool: go short
    long_exits     = signals <= 0   # [T x A] bool: exit long
    short_exits    = signals >= 0   # [T x A] bool: exit short

    # fee_bps is round-trip; divide by 2 for per-side, then by 10_000 for decimal.
    fee_per_side = (config.fee_bps / 2.0) / 10_000.0  # 10 bps -> 0.0005

    return vbt.Portfolio.from_signals(
        close=config.close_prices,       # [T x A] DataFrame, DatetimeIndex
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        size=1.0 / A,                    # equal weight: 1/N per asset (D-11)
        size_type="percent",
        upon_opposite_entry="close",     # close existing position before reversing (Pitfall 2)
        fees=fee_per_side,               # EVAL-02 — inside evaluate, not post-hoc
        freq=config.freq,                # "1D" — REQUIRED for sharpe_ratio() (Pitfall 1)
        init_cash=config.init_cash,
        group_by=True,                   # aggregate to single portfolio-level metrics
        cash_sharing=True,
    )


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def evaluate_with_status(
    individual,
    feature_matrix: np.ndarray,
    config: EvalConfig,
) -> tuple[tuple[float, float, float], str, int]:
    """Evaluate a GP individual, returning the fitness tuple AND why.

    Same computation as evaluate(), but it also reports whether the fitness
    tuple is a real measurement or the worst-fitness ranking sentinel. Use this
    anywhere a Sharpe ratio is going to be reported rather than ranked — the
    sentinel is -inf and must not be recorded as performance.

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
    tuple[tuple[float, float, float], str, int]
        (fitness, status, n_trades) where fitness is
        (sharpe_ratio, total_return, -tree_size) and status is one of:
        - EVAL_OK — fitness is a real measurement
        - EVAL_BELOW_MIN_TRADES — fewer than config.min_trades sign changes (D-14)
        - EVAL_NAN_METRICS — NaN Sharpe or NaN total_return from vectorbt
        For both non-OK statuses fitness is (-np.inf, -np.inf, -tree_size).
        n_trades is the observed sign-change count, reported regardless of status.

    Notes
    -----
    D-15: No deap import in this file. len(individual) works because
          DEAP's PrimitiveTree implements __len__.
    D-16: No per-bar Python loops. The per-asset loop (range(A)) is over assets,
          not timesteps. TreeEvaluator.execute() is vectorized over [T].
    """
    tree_size = len(individual)  # PrimitiveTree.__len__ — no deap import needed
    worst_fitness = (-np.inf, -np.inf, float(-tree_size))

    signals = compute_signals(individual, feature_matrix)

    # Trade filter (D-14): sign changes summed across all assets.
    sign_changes = count_trades(signals)
    if sign_changes < config.min_trades:
        logger.debug(
            "Individual (size=%d) has only %d sign changes — below min_trades=%d. "
            "Returning worst fitness.",
            tree_size, sign_changes, config.min_trades,
        )
        return worst_fitness, EVAL_BELOW_MIN_TRADES, sign_changes

    pf = build_portfolio(signals, config)

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
        return worst_fitness, EVAL_NAN_METRICS, sign_changes

    return (sharpe, total_ret, float(-tree_size)), EVAL_OK, sign_changes


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
        Shape [T x F x A], dtype float32.
    config : EvalConfig
        Backtest configuration. config.close_prices must be set to a [T x A] DataFrame.

    Returns
    -------
    tuple[float, float, float]
        (sharpe_ratio, total_return, -tree_size), or the worst-fitness
        sentinel (-np.inf, -np.inf, -tree_size). This form is for NSGA-II
        ranking. Reporting code wanting to distinguish a real measurement
        from the sentinel must call evaluate_with_status() instead.
    """
    fitness, _status, _n_trades = evaluate_with_status(individual, feature_matrix, config)
    return fitness


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
