"""Backtest runner: vectorbt integration, evaluate(), fitness functions.

ARCHITECTURE INVARIANT: This module must NOT import deap.
Interface from GP: numpy array in -> fitness tuple out.
"""

from .runner import BacktestRunner, EvalConfig, evaluate

__all__ = [
    "evaluate",
    "EvalConfig",
    "BacktestRunner",
]
