"""Evolution engine: DEAP toolbox, NSGA-II loop, checkpointing.

ARCHITECTURE INVARIANT: This module must NOT import vectorbt.
Interface to backtesting: numpy array in -> fitness tuple out.
"""

from .checkpoint import load_checkpoint, save_checkpoint
from .config import EvolutionConfig
from .loop import run_evolution
from .tracker import NoOpTracker, make_tracker

__all__ = [
    "EvolutionConfig",
    "run_evolution",
    "save_checkpoint",
    "load_checkpoint",
    "make_tracker",
    "NoOpTracker",
]
