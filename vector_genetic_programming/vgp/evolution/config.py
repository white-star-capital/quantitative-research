"""EvolutionConfig — NSGA-II hyperparameters and run configuration.

All parameters match CONTEXT.md D-13 defaults. Override for experiment sweeps.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class EvolutionConfig:
    """NSGA-II hyperparameters and run configuration (D-13).

    All parameters have sensible defaults for a validation run.
    Scale pop_size and n_generations up after confirming a successful short run.
    """

    # Population / generations
    pop_size: int = 100          # D-13: 100 individuals per generation
    n_generations: int = 10      # D-13: 10 generations for validation runs

    # Genetic operator probabilities
    cxpb: float = 0.7            # D-13: crossover probability
    mutpb: float = 0.2           # D-13: mutation probability

    # Parallelism (D-06): n_jobs=1 skips Pool entirely (debugging mode)
    n_jobs: int = field(default_factory=lambda: max(1, os.cpu_count() - 1))

    # Checkpointing (D-08): write checkpoint every checkpoint_freq generations
    checkpoint_freq: int = 5
    checkpoint_dir: str = "checkpoints"

    # Reproducibility (D-09, EXP-03)
    seed: int = 42

    # Hall-of-fame (D-14): ParetoFront post-run top-N trim threshold
    hof_size: int = 20

    # Depth limit (D-16, CLAUDE.md #5): enforced via staticLimit on both mate and mutate
    tree_height_limit: int = 8
