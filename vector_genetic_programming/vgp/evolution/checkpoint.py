"""Checkpoint save/load for NSGA-II evolution state — dill serializer (D-07, D-09).

Checkpoint format (D-07):
  {
    'population': list of creator.Individual,
    'halloffame': tools.ParetoFront,
    'logbook': tools.Logbook,
    'rng_state': random.getstate(),       # Python random module state
    'np_rng_state': np.random.get_state(), # numpy legacy RNG state
    'generation': int,                    # generation that was just completed
    'seed': int,                          # original config.seed for audit
  }

D-09: Both rng_state AND np_rng_state must be restored on resume to guarantee
      reproducibility. DEAP operators use Python random; population init uses numpy.
      Missing either causes divergence starting from the next generation.
"""
from __future__ import annotations

import random
from pathlib import Path

import dill
import numpy as np


def save_checkpoint(
    path: str,
    *,
    population,
    halloffame,
    logbook,
    generation: int,
    seed: int,
) -> None:
    """Serialize complete evolution state to disk using dill.

    Parameters
    ----------
    path : str
        File path to write. Parent directories are created if they do not exist.
        Convention: checkpoints/{run_id}/gen_{N:04d}.pkl
    population : list[creator.Individual]
        Current generation population.
    halloffame : tools.ParetoFront
        Accumulated non-dominated individuals.
    logbook : tools.Logbook
        Per-generation statistics log.
    generation : int
        The generation number just completed (resume will start at generation+1).
    seed : int
        Original config.seed — stored for audit; not used for restoration.
    """
    checkpoint = {
        "population": population,
        "halloffame": halloffame,
        "logbook": logbook,
        "rng_state": random.getstate(),           # Python random module state
        "np_rng_state": np.random.get_state(),    # numpy legacy RNG state
        "generation": generation,
        "seed": seed,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        dill.dump(checkpoint, f)


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint dict from disk.

    Returns the raw checkpoint dict. Caller is responsible for:
      1. random.setstate(ckpt['rng_state'])
      2. np.random.set_state(ckpt['np_rng_state'])
      3. Setting start_gen = ckpt['generation'] + 1
    These steps must happen BEFORE any call to toolbox.population(),
    varOr(), or toolbox.evaluate().
    """
    with open(path, "rb") as f:
        return dill.load(f)
