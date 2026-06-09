"""TreeEvaluator — compile and execute a GP tree over a [T x F] feature matrix.

STRUCTURAL INVARIANT (D-05):
  fshift(1) is applied inside execute() after tree execution and before signal
  conversion. This is structural, not configurable. Signal at time t uses only
  tree output from t-1. np.roll(..., 1) wraps the last value to index 0 —
  that wrap is zeroed out explicitly to prevent the periodic artifact.
"""
from __future__ import annotations

import logging

import numpy as np
from deap import gp as deap_gp

logger = logging.getLogger(__name__)


class TreeEvaluator:
    """Compile and execute a GP tree over a single-asset [T x F] feature matrix.

    Parameters
    ----------
    pset : deap.gp.PrimitiveSetTyped
        The primitive set used for tree compilation. Build with build_pset()
        from vgp.gp.gp_types.

    Usage
    -----
    evaluator = TreeEvaluator(build_pset())
    signal = evaluator.execute(individual, feature_matrix)  # shape [T]
    """

    def __init__(self, pset: deap_gp.PrimitiveSetTyped) -> None:
        self._pset = pset

    def execute(self, individual, feature_matrix: np.ndarray) -> np.ndarray:
        """Compile and execute a GP tree; return a [T] signal array.

        Structural invariant (D-05)
        ---------------------------
        fshift(1) is applied inside this method after tree execution.
        Signal at time t uses only tree output from t-1.
        np.roll wraps the last element to index 0; that element is
        explicitly zeroed to prevent circular-wrap lookahead.

        Parameters
        ----------
        individual : creator.Individual (DEAP PrimitiveTree)
            The GP tree to compile and execute.
        feature_matrix : np.ndarray
            Shape [T x F], dtype float32. Single-asset feature slice.
            F must be exactly 12 (matching FEATURE_NAMES from FeatureEngine).

        Returns
        -------
        np.ndarray
            Shape [T], dtype float32, values in {-1.0, 0.0, +1.0}.
        """
        assert feature_matrix.ndim == 2, (
            f"feature_matrix must be 2-D [T x F], got shape {feature_matrix.shape}"
        )
        assert feature_matrix.shape[1] == 12, (
            f"Expected F=12 feature columns (matching FEATURE_NAMES), "
            f"got {feature_matrix.shape[1]}"
        )

        T, F = feature_matrix.shape

        # Compile the tree to a callable Python function.
        # gp.compile returns func(ARG0, ARG1, ..., ARG11) where each arg is
        # the corresponding feature column from FEATURE_NAMES.
        func = deap_gp.compile(individual, self._pset)

        # Vectorized execution: pass each column as a full [T] array.
        # This is a list comprehension for argument unpacking — NOT a
        # per-bar loop. The compiled function applies numpy ops over [T].
        raw_output = func(*[feature_matrix[:, f] for f in range(F)])

        # Coerce to float32 (tree output type depends on which primitives fire)
        raw_output = np.asarray(raw_output, dtype=np.float32)

        # Structural fshift(1): shift output forward by 1 timestep.
        # Signal at time t = tree output from t-1.
        # np.roll is circular: roll(arr, 1)[0] = arr[-1] (last value wraps to index 0).
        # Zero out index 0 to prevent this circular wrap-around lookahead.
        shifted = np.roll(raw_output, 1)
        shifted[0] = 0.0  # no prior output on the first bar — flat (D-05)

        # Convert to 3-state signal: +1 (long), -1 (short), 0 (flat)
        # np.sign returns 0 only when shifted==0.0, which only occurs at index 0
        # after the explicit zero-out above.
        signal = np.sign(shifted).astype(np.float32)
        return signal
