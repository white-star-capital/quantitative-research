"""GP primitive functions — all module-level for multiprocessing.Pool pickling.

INVARIANTS (from CLAUDE.md and D-08):
- All functions are defined at module level (no lambdas, no closures).
- All functions accept and return np.ndarray with dtype float32.
- No pandas objects inside any primitive function.
- No per-bar Python loops — use numpy vectorized operations throughout.

Scalar is defined as a subclass of Vector. This resolves DEAP's PrimitiveSetTyped
type-chain dead-end: rolling primitives return Scalar, and since Scalar IS-A Vector,
arithmetic primitives typed [Vector, Vector] -> Vector will accept Scalar inputs.
Without this subclass relationship, trees using rolling aggregations as arithmetic
inputs would be type-invalid and DEAP would fail to generate valid typed trees.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# Type tokens for DEAP PrimitiveSetTyped
# ---------------------------------------------------------------------------

class Vector:
    """Type token for [T] numpy arrays in the DEAP PrimitiveSetTyped type system.

    At runtime, a Vector is a 1-D np.ndarray of shape [T] and dtype float32.
    This class exists only as a type tag — never instantiated.
    """


class Scalar(Vector):
    """Type token for rolling aggregation outputs.

    Defined as a subclass of Vector so DEAP's issubclass() type check
    allows Scalar outputs to satisfy Vector input slots in arithmetic
    primitives. At runtime, a Scalar is also a [T]-shaped float32 ndarray;
    each element is the rolling statistic at that timestep.
    """


# ---------------------------------------------------------------------------
# Arithmetic primitives: (Vector, Vector) -> Vector  [D-06]
# ---------------------------------------------------------------------------

def prim_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Element-wise addition."""
    return (x + y).astype(np.float32)


def prim_sub(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Element-wise subtraction."""
    return (x - y).astype(np.float32)


def prim_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Element-wise multiplication."""
    return (x * y).astype(np.float32)


def prim_protected_div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Protected division: returns x/y; substitutes 1.0 where |y| < 1e-7.

    Uses np.errstate to suppress divide-by-zero warnings in the numpy layer.
    The np.where guard ensures no NaN or Inf values enter the tree output.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            np.abs(y) < 1e-7,
            np.ones_like(x, dtype=np.float32),
            x / y,
        )
    return result.astype(np.float32)


def prim_neg(x: np.ndarray) -> np.ndarray:
    """Unary negation."""
    return np.negative(x).astype(np.float32)


# ---------------------------------------------------------------------------
# Rolling aggregation primitives: Vector -> Scalar  [D-06]
# All return [T]-shaped float32 arrays. First (window-1) values are padded
# with the corresponding input values (no NaN propagation from warm-up).
# sliding_window_view: no per-bar Python loops — fully vectorized.
# ---------------------------------------------------------------------------

def rolling_mean_5(x: np.ndarray) -> np.ndarray:
    """Rolling 5-period mean. Pads first 4 positions with input values."""
    x32 = x.astype(np.float32)
    window = sliding_window_view(x32, window_shape=5)
    result = np.empty(len(x32), dtype=np.float32)
    result[:4] = x32[:4]
    result[4:] = window.mean(axis=-1)
    return result


def rolling_mean_20(x: np.ndarray) -> np.ndarray:
    """Rolling 20-period mean. Pads first 19 positions with input values."""
    x32 = x.astype(np.float32)
    window = sliding_window_view(x32, window_shape=20)
    result = np.empty(len(x32), dtype=np.float32)
    result[:19] = x32[:19]
    result[19:] = window.mean(axis=-1)
    return result


def rolling_std_5(x: np.ndarray) -> np.ndarray:
    """Rolling 5-period standard deviation. Pads first 4 positions."""
    x32 = x.astype(np.float32)
    window = sliding_window_view(x32, window_shape=5)
    result = np.empty(len(x32), dtype=np.float32)
    result[:4] = np.zeros(4, dtype=np.float32)
    result[4:] = window.std(axis=-1)
    return result


def rolling_std_20(x: np.ndarray) -> np.ndarray:
    """Rolling 20-period standard deviation. Pads first 19 positions."""
    x32 = x.astype(np.float32)
    window = sliding_window_view(x32, window_shape=20)
    result = np.empty(len(x32), dtype=np.float32)
    result[:19] = np.zeros(19, dtype=np.float32)
    result[19:] = window.std(axis=-1)
    return result


def rolling_max_20(x: np.ndarray) -> np.ndarray:
    """Rolling 20-period maximum. Pads first 19 positions with input values."""
    x32 = x.astype(np.float32)
    window = sliding_window_view(x32, window_shape=20)
    result = np.empty(len(x32), dtype=np.float32)
    result[:19] = x32[:19]
    result[19:] = window.max(axis=-1)
    return result


def rolling_min_20(x: np.ndarray) -> np.ndarray:
    """Rolling 20-period minimum. Pads first 19 positions with input values."""
    x32 = x.astype(np.float32)
    window = sliding_window_view(x32, window_shape=20)
    result = np.empty(len(x32), dtype=np.float32)
    result[:19] = x32[:19]
    result[19:] = window.min(axis=-1)
    return result
