"""
GP primitive unit tests — GP-08.

Type-correctness validation: all primitives accept np.ndarray and return np.ndarray
with dtype float32. 1000 random trees are generated and executed to verify
PrimitiveSetTyped enforces type safety throughout.
No network access. No deap evolution — only tree generation and execution.
"""
from __future__ import annotations

import numpy as np
import pytest


_T = 200  # timesteps for all array fixtures


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def x(rng):
    return rng.standard_normal(_T).astype(np.float32)


@pytest.fixture
def y(rng):
    return rng.standard_normal(_T).astype(np.float32) + 0.1  # avoid zero


def test_arithmetic_primitives_dtype(x, y):
    """All arithmetic primitives return float32 [T] arrays (GP-08)."""
    from vgp.gp.primitives import prim_add, prim_sub, prim_mul, prim_protected_div, prim_neg

    for fn_2arg in [prim_add, prim_sub, prim_mul, prim_protected_div]:
        result = fn_2arg(x, y)
        assert result.dtype == np.float32, (
            f"{fn_2arg.__name__} returned {result.dtype}, expected float32"
        )
        assert result.shape == (_T,), (
            f"{fn_2arg.__name__} returned shape {result.shape}, expected ({_T},)"
        )

    result = prim_neg(x)
    assert result.dtype == np.float32, f"prim_neg returned {result.dtype}, expected float32"
    assert result.shape == (_T,), f"prim_neg shape {result.shape}"


def test_rolling_primitives_shape_and_dtype(x):
    """All rolling primitives return float32 [T] arrays (GP-08)."""
    from vgp.gp.primitives import (
        rolling_mean_5, rolling_mean_20, rolling_std_5,
        rolling_std_20, rolling_max_20, rolling_min_20,
    )

    for fn in [rolling_mean_5, rolling_mean_20, rolling_std_5,
               rolling_std_20, rolling_max_20, rolling_min_20]:
        result = fn(x)
        assert result.dtype == np.float32, (
            f"{fn.__name__} returned {result.dtype}, expected float32"
        )
        assert result.shape == (_T,), (
            f"{fn.__name__} returned shape {result.shape}, expected ({_T},)"
        )


def test_protected_div_zero_denominator(x):
    """prim_protected_div with zero denominator produces no NaN or Inf."""
    from vgp.gp.primitives import prim_protected_div

    zeros = np.zeros(_T, dtype=np.float32)
    result = prim_protected_div(x, zeros)

    assert not np.any(np.isnan(result)), "prim_protected_div produced NaN with zero denominator"
    assert not np.any(np.isinf(result)), "prim_protected_div produced Inf with zero denominator"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


def test_protected_div_near_zero_denominator(x):
    """prim_protected_div with near-zero denominator (|y| < 1e-7) returns 1.0."""
    from vgp.gp.primitives import prim_protected_div

    tiny = np.full(_T, 1e-8, dtype=np.float32)  # below epsilon threshold
    result = prim_protected_div(x, tiny)
    assert np.all(result == 1.0), (
        f"Expected 1.0 for |y|<1e-7, got values outside 1.0: {np.unique(result)}"
    )


def test_scalar_is_subclass_of_vector():
    """Scalar must be a subclass of Vector for DEAP type-chain to work (GP-01)."""
    from vgp.gp.primitives import Vector, Scalar

    assert issubclass(Scalar, Vector), (
        "Scalar must be a subclass of Vector so DEAP's issubclass() check allows "
        "rolling primitive outputs to feed into arithmetic primitive Vector slots."
    )


def test_1000_random_trees_no_error():
    """1000 random trees generate and execute without IndexError or dtype failure (GP-08)."""
    from deap import creator, gp
    from vgp.gp.gp_types import build_pset
    from vgp.gp.tree_evaluator import TreeEvaluator

    rng = np.random.default_rng(0)
    pset = build_pset()
    evaluator = TreeEvaluator(pset)
    T, F = 100, 12
    feature_matrix = rng.standard_normal((T, F)).astype(np.float32)

    errors = []
    for i in range(1000):
        try:
            expr = gp.genHalfAndHalf(pset, min_=1, max_=4)
            ind = creator.Individual(expr)
            sig = evaluator.execute(ind, feature_matrix)
            assert sig.shape == (T,), f"Tree {i}: wrong shape {sig.shape}"
            assert sig.dtype == np.float32, f"Tree {i}: dtype {sig.dtype}"
        except Exception as exc:
            errors.append(f"Tree {i}: {type(exc).__name__}: {exc}")

    assert not errors, (
        f"{len(errors)} of 1000 trees failed:\n" + "\n".join(errors[:5])
    )
