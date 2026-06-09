"""DEAP creator definitions and primitive set factory for the VGP GP layer.

ARCHITECTURE INVARIANT: creator.create() must be called at module import time,
not inside functions. multiprocessing.Pool workers re-import this module; if
creator.create() is inside a function it will not run in workers, causing
AttributeError on Individual.fitness.
"""
from __future__ import annotations

import random

from deap import base, creator, gp


def _rand_scalar_int() -> float:
    """Ephemeral constant generator: random integer in [-5, 5] as float.

    Defined at module level (not as a lambda) so it can be pickled by
    multiprocessing.Pool workers. DEAP's addEphemeralConstant requires a
    callable; using a lambda here causes a RuntimeWarning and pickle failure.
    """
    return float(random.randint(-5, 5))

from vgp.gp.primitives import (
    Vector,
    Scalar,
    prim_add,
    prim_sub,
    prim_mul,
    prim_protected_div,
    prim_neg,
    rolling_mean_5,
    rolling_mean_20,
    rolling_std_5,
    rolling_std_20,
    rolling_max_20,
    rolling_min_20,
)

# ---------------------------------------------------------------------------
# Module-level creator registration (D-09 — DEAP pickling requirement)
# Guard: repeated import in interactive sessions must not raise TypeError
# ---------------------------------------------------------------------------
if not hasattr(creator, "FitnessMulti"):
    # weights=(1.0, 1.0, 1.0): maximize all three objectives.
    # Fitness values stored as (sharpe, total_return, -tree_size); the
    # negative sign on tree_size already encodes parsimony pressure so all
    # three weights are positive (see RESEARCH.md Pattern 6).
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))

if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)


def build_pset() -> gp.PrimitiveSetTyped:
    """Build and return the typed primitive set for Phase 3.

    The primitive set defines 12 input terminals (one per feature column,
    all typed as Vector), the arithmetic and rolling primitive functions,
    and a scalar ephemeral constant terminal.

    Primitive set structure (D-06):
    - Arithmetic:  add, sub, mul, protected_div (Vector x Vector -> Vector)
                   neg (Vector -> Vector)
    - Rolling agg: rolling_mean_5/20, rolling_std_5/20,
                   rolling_max_20, rolling_min_20  (Vector -> Scalar)
    - Constants:   scalar_int — random integer in [-5, 5] (Scalar terminal)

    NOTE: Scalar is defined as a subclass of Vector so that DEAP's issubclass
    type check allows Scalar outputs to satisfy Vector input requirements on
    arithmetic primitives. Without this, trees that use rolling aggregations
    as inputs to arithmetic ops would be type-invalid and DEAP would dead-end
    during tree generation.

    NOTE on GP-04: Conditional/comparison primitives are deferred per D-07.
    They are NOT registered in this primitive set.

    Returns
    -------
    gp.PrimitiveSetTyped
        Fully configured primitive set. Pass to TreeEvaluator and to
        gp.genHalfAndHalf / gp.compile in tests and evolution.
    """
    # 12 inputs — one per feature column from FEATURE_NAMES; all are Vector type
    pset = gp.PrimitiveSetTyped(
        "MAIN",
        in_types=[Vector] * 12,
        ret_type=Vector,
    )

    # Arithmetic primitives: Vector x Vector -> Vector
    pset.addPrimitive(prim_add, [Vector, Vector], Vector, name="add")
    pset.addPrimitive(prim_sub, [Vector, Vector], Vector, name="sub")
    pset.addPrimitive(prim_mul, [Vector, Vector], Vector, name="mul")
    pset.addPrimitive(prim_protected_div, [Vector, Vector], Vector, name="div")
    pset.addPrimitive(prim_neg, [Vector], Vector, name="neg")

    # Rolling aggregation primitives: Vector -> Scalar
    # (Scalar is a subclass of Vector; output can feed back into Vector slots)
    pset.addPrimitive(rolling_mean_5, [Vector], Scalar, name="rmean5")
    pset.addPrimitive(rolling_mean_20, [Vector], Scalar, name="rmean20")
    pset.addPrimitive(rolling_std_5, [Vector], Scalar, name="rstd5")
    pset.addPrimitive(rolling_std_20, [Vector], Scalar, name="rstd20")
    pset.addPrimitive(rolling_max_20, [Vector], Scalar, name="rmax20")
    pset.addPrimitive(rolling_min_20, [Vector], Scalar, name="rmin20")

    # Scalar terminal: ephemeral integer constant in [-5, 5]
    # _rand_scalar_int is module-level (not a lambda) so multiprocessing.Pool
    # workers can pickle it. DEAP warns and fails to pickle lambda-based
    # ephemeral constants in parallel evolution (Phase 4).
    pset.addEphemeralConstant(
        "scalar_int",
        _rand_scalar_int,
        Scalar,
    )

    # Rename ARG0..ARG11 to match FEATURE_NAMES for readable tree representations
    pset.renameArguments(
        ARG0="ret_1d",
        ARG1="ret_5d",
        ARG2="ret_20d",
        ARG3="log_close",
        ARG4="vol_5d",
        ARG5="vol_20d",
        ARG6="atr_14",
        ARG7="parkinson_14",
        ARG8="rsi_14",
        ARG9="norm_close",
        ARG10="vol_ratio_20d",
        ARG11="obv_signal",
    )

    return pset
