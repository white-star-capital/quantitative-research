"""
TreeEvaluator tests — GP-05 (vectorized execution), GP-06 (fshift), GP-07 (lookahead detection).

All tests use synthetic [T x 12] float32 feature matrices.
No network access. No parquet files required.
"""
from __future__ import annotations

import numpy as np
import pytest


_T = 300   # timesteps for tree evaluator fixtures
_F = 12    # feature columns — must match FEATURE_NAMES length


@pytest.fixture(scope="module")
def pset():
    from vgp.gp.gp_types import build_pset as _build
    return _build()


@pytest.fixture(scope="module")
def evaluator(pset):
    from vgp.gp.tree_evaluator import TreeEvaluator
    return TreeEvaluator(pset)


@pytest.fixture
def feature_matrix():
    rng = np.random.default_rng(99)
    return rng.standard_normal((_T, _F)).astype(np.float32)


@pytest.fixture
def random_individual(pset):
    from deap import creator, gp
    import random
    random.seed(7)
    expr = gp.genHalfAndHalf(pset, min_=1, max_=3)
    return creator.Individual(expr)


# ---------------------------------------------------------------------------
# GP-05: Vectorized execution
# ---------------------------------------------------------------------------

def test_execute_output_shape_gp05(evaluator, random_individual, feature_matrix):
    """execute() returns a 1-D array of length T (GP-05)."""
    sig = evaluator.execute(random_individual, feature_matrix)
    assert sig.shape == (_T,), f"Expected shape ({_T},), got {sig.shape}"


def test_execute_output_dtype_gp05(evaluator, random_individual, feature_matrix):
    """execute() returns float32 (GP-05)."""
    sig = evaluator.execute(random_individual, feature_matrix)
    assert sig.dtype == np.float32, f"Expected float32, got {sig.dtype}"


def test_execute_output_values_gp05(evaluator, random_individual, feature_matrix):
    """execute() output values are in {-1.0, 0.0, +1.0} (GP-05)."""
    sig = evaluator.execute(random_individual, feature_matrix)
    unique = set(np.unique(sig))
    assert unique.issubset({-1.0, 0.0, 1.0}), (
        f"Signal values outside {{-1, 0, 1}}: {unique}"
    )


# ---------------------------------------------------------------------------
# GP-06: Structural fshift(1)
# ---------------------------------------------------------------------------

def test_fshift_index0_is_zero_gp06(evaluator, random_individual, feature_matrix):
    """Signal at index 0 must be 0.0 — no prior output on the first bar (GP-06)."""
    sig = evaluator.execute(random_individual, feature_matrix)
    assert sig[0] == 0.0, (
        f"signal[0] = {sig[0]} but must be 0.0 after structural fshift zero-out (D-05)"
    )


def test_fshift_roll_boundary_no_lookahead_gp06(evaluator, feature_matrix):
    """Verify that np.roll wraps the last value and is explicitly zeroed (GP-06, Pitfall 3).

    Construct a degenerate tree whose raw output is a constant non-zero array.
    After np.roll(arr, 1), index 0 would be arr[-1] (the wrapped last value)
    if we did NOT zero it out. The tree evaluator must zero it, so signal[0] == 0.
    """
    from deap import creator, gp
    from vgp.gp.gp_types import build_pset
    from vgp.gp.tree_evaluator import TreeEvaluator

    local_pset = build_pset()
    # genFull with depth 0 produces a terminal; ensure we get a feature column
    # terminal (not a scalar constant) by looping
    ind = None
    for _ in range(50):
        expr = gp.genFull(local_pset, min_=0, max_=0)
        candidate = creator.Individual(expr)
        # Check it's NOT a scalar constant terminal (which has a .value float attribute)
        if hasattr(candidate[0], 'value') and isinstance(candidate[0].value, float):
            continue  # skip ephemeral scalar constant terminals
        ind = candidate
        break

    if ind is None:
        pytest.skip("Could not generate a feature terminal in 50 attempts")

    # Use a feature matrix where all feature columns are all-positive
    fm = np.ones((_T, _F), dtype=np.float32) * 2.0

    local_evaluator = TreeEvaluator(local_pset)
    sig = local_evaluator.execute(ind, fm)

    # After fshift, index 0 must be 0.0 regardless of the raw output value
    assert sig[0] == 0.0, (
        f"signal[0] = {sig[0]} — np.roll wrapped the last value to index 0 "
        f"without zeroing it. Structural fshift is broken."
    )


def test_shape_assertion_wrong_ndim(evaluator, random_individual):
    """execute() raises AssertionError for non-2D feature matrix."""
    with pytest.raises(AssertionError, match="2-D"):
        evaluator.execute(random_individual, np.ones((_T,), dtype=np.float32))


def test_shape_assertion_wrong_F(evaluator, random_individual):
    """execute() raises AssertionError when F != 12."""
    with pytest.raises(AssertionError, match="F=12"):
        evaluator.execute(random_individual, np.ones((_T, 10), dtype=np.float32))


# ---------------------------------------------------------------------------
# GP-07: Lookahead detection
# ---------------------------------------------------------------------------

def test_lookahead_detection_gp07():
    """Injecting a future-leak primitive produces higher in-sample fitness (GP-07).

    Design:
    - Build a 'leaky pset' that includes leak_future(x) = np.roll(x, -1),
      a primitive that returns tomorrow's value today.
    - Construct a trivial tree that uses leak_future on a single feature.
    - Construct an equivalent no-leak tree using the same feature (no roll).
    - Evaluate both trees with a simple IS fitness proxy:
      the correlation of signal[t] with return[t+1].
    - Assert: the leaky tree's proxy fitness is higher (closer to 1) than the
      no-leak tree's proxy fitness. This confirms the lookahead is detectable.

    The test does NOT call evaluate() (that's in test_evaluate.py). It confirms
    that a leaky tree systematically produces signals that correlate with future
    returns — which would artificially inflate in-sample Sharpe in the full system.
    """
    from deap import creator, gp
    from vgp.gp.primitives import Vector, Scalar
    from vgp.gp.gp_types import build_pset
    from vgp.gp.tree_evaluator import TreeEvaluator

    # ---------- define the future-leak primitive ----------
    def leak_future(x: np.ndarray) -> np.ndarray:
        """Returns np.roll(x, -1) — uses tomorrow's value today."""
        return np.roll(x.astype(np.float32), -1)

    # Build leaky pset: same as standard pset + leak_future
    leaky_pset = build_pset()
    leaky_pset.addPrimitive(leak_future, [Vector], Vector, name="leak_future")

    # ---------- synthetic data: returns follow a trend pattern ----------
    rng = np.random.default_rng(123)
    T, F = 600, 12
    # ARG0 = simple return signal; build a slightly mean-reverting series
    trend = np.cumsum(rng.standard_normal(T) * 0.01).astype(np.float32)
    fm = rng.standard_normal((T, F)).astype(np.float32)
    fm[:, 0] = trend  # ARG0 = trend

    # ---------- leaky tree: leak_future(ARG0) ----------
    # Manually build: Individual([leak_future_prim, ret_1d_terminal])
    leak_prim = leaky_pset.primitives[Vector][-1]  # leak_future is last added
    ret1d_terminal = leaky_pset.terminals[Vector][0]  # first Vector terminal = ret_1d
    leaky_ind = creator.Individual([leak_prim, ret1d_terminal])

    leaky_eval = TreeEvaluator(leaky_pset)
    leaky_signal = leaky_eval.execute(leaky_ind, fm)

    # ---------- no-leak tree: just ARG0 (ret_1d terminal) ----------
    clean_pset = build_pset()
    clean_eval = TreeEvaluator(clean_pset)
    clean_ind = creator.Individual([clean_pset.terminals[Vector][0]])
    clean_signal = clean_eval.execute(clean_ind, fm)

    # ---------- proxy fitness: correlation of signal[t] with return[t+1] ----------
    # Future return = fm[1:, 0] (next day's ret_1d)
    # Align: signal[:-1] vs fm[1:, 0]
    future_ret = fm[1:, 0]
    leaky_corr = float(np.corrcoef(leaky_signal[:-1], future_ret)[0, 1])
    clean_corr = float(np.corrcoef(clean_signal[:-1], future_ret)[0, 1])

    # The leaky tree sees tomorrow's value; it should have significantly higher
    # absolute correlation with next-day returns than the clean tree.
    assert abs(leaky_corr) > abs(clean_corr), (
        f"Lookahead detection FAILED: leaky_corr={leaky_corr:.4f} not > "
        f"clean_corr={clean_corr:.4f}. The future-leak primitive should "
        f"produce better correlation with next-period returns."
    )
    # The leaky tree should have high absolute correlation (> 0.5 expected for perfect leak)
    assert abs(leaky_corr) > 0.5, (
        f"Leaky tree correlation with future returns is only {leaky_corr:.4f}; "
        f"expected > 0.5 for a direct future-leak primitive."
    )
