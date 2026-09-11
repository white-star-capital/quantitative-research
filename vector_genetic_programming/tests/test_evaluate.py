"""
evaluate() contract tests — EVAL-01, EVAL-02, EVAL-03, EVAL-04.

Tests use synthetic data: no parquet files, no network access.
The feature matrix is a random [T x F x A] float32 array.
Close prices are a synthetic pd.DataFrame with a DatetimeIndex.

EVAL-01: evaluate() compiles tree, generates signals, calls vectorbt Portfolio.from_signals
EVAL-02: Transaction costs applied inside evaluate() via fees parameter, not post-hoc
EVAL-03: Individuals with < 50 trades receive worst-possible fitness (not NaN, not exception)
EVAL-04: Fitness tuple is (Sharpe, total_return, -tree_size)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_T = 400     # timesteps — enough for meaningful Sharpe calculation
_F = 12      # feature columns (must match FEATURE_NAMES length)
_A = 3       # assets — use a small number to keep tests fast


@pytest.fixture(scope="module")
def pset():
    from vgp.gp.gp_types import build_pset as _build
    return _build()


@pytest.fixture(scope="module")
def feature_matrix():
    """Random [T x F x A] float32 feature matrix."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((_T, _F, _A)).astype(np.float32)


@pytest.fixture(scope="module")
def close_prices():
    """Synthetic [T x A] close-price DataFrame with DatetimeIndex.

    Prices are random walks starting from 100.0.
    Uses daily DatetimeIndex matching the feature matrix length.
    """
    rng = np.random.default_rng(7)
    dates = pd.date_range(start="2024-01-01", periods=_T, freq="D")
    prices = 100.0 * np.exp(np.cumsum(rng.standard_normal((_T, _A)) * 0.01, axis=0))
    return pd.DataFrame(
        prices.astype(np.float64),
        index=dates,
        columns=[f"asset_{i}" for i in range(_A)],
    )


@pytest.fixture(scope="module")
def eval_config(close_prices):
    """Standard EvalConfig with default fees and the synthetic close prices."""
    from vgp.backtest.runner import EvalConfig
    return EvalConfig(
        fee_bps=10.0,
        min_trades=50,
        freq="1D",
        init_cash=10_000.0,
        close_prices=close_prices,
    )


@pytest.fixture(scope="module")
def valid_individual(pset):
    """A randomly generated GP individual likely to produce enough sign changes."""
    from deap import creator, gp
    import random
    random.seed(55)
    expr = gp.genHalfAndHalf(pset, min_=2, max_=4)
    return creator.Individual(expr)


# ---------------------------------------------------------------------------
# EVAL-01: evaluate() interface + import boundary audit (D-15)
# ---------------------------------------------------------------------------

def test_backtest_runner_does_not_import_deap_eval01():
    """vgp.backtest.runner must NOT import deap at module level (EVAL-01, D-15).

    The module is evicted from sys.modules and re-imported so side-effect
    imports are observable, then the ORIGINAL module object is restored.

    Restoring matters. Without it, the re-import leaves a different module
    object in sys.modules while every module that did
    `from vgp.backtest.runner import evaluate` still holds the old function.
    functools.partial(evaluate, ...) then fails to pickle — pickle resolves the
    function by qualified name and checks identity — so any LATER test that
    sends work to a multiprocessing worker dies with

        PicklingError: Can't pickle <function evaluate ...>:
        it's not the same object as vgp.backtest.runner.evaluate

    That is exactly how the real-spawn-pool reproducibility test came to pass
    in isolation and fail in the full suite, which cost a long detour into
    floating-point and RNG hypotheses before the traceback was read.
    """
    import vgp.backtest as backtest_pkg

    saved = sys.modules.get("vgp.backtest.runner")
    saved_attr = getattr(backtest_pkg, "runner", None)
    try:
        if "vgp.backtest.runner" in sys.modules:
            del sys.modules["vgp.backtest.runner"]

        mods_before = set(sys.modules.keys())
        import vgp.backtest.runner  # noqa: F401
        mods_after = set(sys.modules.keys())

        new_deap_mods = [m for m in (mods_after - mods_before) if "deap" in m]
        assert not new_deap_mods, (
            f"vgp.backtest.runner imported deap modules as a side effect: "
            f"{new_deap_mods}. Architecture invariant D-15 violated — "
            f"BacktestRunner must NOT import deap."
        )
    finally:
        # Both bindings must be restored. `import a.b` consults sys.modules but
        # ALSO rebinds the attribute on the parent package, so restoring only
        # sys.modules leaves `vgp.backtest.runner` pointing at the re-imported
        # copy and pickle-by-reference still fails.
        if saved is not None:
            sys.modules["vgp.backtest.runner"] = saved
        if saved_attr is not None:
            backtest_pkg.runner = saved_attr


def test_evaluate_is_picklable_by_reference():
    """functools.partial(evaluate, ...) must survive pickling.

    This is what multiprocessing does on every generation: the toolbox binds
    evaluate into a partial and ships it to workers. It breaks if anything has
    left a different module object in sys.modules, and the failure surfaces far
    from its cause — in whichever test next uses a pool.
    """
    import functools
    import pickle

    import vgp.backtest.runner as runner_mod
    from vgp.backtest.runner import EvalConfig, evaluate

    assert evaluate is runner_mod.evaluate, (
        "the imported evaluate is not the module's current attribute — "
        "something replaced or reloaded vgp.backtest.runner without restoring it"
    )
    pickle.loads(pickle.dumps(functools.partial(evaluate, config=EvalConfig())))


def test_evaluate_returns_tuple_eval01(valid_individual, feature_matrix, eval_config):
    """evaluate() returns a 3-tuple of floats (EVAL-01, EVAL-04)."""
    from vgp.backtest.runner import evaluate

    result = evaluate(valid_individual, feature_matrix, eval_config)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
    for i, val in enumerate(result):
        assert isinstance(val, float), (
            f"Fitness tuple element {i} is {type(val)}, expected float"
        )


# ---------------------------------------------------------------------------
# EVAL-04: Fitness tuple is (Sharpe, total_return, -tree_size)
# ---------------------------------------------------------------------------

def test_negative_tree_size_in_fitness_eval04(valid_individual, feature_matrix, eval_config):
    """Third fitness element must be -len(individual) (EVAL-04)."""
    from vgp.backtest.runner import evaluate

    result = evaluate(valid_individual, feature_matrix, eval_config)
    expected_neg_tree_size = float(-len(valid_individual))

    assert result[2] == expected_neg_tree_size, (
        f"Fitness[2] = {result[2]}, expected -tree_size = {expected_neg_tree_size}"
    )


def test_sharpe_not_nan_valid_portfolio_eval04(valid_individual, feature_matrix, eval_config):
    """A valid portfolio with freq='1D' must return non-NaN Sharpe (EVAL-04, Pitfall 1)."""
    from vgp.backtest.runner import evaluate

    result = evaluate(valid_individual, feature_matrix, eval_config)

    # Result could be worst_fitness if the random individual doesn't trade enough;
    # check if it's a proper fitness result (not -inf)
    if result[0] != -np.inf:
        assert not np.isnan(result[0]), (
            "sharpe_ratio() returned NaN — check that freq='1D' is set in Portfolio.from_signals. "
            "This is the silent failure mode documented in test_smoke.py."
        )
        assert not np.isnan(result[1]), (
            f"total_return() returned NaN for valid individual: {result}"
        )


# ---------------------------------------------------------------------------
# EVAL-02: Transaction costs applied inside evaluate() — not post-hoc
# ---------------------------------------------------------------------------

def test_transaction_costs_applied_inside_eval02(pset, feature_matrix, close_prices):
    """Evaluating with fee_bps=0 vs fee_bps=200 produces different fitness (EVAL-02).

    If transaction costs are applied post-hoc instead of inside evaluate(),
    the fitness calculation would not reflect them during GP tree selection.
    This test confirms that different fee_bps values produce different fitness values.
    """
    from deap import creator, gp
    from vgp.backtest.runner import EvalConfig, evaluate
    import random

    random.seed(999)
    expr = gp.genHalfAndHalf(pset, min_=2, max_=4)
    ind = creator.Individual(expr)

    cfg_zero_fees = EvalConfig(fee_bps=0.0, min_trades=1, freq="1D",
                               init_cash=10_000.0, close_prices=close_prices)
    cfg_high_fees = EvalConfig(fee_bps=200.0, min_trades=1, freq="1D",
                               init_cash=10_000.0, close_prices=close_prices)

    result_zero = evaluate(ind, feature_matrix, cfg_zero_fees)
    result_high = evaluate(ind, feature_matrix, cfg_high_fees)

    # If both return worst fitness (< 1 trade), the test is inconclusive —
    # skip rather than false-fail
    if result_zero[0] == -np.inf or result_high[0] == -np.inf:
        pytest.skip(
            "Individual produced worst fitness — likely < 1 trade for this random seed. "
            "Cannot compare fee effect on worst-fitness individuals."
        )

    assert result_zero[0] != result_high[0], (
        f"fee_bps=0 and fee_bps=200 produced identical Sharpe ({result_zero[0]:.4f}). "
        f"Transaction costs are not being applied inside evaluate(). "
        f"This violates EVAL-02 and CLAUDE.md constraint #3."
    )

    # Higher fees should reduce or equal Sharpe (never improve it)
    assert result_high[0] <= result_zero[0], (
        f"Higher fees produced higher Sharpe: zero_fees={result_zero[0]:.4f}, "
        f"high_fees={result_high[0]:.4f}. This is unexpected — costs should reduce returns."
    )


# ---------------------------------------------------------------------------
# EVAL-03: < 50 trades → worst-possible fitness (not NaN, not exception)
# ---------------------------------------------------------------------------

def test_below_50_trades_returns_worst_fitness_eval03(pset, feature_matrix, close_prices):
    """An individual that produces < min_trades sign changes must return (-inf, -inf, -size) (EVAL-03).

    Forces worst-fitness path by setting min_trades=99999 — impossibly high for any tree.
    """
    from deap import creator, gp
    from vgp.backtest.runner import EvalConfig, evaluate
    import random
    random.seed(2025)

    ind = creator.Individual(gp.genHalfAndHalf(pset, min_=1, max_=3))
    cfg_high_threshold = EvalConfig(
        fee_bps=10.0,
        min_trades=99999,   # impossibly high — forces worst-fitness path
        freq="1D",
        init_cash=10_000.0,
        close_prices=close_prices,
    )
    result = evaluate(ind, feature_matrix, cfg_high_threshold)

    expected_neg_size = float(-len(ind))
    assert result[0] == -np.inf, (
        f"Fitness[0] = {result[0]}, expected -inf for < min_trades individual (EVAL-03)"
    )
    assert result[1] == -np.inf, (
        f"Fitness[1] = {result[1]}, expected -inf for < min_trades individual (EVAL-03)"
    )
    assert result[2] == expected_neg_size, (
        f"Fitness[2] = {result[2]}, expected {expected_neg_size} (-tree_size preserved)"
    )
    # Must be a tuple of Python floats, not NaN — rankable by NSGA-II
    for i, val in enumerate(result):
        assert not np.isnan(val), (
            f"Fitness[{i}] = NaN — worst fitness must be -inf, not NaN (EVAL-03). "
            f"NaN breaks NSGA-II domination checks."
        )


def test_worst_fitness_is_rankable_eval03(pset, feature_matrix, close_prices):
    """Worst-fitness individuals can be compared and ranked — not NaN (EVAL-03).

    NSGA-II domination check requires fitness.values to contain floats that
    support < comparison. -inf < valid_sharpe must be True.
    """
    from deap import creator, gp
    from vgp.backtest.runner import EvalConfig, evaluate
    import random
    random.seed(77)

    # Two different-sized individuals, both hitting worst-fitness path
    expr1 = gp.genFull(pset, min_=1, max_=2)
    ind1 = creator.Individual(expr1)
    expr2 = gp.genFull(pset, min_=2, max_=4)
    ind2 = creator.Individual(expr2)

    cfg = EvalConfig(
        fee_bps=10.0, min_trades=99999, freq="1D",
        init_cash=10_000.0, close_prices=close_prices
    )
    r1 = evaluate(ind1, feature_matrix, cfg)
    r2 = evaluate(ind2, feature_matrix, cfg)

    # Both are worst-fitness; -tree_size should differ if tree sizes differ
    # Third objective differentiates by size (parsimony pressure still active)
    assert r1[0] == -np.inf and r2[0] == -np.inf, "Both should hit worst-fitness path"

    # Verify -inf is comparable (doesn't raise TypeError)
    try:
        _ = r1[0] < 0.0  # -inf < 0.0 must be True
        _ = r1[0] < r2[0]  # -inf < -inf is False (equal — fine)
    except TypeError as exc:
        pytest.fail(f"Worst-fitness values are not comparable: {exc}")


# ---------------------------------------------------------------------------
# EVAL-03: evaluate_with_status() distinguishes the ranking sentinel
# from a real measurement
# ---------------------------------------------------------------------------

def test_evaluate_with_status_flags_below_min_trades(pset, feature_matrix, close_prices):
    """evaluate_with_status must report WHY fitness is the worst-fitness tuple.

    evaluate() has to keep returning -inf so NSGA-II can rank unusable
    individuals. Reporting code needs to tell that sentinel apart from a real
    Sharpe of -inf, which is what evaluate_with_status() is for.
    """
    import random

    from deap import creator, gp

    from vgp.backtest.runner import (
        EVAL_BELOW_MIN_TRADES,
        EvalConfig,
        evaluate,
        evaluate_with_status,
    )
    random.seed(2025)

    ind = creator.Individual(gp.genHalfAndHalf(pset, min_=1, max_=3))
    cfg = EvalConfig(
        fee_bps=10.0, min_trades=99999, freq="1D",
        init_cash=10_000.0, close_prices=close_prices,
    )

    fitness, status, n_trades = evaluate_with_status(ind, feature_matrix, cfg)

    assert status == EVAL_BELOW_MIN_TRADES, (
        f"status = {status!r}, expected {EVAL_BELOW_MIN_TRADES!r}"
    )
    assert fitness[0] == -np.inf, "fitness contract unchanged — still -inf for NSGA-II"
    assert fitness[2] == float(-len(ind)), "-tree_size preserved"
    assert n_trades < 99999, f"observed trade count must be reported, got {n_trades}"

    # evaluate() remains a thin wrapper with the original 3-float contract
    assert evaluate(ind, feature_matrix, cfg) == fitness


def test_evaluate_with_status_reports_ok_for_valid_individual(
    pset, feature_matrix, close_prices
):
    """A measurable individual gets EVAL_OK and a finite Sharpe."""
    import random

    from deap import creator, gp

    from vgp.backtest.runner import EVAL_OK, EvalConfig, evaluate_with_status
    random.seed(11)

    cfg = EvalConfig(
        fee_bps=10.0, min_trades=1, freq="1D",
        init_cash=10_000.0, close_prices=close_prices,
    )

    # Try a handful of random trees until one is measurable on synthetic data
    for _ in range(30):
        ind = creator.Individual(gp.genHalfAndHalf(pset, min_=2, max_=4))
        fitness, status, n_trades = evaluate_with_status(ind, feature_matrix, cfg)
        if status == EVAL_OK:
            assert np.isfinite(fitness[0]), f"EVAL_OK with non-finite Sharpe {fitness[0]}"
            assert np.isfinite(fitness[1]), f"EVAL_OK with non-finite return {fitness[1]}"
            assert n_trades >= 1
            return

    pytest.skip("no randomly generated tree was measurable on this synthetic data")


# ---------------------------------------------------------------------------
# The vectorbt side of the annualization coupling.
#
# vgp/analysis/dsr.py de-annualizes with PERIODS_PER_YEAR_DAILY, which is only
# correct while vectorbt annualizes freq="1D" the same way. Pin it here so a
# change in vectorbt's convention fails loudly rather than silently rescaling
# every DSR by a constant factor.
# ---------------------------------------------------------------------------

def test_vectorbt_daily_sharpe_uses_the_documented_annualization(
    pset, feature_matrix, close_prices
):
    """pf.sharpe_ratio() must equal the per-period Sharpe times sqrt(365)."""
    import random

    from deap import creator, gp

    from vgp.analysis.dsr import PERIODS_PER_YEAR_DAILY
    from vgp.backtest.runner import EVAL_OK, EvalConfig, evaluate_with_status
    random.seed(3)

    cfg = EvalConfig(fee_bps=10.0, min_trades=1, freq="1D",
                     init_cash=10_000.0, close_prices=close_prices)

    for _ in range(40):
        ind = creator.Individual(gp.genHalfAndHalf(pset, min_=2, max_=4))
        fitness, status, _ = evaluate_with_status(ind, feature_matrix, cfg)
        if status != EVAL_OK or not np.isfinite(fitness[0]) or abs(fitness[0]) < 1e-6:
            continue

        # Rebuild the same portfolio's return series via the DSR helper
        from vgp.analysis.runner import _get_is_returns
        rets = _get_is_returns(ind, feature_matrix, cfg)
        sd = float(np.std(rets, ddof=1))
        if sd == 0.0:
            continue
        per_period = float(np.mean(rets)) / sd

        implied = fitness[0] / per_period
        assert implied == pytest.approx(np.sqrt(PERIODS_PER_YEAR_DAILY), rel=1e-6), (
            f"vectorbt annualized with {implied ** 2:.1f} periods/year but "
            f"vgp.analysis.dsr de-annualizes with {PERIODS_PER_YEAR_DAILY}. "
            f"Every DSR would be rescaled by "
            f"{implied / np.sqrt(PERIODS_PER_YEAR_DAILY):.4f}."
        )
        return

    pytest.skip("no randomly generated tree produced a measurable portfolio")
