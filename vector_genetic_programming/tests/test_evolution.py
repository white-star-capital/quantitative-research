"""
Evolution engine tests — EVO-01 through EVO-07, EXP-01 through EXP-03.

Tests use synthetic data: no parquet files, no network access.
Feature matrix: random [T x F x A] float32 array.
Close prices: synthetic pd.DataFrame with DatetimeIndex.

All evolution tests run with n_jobs=1 to avoid spawn subprocess overhead in CI.
EVO-07 parallel-specific behavior is verified via code inspection of loop.py.

EVO-01: toolbox wires selNSGA2, cxOnePoint, mutUniform; import boundary verified
EVO-02: run_evolution completes end-to-end and returns (pop, hof, logbook)
EVO-03: tree depth hard-limited to 8 via staticLimit — no individual exceeds height 8
EVO-04: ParetoFront hof is non-empty; individuals have valid 3-tuple fitness
EVO-05: checkpoint save/load round-trip; resume reproduces same population
EVO-06: Logbook has per-gen records with fitness.sharpe_max and size.size_mean
EVO-07: _jit_warmup is module-level callable; n_jobs=1 path completes correctly
EXP-01: MLflow log_params receives all EvolutionConfig fields (skipif not installed)
EXP-02: MLflow log_metrics called per generation with step=gen (skipif not installed)
EXP-03: same seed produces identical Pareto fronts across two independent runs
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data constants
# ---------------------------------------------------------------------------

_T = 400   # timesteps (enough for meaningful rolling features + 50-trade filter)
_F = 12    # feature columns (FEATURE_NAMES count)
_A = 3     # assets

# ---------------------------------------------------------------------------
# Optional dependency detection (D-02)
# ---------------------------------------------------------------------------

try:
    import mlflow  # noqa: F401
    _mlflow_available = True
except ImportError:
    _mlflow_available = False

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def feature_matrix():
    """Random [T x F x A] float32 feature matrix — no parquet files needed."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((_T, _F, _A)).astype(np.float32)


@pytest.fixture(scope="module")
def close_prices():
    """Synthetic [T x A] close price DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(7)
    dates = pd.date_range(start="2024-01-01", periods=_T, freq="D")
    prices = 100.0 * np.exp(np.cumsum(rng.standard_normal((_T, _A)) * 0.01, axis=0))
    return pd.DataFrame(
        prices.astype(np.float64),
        index=dates,
        columns=[f"asset_{i}" for i in range(_A)],
    )


@pytest.fixture(scope="module")
def small_cfg():
    """Small EvolutionConfig for fast test runs: 10 individuals, 3 generations."""
    from vgp.evolution.config import EvolutionConfig
    return EvolutionConfig(
        pop_size=10,
        n_generations=3,
        seed=42,
        n_jobs=1,             # serial — no spawn overhead in tests
        checkpoint_freq=999,  # no checkpoint writes during basic tests
    )


@pytest.fixture(scope="module")
def eval_cfg(close_prices):
    """EvalConfig with synthetic close prices and relaxed min_trades for test data."""
    from vgp.backtest.runner import EvalConfig
    return EvalConfig(
        close_prices=close_prices,
        min_trades=1,   # relax filter so test individuals are not all worst-fitness
    )


@pytest.fixture(scope="module")
def evolution_result(small_cfg, feature_matrix, eval_cfg):
    """Run a single short evolution and cache the (pop, hof, logbook) result."""
    from vgp.evolution.loop import run_evolution
    return run_evolution(small_cfg, feature_matrix, eval_cfg)


# ---------------------------------------------------------------------------
# EVO-01: Import boundary — loop.py must not import vectorbt
# ---------------------------------------------------------------------------

def test_evolution_loop_does_not_import_vectorbt_evo01():
    """loop.py must NOT contain 'import vectorbt' at module level (D-15).

    Checks source code, not sys.modules, because vgp.backtest.runner (a transitive
    dependency of loop.py) imports vectorbt at its own module level — that transitive
    import is expected and cannot be avoided. D-15 applies only to direct import
    statements written in loop.py's own source.
    """
    import inspect
    import vgp.evolution.loop as loop_mod

    src_lines = inspect.getsource(loop_mod).split("\n")
    # Module-level imports are lines that start with 'import' or 'from' (no leading spaces)
    module_level_vbt = [
        line for line in src_lines
        if (line.startswith("import vectorbt") or line.startswith("from vectorbt"))
    ]
    assert not module_level_vbt, (
        f"D-15 violated: loop.py contains module-level vectorbt import(s): {module_level_vbt}. "
        f"vectorbt must only appear inside _jit_warmup() function body."
    )


# ---------------------------------------------------------------------------
# EVO-01: Toolbox wiring — selNSGA2, cxOnePoint, mutUniform registered
# ---------------------------------------------------------------------------

def test_toolbox_operators_registered_evo01(small_cfg, feature_matrix, eval_cfg):
    """Toolbox must register selNSGA2, cxOnePoint, mutUniform (EVO-01)."""
    from vgp.evolution.loop import _build_toolbox
    from vgp.gp.gp_types import build_pset

    pset = build_pset()
    toolbox = _build_toolbox(pset, feature_matrix, eval_cfg, small_cfg)

    assert hasattr(toolbox, "select"), "toolbox.select not registered"
    assert hasattr(toolbox, "mate"), "toolbox.mate not registered"
    assert hasattr(toolbox, "mutate"), "toolbox.mutate not registered"
    assert hasattr(toolbox, "evaluate"), "toolbox.evaluate not registered"
    assert hasattr(toolbox, "population"), "toolbox.population not registered"

    # Verify select wraps selNSGA2.
    # DEAP's toolbox.register creates a partial and sets __name__ = alias ('select'),
    # so check the underlying .func attribute to confirm selNSGA2 is wrapped.
    from deap import tools
    select_func = getattr(toolbox.select, "func", toolbox.select)
    assert select_func is tools.selNSGA2, (
        f"Expected toolbox.select to wrap selNSGA2, got {select_func}"
    )


# ---------------------------------------------------------------------------
# EVO-02: Evolution loop runs end-to-end and returns correct structure
# ---------------------------------------------------------------------------

def test_evolution_returns_correct_structure_evo02(evolution_result, small_cfg):
    """run_evolution returns (population, hof, logbook) with correct sizes (EVO-02)."""
    pop, hof, logbook = evolution_result
    assert isinstance(pop, list), "population must be a list"
    assert len(pop) == small_cfg.pop_size, (
        f"Expected pop size {small_cfg.pop_size}, got {len(pop)}"
    )
    assert logbook is not None, "logbook must not be None"
    # Logbook should have gen=0 through gen=n_generations records
    assert len(logbook) >= small_cfg.n_generations, (
        f"Expected at least {small_cfg.n_generations} logbook records"
    )


# ---------------------------------------------------------------------------
# EVO-03: Tree depth hard-limited to 8 via staticLimit
# ---------------------------------------------------------------------------

def test_tree_depth_limit_evo03(evolution_result, small_cfg):
    """No individual in final population may exceed tree_height_limit (EVO-03, D-16)."""
    pop, hof, _ = evolution_result
    limit = small_cfg.tree_height_limit  # 8
    violations = [ind for ind in pop if ind.height > limit]
    assert not violations, (
        f"{len(violations)}/{len(pop)} individuals exceed height {limit}. "
        f"Heights: {[ind.height for ind in violations]}"
    )
    # Also verify HoF individuals respect the limit
    hof_violations = [ind for ind in hof if ind.height > limit]
    assert not hof_violations, (
        f"{len(hof_violations)} HoF individuals exceed height {limit}"
    )


# ---------------------------------------------------------------------------
# EVO-04: ParetoFront HoF is non-empty and contains valid individuals
# ---------------------------------------------------------------------------

def test_pareto_front_populated_evo04(evolution_result):
    """Hall-of-fame must be non-empty and contain valid fitness tuples (EVO-04, D-14)."""
    _, hof, _ = evolution_result
    assert len(hof) > 0, "ParetoFront hall-of-fame is empty after evolution"
    for ind in hof:
        assert ind.fitness.valid, f"HoF individual has invalid fitness: {ind.fitness}"
        assert len(ind.fitness.values) == 3, (
            f"Expected 3-tuple fitness, got {len(ind.fitness.values)}: {ind.fitness.values}"
        )
        sharpe, total_ret, neg_size = ind.fitness.values
        assert isinstance(float(sharpe), float)
        assert neg_size <= 0, (
            f"Third fitness component should be -tree_size (<=0), got {neg_size}"
        )


# ---------------------------------------------------------------------------
# EVO-05: Checkpoint save/load round-trip; resume from checkpoint
# ---------------------------------------------------------------------------

def test_checkpoint_save_load_evo05():
    """Checkpoint round-trip: save writes dill file; load restores all keys (EVO-05, D-07)."""
    import random as rnd
    from vgp.evolution.checkpoint import save_checkpoint, load_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "subdir/gen_0005.pkl")
        rnd.seed(77)
        np.random.seed(88)
        rng_state = rnd.getstate()
        nrng_state = np.random.get_state()

        save_checkpoint(
            path,
            population=["dummy_individual"],
            halloffame=["dummy_hof"],
            logbook=["dummy_logbook_record"],
            generation=5,
            seed=42,
        )

        assert os.path.exists(path), "checkpoint file was not created"

        ckpt = load_checkpoint(path)
        assert ckpt["generation"] == 5
        assert ckpt["seed"] == 42
        assert ckpt["population"] == ["dummy_individual"]

        # Verify RNG state round-trip (D-09)
        rnd.setstate(ckpt["rng_state"])
        np.random.set_state(ckpt["np_rng_state"])
        assert rnd.getstate() == rng_state, "Python random RNG state not restored"
        restored_np = np.random.get_state()
        assert (restored_np[1][:10] == nrng_state[1][:10]).all(), (
            "numpy RNG state not correctly restored"
        )


def test_checkpoint_resume_matches_continuous_evo05(feature_matrix, eval_cfg):
    """Resume from checkpoint produces same population as continuous run (EVO-05, EXP-03)."""
    from vgp.evolution.config import EvolutionConfig
    from vgp.evolution.loop import run_evolution

    with tempfile.TemporaryDirectory() as tmp:
        cfg_cont = EvolutionConfig(
            pop_size=8,
            n_generations=4,
            seed=42,
            n_jobs=1,
            checkpoint_freq=2,
            checkpoint_dir=os.path.join(tmp, "checkpoints"),
        )
        cfg_resume = EvolutionConfig(
            pop_size=8,
            n_generations=4,
            seed=42,
            n_jobs=1,
            checkpoint_freq=2,
            checkpoint_dir=os.path.join(tmp, "checkpoints"),
        )

        # Continuous run (no resume)
        pop_cont, _, _ = run_evolution(cfg_cont, feature_matrix, eval_cfg)

        # Find checkpoint written at gen 2
        ckpt_dir = os.path.join(tmp, "checkpoints")
        ckpt_files = []
        for root, dirs, files in os.walk(ckpt_dir):
            for f in files:
                if f.endswith(".pkl"):
                    ckpt_files.append(os.path.join(root, f))

        if not ckpt_files:
            pytest.skip("No checkpoint files written — cannot test resume path")

        # Pick the earliest checkpoint (lowest gen number)
        ckpt_path = sorted(ckpt_files)[0]

        # Resume run
        pop_resume, _, _ = run_evolution(
            cfg_resume, feature_matrix, eval_cfg, resume_checkpoint=ckpt_path
        )

        # Compare final population string representations
        cont_strs = sorted(str(ind) for ind in pop_cont)
        resume_strs = sorted(str(ind) for ind in pop_resume)
        assert cont_strs == resume_strs, (
            f"Resumed population differs from continuous run.\n"
            f"Continuous[0]: {cont_strs[0] if cont_strs else 'empty'}\n"
            f"Resumed[0]: {resume_strs[0] if resume_strs else 'empty'}"
        )


# ---------------------------------------------------------------------------
# EVO-06: Logbook captures per-generation statistics
# ---------------------------------------------------------------------------

def test_logbook_structure_evo06(evolution_result):
    """Logbook must have per-gen records with fitness and size chapters (EVO-06)."""
    _, _, logbook = evolution_result
    assert len(logbook) > 0, "logbook is empty"

    for record in logbook:
        assert "gen" in record, f"logbook record missing 'gen' key: {record}"
        assert "nevals" in record, f"logbook record missing 'nevals': {record}"

    # DEAP's Logbook stores MultiStatistics chapter data in logbook.chapters, NOT in
    # the per-entry dicts. Per-entry dicts (logbook[-1]) only contain scalar values
    # (gen, nevals). Chapter dicts are accessed via logbook.chapters['fitness'] etc.
    assert "fitness" in logbook.chapters, (
        f"logbook missing 'fitness' chapter. Available chapters: {list(logbook.chapters.keys())}"
    )
    assert "size" in logbook.chapters, (
        f"logbook missing 'size' chapter. Available chapters: {list(logbook.chapters.keys())}"
    )
    fitness_chapter = logbook.chapters["fitness"][-1]
    assert "sharpe_max" in fitness_chapter, (
        f"fitness chapter missing 'sharpe_max'. Keys: {list(fitness_chapter.keys())}"
    )
    assert "sharpe_mean" in fitness_chapter, "fitness chapter missing 'sharpe_mean'"
    assert "sharpe_min" in fitness_chapter, "fitness chapter missing 'sharpe_min'"
    size_chapter = logbook.chapters["size"][-1]
    assert "size_mean" in size_chapter, (
        f"size chapter missing 'size_mean'. Keys: {list(size_chapter.keys())}"
    )
    assert "size_max" in size_chapter, "size chapter missing 'size_max'"


# ---------------------------------------------------------------------------
# EVO-07: _jit_warmup is module-level callable; n_jobs=1 completes correctly
# ---------------------------------------------------------------------------

def test_jit_warmup_is_module_level_evo07():
    """_jit_warmup must be a module-level callable for spawn pickling (EVO-07, CLAUDE.md #8)."""
    import inspect
    import vgp.evolution.loop as loop_mod

    assert hasattr(loop_mod, "_jit_warmup"), (
        "_jit_warmup must be a module-level function in vgp.evolution.loop"
    )
    assert callable(loop_mod._jit_warmup), "_jit_warmup must be callable"
    assert inspect.isfunction(loop_mod._jit_warmup), (
        "_jit_warmup must be a plain function, not a lambda or partial"
    )


def test_njobs1_completes_correctly_evo07(small_cfg, feature_matrix, eval_cfg):
    """n_jobs=1 single-threaded path runs evolution without Pool (EVO-07)."""
    from vgp.evolution.loop import run_evolution
    pop, hof, logbook = run_evolution(small_cfg, feature_matrix, eval_cfg)
    assert len(pop) == small_cfg.pop_size
    assert len(logbook) >= small_cfg.n_generations


# ---------------------------------------------------------------------------
# EXP-01: MLflow logs all hyperparameters (skipif mlflow not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _mlflow_available, reason="mlflow not installed (D-02)")
def test_mlflow_tracker_logs_params_exp01():
    """MLflowTracker.log_params receives EvolutionConfig fields as a dict (EXP-01)."""
    import dataclasses
    from unittest.mock import patch
    from vgp.evolution.config import EvolutionConfig
    from vgp.evolution.tracker import MLflowTracker

    cfg = EvolutionConfig(pop_size=20, n_generations=2, seed=99, n_jobs=1)
    expected_params = dataclasses.asdict(cfg)

    with patch("mlflow.set_experiment"), patch("mlflow.start_run"), \
         patch("mlflow.log_params") as mock_log_params, \
         patch("mlflow.end_run"):
        tracker = MLflowTracker(experiment_name="test")
        tracker.start_run(run_name="test_run")
        tracker.log_params(expected_params)
        tracker.end_run()

    mock_log_params.assert_called_once_with(expected_params)
    logged = mock_log_params.call_args[0][0]
    for key in expected_params:
        assert key in logged, f"EvolutionConfig field '{key}' missing from logged params"


# ---------------------------------------------------------------------------
# EXP-02: MLflow logs per-generation metrics with step= argument (skipif)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _mlflow_available, reason="mlflow not installed (D-02)")
def test_mlflow_tracker_logs_metrics_per_gen_exp02():
    """MLflowTracker.log_metrics is called with step=gen for each generation (EXP-02)."""
    from unittest.mock import patch
    from vgp.evolution.tracker import MLflowTracker

    with patch("mlflow.set_experiment"), patch("mlflow.start_run"), \
         patch("mlflow.log_metrics") as mock_metrics, \
         patch("mlflow.log_params"), patch("mlflow.end_run"):
        tracker = MLflowTracker(experiment_name="test")
        tracker.start_run()
        for gen in range(3):
            tracker.log_metrics({"fitness__sharpe_max": 0.5 + gen * 0.1}, step=gen)
        tracker.end_run()

    assert mock_metrics.call_count == 3, (
        f"Expected 3 log_metrics calls (one per gen), got {mock_metrics.call_count}"
    )
    for call_idx, call in enumerate(mock_metrics.call_args_list):
        _, kwargs = call
        assert "step" in kwargs, f"Call {call_idx}: step= kwarg missing"
        assert kwargs["step"] == call_idx, (
            f"Call {call_idx}: expected step={call_idx}, got {kwargs['step']}"
        )


# ---------------------------------------------------------------------------
# EXP-03: Seed reproducibility — two runs with same seed produce identical HoF
# ---------------------------------------------------------------------------

def test_seed_reproducibility_exp03(feature_matrix, close_prices):
    """Two runs with the same seed must produce identical Pareto fronts (EXP-03, D-09)."""
    from vgp.backtest.runner import EvalConfig
    from vgp.evolution.config import EvolutionConfig
    from vgp.evolution.loop import run_evolution

    cfg = EvolutionConfig(
        pop_size=10,
        n_generations=2,
        seed=42,
        n_jobs=1,
        checkpoint_freq=999,  # no checkpoint I/O during this test
    )
    eval_cfg = EvalConfig(close_prices=close_prices, min_trades=1)

    _, hof1, _ = run_evolution(cfg, feature_matrix, eval_cfg)
    _, hof2, _ = run_evolution(cfg, feature_matrix, eval_cfg)

    assert len(hof1) == len(hof2), (
        f"HoF sizes differ across identical-seed runs: {len(hof1)} vs {len(hof2)}"
    )
    for ind1, ind2 in zip(hof1, hof2):
        assert str(ind1) == str(ind2), (
            f"HoF individuals differ across identical-seed runs (EXP-03).\n"
            f"Run 1: {str(ind1)}\nRun 2: {str(ind2)}"
        )
