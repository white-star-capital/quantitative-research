"""
Walk-forward validation tests — VAL-01 through VAL-04.

Tests use synthetic data only: no parquet files, no network access.
Feature matrix: random [T x F x A] float32 array.
Close prices: synthetic pd.DataFrame with DatetimeIndex.

VAL-01: generate_windows() produces exactly 4 non-overlapping OOS windows
        with correct first-window dates
VAL-02: WalkForwardRunner.run_window() never passes test data to run_evolution()
        (structural OOS invariant, not advisory)
VAL-03: run_window() iterates over list of seeds; returns one dict per seed
VAL-04: compute_dsr() returns float in [0.0, 1.0]; returns 0.0 for flat returns;
        aggregate_seeds() correctly counts positive-OOS seeds

All evolution tests (VAL-02, VAL-03) use unittest.mock.patch so no actual
DEAP/vectorbt runs occur. VAL-04 uses compute_dsr() directly with numpy arrays.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data constants
# ---------------------------------------------------------------------------

_T = 600   # timesteps — covers 2024-01-01 to ~2025-08-22, enough for window splits
_F = 12    # feature columns (FEATURE_NAMES count)
_A = 3     # assets

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def feature_matrix():
    """Random [T x F x A] float32 feature matrix — no parquet files needed."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((_T, _F, _A)).astype(np.float32)


@pytest.fixture(scope="module")
def dates():
    """DatetimeIndex from '2024-01-01' with _T daily periods."""
    return pd.date_range(start="2024-01-01", periods=_T, freq="D")


@pytest.fixture(scope="module")
def close_prices(dates):
    """Synthetic [T x A] close price DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(7)
    prices = 100.0 * np.exp(
        np.cumsum(rng.standard_normal((_T, _A)) * 0.01, axis=0)
    )
    return pd.DataFrame(
        prices.astype(np.float64),
        index=dates,
        columns=[f"asset_{i}" for i in range(_A)],
    )


@pytest.fixture(scope="module")
def eval_cfg(close_prices):
    """EvalConfig template with relaxed min_trades for test data."""
    from vgp.backtest.runner import EvalConfig
    return EvalConfig(
        close_prices=close_prices,
        min_trades=1,  # relax so mock individuals are not all worst-fitness
    )


@pytest.fixture(scope="module")
def base_evo_kwargs():
    """Small EvolutionConfig kwargs for fast test runs."""
    return dict(
        pop_size=10,
        n_generations=3,
        n_jobs=1,
        checkpoint_freq=999,  # no checkpoint writes during tests
    )


# ---------------------------------------------------------------------------
# Mock helper: build a mock HOF, logbook, and individual for VAL-02/VAL-03
# ---------------------------------------------------------------------------

def _make_mock_evolution_return():
    """Return (pop=[], mock_hof, mock_logbook) for patching run_evolution().

    mock_hof:
      - bool(hof) is True (non-empty)
      - len(hof) == 1
      - hof[0] returns mock_individual

    mock_individual:
      - len(individual) == 5 (n_nodes_best)
      - individual.fitness.values == (0.5, 0.1, -5.0)

    mock_logbook:
      - logbook.chapters == {'fitness': [{'sharpe_max': 0.5}]}
    """
    mock_individual = MagicMock()
    mock_individual.__len__ = lambda s: 5
    mock_individual.fitness = MagicMock()
    mock_individual.fitness.values = (0.5, 0.1, -5.0)

    mock_hof = MagicMock()
    mock_hof.__bool__ = lambda s: True
    mock_hof.__len__ = lambda s: 1
    mock_hof.__getitem__ = lambda s, i: mock_individual

    mock_logbook = MagicMock()
    mock_logbook.chapters = {"fitness": [{"sharpe_max": 0.5}]}

    return [], mock_hof, mock_logbook


# ---------------------------------------------------------------------------
# VAL-01: generate_windows() produces correct count and non-overlapping OOS
# ---------------------------------------------------------------------------


def test_generate_windows_count():
    """VAL-01: generate_windows('2024-01-01', '2026-04-01') returns exactly 4 windows."""
    from vgp.analysis import generate_windows

    windows = generate_windows("2024-01-01", "2026-04-01")
    assert len(windows) == 4, (
        f"Expected 4 windows with default params over 2024-01-01 to 2026-04-01, "
        f"got {len(windows)}"
    )


def test_generate_windows_non_overlapping():
    """VAL-01: consecutive windows have non-overlapping OOS periods."""
    from vgp.analysis import generate_windows

    windows = generate_windows("2024-01-01", "2026-04-01")
    for i in range(len(windows) - 1):
        end_i = pd.Timestamp(windows[i].test_end)
        start_next = pd.Timestamp(windows[i + 1].test_start)
        assert start_next > end_i, (
            f"Window {i} OOS ends {end_i.date()} but window {i+1} OOS starts "
            f"{start_next.date()} — windows overlap"
        )


def test_generate_windows_dates():
    """VAL-01: first window has correct dates matching 12-month train / 2-month val / 3-month OOS."""
    from vgp.analysis import generate_windows, WindowSpec

    windows = generate_windows("2024-01-01", "2026-04-01")
    w0 = windows[0]

    assert w0.window_id == 0
    assert w0.train_end == "2024-12-31", (
        f"Expected train_end='2024-12-31', got '{w0.train_end}'"
    )
    assert w0.test_start == "2025-03-01", (
        f"Expected test_start='2025-03-01', got '{w0.test_start}'"
    )
    assert w0.test_end == "2025-05-31", (
        f"Expected test_end='2025-05-31', got '{w0.test_end}'"
    )


# ---------------------------------------------------------------------------
# VAL-02: run_window() never passes test data to run_evolution()
# ---------------------------------------------------------------------------


def test_runner_oos_not_passed_to_evolution(feature_matrix, close_prices, dates, eval_cfg, base_evo_kwargs):
    """VAL-02: structural OOS invariant — feature_matrix arg to run_evolution must be
    train_fm (shape T_train), never the full feature_matrix (shape T_full)."""
    from vgp.analysis import generate_windows
    from vgp.analysis.runner import WalkForwardRunner

    runner = WalkForwardRunner(dates=dates)
    windows = generate_windows("2024-01-01", "2026-04-01")
    window = windows[0]  # train_end=2024-12-31

    full_T = feature_matrix.shape[0]  # 600

    mock_return = _make_mock_evolution_return()

    with patch("vgp.analysis.runner.run_evolution", return_value=mock_return) as mock_run_evo, \
         patch("vgp.analysis.runner.evaluate", return_value=(0.3, 0.05, -5.0)), \
         patch("vgp.analysis.runner._get_is_returns", return_value=np.random.default_rng(0).standard_normal(250)):

        runner.run_window(
            window=window,
            feature_matrix=feature_matrix,
            close_prices=close_prices,
            base_eval_config=eval_cfg,
            seeds=[42],
            evo_config_kwargs=base_evo_kwargs,
        )

    assert mock_run_evo.called, "run_evolution was never called"

    for call in mock_run_evo.call_args_list:
        # feature_matrix is positional arg 1 (index 1: config=0, feature_matrix=1, eval_config=2)
        if len(call.args) >= 2:
            fm_arg = call.args[1]
        else:
            fm_arg = call.kwargs.get("feature_matrix")

        assert fm_arg is not None, "feature_matrix arg not found in run_evolution call"
        assert fm_arg.shape[0] < full_T, (
            f"run_evolution received feature_matrix with {fm_arg.shape[0]} rows "
            f"(== full_T={full_T}). Test data must NOT be passed to run_evolution. "
            f"Only train_fm (shape T_train < T_full) should be passed."
        )


# ---------------------------------------------------------------------------
# VAL-03: run_window() iterates over all seeds, returns one dict per seed
# ---------------------------------------------------------------------------


def test_runner_iterates_seeds(feature_matrix, close_prices, dates, eval_cfg, base_evo_kwargs):
    """VAL-03: seeds=[0, 1, 2] -> 3 result dicts with seed values 0, 1, 2."""
    from vgp.analysis import generate_windows
    from vgp.analysis.runner import WalkForwardRunner

    runner = WalkForwardRunner(dates=dates)
    windows = generate_windows("2024-01-01", "2026-04-01")
    window = windows[0]

    mock_return = _make_mock_evolution_return()
    seeds = [0, 1, 2]

    with patch("vgp.analysis.runner.run_evolution", return_value=mock_return), \
         patch("vgp.analysis.runner.evaluate", return_value=(0.3, 0.05, -5.0)), \
         patch("vgp.analysis.runner._get_is_returns", return_value=np.random.default_rng(0).standard_normal(250)):

        results = runner.run_window(
            window=window,
            feature_matrix=feature_matrix,
            close_prices=close_prices,
            base_eval_config=eval_cfg,
            seeds=seeds,
            evo_config_kwargs=base_evo_kwargs,
        )

    assert len(results) == 3, (
        f"Expected 3 result dicts (one per seed), got {len(results)}"
    )

    result_seeds = [r["seed"] for r in results]
    assert result_seeds == seeds, (
        f"Expected seed values {seeds}, got {result_seeds}"
    )

    # Verify each result dict has all required keys
    required_keys = {
        "window_id", "seed", "train_end", "test_start", "test_end",
        "is_sharpe", "oos_sharpe", "dsr", "n_nodes_best",
    }
    for r in results:
        missing = required_keys - set(r.keys())
        assert not missing, f"Result dict missing keys: {missing}"


# ---------------------------------------------------------------------------
# VAL-04: compute_dsr() returns float in [0, 1]; flat returns guard; aggregate_seeds
# ---------------------------------------------------------------------------


def test_compute_dsr_returns_float_in_range():
    """VAL-04: compute_dsr with standard-normal returns returns float in [0.0, 1.0]."""
    from vgp.analysis import compute_dsr

    rng = np.random.default_rng(0)
    returns = rng.standard_normal(252)
    result = compute_dsr(returns, sr_hat=1.0, n_trials=10)

    assert isinstance(result, float), (
        f"compute_dsr must return float, got {type(result)}"
    )
    assert 0.0 <= result <= 1.0, (
        f"compute_dsr must return value in [0.0, 1.0], got {result}"
    )


def test_compute_dsr_flat_returns_zero():
    """VAL-04: compute_dsr with all-zero returns must return 0.0 (flat portfolio guard)."""
    from vgp.analysis import compute_dsr

    result = compute_dsr(np.zeros(100), sr_hat=0.0, n_trials=1)
    assert result == 0.0, (
        f"compute_dsr with flat returns (std=0) must return 0.0, got {result}"
    )


def test_aggregate_seeds_positive_count():
    """VAL-04: aggregate_seeds correctly counts seeds with positive OOS Sharpe."""
    from vgp.analysis import aggregate_seeds

    seed_results = [
        {"oos_sharpe": 1.0, "dsr": 0.9},
        {"oos_sharpe": -0.5, "dsr": 0.3},
    ]
    agg = aggregate_seeds(seed_results)

    assert "n_seeds_positive_oos" in agg, "aggregate_seeds must return n_seeds_positive_oos"
    assert agg["n_seeds_positive_oos"] == 1, (
        f"Expected 1 seed with positive OOS Sharpe, got {agg['n_seeds_positive_oos']}"
    )
    assert "median_oos_sharpe" in agg
    assert "iqr_oos_sharpe" in agg
    assert "median_dsr" in agg
    # Median of [1.0, -0.5] = 0.25
    assert abs(agg["median_oos_sharpe"] - 0.25) < 1e-9, (
        f"Expected median_oos_sharpe=0.25, got {agg['median_oos_sharpe']}"
    )
