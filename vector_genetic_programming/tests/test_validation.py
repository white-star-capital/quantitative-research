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

_T = 600  # timesteps — covers 2024-01-01 to ~2025-08-22, enough for window splits
_F = 12  # feature columns (FEATURE_NAMES count)
_A = 3  # assets

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
    prices = 100.0 * np.exp(np.cumsum(rng.standard_normal((_T, _A)) * 0.01, axis=0))
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
    """VAL-01: first window dates match 12-month train / 2-month val / 3-month OOS."""
    from vgp.analysis import generate_windows

    windows = generate_windows("2024-01-01", "2026-04-01")
    w0 = windows[0]

    assert w0.window_id == 0
    assert w0.train_end == "2024-12-31", f"Expected train_end='2024-12-31', got '{w0.train_end}'"
    assert w0.test_start == "2025-03-01", f"Expected test_start='2025-03-01', got '{w0.test_start}'"
    assert w0.test_end == "2025-05-31", f"Expected test_end='2025-05-31', got '{w0.test_end}'"


# ---------------------------------------------------------------------------
# VAL-02: run_window() never passes test data to run_evolution()
# ---------------------------------------------------------------------------


def test_runner_oos_not_passed_to_evolution(
    feature_matrix, close_prices, dates, eval_cfg, base_evo_kwargs
):
    """VAL-02: structural OOS invariant — feature_matrix arg to run_evolution must be
    train_fm (shape T_train), never the full feature_matrix (shape T_full)."""
    from vgp.analysis import generate_windows
    from vgp.analysis.runner import WalkForwardRunner

    runner = WalkForwardRunner(dates=dates)
    windows = generate_windows("2024-01-01", "2026-04-01")
    window = windows[0]  # train_end=2024-12-31

    full_T = feature_matrix.shape[0]  # 600

    mock_return = _make_mock_evolution_return()

    with (
        patch("vgp.analysis.runner.run_evolution", return_value=mock_return) as mock_run_evo,
        patch(
            "vgp.analysis.runner.evaluate_with_status", return_value=((0.3, 0.05, -5.0), "ok", 120)
        ),
        patch(
            "vgp.analysis.runner._get_is_returns",
            return_value=np.random.default_rng(0).standard_normal(250),
        ),
    ):

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

    with (
        patch("vgp.analysis.runner.run_evolution", return_value=mock_return),
        patch(
            "vgp.analysis.runner.evaluate_with_status", return_value=((0.3, 0.05, -5.0), "ok", 120)
        ),
        patch(
            "vgp.analysis.runner._get_is_returns",
            return_value=np.random.default_rng(0).standard_normal(250),
        ),
    ):

        results = runner.run_window(
            window=window,
            feature_matrix=feature_matrix,
            close_prices=close_prices,
            base_eval_config=eval_cfg,
            seeds=seeds,
            evo_config_kwargs=base_evo_kwargs,
        )

    assert len(results) == 3, f"Expected 3 result dicts (one per seed), got {len(results)}"

    result_seeds = [r["seed"] for r in results]
    assert result_seeds == seeds, f"Expected seed values {seeds}, got {result_seeds}"

    # Verify each result dict has all required keys
    required_keys = {
        "window_id",
        "seed",
        "train_end",
        "test_start",
        "test_end",
        "is_sharpe",
        "oos_sharpe",
        "oos_status",
        "oos_n_trades",
        "oos_min_trades",
        "dsr",
        "n_nodes_best",
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
    result = compute_dsr(
        returns, sr_hat=1.0, trial_sharpes=[0.4, 1.0, 0.7, 1.3, 0.2, 0.9, 1.1, 0.5, 0.8, 0.6]
    )

    assert isinstance(result, float), f"compute_dsr must return float, got {type(result)}"
    assert 0.0 <= result <= 1.0, f"compute_dsr must return value in [0.0, 1.0], got {result}"


def test_compute_dsr_flat_returns_zero():
    """VAL-04: compute_dsr with all-zero returns must return 0.0 (flat portfolio guard)."""
    from vgp.analysis import compute_dsr

    result = compute_dsr(np.zeros(100), sr_hat=0.0, trial_sharpes=[0.0, 1.0, 2.0])
    assert result == 0.0, f"compute_dsr with flat returns (std=0) must return 0.0, got {result}"


def test_aggregate_seeds_positive_count():
    """VAL-04: aggregate_seeds correctly counts seeds with positive OOS Sharpe."""
    from vgp.analysis import aggregate_seeds

    seed_results = [
        {"oos_sharpe": 1.0, "dsr": 0.9},
        {"oos_sharpe": -0.5, "dsr": 0.3},
    ]
    agg = aggregate_seeds(seed_results)

    assert "n_seeds_positive_oos" in agg, "aggregate_seeds must return n_seeds_positive_oos"
    assert (
        agg["n_seeds_positive_oos"] == 1
    ), f"Expected 1 seed with positive OOS Sharpe, got {agg['n_seeds_positive_oos']}"
    assert "median_oos_sharpe" in agg
    assert "iqr_oos_sharpe" in agg
    assert "median_dsr" in agg
    # Median of [1.0, -0.5] = 0.25
    assert (
        abs(agg["median_oos_sharpe"] - 0.25) < 1e-9
    ), f"Expected median_oos_sharpe=0.25, got {agg['median_oos_sharpe']}"


# ---------------------------------------------------------------------------
# VAL-04 (regression): E[SR_max] must be scaled by sigma_SR
#
# The bracket of Bailey & Lopez de Prado Proposition 3 is a dimensionless
# multiplier on the cross-sectional std of the trial Sharpes. Comparing SR_hat
# to the bare bracket puts the hurdle at ~1.52 in per-period units (~24
# annualized), which nothing clears — every DSR collapses to ~0 regardless of
# the strategy. These tests pin the scaling so that failure mode cannot return.
# ---------------------------------------------------------------------------


def _synthetic_returns(ann_sharpe: float, T: int = 300, seed: int = 0) -> np.ndarray:
    """Daily returns whose realized annualized Sharpe equals ann_sharpe."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(T) * 0.01
    target_pp = ann_sharpe / np.sqrt(252)
    return r - r.mean() + target_pp * r.std()


def test_compute_dsr_not_degenerate_for_realistic_sharpe():
    """VAL-04 regression: a strong IS Sharpe over clustered trials must not give DSR ~ 0.

    These are the actual IS Sharpe ratios from a 3-seed x 3-window run. Under
    the unscaled formula every one of them produced DSR < 1e-170.
    """
    from vgp.analysis import compute_dsr

    trials = [4.045, 3.744, 4.063, 3.312, 3.686, 3.629, 3.222, 3.113, 3.254]
    dsr = compute_dsr(_synthetic_returns(4.045), sr_hat=4.045, trial_sharpes=trials)

    assert np.isfinite(dsr), f"DSR must be finite for a valid trial set, got {dsr}"
    assert dsr > 0.5, (
        f"DSR = {dsr:.3e} for IS Sharpe 4.045 against trials with annualized "
        f"spread {np.std(trials, ddof=1):.3f}. A value near zero means E[SR_max] "
        f"is not being scaled by sigma_SR — the hurdle has become ~24 annualized."
    )


def test_compute_dsr_decreases_with_trial_spread():
    """VAL-04: a wider spread of trial Sharpes raises the hurdle, lowering DSR."""
    from vgp.analysis import compute_dsr

    returns = _synthetic_returns(4.0)
    tight = [3.8, 4.0, 4.2, 3.9, 4.1, 4.05, 3.95, 4.15, 3.85]
    wide = [0.5, 8.0, 1.5, 7.0, 2.5, 6.5, 3.5, 7.5, 4.0]

    dsr_tight = compute_dsr(returns, sr_hat=4.0, trial_sharpes=tight)
    dsr_wide = compute_dsr(returns, sr_hat=4.0, trial_sharpes=wide)

    assert dsr_tight > dsr_wide, (
        f"DSR must fall as the trial Sharpe spread grows (more scope for "
        f"selection bias): tight={dsr_tight:.4f}, wide={dsr_wide:.4f}"
    )


def test_compute_dsr_requires_two_finite_trials():
    """VAL-04: sigma_SR is undefined below 2 trials — DSR is NaN, not 0.0.

    NaN means "not computable"; 0.0 would assert "no skill", which is a
    different claim and would be read as a result.
    """
    from vgp.analysis import compute_dsr

    returns = _synthetic_returns(2.0)
    assert np.isnan(compute_dsr(returns, sr_hat=2.0, trial_sharpes=[2.0]))
    assert np.isnan(compute_dsr(returns, sr_hat=2.0, trial_sharpes=[]))


def test_compute_dsr_ignores_worst_fitness_trials():
    """VAL-04: -inf trial Sharpes are sentinels, not data — they must be dropped."""
    from vgp.analysis import compute_dsr

    returns = _synthetic_returns(4.0)
    clean = [3.8, 4.0, 4.2, 3.9, 4.1]
    with_sentinels = clean + [-np.inf, -np.inf, np.nan]

    dsr_clean = compute_dsr(returns, sr_hat=4.0, trial_sharpes=clean)
    dsr_dirty = compute_dsr(returns, sr_hat=4.0, trial_sharpes=with_sentinels)

    assert dsr_clean == pytest.approx(dsr_dirty), (
        f"Sentinel trial values changed the result: clean={dsr_clean}, "
        f"with sentinels={dsr_dirty}. -inf would blow up sigma_SR."
    )


def test_compute_dsr_nonfinite_sr_hat_is_nan():
    """VAL-04: a worst-fitness IS Sharpe is not a measurement — DSR is NaN."""
    from vgp.analysis import compute_dsr

    returns = _synthetic_returns(4.0)
    trials = [3.8, 4.0, 4.2, 3.9, 4.1]
    assert np.isnan(compute_dsr(returns, sr_hat=-np.inf, trial_sharpes=trials))
    assert np.isnan(compute_dsr(returns, sr_hat=np.nan, trial_sharpes=trials))


def test_compute_dsr_identical_trials_is_nan():
    """VAL-04: sigma_SR == 0 makes the hurdle degenerate — NaN, not a free pass."""
    from vgp.analysis import compute_dsr

    dsr = compute_dsr(_synthetic_returns(4.0), sr_hat=4.0, trial_sharpes=[4.0, 4.0, 4.0])
    assert np.isnan(dsr), f"Zero trial spread must give NaN, got {dsr}"


# ---------------------------------------------------------------------------
# VAL-04: attach_dsr() computes DSR across the whole trial set
# ---------------------------------------------------------------------------


def test_attach_dsr_fills_rows_and_drops_returns():
    """VAL-04: attach_dsr fills dsr from all trials and removes the scratch returns."""
    from vgp.analysis import IS_RETURNS_KEY, attach_dsr

    sharpes = [4.0, 3.6, 3.9, 3.2, 3.7, 3.5]
    results = [
        {
            "is_sharpe": sr,
            "oos_sharpe": 0.5,
            "dsr": float("nan"),
            IS_RETURNS_KEY: _synthetic_returns(sr, seed=i),
        }
        for i, sr in enumerate(sharpes)
    ]

    attach_dsr(results)

    for row in results:
        assert IS_RETURNS_KEY not in row, "scratch IS returns must be popped"
        assert np.isfinite(row["dsr"]), f"dsr not filled in: {row['dsr']}"
        assert 0.0 <= row["dsr"] <= 1.0
        assert row["dsr_n_trials"] == len(
            sharpes
        ), "n_trials must be the whole trial set, not one row"
        assert row["dsr_trial_sr_std"] == pytest.approx(np.std(sharpes, ddof=1))


def test_attach_dsr_excludes_sentinel_rows():
    """VAL-04: rows with a worst-fitness IS Sharpe get NaN DSR and do not count as trials."""
    from vgp.analysis import IS_RETURNS_KEY, attach_dsr

    results = [
        {"is_sharpe": 4.0, "dsr": float("nan"), IS_RETURNS_KEY: _synthetic_returns(4.0, seed=1)},
        {"is_sharpe": 3.6, "dsr": float("nan"), IS_RETURNS_KEY: _synthetic_returns(3.6, seed=2)},
        {"is_sharpe": 3.8, "dsr": float("nan"), IS_RETURNS_KEY: _synthetic_returns(3.8, seed=3)},
        {
            "is_sharpe": float("nan"),
            "dsr": float("nan"),
            IS_RETURNS_KEY: _synthetic_returns(1.0, seed=4),
        },
    ]

    attach_dsr(results)

    assert all(
        r["dsr_n_trials"] == 3 for r in results
    ), "the unmeasured row must not be counted as a trial"
    assert np.isnan(results[-1]["dsr"]), "unmeasured IS Sharpe must give NaN DSR"
    assert all(np.isfinite(r["dsr"]) for r in results[:3])


def test_attach_dsr_row_without_returns_is_nan():
    """VAL-04: a row whose IS backtest failed gets NaN DSR, not 0.0."""
    from vgp.analysis import attach_dsr

    results = [{"is_sharpe": 4.0, "dsr": float("nan")}, {"is_sharpe": 3.5, "dsr": float("nan")}]
    attach_dsr(results)
    assert all(np.isnan(r["dsr"]) for r in results)


# ---------------------------------------------------------------------------
# VAL-03 (regression): the worst-fitness sentinel must never be reported
# as an OOS Sharpe ratio
# ---------------------------------------------------------------------------


def test_run_window_records_nan_not_inf_when_oos_unmeasurable(
    feature_matrix, close_prices, dates, eval_cfg, base_evo_kwargs
):
    """VAL-03 regression: an unmeasurable OOS window yields NaN, never -inf.

    evaluate() returns (-inf, -inf, -size) so NSGA-II can still RANK an unusable
    individual. That sentinel is not a Sharpe ratio of minus infinity. Writing it
    into results.csv reports "the trade filter tripped" as catastrophic
    performance, and poisons every median and IQR computed from the column.
    """
    from vgp.analysis import generate_windows
    from vgp.analysis.runner import WalkForwardRunner

    runner = WalkForwardRunner(dates=dates)
    window = generate_windows("2024-01-01", "2026-04-01")[0]
    mock_return = _make_mock_evolution_return()

    # evaluate_with_status reports the sentinel plus the reason for it
    sentinel = ((-np.inf, -np.inf, -5.0), "below_min_trades", 3)

    with (
        patch("vgp.analysis.runner.run_evolution", return_value=mock_return),
        patch("vgp.analysis.runner.evaluate_with_status", return_value=sentinel),
        patch(
            "vgp.analysis.runner._get_is_returns",
            return_value=np.random.default_rng(0).standard_normal(250),
        ),
    ):

        results = runner.run_window(
            window=window,
            feature_matrix=feature_matrix,
            close_prices=close_prices,
            base_eval_config=eval_cfg,
            seeds=[0],
            evo_config_kwargs=base_evo_kwargs,
        )

    row = results[0]
    assert not np.isinf(row["oos_sharpe"]), (
        f"oos_sharpe = {row['oos_sharpe']} — the worst-fitness sentinel leaked "
        f"into reporting as a performance number"
    )
    assert np.isnan(row["oos_sharpe"]), (
        f"oos_sharpe must be NaN ('not measured') when the OOS evaluation is "
        f"invalid, got {row['oos_sharpe']}"
    )
    assert row["oos_status"] == "below_min_trades"
    assert row["oos_n_trades"] == 3, "the observed trade count must be reported"


def test_run_window_reports_ok_status_and_real_sharpe(
    feature_matrix, close_prices, dates, eval_cfg, base_evo_kwargs
):
    """VAL-03: a valid OOS evaluation is recorded verbatim with status 'ok'."""
    from vgp.analysis import generate_windows
    from vgp.analysis.runner import WalkForwardRunner

    runner = WalkForwardRunner(dates=dates)
    window = generate_windows("2024-01-01", "2026-04-01")[0]
    mock_return = _make_mock_evolution_return()

    with (
        patch("vgp.analysis.runner.run_evolution", return_value=mock_return),
        patch(
            "vgp.analysis.runner.evaluate_with_status", return_value=((1.23, 0.05, -5.0), "ok", 87)
        ),
        patch(
            "vgp.analysis.runner._get_is_returns",
            return_value=np.random.default_rng(0).standard_normal(250),
        ),
    ):

        results = runner.run_window(
            window=window,
            feature_matrix=feature_matrix,
            close_prices=close_prices,
            base_eval_config=eval_cfg,
            seeds=[0],
            evo_config_kwargs=base_evo_kwargs,
        )

    assert results[0]["oos_sharpe"] == pytest.approx(1.23)
    assert results[0]["oos_status"] == "ok"
    assert results[0]["oos_n_trades"] == 87


def test_run_window_records_nan_for_worst_fitness_is_sharpe(
    feature_matrix, close_prices, dates, eval_cfg, base_evo_kwargs
):
    """VAL-03: a -inf IS Sharpe on hof[0] is also a sentinel — record NaN."""
    from vgp.analysis import generate_windows
    from vgp.analysis.runner import WalkForwardRunner

    _pop, mock_hof, mock_logbook = _make_mock_evolution_return()
    mock_hof[0].fitness.values = (-np.inf, -np.inf, -5.0)

    runner = WalkForwardRunner(dates=dates)
    window = generate_windows("2024-01-01", "2026-04-01")[0]

    with (
        patch("vgp.analysis.runner.run_evolution", return_value=([], mock_hof, mock_logbook)),
        patch(
            "vgp.analysis.runner.evaluate_with_status", return_value=((0.3, 0.05, -5.0), "ok", 90)
        ),
        patch(
            "vgp.analysis.runner._get_is_returns",
            return_value=np.random.default_rng(0).standard_normal(250),
        ),
    ):

        results = runner.run_window(
            window=window,
            feature_matrix=feature_matrix,
            close_prices=close_prices,
            base_eval_config=eval_cfg,
            seeds=[0],
            evo_config_kwargs=base_evo_kwargs,
        )

    assert np.isnan(results[0]["is_sharpe"]), (
        f"is_sharpe = {results[0]['is_sharpe']} — worst-fitness sentinel must "
        f"not be reported as an IS Sharpe"
    )


def test_run_window_scales_oos_min_trades(feature_matrix, close_prices, dates, base_evo_kwargs):
    """VAL-03: the OOS trade threshold scales to the OOS window length.

    min_trades is a RATE requirement written for the ~12-month train window.
    A 3-month OOS window has roughly a quarter of the bars, so requiring the
    same 50 sign changes is mechanically unreachable for a strategy trading at
    exactly its in-sample frequency.
    """
    from vgp.analysis import generate_windows
    from vgp.analysis.runner import WalkForwardRunner
    from vgp.backtest.runner import EvalConfig

    runner = WalkForwardRunner(dates=dates)
    window = generate_windows("2024-01-01", "2026-04-01")[0]
    base_cfg = EvalConfig(close_prices=close_prices, min_trades=50)
    mock_return = _make_mock_evolution_return()

    with (
        patch("vgp.analysis.runner.run_evolution", return_value=mock_return),
        patch(
            "vgp.analysis.runner.evaluate_with_status", return_value=((0.3, 0.05, -5.0), "ok", 90)
        ) as mock_eval,
        patch(
            "vgp.analysis.runner._get_is_returns",
            return_value=np.random.default_rng(0).standard_normal(250),
        ),
    ):

        results = runner.run_window(
            window=window,
            feature_matrix=feature_matrix,
            close_prices=close_prices,
            base_eval_config=base_cfg,
            seeds=[0],
            evo_config_kwargs=base_evo_kwargs,
        )

    oos_cfg = mock_eval.call_args.args[2]
    assert oos_cfg.min_trades < 50, (
        f"OOS min_trades = {oos_cfg.min_trades}; the train-window threshold of "
        f"50 was applied verbatim to a 3-month OOS window"
    )
    assert oos_cfg.min_trades >= 1, "threshold must stay at least 1"
    assert (
        results[0]["oos_min_trades"] == oos_cfg.min_trades
    ), "the threshold actually applied must be recorded alongside the result"


def test_run_window_honours_explicit_oos_min_trades(
    feature_matrix, close_prices, dates, base_evo_kwargs
):
    """VAL-03: oos_min_trades=0 measures whatever the OOS window produced."""
    from vgp.analysis import generate_windows
    from vgp.analysis.runner import WalkForwardRunner
    from vgp.backtest.runner import EvalConfig

    runner = WalkForwardRunner(dates=dates)
    window = generate_windows("2024-01-01", "2026-04-01")[0]
    base_cfg = EvalConfig(close_prices=close_prices, min_trades=50)
    mock_return = _make_mock_evolution_return()

    with (
        patch("vgp.analysis.runner.run_evolution", return_value=mock_return),
        patch(
            "vgp.analysis.runner.evaluate_with_status", return_value=((0.3, 0.05, -5.0), "ok", 4)
        ) as mock_eval,
        patch(
            "vgp.analysis.runner._get_is_returns",
            return_value=np.random.default_rng(0).standard_normal(250),
        ),
    ):

        runner.run_window(
            window=window,
            feature_matrix=feature_matrix,
            close_prices=close_prices,
            base_eval_config=base_cfg,
            seeds=[0],
            evo_config_kwargs=base_evo_kwargs,
            oos_min_trades=0,
        )

    assert mock_eval.call_args.args[2].min_trades == 0


# ---------------------------------------------------------------------------
# VAL-04: aggregate_seeds() and save_results_csv() must not treat
# "not measured" as a data point
# ---------------------------------------------------------------------------


def test_aggregate_seeds_ignores_unmeasured_oos():
    """VAL-04: NaN OOS Sharpes are excluded from median, IQR and positive count."""
    from vgp.analysis import aggregate_seeds

    agg = aggregate_seeds(
        [
            {"oos_sharpe": 1.0, "dsr": 0.9},
            {"oos_sharpe": float("nan"), "dsr": float("nan")},
            {"oos_sharpe": 0.5, "dsr": 0.8},
        ]
    )

    assert agg["n_seeds_valid_oos"] == 2, "only measured seeds count as valid"
    assert agg["n_seeds"] == 3
    assert agg["n_seeds_positive_oos"] == 2
    assert agg["median_oos_sharpe"] == pytest.approx(
        0.75
    ), f"median must be over measured seeds only, got {agg['median_oos_sharpe']}"
    assert np.isfinite(agg["iqr_oos_sharpe"])
    assert agg["median_dsr"] == pytest.approx(0.85)


def test_aggregate_seeds_all_unmeasured():
    """VAL-04: when no seed was measurable, medians are NaN and counts are 0."""
    from vgp.analysis import aggregate_seeds

    agg = aggregate_seeds(
        [
            {"oos_sharpe": float("nan"), "dsr": float("nan")},
            {"oos_sharpe": float("nan"), "dsr": float("nan")},
        ]
    )

    assert np.isnan(agg["median_oos_sharpe"])
    assert np.isnan(agg["iqr_oos_sharpe"])
    assert agg["n_seeds_valid_oos"] == 0
    assert agg["n_seeds_positive_oos"] == 0
    assert agg["n_seeds"] == 2


def test_aggregate_seeds_would_be_poisoned_by_inf():
    """VAL-04: an -inf that somehow reaches aggregation must not silently
    produce -inf / NaN summaries for the whole window."""
    from vgp.analysis import aggregate_seeds

    agg = aggregate_seeds(
        [
            {"oos_sharpe": 1.0, "dsr": 0.9},
            {"oos_sharpe": -np.inf, "dsr": 0.1},
            {"oos_sharpe": 1.4, "dsr": 0.8},
        ]
    )

    assert np.isfinite(agg["median_oos_sharpe"]), (
        f"median_oos_sharpe = {agg['median_oos_sharpe']} — an infinite sentinel "
        f"destroyed the window summary"
    )
    assert agg["median_oos_sharpe"] == pytest.approx(1.2)
    assert agg["n_seeds_valid_oos"] == 2


def test_save_results_csv_drops_scratch_keys(tmp_path):
    """VAL-04: the per-period returns array is scratch, not a CSV column."""
    from vgp.analysis import IS_RETURNS_KEY, save_results_csv

    path = tmp_path / "results.csv"
    save_results_csv(
        [
            {
                "window_id": 0,
                "seed": 0,
                "is_sharpe": 4.0,
                "oos_sharpe": float("nan"),
                "oos_status": "below_min_trades",
                "dsr": 0.9,
                IS_RETURNS_KEY: np.zeros(10),
            }
        ],
        str(path),
    )

    header = path.read_text().splitlines()[0]
    assert IS_RETURNS_KEY not in header, f"scratch key written to CSV: {header}"
    assert "oos_status" in header, "the status column must be persisted"


# ---------------------------------------------------------------------------
# VAL-04 (regression): the trial count must reflect the SEARCH, not the winners
#
# N in Bailey & Lopez de Prado is the number of configurations effectively
# searched. For a GP that is every individual evaluated — thousands — not the
# per-seed winners in the results table. With N=9 the hurdle multiplier is 1.52;
# with N=3600 it is 3.24. On signal-free data the winners-only convention
# certifies noise at DSR>0.95 while the evaluation-based one rejects it at <0.05.
# ---------------------------------------------------------------------------


def _row(is_sharpe: float, seed: int = 0, acc=None) -> dict:
    from vgp.analysis.dsr import IS_RETURNS_KEY, TRIALS_KEY

    row = {
        "window_id": 0,
        "seed": seed,
        "is_sharpe": is_sharpe,
        "oos_sharpe": 0.5,
        "dsr": float("nan"),
        IS_RETURNS_KEY: _synthetic_returns(is_sharpe, seed=seed + 1),
    }
    if acc is not None:
        row[TRIALS_KEY] = acc
    return row


def _accumulator(sharpes):
    from vgp.trials import TrialAccumulator

    acc = TrialAccumulator()
    acc.extend(sharpes)
    return acc


def test_attach_dsr_sizes_correction_from_all_evaluations():
    """The primary trial set must be every evaluated individual, merged."""
    from vgp.analysis import attach_dsr
    from vgp.analysis.dsr import TRIALS_KEY

    rng = np.random.default_rng(0)
    rows = [
        _row(4.0, seed=0, acc=_accumulator(rng.standard_normal(500) * 1.5)),
        _row(3.7, seed=1, acc=_accumulator(rng.standard_normal(500) * 1.5)),
        _row(3.9, seed=2, acc=_accumulator(rng.standard_normal(500) * 1.5)),
    ]

    attach_dsr(rows)

    for r in rows:
        assert TRIALS_KEY not in r, "accumulator must be popped, not written to CSV"
        assert r["dsr_trial_source"] == "all_evaluations"
        assert r["dsr_n_trials"] == 1500, (
            f"trial count {r['dsr_n_trials']} does not reflect the 1500 evaluations; "
            f"sizing the correction to the 3 winners would leave it nearly inert"
        )
        assert r["dsr_n_trials_bests"] == 3
        assert r["dsr_n_evaluations"] == 1500


def test_attach_dsr_primary_is_conservative_relative_to_bests_only():
    """More trials means a higher hurdle, so dsr <= dsr_bests_only always."""
    from vgp.analysis import attach_dsr

    rng = np.random.default_rng(1)
    rows = [
        _row(sr, seed=i, acc=_accumulator(rng.standard_normal(400) * 1.2))
        for i, sr in enumerate([4.0, 3.8, 3.6, 3.9])
    ]

    attach_dsr(rows)

    for r in rows:
        assert r["dsr"] <= r["dsr_bests_only"] + 1e-9, (
            f"primary DSR {r['dsr']:.4f} exceeds the bests-only bound "
            f"{r['dsr_bests_only']:.4f} — the conservative set must not be laxer"
        )
        assert np.isfinite(r["dsr"]) and np.isfinite(r["dsr_bests_only"])


def test_attach_dsr_evaluation_count_rejects_what_winners_only_certifies():
    """The regression that matters: noise certified by one convention, rejected by the other.

    Reproduces the observed situation — nine winners clustered at a modest IS
    Sharpe drawn from a wide search. Counting only the winners gives a small
    hurdle and a high DSR; counting the search rejects it.
    """
    from vgp.analysis import attach_dsr

    winners = [1.94, 1.72, 1.94, 1.48, 1.30, 1.59, 1.18, 1.02, 1.09]
    rng = np.random.default_rng(7)
    rows = [
        _row(sr, seed=i, acc=_accumulator(rng.standard_normal(190) * 1.0))
        for i, sr in enumerate(winners)
    ]

    attach_dsr(rows)

    best_primary = max(r["dsr"] for r in rows)
    best_bests = max(r["dsr_bests_only"] for r in rows)

    assert rows[0]["dsr_n_trials"] > 1000
    assert best_bests > best_primary, "the two conventions must differ materially"
    assert (
        best_primary < 0.95
    ), f"the evaluation-sized correction still certifies this at {best_primary:.4f}"


def test_attach_dsr_falls_back_to_bests_and_warns(caplog):
    """With no accumulators the fallback must be explicit, not silent.

    A run whose trial accumulators are missing gets a correction sized to the
    winners. That is the wrong trial population, so it must be labelled and
    logged rather than passed off as the primary figure.
    """
    import logging

    from vgp.analysis import attach_dsr

    rows = [_row(sr, seed=i) for i, sr in enumerate([4.0, 3.7, 3.9])]

    with caplog.at_level(logging.WARNING, logger="vgp.analysis.dsr"):
        attach_dsr(rows)

    assert rows[0]["dsr_trial_source"] == "reported_bests"
    assert rows[0]["dsr_n_trials"] == 3
    assert rows[0]["dsr_n_evaluations"] == 0
    assert any(
        "no trial accumulators" in m for m in caplog.messages
    ), "falling back to the winners must be logged — it changes what DSR means"


def test_run_evolution_records_every_evaluation():
    """EVO: the logbook must carry an accumulator covering all evaluated individuals."""
    from vgp.backtest.runner import EvalConfig
    from vgp.evolution.config import EvolutionConfig
    from vgp.evolution.loop import run_evolution

    T, F, A = 220, 12, 2
    rng = np.random.default_rng(3)
    dates = pd.date_range("2024-01-01", periods=T, freq="D")
    close = pd.DataFrame(
        (100.0 * np.exp(np.cumsum(rng.standard_normal((T, A)) * 0.01, axis=0))),
        index=dates,
        columns=[f"a{i}" for i in range(A)],
    )
    fm = rng.standard_normal((T, F, A)).astype(np.float32)

    cfg = EvolutionConfig(pop_size=12, n_generations=3, seed=0, n_jobs=1, checkpoint_freq=999)
    _pop, _hof, logbook = run_evolution(cfg, fm, EvalConfig(close_prices=close, min_trades=1))

    acc = getattr(logbook, "trial_accumulator", None)
    assert acc is not None, "run_evolution must attach a trial accumulator to the logbook"

    # gen 0 evaluates the whole population; each later generation evaluates the
    # invalidated offspring, so the total must exceed one population.
    assert acc.n_evaluations > cfg.pop_size, (
        f"only {acc.n_evaluations} evaluations recorded for pop_size={cfg.pop_size} "
        f"over {cfg.n_generations} generations — later generations are not counted"
    )
    logged = sum(rec["nevals"] for rec in logbook)
    assert acc.n_evaluations == logged, (
        f"accumulator counted {acc.n_evaluations} evaluations but the logbook "
        f"recorded {logged} — the two must agree"
    )


# ---------------------------------------------------------------------------
# VAL-01: window geometry is configurable, and step == oos keeps OOS periods
# independent at ANY geometry — not just the defaults.
#
# scripts/run.py now runs 9m/2m/2m to get 6 windows out of a ~23-month panel
# instead of 3, so the non-overlap guarantee has to hold there too. Overlapping
# OOS periods would correlate the per-window results and break aggregation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "train,val,oos",
    [(12, 2, 3), (9, 2, 2), (9, 1, 2), (6, 1, 2), (6, 2, 3)],
)
def test_generate_windows_oos_never_overlaps_at_any_geometry(train, val, oos):
    """step == oos must produce strictly non-overlapping OOS periods."""
    from vgp.analysis import generate_windows

    windows = generate_windows(
        "2024-05-01",
        "2026-04-01",
        train_months=train,
        val_months=val,
        oos_months=oos,
        step_months=oos,
    )
    assert windows, f"no windows for {train}/{val}/{oos} — geometry unusable"

    for a, b in zip(windows, windows[1:]):
        assert pd.Timestamp(b.test_start) > pd.Timestamp(a.test_end), (
            f"{train}/{val}/{oos}: window {a.window_id} OOS ends "
            f"{a.test_end} but {b.window_id} starts {b.test_start}"
        )


@pytest.mark.parametrize(
    "train,val,oos",
    [(12, 2, 3), (9, 2, 2), (6, 1, 2)],
)
def test_generate_windows_respects_requested_lengths(train, val, oos):
    """Each window's train/val/OOS spans must match what was asked for.

    Guards the config actually meaning something: a geometry that silently
    ignored train_months would change the experiment without changing the
    printed setup.
    """
    from vgp.analysis import generate_windows

    windows = generate_windows(
        "2024-05-01",
        "2026-04-01",
        train_months=train,
        val_months=val,
        oos_months=oos,
        step_months=oos,
    )
    for w in windows:
        val_span = (pd.Timestamp(w.val_end) - pd.Timestamp(w.val_start)).days
        oos_span = (pd.Timestamp(w.test_end) - pd.Timestamp(w.test_start)).days
        # Calendar months vary in length; allow a few days of slack
        assert (
            abs(val_span - val * 30.44) < 8
        ), f"window {w.window_id} val span {val_span}d, expected ~{val * 30.44:.0f}d"
        assert (
            abs(oos_span - oos * 30.44) < 8
        ), f"window {w.window_id} OOS span {oos_span}d, expected ~{oos * 30.44:.0f}d"
        # train_end must precede val_start, which must precede test_start
        assert pd.Timestamp(w.train_end) < pd.Timestamp(w.val_start)
        assert pd.Timestamp(w.val_end) < pd.Timestamp(w.test_start)


def test_more_windows_from_a_shorter_train_window():
    """The trade-off the run config makes, pinned.

    A ~23-month panel fits only 3 non-overlapping 12m/2m/3m windows. Shortening
    to 9m/2m/2m doubles that. If generate_windows ever stopped honouring this,
    the run would quietly fall back to too few OOS periods to say anything.
    """
    from vgp.analysis import generate_windows

    wide = generate_windows(
        "2024-05-01", "2026-04-01", train_months=12, val_months=2, oos_months=3, step_months=3
    )
    narrow = generate_windows(
        "2024-05-01", "2026-04-01", train_months=9, val_months=2, oos_months=2, step_months=2
    )

    assert len(wide) == 3, f"expected 3 wide windows, got {len(wide)}"
    assert len(narrow) == 6, f"expected 6 narrow windows, got {len(narrow)}"

    # And the narrow geometry must cover more OOS calendar in total
    def coverage(ws):
        return sum((pd.Timestamp(w.test_end) - pd.Timestamp(w.test_start)).days for w in ws)

    assert coverage(narrow) > coverage(wide), (
        f"narrow geometry covers {coverage(narrow)}d of OOS vs {coverage(wide)}d "
        f"— more windows should mean more out-of-sample calendar, not less"
    )


# ---------------------------------------------------------------------------
# VAL-04: de-annualization must match how the Sharpe was annualized
#
# vectorbt annualizes a freq="1D" Sharpe with 365 periods per year (verified:
# pf.sharpe_ratio() / per-period Sharpe of pf.returns() == sqrt(365), constant
# to 1e-14). compute_dsr previously de-annualized with 252, rescaling sr_hat
# and sigma_SR by sqrt(365/252) = 1.2035 and inflating the DSR z-statistic by
# about 20%. A unit mismatch like this is invisible in the output — the number
# is simply wrong by a constant factor — so it needs pinning.
# ---------------------------------------------------------------------------


def test_periods_per_year_matches_vectorbt_daily_convention():
    """The constant must be 365, not the equity-market 252."""
    from vgp.analysis import PERIODS_PER_YEAR_DAILY

    assert PERIODS_PER_YEAR_DAILY == 365, (
        f"PERIODS_PER_YEAR_DAILY is {PERIODS_PER_YEAR_DAILY}; vectorbt "
        f"annualizes freq='1D' with 365, and crypto trades every calendar day"
    )


def test_de_annualization_is_self_consistent():
    """Annualizing then de-annualizing must be a round trip.

    compute_dsr with Sharpes annualized by sqrt(ppy) and periods_per_year=ppy
    must equal compute_dsr with per-period Sharpes and periods_per_year=1. If
    the two disagree the de-annualization is not inverting the annualization.
    """
    from vgp.analysis import compute_dsr

    rng = np.random.default_rng(0)
    returns = rng.standard_normal(400) * 0.01
    sr_pp = 0.06
    trials_pp = [0.05, 0.06, 0.07, 0.055, 0.065, 0.045, 0.075, 0.06, 0.058]

    for ppy in (252, 365):
        scale = np.sqrt(ppy)
        annualized = compute_dsr(
            returns,
            sr_hat=sr_pp * scale,
            trial_sharpes=[t * scale for t in trials_pp],
            periods_per_year=ppy,
        )
        per_period = compute_dsr(returns, sr_hat=sr_pp, trial_sharpes=trials_pp, periods_per_year=1)
        assert annualized == pytest.approx(per_period, rel=1e-9), (
            f"ppy={ppy}: annualized path gave {annualized}, per-period path "
            f"gave {per_period} — de-annualization is not the inverse"
        )


def test_wrong_periods_per_year_changes_the_answer_materially():
    """Documents the size of the bug this replaced, so it is not dismissed.

    Same inputs, only the assumed annualization differing, must move the DSR —
    otherwise the constant would not matter and the test above would be
    pointless ceremony.
    """
    from vgp.analysis import compute_dsr

    rng = np.random.default_rng(1)
    returns = rng.standard_normal(400) * 0.01
    sr_ann = 3.0
    trials = [2.6, 2.8, 3.0, 3.1, 2.7, 2.9, 3.2, 2.5, 3.05]

    correct = compute_dsr(returns, sr_hat=sr_ann, trial_sharpes=trials, periods_per_year=365)
    wrong = compute_dsr(returns, sr_hat=sr_ann, trial_sharpes=trials, periods_per_year=252)

    assert correct != pytest.approx(wrong, rel=1e-6), (
        "the annualization constant made no difference; the de-annualization "
        "is not actually being applied"
    )
