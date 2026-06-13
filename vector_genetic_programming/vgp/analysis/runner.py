"""Walk-forward multi-seed evolution runner.

Implements the walk-forward validation harness (VAL-01, VAL-02, VAL-03).
OOS structural invariant: test_fm is never passed to run_evolution() —
it is only used in the single evaluate() call after evolution completes.

python-dateutil is a pandas transitive dependency (not in pyproject.toml directly).
Available in all environments that have pandas>=3.0.0 installed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
# python-dateutil is a pandas transitive dependency (not in pyproject.toml directly).
# Available in all environments that have pandas>=3.0.0 installed.
from dateutil.relativedelta import relativedelta

from vgp.analysis.dsr import compute_dsr
from vgp.backtest.runner import EvalConfig, evaluate
from vgp.data.splitter import WalkForwardSplitter
from vgp.evolution.config import EvolutionConfig
from vgp.evolution.loop import run_evolution

logger = logging.getLogger(__name__)


@dataclass
class WindowSpec:
    """One walk-forward window's date boundaries."""

    window_id: int
    train_end: str        # inclusive ISO e.g. "2024-12-31"
    val_start: str
    val_end: str
    test_start: str       # stored; NEVER passed to run_evolution()
    test_end: str         # stored; NEVER passed to run_evolution()


def generate_windows(
    total_start: str,
    total_end: str,
    train_months: int = 12,
    val_months: int = 2,
    oos_months: int = 3,
    step_months: int = 3,
) -> list[WindowSpec]:
    """Generate non-overlapping walk-forward window specs.

    With default params and data range 2024-01-01 to 2026-04-01,
    produces exactly 4 windows with non-overlapping OOS periods.

    Parameters
    ----------
    total_start : str
        First date of usable data, ISO format e.g. "2024-01-01".
    total_end : str
        Last date of usable data, ISO format.
    train_months, val_months, oos_months : int
        Window sizes in calendar months.
    step_months : int
        How many months to advance the window start per step.
        When step_months == oos_months, OOS periods are non-overlapping.
    """
    windows: list[WindowSpec] = []
    start = pd.Timestamp(total_start)
    total_end_ts = pd.Timestamp(total_end)
    window_id = 0

    while True:
        train_end_ts = start + relativedelta(months=train_months) - pd.Timedelta(days=1)
        val_start_ts = train_end_ts + pd.Timedelta(days=1)
        val_end_ts = val_start_ts + relativedelta(months=val_months) - pd.Timedelta(days=1)
        test_start_ts = val_end_ts + pd.Timedelta(days=1)
        test_end_ts = test_start_ts + relativedelta(months=oos_months) - pd.Timedelta(days=1)

        if test_end_ts > total_end_ts:
            break

        windows.append(WindowSpec(
            window_id=window_id,
            train_end=train_end_ts.strftime("%Y-%m-%d"),
            val_start=val_start_ts.strftime("%Y-%m-%d"),
            val_end=val_end_ts.strftime("%Y-%m-%d"),
            test_start=test_start_ts.strftime("%Y-%m-%d"),
            test_end=test_end_ts.strftime("%Y-%m-%d"),
        ))
        window_id += 1
        start = start + relativedelta(months=step_months)

    return windows


def _get_is_returns(
    individual,
    train_fm: np.ndarray,
    train_eval_cfg: EvalConfig,
) -> np.ndarray:
    """Re-run IS backtest to extract per-period returns for DSR computation.

    This uses ONLY train data — no OOS data is accessed here.
    Separated into its own function so tests can patch it without
    requiring actual GP tree execution.

    Parameters
    ----------
    individual : creator.Individual
        Best individual from hof[0].
    train_fm : np.ndarray
        Training feature matrix [T_train x F x A].
    train_eval_cfg : EvalConfig
        EvalConfig with train close prices set.

    Returns
    -------
    np.ndarray
        Per-period portfolio returns shape [T_train].
    """
    import vectorbt as vbt  # noqa: PLC0415 — deferred import (D-15 pattern)

    from vgp.gp.gp_types import build_pset  # noqa: PLC0415
    from vgp.gp.tree_evaluator import TreeEvaluator  # noqa: PLC0415

    pset = build_pset()
    evaluator = TreeEvaluator(pset)
    T_train, F, A = train_fm.shape
    train_signals = np.zeros((T_train, A), dtype=np.float32)
    for a in range(A):
        train_signals[:, a] = evaluator.execute(individual, train_fm[:, :, a])

    long_entries = train_signals > 0
    short_entries = train_signals < 0
    long_exits = train_signals <= 0
    short_exits = train_signals >= 0
    fee_per_side = (train_eval_cfg.fee_bps / 2.0) / 10_000.0

    pf = vbt.Portfolio.from_signals(
        close=train_eval_cfg.close_prices,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        size=1.0 / A,
        size_type="percent",
        upon_opposite_entry="close",
        fees=fee_per_side,
        freq=train_eval_cfg.freq,
        init_cash=train_eval_cfg.init_cash,
        group_by=True,
        cash_sharing=True,
    )
    return pf.returns().to_numpy()


class WalkForwardRunner:
    """Multi-seed walk-forward evolution runner.

    Structural OOS invariant (VAL-02):
    - run_window() receives feature_matrix and close_prices for the FULL data range.
    - It splits into train/val/test at the start of each window.
    - Only train_fm and train_close are forwarded to run_evolution().
    - test_fm and test_close are held in local variables and used ONCE for OOS evaluate().
    - After evaluate() the test slice variables go out of scope.
    """

    def __init__(self, dates: pd.DatetimeIndex) -> None:
        """
        Parameters
        ----------
        dates : pd.DatetimeIndex
            DatetimeIndex from FeatureEngine.dates_ — required for ndarray splits.
        """
        self._splitter = WalkForwardSplitter()
        self._dates = dates

    def run_window(
        self,
        window: WindowSpec,
        feature_matrix: np.ndarray,    # full [T x F x A] float32
        close_prices: pd.DataFrame,    # full [T x A] with DatetimeIndex
        base_eval_config: EvalConfig,
        seeds: list[int],
        evo_config_kwargs: dict,
    ) -> list[dict]:
        """Run evolution for all seeds on one window. Returns one dict per seed.

        VAL-02 structural enforcement: this method body has exactly ONE call
        to evaluate() that uses test_fm. It is called after run_evolution()
        completes. test_fm is a local variable that goes out of scope after
        result is stored.

        Parameters
        ----------
        window : WindowSpec
            Date boundaries for this walk-forward window.
        feature_matrix : np.ndarray
            Full feature matrix [T x F x A]. Splitter extracts train/test slices.
        close_prices : pd.DataFrame
            Full close prices [T x A] with DatetimeIndex. Splitter slices to train/test.
        base_eval_config : EvalConfig
            EvalConfig template. close_prices will be replaced per-slice.
        seeds : list[int]
            Seeds to iterate over. Length = n_seeds.
        evo_config_kwargs : dict
            Keyword args for EvolutionConfig (excluding seed, which is set per iteration).
        """
        logger.info(
            "Window %d: train_end=%s test_start=%s test_end=%s n_seeds=%d",
            window.window_id, window.train_end, window.test_start, window.test_end, len(seeds),
        )

        # --- Split feature matrix (ndarray) --- #
        train_fm, _val_fm, test_fm = self._splitter.split(
            feature_matrix,
            train_end=window.train_end,
            val_start=window.val_start,
            val_end=window.val_end,
            test_start=window.test_start,
            dates=self._dates,
        )

        # --- Split close prices (DataFrame) --- #
        train_close, _val_close, test_close = self._splitter.split(
            close_prices,
            train_end=window.train_end,
            val_start=window.val_start,
            val_end=window.val_end,
            test_start=window.test_start,
        )

        # Build train/test EvalConfigs (close_prices must match the data slice)
        train_eval_cfg = EvalConfig(
            fee_bps=base_eval_config.fee_bps,
            min_trades=base_eval_config.min_trades,
            freq=base_eval_config.freq,
            init_cash=base_eval_config.init_cash,
            close_prices=train_close.copy(),
        )
        test_eval_cfg = EvalConfig(
            fee_bps=base_eval_config.fee_bps,
            min_trades=base_eval_config.min_trades,
            freq=base_eval_config.freq,
            init_cash=base_eval_config.init_cash,
            close_prices=test_close.copy(),
        )

        seed_results: list[dict] = []
        for seed in seeds:
            cfg = EvolutionConfig(seed=seed, **evo_config_kwargs)

            # --- Evolution on train data ONLY --- #
            pop, hof, logbook = run_evolution(cfg, train_fm, train_eval_cfg)

            if not hof:
                logger.warning(
                    "Window %d seed %d: HOF is empty — skipping OOS eval",
                    window.window_id, seed,
                )
                continue

            best_ind = hof[0]

            # IS Sharpe: last generation's max from logbook
            is_sharpe = float(logbook.chapters["fitness"][-1]["sharpe_max"])

            # --- OOS evaluate: called EXACTLY ONCE per (window, seed) --- #
            oos_fitness = evaluate(best_ind, test_fm, test_eval_cfg)
            oos_sharpe = float(oos_fitness[0])
            # test_fm is no longer referenced after this line

            # DSR: re-run IS backtest to get per-period returns for DSR formula
            # Uses train data only (no OOS leakage). _get_is_returns() is
            # a separate helper so tests can patch it without requiring
            # actual GP tree execution.
            n_trials = max(1, len(seeds))
            try:
                is_returns = _get_is_returns(best_ind, train_fm, train_eval_cfg)
                dsr = compute_dsr(is_returns, sr_hat=is_sharpe, n_trials=n_trials)
            except Exception as exc:  # pragma: no cover — only fires if vbt/eval fails
                logger.warning(
                    "Window %d seed %d: DSR computation failed (%s) — defaulting to 0.0",
                    window.window_id, seed, exc,
                )
                dsr = 0.0

            seed_results.append({
                "window_id": window.window_id,
                "seed": seed,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_sharpe,
                "dsr": dsr,
                "n_nodes_best": len(best_ind),
            })

        return seed_results
