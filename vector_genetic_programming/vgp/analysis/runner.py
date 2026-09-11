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

from vgp.analysis.dsr import IS_RETURNS_KEY, TRIALS_KEY
from vgp.backtest.runner import EVAL_OK, EvalConfig, evaluate_with_status
from vgp.data.splitter import WalkForwardSplitter
from vgp.evolution.config import EvolutionConfig
from vgp.evolution.loop import run_evolution

logger = logging.getLogger(__name__)


@dataclass
class WindowSpec:
    """One walk-forward window's date boundaries."""

    window_id: int
    train_end: str  # inclusive ISO e.g. "2024-12-31"
    val_start: str
    val_end: str
    test_start: str  # stored; NEVER passed to run_evolution()
    test_end: str  # stored; NEVER passed to run_evolution()


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

        windows.append(
            WindowSpec(
                window_id=window_id,
                train_end=train_end_ts.strftime("%Y-%m-%d"),
                val_start=val_start_ts.strftime("%Y-%m-%d"),
                val_end=val_end_ts.strftime("%Y-%m-%d"),
                test_start=test_start_ts.strftime("%Y-%m-%d"),
                test_end=test_end_ts.strftime("%Y-%m-%d"),
            )
        )
        window_id += 1
        start = start + relativedelta(months=step_months)

    return windows


def _get_is_returns(
    individual,
    train_fm: np.ndarray,
    train_eval_cfg: EvalConfig,
) -> np.ndarray:
    """Per-period IS portfolio returns, for the DSR.

    Uses ONLY train data — no OOS is touched here. Delegates to the same
    compute_signals/build_portfolio used by evaluate(), so the returns the DSR
    deflates are by construction those of the portfolio whose Sharpe it is
    deflating. This function previously re-derived the signal conversion, fee
    handling and Portfolio.from_signals call itself; the two copies did agree,
    but nothing kept them in step.

    Kept as a separate function so tests can patch it without running a GP tree.

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
        Per-period portfolio returns, shape [T_train].
    """
    from vgp.backtest.runner import (  # noqa: PLC0415 — deferred (D-15 pattern)
        build_portfolio,
        compute_signals,
    )

    signals = compute_signals(individual, train_fm)
    pf = build_portfolio(signals, train_eval_cfg)
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
        feature_matrix: np.ndarray,  # full [T x F x A] float32
        close_prices: pd.DataFrame,  # full [T x A] with DatetimeIndex
        base_eval_config: EvalConfig,
        seeds: list[int],
        evo_config_kwargs: dict,
        oos_min_trades: int | None = None,
        pool: object | None = None,
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
        oos_min_trades : int | None
            Minimum sign changes required for the OOS Sharpe to count as a
            measurement. Defaults to ``base_eval_config.min_trades`` scaled by
            the OOS/train length ratio, because the same trade RATE produces
            proportionally fewer trades in a shorter window — applying the
            train-window threshold (50) verbatim to a 3-month OOS window is
            mechanically unreachable and reports every strategy as invalid.
            Pass 0 to measure whatever the window produced.
        pool : multiprocessing.Pool | None
            A warm worker pool from `vgp.evolution.evolution_pool()`, forwarded
            to every seed's evolution. Pass one when running a grid: otherwise
            each (window, seed) pair creates and tears down its own pool and
            re-pays the numba JIT warmup, which for a modest search costs more
            than the parallelism returns.

        Returns
        -------
        list[dict]
            One row per seed. ``oos_sharpe`` is NaN — never the -inf
            worst-fitness sentinel — whenever ``oos_status != EVAL_OK``; the
            status and trade count say why. ``dsr`` is left as NaN and must be
            filled in by ``attach_dsr()`` once every window and seed has run,
            since the multiple-testing correction needs the whole trial set.
        """
        logger.info(
            "Window %d: train_end=%s test_start=%s test_end=%s n_seeds=%d",
            window.window_id,
            window.train_end,
            window.test_start,
            window.test_end,
            len(seeds),
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
        # Scale the OOS trade threshold to the OOS window length. min_trades is a
        # RATE requirement expressed for the train window; a 3-month OOS window has
        # ~1/4 the bars, so the unscaled 50 can be impossible to reach even for a
        # strategy trading exactly as often as it did in-sample.
        if oos_min_trades is None:
            T_train = int(train_fm.shape[0])
            T_test = int(test_fm.shape[0])
            ratio = (T_test / T_train) if T_train > 0 else 1.0
            oos_min_trades = max(1, int(round(base_eval_config.min_trades * ratio)))
            logger.info(
                "Window %d: OOS min_trades scaled %d -> %d (T_test=%d / T_train=%d)",
                window.window_id,
                base_eval_config.min_trades,
                oos_min_trades,
                T_test,
                T_train,
            )

        test_eval_cfg = EvalConfig(
            fee_bps=base_eval_config.fee_bps,
            min_trades=oos_min_trades,
            freq=base_eval_config.freq,
            init_cash=base_eval_config.init_cash,
            close_prices=test_close.copy(),
        )

        seed_results: list[dict] = []
        for seed in seeds:
            cfg = EvolutionConfig(seed=seed, **evo_config_kwargs)

            # --- Evolution on train data ONLY --- #
            pop, hof, logbook = run_evolution(cfg, train_fm, train_eval_cfg, pool=pool)

            if not hof:
                logger.warning(
                    "Window %d seed %d: HOF is empty — skipping OOS eval",
                    window.window_id,
                    seed,
                )
                continue

            best_ind = hof[0]

            # IS Sharpe: read from the individual's fitness tuple (index 0 = Sharpe).
            # Using hof[0].fitness.values[0] rather than the logbook population-max
            # because hof[0] is the specific individual evaluated OOS — the two can
            # diverge under multi-objective (NSGA-II) selection.
            # The trial accumulator rides on the logbook (run_evolution keeps its
            # 3-tuple signature). Absent only for a mocked or pre-accounting run.
            trial_acc = getattr(logbook, "trial_accumulator", None)
            if trial_acc is None:
                logger.warning(
                    "Window %d seed %d: no trial accumulator on the logbook — "
                    "this seed's evaluations will not size the DSR correction",
                    window.window_id,
                    seed,
                )

            # A non-finite value here is the worst-fitness sentinel, not a
            # measurement — record NaN so it cannot be averaged or plotted.
            is_sharpe = float(best_ind.fitness.values[0])
            if not np.isfinite(is_sharpe):
                logger.warning(
                    "Window %d seed %d: best individual carries worst-fitness IS Sharpe "
                    "(%s) — recording NaN, not a measurement",
                    window.window_id,
                    seed,
                    is_sharpe,
                )
                is_sharpe = float("nan")

            # --- OOS evaluate: called EXACTLY ONCE per (window, seed) --- #
            oos_fitness, oos_status, oos_n_trades = evaluate_with_status(
                best_ind, test_fm, test_eval_cfg
            )
            # test_fm is no longer referenced after this line

            if oos_status == EVAL_OK:
                oos_sharpe = float(oos_fitness[0])
            else:
                # evaluate() returns -inf to keep unusable individuals RANKABLE by
                # NSGA-II. That sentinel is not an OOS Sharpe of minus infinity, so
                # it must not be reported as one — NaN means "not measured".
                oos_sharpe = float("nan")
                logger.info(
                    "Window %d seed %d: OOS not measured (status=%s, n_trades=%d, "
                    "min_trades=%d)",
                    window.window_id,
                    seed,
                    oos_status,
                    oos_n_trades,
                    oos_min_trades,
                )

            # Per-period IS returns for DSR. Uses train data only (no OOS leakage).
            # DSR itself is deferred to attach_dsr(): the multiple-testing correction
            # needs sigma_SR across every trial in the experiment, which is not
            # knowable inside this loop. _get_is_returns() is a separate function so
            # tests can patch it without running an actual GP tree.
            try:
                is_returns = _get_is_returns(best_ind, train_fm, train_eval_cfg)
            except Exception as exc:  # pragma: no cover — only fires if vbt/eval fails
                logger.warning(
                    "Window %d seed %d: IS returns unavailable (%s) — DSR will be NaN",
                    window.window_id,
                    seed,
                    exc,
                )
                is_returns = None

            seed_results.append(
                {
                    "window_id": window.window_id,
                    "seed": seed,
                    "train_end": window.train_end,
                    "test_start": window.test_start,
                    "test_end": window.test_end,
                    "is_sharpe": is_sharpe,
                    "oos_sharpe": oos_sharpe,
                    "oos_status": oos_status,
                    "oos_n_trades": oos_n_trades,
                    "oos_min_trades": oos_min_trades,
                    "dsr": float("nan"),  # filled in by attach_dsr()
                    "n_nodes_best": len(best_ind),
                    # Every individual this seed evaluated is a trial for the
                    # multiple-testing correction; attach_dsr() merges these across
                    # seeds and windows. See vgp/trials.py.
                    "n_evaluations": trial_acc.n_evaluations if trial_acc else 0,
                    IS_RETURNS_KEY: is_returns,
                    TRIALS_KEY: trial_acc,
                }
            )

        return seed_results
