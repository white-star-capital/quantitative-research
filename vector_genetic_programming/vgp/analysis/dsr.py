"""Deflated Sharpe Ratio (DSR) implementation and results reporting.

Reference: Bailey, D.H. & Lopez de Prado, M. (2014).
"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
Overfitting and Non-Normality." Journal of Portfolio Management, 40(5), 94-107.

Formula: DSR = Phi((SR_hat - E[SR_max]) / sqrt(Var(SR_hat)))

where, with N independent trials and gamma the Euler-Mascheroni constant,

    E[SR_max] = sigma_SR * [ (1 - gamma) * Phi^-1(1 - 1/N)
                             + gamma * Phi^-1(1 - 1/(N*e)) ]

The bracket is a dimensionless multiplier. `sigma_SR` — the cross-sectional
standard deviation of the Sharpe ratios ACROSS the N trials — is what gives
E[SR_max] its units. Dropping it makes the hurdle ~1.5 in per-period units
(roughly 24 annualized on daily data), which no real strategy clears, so every
DSR collapses to zero regardless of input. That is why compute_dsr() requires
the trial Sharpe ratios and not merely a trial count: DSR is a property of the
whole trial set, and cannot be computed for one trial in isolation.

All internal arithmetic is in PER-PERIOD units. Callers pass annualized Sharpe
ratios (as vectorbt reports them) and this module de-annualizes consistently.

INTERPRETIVE CAVEAT
-------------------
DSR corrects for selection ACROSS trials, and nothing else. Because the hurdle
is proportional to sigma_SR, a trial set whose Sharpe ratios all cluster tightly
at a high value produces a SMALL hurdle and therefore a DSR near 1.0 — the test
is saying "picking the best of these N draws does not explain this level", which
is not the same as "this is not overfit". Bias shared by every trial is invisible
to DSR: a common lookahead, one training window reused by all seeds, a survivor-
biased universe, or a fee assumption that is wrong for all of them. A high DSR
over few, tightly-clustered trials is weak evidence, not strong evidence. Read it
alongside n_trials and the spread of the trial Sharpes, both of which
attach_dsr() records.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

logger = logging.getLogger(__name__)

# Euler-Mascheroni constant
_EULER_GAMMA = 0.5772156649

# Key under which run_window() stashes per-period IS returns on each result row.
# attach_dsr() consumes and removes it; save_results_csv() drops any leftovers.
IS_RETURNS_KEY = "_is_returns"


def _expected_max_sr_multiplier(n_trials: int) -> float:
    """Dimensionless bracket of Bailey & Lopez de Prado (2014), Proposition 3.

    Multiply by sigma_SR (per-period) to obtain E[SR_max] in per-period units.
    Requires n_trials >= 2 — with a single trial there is no maximum to correct
    for and Phi^-1(1 - 1/1) = Phi^-1(0) is -inf.
    """
    if n_trials < 2:
        return float("nan")
    return float(
        (1 - _EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
        + _EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))
    )


def compute_dsr(
    returns: np.ndarray,
    sr_hat: float,
    trial_sharpes: np.ndarray | list[float],
    periods_per_year: int = 252,
) -> float:
    """Deflated Sharpe Ratio — probability that SR_hat exceeds E[max SR under H0].

    Parameters
    ----------
    returns : np.ndarray
        Per-period IS portfolio returns for THIS trial, shape [T].
        Use pf.returns().to_numpy().
    sr_hat : float
        Annualized IS Sharpe ratio of this trial (from the fitness tuple or
        pf.sharpe_ratio()). Must be finite — a worst-fitness sentinel (-inf)
        is not a measurement and yields NaN.
    trial_sharpes : np.ndarray | list[float]
        Annualized IS Sharpe ratios of ALL trials in the experiment
        (seeds x windows), including this one. Non-finite entries — e.g. the
        (-inf, -inf, -size) worst-fitness sentinel — are dropped before
        sigma_SR is estimated. At least 2 finite entries are required.
    periods_per_year : int
        252 for daily data. Used to de-annualize sr_hat and trial_sharpes.

    Returns
    -------
    float
        DSR in [0.0, 1.0]. Values above 0.95 indicate statistical significance.
        Returns 0.0 if returns are flat (std == 0) — genuinely no evidence of
        skill. Returns NaN when DSR is not computable (non-finite sr_hat, fewer
        than 2 finite trials, degenerate sigma_SR, too few observations);
        NaN means "not measured" and must not be read as "no skill".
    """
    if not np.isfinite(sr_hat):
        logger.debug("compute_dsr: non-finite sr_hat (%s) — returning NaN", sr_hat)
        return float("nan")

    returns = np.asarray(returns, dtype=np.float64)
    if np.std(returns) == 0.0:
        logger.debug("compute_dsr: flat returns (std=0) — returning 0.0")
        return 0.0

    T = len(returns)
    if T < 2:
        logger.debug("compute_dsr: too few observations (%d) — returning NaN", T)
        return float("nan")

    # --- sigma_SR: cross-sectional spread of trial Sharpes (per-period) --- #
    trials = np.asarray(trial_sharpes, dtype=np.float64).ravel()
    trials = trials[np.isfinite(trials)]
    n_trials = int(trials.size)
    if n_trials < 2:
        logger.debug(
            "compute_dsr: %d finite trial Sharpe(s) — sigma_SR undefined, returning NaN",
            n_trials,
        )
        return float("nan")

    sqrt_ppy = np.sqrt(periods_per_year)
    sigma_sr_pp = float(np.std(trials / sqrt_ppy, ddof=1))
    if not np.isfinite(sigma_sr_pp) or sigma_sr_pp <= 0.0:
        logger.debug(
            "compute_dsr: degenerate sigma_SR (%s) across %d trials — returning NaN",
            sigma_sr_pp, n_trials,
        )
        return float("nan")

    multiplier = _expected_max_sr_multiplier(n_trials)
    if not np.isfinite(multiplier):
        logger.debug("compute_dsr: bad E[SR_max] multiplier for n_trials=%d", n_trials)
        return float("nan")

    # E[SR_max] under H0, in per-period units (Proposition 3)
    expected_max_sr = sigma_sr_pp * multiplier

    # De-annualize: the formula requires per-period SR (Bailey & Lopez de Prado 2014, eq. 5)
    sr_hat_pp = sr_hat / sqrt_ppy

    # Return distribution moments
    ret_skew = float(skew(returns))
    # kurtosis(fisher=True) returns EXCESS kurtosis — matches DSR formula
    ret_kurt = float(kurtosis(returns, fisher=True))

    # Variance of the SR estimate (non-normality adjustment, per-period units).
    # Equivalent to (1 - skew*SR + (kurt_nonexcess - 1)/4 * SR^2) / (T - 1).
    sr_var = (
        1 + (0.5 * sr_hat_pp**2) - ret_skew * sr_hat_pp + ((ret_kurt / 4) * sr_hat_pp**2)
    ) / (T - 1)

    if sr_var <= 0.0:
        logger.debug("compute_dsr: non-positive sr_var (%f) — returning NaN", sr_var)
        return float("nan")

    dsr = float(norm.cdf((sr_hat_pp - expected_max_sr) / np.sqrt(sr_var)))
    return float(np.clip(dsr, 0.0, 1.0))


def attach_dsr(
    results: list[dict],
    periods_per_year: int = 252,
) -> list[dict]:
    """Fill in the ``dsr`` field on every result row, in place.

    DSR needs sigma_SR across the whole trial set, so it cannot be computed
    inside the per-seed loop — run_window() leaves ``dsr`` as NaN and stashes
    per-period IS returns under ``IS_RETURNS_KEY``. Call this once after all
    windows and seeds have run.

    Rows whose IS Sharpe is non-finite (worst-fitness sentinel) are excluded
    from the sigma_SR estimate and receive ``dsr = NaN``.

    Parameters
    ----------
    results : list[dict]
        All result rows from every run_window() call in the experiment.
        Mutated in place: ``dsr`` and ``dsr_n_trials`` are set, and
        ``IS_RETURNS_KEY`` is removed.
    periods_per_year : int
        252 for daily data.

    Returns
    -------
    list[dict]
        The same list, for convenient chaining.
    """
    if not results:
        return results

    trial_sharpes = np.array(
        [float(r.get("is_sharpe", np.nan)) for r in results], dtype=np.float64
    )
    n_finite = int(np.sum(np.isfinite(trial_sharpes)))

    finite = trial_sharpes[np.isfinite(trial_sharpes)]
    trial_sr_std = float(np.std(finite, ddof=1)) if n_finite >= 2 else float("nan")

    for row in results:
        is_returns = row.pop(IS_RETURNS_KEY, None)
        row["dsr_n_trials"] = n_finite
        # Annualized spread of trial Sharpes — the scale of the DSR hurdle.
        # A tiny value with few trials means a low hurdle and a DSR that should
        # not be read as strong evidence (see module docstring).
        row["dsr_trial_sr_std"] = trial_sr_std
        if is_returns is None:
            row["dsr"] = float("nan")
            continue
        row["dsr"] = compute_dsr(
            is_returns,
            sr_hat=float(row.get("is_sharpe", np.nan)),
            trial_sharpes=trial_sharpes,
            periods_per_year=periods_per_year,
        )

    logger.info(
        "attach_dsr: computed DSR for %d rows using %d finite trial Sharpe(s) "
        "(annualized trial SR std = %s)",
        len(results), n_finite, trial_sr_std,
    )
    if n_finite < 10:
        logger.warning(
            "attach_dsr: only %d finite trial(s) — the multiple-testing correction "
            "is weak at this sample size; DSR should not be read as strong evidence",
            n_finite,
        )
    return results


def aggregate_seeds(seed_results: list[dict]) -> dict:
    """Aggregate per-seed results for one window into summary statistics.

    NaN-safe: an OOS Sharpe of NaN means "not measured" (e.g. the OOS trade
    filter tripped) and is excluded from the median, IQR and positive count
    rather than being treated as a bad result. Rows are never summarized with
    a worst-fitness sentinel in them — see run_window().

    Parameters
    ----------
    seed_results : list[dict]
        Each dict has at minimum: oos_sharpe, dsr.

    Returns
    -------
    dict
        Keys: median_oos_sharpe, iqr_oos_sharpe, median_dsr,
        n_seeds_positive_oos, n_seeds_valid_oos, n_seeds.
        Medians and IQR are NaN when no seed produced a valid measurement.
    """
    empty = {
        "median_oos_sharpe": float("nan"),
        "iqr_oos_sharpe": float("nan"),
        "median_dsr": float("nan"),
        "n_seeds_positive_oos": 0,
        "n_seeds_valid_oos": 0,
        "n_seeds": len(seed_results),
    }
    if not seed_results:
        return empty

    oos_sharpes = np.array(
        [float(r.get("oos_sharpe", np.nan)) for r in seed_results], dtype=np.float64
    )
    dsrs = np.array([float(r.get("dsr", np.nan)) for r in seed_results], dtype=np.float64)

    valid_oos = oos_sharpes[np.isfinite(oos_sharpes)]
    valid_dsr = dsrs[np.isfinite(dsrs)]

    if valid_oos.size == 0:
        logger.warning(
            "aggregate_seeds: no seed produced a valid OOS Sharpe (%d rows) — "
            "medians are NaN",
            len(seed_results),
        )
        out = dict(empty)
        out["median_dsr"] = float(np.median(valid_dsr)) if valid_dsr.size else float("nan")
        return out

    q25 = float(np.percentile(valid_oos, 25))
    q75 = float(np.percentile(valid_oos, 75))

    return {
        "median_oos_sharpe": float(np.median(valid_oos)),
        "iqr_oos_sharpe": float(q75 - q25),
        "median_dsr": float(np.median(valid_dsr)) if valid_dsr.size else float("nan"),
        "n_seeds_positive_oos": int(np.sum(valid_oos > 0)),
        "n_seeds_valid_oos": int(valid_oos.size),
        "n_seeds": len(seed_results),
    }


def save_results_csv(results: list[dict], path: str) -> None:
    """Save per-(window, seed) results to CSV.

    Keys prefixed with ``_`` are internal scratch (e.g. IS_RETURNS_KEY holds a
    per-period returns array) and are dropped rather than written.

    Parameters
    ----------
    results : list[dict]
        Each dict is one (window_id, seed) pair with keys:
        window_id, seed, train_end, test_start, test_end,
        is_sharpe, oos_sharpe, oos_status, oos_n_trades, oos_min_trades,
        dsr, dsr_n_trials, dsr_trial_sr_std, n_nodes_best
    path : str
        Output file path. Parent directory must exist.
    """
    public = [
        {k: v for k, v in row.items() if not k.startswith("_")} for row in results
    ]
    df = pd.DataFrame(public)

    # A -inf here would mean a worst-fitness sentinel leaked into reporting.
    for col in ("is_sharpe", "oos_sharpe", "dsr"):
        if col in df.columns:
            n_inf = int(np.isinf(pd.to_numeric(df[col], errors="coerce")).sum())
            if n_inf:
                logger.error(
                    "save_results_csv: %d infinite value(s) in '%s' — a worst-fitness "
                    "sentinel leaked into reporting; these are not measurements",
                    n_inf, col,
                )

    df.to_csv(path, index=False)
    logger.info("Results saved to %s (%d rows)", path, len(df))
