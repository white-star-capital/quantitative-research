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

from vgp.trials import TrialAccumulator, TrialSet

logger = logging.getLogger(__name__)

# Euler-Mascheroni constant
_EULER_GAMMA = 0.5772156649

# Key under which run_window() stashes per-period IS returns on each result row.
# attach_dsr() consumes and removes it; save_results_csv() drops any leftovers.
IS_RETURNS_KEY = "_is_returns"

# Key under which run_window() stashes that seed's TrialAccumulator (every
# individual the GP evaluated). attach_dsr() merges these across rows to size
# the multiple-testing correction, then removes the key.
TRIALS_KEY = "_trial_accumulator"


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
    trial_sharpes: TrialSet | np.ndarray | list[float],
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
    trial_sharpes : TrialSet | np.ndarray | list[float]
        The trial set this Sharpe is being deflated against. Either a
        `vgp.trials.TrialSet` (n_trials + annualized sr_std, as produced by the
        streaming accumulator over every evaluated individual) or a raw array of
        annualized trial Sharpe ratios, from which both are derived.
        Non-finite entries — e.g. the (-inf, -inf, -size) worst-fitness
        sentinel — are dropped; at least 2 finite trials are required.

        A GP evaluates thousands of individuals, and each is a trial. Passing
        only the per-seed winners understates N by orders of magnitude and
        leaves the correction nearly inert — see vgp/trials.py.
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
    trial_set = (
        trial_sharpes
        if isinstance(trial_sharpes, TrialSet)
        else TrialSet.from_sharpes(trial_sharpes)
    )
    if not trial_set.is_usable:
        logger.debug(
            "compute_dsr: unusable trial set (n=%d, sr_std=%s, label=%s) — "
            "returning NaN",
            trial_set.n_trials, trial_set.sr_std, trial_set.label,
        )
        return float("nan")

    sqrt_ppy = np.sqrt(periods_per_year)
    sigma_sr_pp = trial_set.sr_std / sqrt_ppy

    multiplier = _expected_max_sr_multiplier(trial_set.n_trials)
    if not np.isfinite(multiplier):
        logger.debug(
            "compute_dsr: bad E[SR_max] multiplier for n_trials=%d",
            trial_set.n_trials,
        )
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
    """Fill in the DSR fields on every result row, in place.

    DSR needs the trial set across the whole experiment, so it cannot be
    computed inside the per-seed loop — run_window() leaves ``dsr`` as NaN and
    stashes both the per-period IS returns (``IS_RETURNS_KEY``) and that seed's
    trial accumulator (``TRIALS_KEY``). Call this once after all windows and
    seeds have run.

    TWO BOUNDS ARE REPORTED, because the honest answer is an interval:

    ``dsr`` deflates against every individual the GP evaluated — thousands of
    trials. This is the primary figure. It is conservative, because
    Proposition 3 assumes independent trials while GP individuals are
    correlated by descent, so the effective N is below the raw count.

    ``dsr_bests_only`` deflates against just the per-seed winners that appear in
    the results table — as many trials as there are rows. This is an upper bound
    and on its own it is close to meaningless: with a handful of trials the
    hurdle multiplier is small, so this figure flatters the strategy. It is
    reported so the gap between the two is visible rather than hidden by the
    choice of one convention.

    If the two bracket 0.95, the experiment has not settled the question.

    Parameters
    ----------
    results : list[dict]
        All result rows from every run_window() call in the experiment.
        Mutated in place: the ``dsr*`` fields are set, and ``IS_RETURNS_KEY``
        and ``TRIALS_KEY`` are removed.
    periods_per_year : int
        252 for daily data.

    Returns
    -------
    list[dict]
        The same list, for convenient chaining.
    """
    if not results:
        return results

    # --- trial set 1: every individual evaluated, merged across seeds/windows ---
    evaluated = TrialAccumulator()
    n_rows_with_trials = 0
    for row in results:
        acc = row.pop(TRIALS_KEY, None)
        if acc is not None:
            evaluated.merge(acc)
            n_rows_with_trials += 1

    # --- trial set 2: the reported winners only ---
    winner_sharpes = np.array(
        [float(r.get("is_sharpe", np.nan)) for r in results], dtype=np.float64
    )
    bests_set = TrialSet.from_sharpes(winner_sharpes, label="reported_bests")

    if n_rows_with_trials:
        evaluated_set = evaluated.to_trial_set(label="all_evaluations")
    else:
        # No accumulator reached us (a mocked run, or checkpoints predating
        # trial accounting). Fall back to the winners and say so, rather than
        # silently reporting a correction sized to the wrong trial population.
        evaluated_set = bests_set
        logger.warning(
            "attach_dsr: no trial accumulators found on %d row(s) — falling back "
            "to the %d reported best(s) as the trial set. The multiple-testing "
            "correction is then sized to the winners, not to the search, and "
            "understates N by orders of magnitude",
            len(results), bests_set.n_trials,
        )

    for row in results:
        is_returns = row.pop(IS_RETURNS_KEY, None)
        sr_hat = float(row.get("is_sharpe", np.nan))

        row["dsr_n_trials"] = evaluated_set.n_trials
        row["dsr_trial_sr_std"] = evaluated_set.sr_std
        row["dsr_trial_source"] = evaluated_set.label
        row["dsr_n_evaluations"] = evaluated.n_evaluations if n_rows_with_trials else 0
        row["dsr_n_trials_bests"] = bests_set.n_trials
        row["dsr_trial_sr_std_bests"] = bests_set.sr_std

        if is_returns is None:
            row["dsr"] = float("nan")
            row["dsr_bests_only"] = float("nan")
            continue

        row["dsr"] = compute_dsr(
            is_returns, sr_hat=sr_hat,
            trial_sharpes=evaluated_set, periods_per_year=periods_per_year,
        )
        row["dsr_bests_only"] = compute_dsr(
            is_returns, sr_hat=sr_hat,
            trial_sharpes=bests_set, periods_per_year=periods_per_year,
        )

    logger.info(
        "attach_dsr: %d rows | primary trial set '%s' n=%d sr_std=%.4f "
        "(%d evaluations) | bests-only n=%d sr_std=%.4f",
        len(results), evaluated_set.label, evaluated_set.n_trials,
        evaluated_set.sr_std, evaluated.n_evaluations,
        bests_set.n_trials, bests_set.sr_std,
    )

    finite_primary = [r["dsr"] for r in results if np.isfinite(r.get("dsr", np.nan))]
    finite_bests = [
        r["dsr_bests_only"] for r in results
        if np.isfinite(r.get("dsr_bests_only", np.nan))
    ]
    if finite_primary and finite_bests:
        lo, hi = max(finite_primary), max(finite_bests)
        if lo < 0.95 <= hi:
            logger.warning(
                "attach_dsr: the two trial-set conventions straddle 0.95 "
                "(best dsr=%.4f, best dsr_bests_only=%.4f) — significance is "
                "an artifact of how trials are counted, not a result",
                lo, hi,
            )

    if evaluated_set.n_trials < 100:
        logger.warning(
            "attach_dsr: only %d finite trial(s) in the primary set — the "
            "multiple-testing correction is weak at this sample size; DSR "
            "should not be read as strong evidence",
            evaluated_set.n_trials,
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
        dsr, dsr_bests_only, dsr_n_trials, dsr_trial_sr_std, dsr_trial_source,
        dsr_n_evaluations, n_evaluations, n_nodes_best, and — when stamped by
        UniverseRecord.stamp_rows() — n_assets and universe_fingerprint
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
