"""Null control: does the pipeline find the same "alpha" in data with no signal?

WHY THIS EXISTS
---------------
The Deflated Sharpe Ratio corrects for selection across trials and nothing
else. Bias shared by every trial is invisible to it, because such bias shifts
all trials together and leaves sigma_SR — the spread the hurdle is built from —
unchanged. Lookahead in a primitive, one training window reused by every seed,
a survivor-biased universe, a fee model that is wrong for all of them: each
inflates every trial equally, and DSR reports significance anyway.

The only way to catch that class of error is to run the identical pipeline on
data where the answer is known to be "nothing here", and check that it reports
nothing. If evolving on signal-free surrogates produces Sharpe ratios as good
as the real run, the pipeline is measuring its own construction rather than the
market — and that verdict holds regardless of what DSR says.

This is a falsification test, not a performance metric. A real result must
clear the null distribution, not merely clear zero.

WHAT THE SURROGATE PRESERVES
----------------------------
`block_bootstrap_ohlcv()` resamples circular blocks of log returns, drawing one
shared block sequence over the dates all assets have in common and mapping it
through each asset's own index, then rebuilds bars from them. Preserved: each
asset's return distribution (so volatility, fat tails and skew are realistic),
the cross-asset correlation structure over that common window — every asset
takes its return from the same source date, so a market factor still exists —
within-block autocorrelation and volatility clustering, and the bar geometry:
open/high/low and volume ride along as ratios to their own close, so every
surrogate bar is a real bar's shape.

Destroyed: the specific ordering that makes returns predictable from any signal
computed on earlier bars, at horizons longer than the block. That is exactly the
thing the GP is supposed to be discovering.

Block size is the knob that matters, and it is not a free parameter. A block
bootstrap severs serial dependence only at block boundaries, so with L-bar
blocks roughly 1 in L transitions is broken and structure shorter than L
survives into the surrogate largely intact — measurably so: an AR(1) return
series with lag-1 autocorrelation 0.82 retains 0.67 under 5-bar blocks and
essentially all of it under 50-bar blocks.

The consequence is concrete: **the block must be shorter than the horizon of
the effect being tested.** With the default 20 bars, a strategy exploiting
1-5 day momentum survives into the null and the control will NOT flag it. Set
`block_size` below the holding period you are testing; `block_size=1` is a full
iid shuffle, which destroys everything including genuine short-horizon
autocorrelation and makes the null too easy to beat. The 20-bar default is
deliberately conservative on the "too hard" side — it will not manufacture
significance, but it can miss short-horizon bias.

COST
----
A null control costs `n_runs` times a full experiment. That is the price of the
claim; a cheap significance test that cannot detect shared bias is not a
substitute. Use fewer seeds per null run than the real run if you must — the
statistic is the BEST Sharpe the search finds, and the null only needs to
represent the same search procedure, not the same wall-clock budget.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _circular_block_indices(
    n: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """`n` source positions drawn as circular blocks of `block_size` from [0, n).

    Wrapping at the end (rather than truncating) keeps every position equally
    likely to be drawn, which is what makes the surrogate's marginal
    distribution match the original's.
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = (starts[:, None] + np.arange(block_size)[None, :]) % n
    return offsets.reshape(-1)[:n].astype(np.int64)

# One experiment: features + close + dates in, result rows out.
ExperimentFn = Callable[[np.ndarray, pd.DataFrame, pd.DatetimeIndex], list[dict]]


# ---------------------------------------------------------------------------
# Surrogate data
# ---------------------------------------------------------------------------

def block_bootstrap_ohlcv(
    ohlcv: dict[str, pd.DataFrame],
    rng: np.random.Generator,
    block_size: int = 20,
) -> dict[str, pd.DataFrame]:
    """Signal-free surrogate OHLCV via circular block bootstrap of log returns.

    Parameters
    ----------
    ohlcv : dict[str, pd.DataFrame]
        Real data: ticker -> DataFrame with columns [open, high, low, close,
        volume] on a DatetimeIndex, as produced by DataLoader.
    rng : np.random.Generator
        Source of randomness. Pass a seeded generator for reproducibility.
    block_size : int
        Length of each resampled block in bars. Must be >= 1. See the module
        docstring on choosing it.

    Returns
    -------
    dict[str, pd.DataFrame]
        Surrogates with the same tickers, index and columns as the input. Each
        asset starts at its real first close, so price levels stay plausible.

    Notes
    -----
    The same block offsets are applied to every asset, which is what preserves
    the cross-sectional correlation structure. Drawing per-asset blocks
    independently would destroy the market factor and produce a null that is
    far too easy to beat.

    RAGGED PANELS. Assets are not required to share an index length —
    staggered listing dates are the normal case in crypto, and the fetcher
    returns them as-is. Each asset keeps its OWN index and row count in the
    surrogate, which matters because FeatureEngine's min_obs_fraction filter
    then makes the same retention decision on the surrogate as on the real
    data: the null run is computed on the same universe as the observed run,
    which is what makes the comparison meaningful.

    Co-movement is preserved EXACTLY over the intersection window — the dates
    every asset has in common. That is the span that reaches the GP, because
    FeatureEngine intersects to a common date index before stacking, so getting
    it exact there is what counts. Concretely: one shared block sequence is
    drawn over the intersection, and every asset maps it through its own date
    index, so on any given surrogate bar all assets take their return from the
    same source date.

    Each asset's history BEFORE the intersection (only the longer-listed assets
    have any) is resampled independently from its own pre-intersection returns.
    Those bars only feed the rolling-window warm-up that the lookback trim
    discards, and there is nothing to co-move with there — the shorter assets
    do not exist yet.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if not ohlcv:
        return {}

    tickers = list(ohlcv.keys())
    lengths = {t: len(ohlcv[t]) for t in tickers}
    if max(lengths.values()) < 2:
        return {t: df.copy() for t, df in ohlcv.items()}

    # Intersection of all asset dates — the span that survives FeatureEngine.
    common: pd.DatetimeIndex | None = None
    for t in tickers:
        idx = ohlcv[t].index
        common = idx if common is None else common.intersection(idx)
    common = common.sort_values() if common is not None else pd.DatetimeIndex([])

    shared_src: np.ndarray | None = None
    if len(common) >= 2:
        shared_src = _circular_block_indices(len(common) - 1, block_size, rng)
    else:
        logger.warning(
            "block_bootstrap_ohlcv: assets share %d common date(s) — no joint "
            "window, so each asset is resampled independently and cross-asset "
            "co-movement is NOT preserved. The null will be easier to beat "
            "than it should be; check the universe's date alignment.",
            len(common),
        )

    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = ohlcv[ticker]
        index = df.index
        T = lengths[ticker]
        if T < 2:
            # Nothing to resample; pass it through so the asset still exists
            # and the retention decision downstream is unchanged.
            out[ticker] = df.copy()
            continue

        # src[i] is the source return slot for output return slot i, where slot
        # i is the return arriving at index[i + 1].
        src = np.full(T - 1, -1, dtype=np.int64)

        if shared_src is not None:
            # Map the shared sequence through this asset's own dates. Every
            # common date is present in this asset by construction, so the
            # lookup cannot miss.
            pos = index.get_indexer(common)
            target_slots = pos[1:] - 1                     # arrival at common[j+1]
            source_slots = pos[shared_src + 1] - 1         # arrival at common[src+1]
            src[target_slots] = source_slots

        # Slots the shared sequence did not cover: this asset's pre-intersection
        # history. Resample from among themselves so the returns still come from
        # the same regime, rather than importing intersection-window returns.
        uncovered = np.flatnonzero(src < 0)
        if uncovered.size:
            local = _circular_block_indices(uncovered.size, block_size, rng)
            src[uncovered] = uncovered[local]

        close = df["close"].to_numpy(dtype=np.float64)

        # Log returns, guarded against non-positive prices in the source data.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.diff(np.log(close))
        log_ret = np.nan_to_num(log_ret, nan=0.0, posinf=0.0, neginf=0.0)

        # Rebuild a price path from the resampled returns.
        new_close = np.empty(T, dtype=np.float64)
        new_close[0] = close[0] if np.isfinite(close[0]) and close[0] > 0 else 1.0
        new_close[1:] = new_close[0] * np.exp(np.cumsum(log_ret[src]))

        # Bar geometry travels with the return that was drawn: each surrogate
        # bar reuses a real bar's open/high/low/volume RATIOS to its own close,
        # so the surrogate bars remain internally consistent (low <= close <=
        # high) instead of being synthesized.
        # Return slot i came from source slot src[i], i.e. source bar src[i]+1.
        geom_src = np.empty(T, dtype=np.int64)
        geom_src[0] = 0
        geom_src[1:] = src + 1

        surrogate = {"close": new_close}
        for col in ("open", "high", "low"):
            if col not in df.columns:
                continue
            ratio = df[col].to_numpy(dtype=np.float64) / np.where(close > 0, close, np.nan)
            ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
            surrogate[col] = new_close * ratio[geom_src]
        if "volume" in df.columns:
            vol = df["volume"].to_numpy(dtype=np.float64)
            surrogate["volume"] = vol[geom_src]

        cols = [c for c in _OHLCV_COLUMNS if c in surrogate]
        out[ticker] = pd.DataFrame(
            {c: surrogate[c] for c in cols}, index=index
        )

    logger.debug(
        "block_bootstrap_ohlcv: %d assets, %d common dates, block_size=%d",
        len(tickers), len(common), block_size,
    )
    return out


# ---------------------------------------------------------------------------
# Empirical p-value
# ---------------------------------------------------------------------------

def empirical_p_value(observed: float, null_samples: Sequence[float]) -> float:
    """One-sided empirical p-value: P(null >= observed).

    Uses the (1 + count) / (1 + n) form (Davison & Hinkley 1997), which never
    returns exactly 0 — with n surrogate runs the smallest reportable p-value is
    1 / (1 + n), and claiming anything smaller would assert more resolution than
    the number of runs supports.

    Returns NaN if `observed` is not finite or no finite null sample exists;
    NaN means "not measured", never "not significant".
    """
    null = np.asarray(list(null_samples), dtype=np.float64).ravel()
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or null.size == 0:
        return float("nan")
    n_ge = int(np.sum(null >= observed))
    return float((1 + n_ge) / (1 + null.size))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class NullControlResult:
    """Outcome of a null control run.

    `p_value_is` is the headline number: the fraction of signal-free runs whose
    best in-sample Sharpe matched or beat the real run's. Large means the search
    finds the same thing in noise.
    """

    n_runs: int
    block_size: int
    observed_best_is_sharpe: float
    observed_best_oos_sharpe: float
    null_best_is_sharpe: np.ndarray = field(default_factory=lambda: np.array([]))
    null_best_oos_sharpe: np.ndarray = field(default_factory=lambda: np.array([]))
    n_runs_failed: int = 0

    @property
    def p_value_is(self) -> float:
        return empirical_p_value(self.observed_best_is_sharpe, self.null_best_is_sharpe)

    @property
    def p_value_oos(self) -> float:
        return empirical_p_value(self.observed_best_oos_sharpe, self.null_best_oos_sharpe)

    @property
    def resolution(self) -> float:
        """Smallest p-value this many runs can express."""
        n = int(np.sum(np.isfinite(self.null_best_is_sharpe)))
        return 1.0 / (1 + n) if n else float("nan")

    def summary(self) -> str:
        """Human-readable verdict, honest about what the run count supports."""
        def _pct(a: np.ndarray, q: float) -> float:
            a = a[np.isfinite(a)]
            return float(np.percentile(a, q)) if a.size else float("nan")

        lines = [
            f"Null control: {self.n_runs} signal-free run(s), "
            f"block_size={self.block_size} bars"
            + (f", {self.n_runs_failed} failed" if self.n_runs_failed else ""),
            f"  observed best IS Sharpe   {self.observed_best_is_sharpe:+.3f}"
            f"   null median {_pct(self.null_best_is_sharpe, 50):+.3f}"
            f"   null p95 {_pct(self.null_best_is_sharpe, 95):+.3f}"
            f"   p = {self.p_value_is:.3f}",
            f"  observed best OOS Sharpe  {self.observed_best_oos_sharpe:+.3f}"
            f"   null median {_pct(self.null_best_oos_sharpe, 50):+.3f}"
            f"   null p95 {_pct(self.null_best_oos_sharpe, 95):+.3f}"
            f"   p = {self.p_value_oos:.3f}",
        ]
        p_is = self.p_value_is
        if not np.isfinite(p_is):
            lines.append("  VERDICT: not measured — no usable null runs.")
        elif p_is > 0.10:
            lines.append(
                "  VERDICT: the search finds comparable performance in data with "
                "no signal. Treat the headline result as unproven regardless of DSR."
            )
        elif p_is > 0.05:
            lines.append("  VERDICT: marginal — the null is not clearly beaten.")
        else:
            lines.append(
                f"  VERDICT: the observed result is outside the null distribution "
                f"(p <= {max(p_is, self.resolution):.3f}). This rules out bias "
                f"shared by all trials; it does not by itself establish "
                f"tradeable alpha."
            )
        if np.isfinite(self.resolution) and self.resolution > 0.05:
            lines.append(
                f"  NOTE: {self.n_runs - self.n_runs_failed} usable run(s) cannot "
                f"resolve p below {self.resolution:.3f} — too few to claim "
                f"significance at the 5% level whatever the outcome."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def best_sharpes(results: list[dict]) -> tuple[float, float]:
    """Best measurable IS and OOS Sharpe across result rows.

    Non-finite entries mean "not measured" (a worst-fitness sentinel, or a
    tripped OOS trade filter) and are skipped rather than treated as bad
    results. Returns NaN for a field with no measurable row.
    """
    def _best(key: str) -> float:
        vals = [
            float(r[key]) for r in results
            if key in r and np.isfinite(float(r.get(key, np.nan)))
        ]
        return max(vals) if vals else float("nan")

    return _best("is_sharpe"), _best("oos_sharpe")


def run_null_control(
    ohlcv: dict[str, pd.DataFrame],
    experiment_fn: ExperimentFn,
    observed_results: list[dict],
    n_runs: int = 10,
    block_size: int = 20,
    seed: int = 0,
    feature_builder: Callable[[dict[str, pd.DataFrame]], tuple] | None = None,
) -> NullControlResult:
    """Run the same experiment on `n_runs` signal-free surrogates.

    Parameters
    ----------
    ohlcv : dict[str, pd.DataFrame]
        The REAL data. Surrogates are bootstrapped from it so they inherit its
        distributional properties.
    experiment_fn : callable
        `(feature_matrix, close_prices, dates) -> list[dict]` — runs one full
        experiment and returns result rows with `is_sharpe` / `oos_sharpe`.
        Injected rather than constructed here so the null run uses exactly the
        same code path as the real run, and so tests can substitute a stub.
    observed_results : list[dict]
        Result rows from the real run, to extract the observed statistics.
    n_runs : int
        Number of surrogate experiments. Cost is n_runs x one experiment.
        Note that p-values cannot resolve below 1/(1+n_runs).
    block_size : int
        Bootstrap block length in bars.
    seed : int
        Base seed; surrogate r uses seed + r so runs are independent and the
        whole control is reproducible.
    feature_builder : callable, optional
        `ohlcv -> (feature_matrix, close_prices, dates)`. Defaults to the
        standard FeatureEngine path. Surrogates MUST go through the same
        feature construction as the real run, or the comparison is invalid.

    Returns
    -------
    NullControlResult
    """
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}")

    if feature_builder is None:
        feature_builder = _default_feature_builder

    obs_is, obs_oos = best_sharpes(observed_results)

    null_is: list[float] = []
    null_oos: list[float] = []
    n_failed = 0

    for r in range(n_runs):
        rng = np.random.default_rng(seed + r)
        surrogate = block_bootstrap_ohlcv(ohlcv, rng, block_size=block_size)
        try:
            fm, close, dates = feature_builder(surrogate)
            rows = experiment_fn(fm, close, dates)
        except Exception as exc:
            # One bad surrogate must not abandon the control; it is recorded so
            # the reported run count stays honest.
            logger.warning("Null run %d/%d failed (%s) — skipping", r + 1, n_runs, exc)
            n_failed += 1
            continue

        b_is, b_oos = best_sharpes(rows)
        null_is.append(b_is)
        null_oos.append(b_oos)
        logger.info(
            "Null run %d/%d: best IS Sharpe %+.3f, best OOS Sharpe %+.3f",
            r + 1, n_runs, b_is, b_oos,
        )

    return NullControlResult(
        n_runs=n_runs,
        block_size=block_size,
        observed_best_is_sharpe=obs_is,
        observed_best_oos_sharpe=obs_oos,
        null_best_is_sharpe=np.asarray(null_is, dtype=np.float64),
        null_best_oos_sharpe=np.asarray(null_oos, dtype=np.float64),
        n_runs_failed=n_failed,
    )


def _default_feature_builder(
    ohlcv: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, pd.DataFrame, pd.DatetimeIndex]:
    """Standard FeatureEngine path — identical to what scripts/run.py does."""
    from vgp.data import FeatureEngine  # noqa: PLC0415 — deferred, keeps imports light

    fe = FeatureEngine()
    fm = fe.fit_transform(ohlcv)
    close_prices = pd.DataFrame(
        {ticker: ohlcv[ticker]["close"] for ticker in fe.retained_assets_}
    ).reindex(fe.dates_).ffill(limit=3)
    return fm, close_prices, fe.dates_
