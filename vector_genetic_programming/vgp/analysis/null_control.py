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

Because the null dominates the runtime, share ONE warm worker pool
(`vgp.evolution.evolution_pool()`) across the observed run and every null run
by capturing it in the `experiment_fn` closure. A pool created per run re-pays
the numba JIT warmup each time, and there are `n_runs x n_windows` of them.
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

    Co-movement is preserved by STRATIFYING on the set of listed assets. The
    sample is split at each asset's listing date; within a stratum the alive set
    is constant, and one shared block sequence is drawn over dates where every
    alive asset has data. Assets that co-exist therefore always take their
    return from the same source date, at every point in the sample rather than
    only where all 27 overlap.

    That distinction is not academic. An earlier version drew a single shared
    sequence over the intersection of ALL assets and resampled everything
    outside it independently per asset. On the real Binance panel EUL lists
    2025-10-13, so the intersection was the final 171 days — after every
    training window — and effectively the entire usable history was drawn
    per-asset. Measured in the real training window: mean pairwise correlation
    0.615 against 0.001 in the surrogate, 2.4 effective bets against 20. A
    cross-sectional book with twenty independent bets instead of two reaches a
    far higher in-sample Sharpe, which is why the null control was unbeatable
    in-sample (p = 1.000) while out-of-sample behaved sensibly.

    The lesson generalises: a surrogate that preserves the UNCONDITIONAL
    correlation matrix can still destroy it within any particular window.
    Correlation is time-varying, and the window is what the GP trains on.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if not ohlcv:
        return {}

    tickers = list(ohlcv.keys())
    indices = {t: ohlcv[t].index for t in tickers}
    lengths = {t: len(indices[t]) for t in tickers}
    if max(lengths.values()) < 2:
        return {t: df.copy() for t, df in ohlcv.items()}

    # ---- Strata: periods over which the set of listed assets is constant ----
    # Draw ONE shared source sequence per stratum, over dates where every asset
    # alive in that stratum has data. Assets alive together therefore always
    # take their return from the same source date, which is what preserves
    # contemporaneous cross-asset correlation.
    #
    # An earlier version drew one shared sequence over the intersection of ALL
    # assets and resampled everything outside it per-asset independently. With
    # one very late listing that intersection collapses — on the 27-asset
    # Binance panel EUL lists 2025-10-13, leaving a 171-day intersection that
    # excludes every training window — so effectively the whole usable history
    # was drawn independently per asset. Measured effect: mean pairwise
    # correlation 0.615 in the real training window against 0.001 in the
    # surrogate, 2.4 effective bets against 20. That inflated the achievable
    # in-sample Sharpe and made the null control unbeatable in-sample.
    calendar = indices[tickers[0]]
    for t in tickers[1:]:
        calendar = calendar.union(indices[t])
    calendar = calendar.sort_values()

    starts = sorted({indices[t][0] for t in tickers})

    # src[t][i] = source return slot for asset t's output slot i (arrival at
    # indices[t][i + 1]). -1 until assigned.
    src = {t: np.full(max(lengths[t] - 1, 0), -1, dtype=np.int64) for t in tickers}
    degraded: list[str] = []

    for k, stratum_start in enumerate(starts):
        stratum_end = starts[k + 1] if k + 1 < len(starts) else None

        alive = [t for t in tickers if indices[t][0] <= stratum_start]
        if not alive:
            continue

        # Source region: dates from stratum_start on that EVERY alive asset has.
        # Intersecting guards the general case of differing end dates; for the
        # usual nested panel (staggered starts, common end) it is just the tail.
        region = calendar[calendar >= stratum_start]
        for t in alive:
            region = region.intersection(indices[t])
        region = region.sort_values()

        # Output dates this stratum covers, on the shared calendar.
        in_stratum = calendar >= stratum_start
        if stratum_end is not None:
            in_stratum &= calendar < stratum_end
        out_dates = calendar[in_stratum]
        if len(out_dates) == 0:
            continue

        if len(region) < 2:
            # No shared source window for this stratum: fall back per asset to
            # its own history so bars stay valid, and record the degradation.
            for t in alive:
                idx = indices[t]
                pos = idx.get_indexer(out_dates)
                pos = pos[pos >= 1]
                if pos.size == 0:
                    continue
                local = _circular_block_indices(pos.size, block_size, rng)
                src[t][pos - 1] = (pos - 1)[local % pos.size]
                if t not in degraded:
                    degraded.append(t)
            continue

        # ONE shared draw for the stratum, mapping each output date to a source
        # date. Every alive asset reads this same mapping, so on any surrogate
        # bar they all take their return from the same source date — which is
        # what preserves contemporaneous cross-asset correlation.
        shared = _circular_block_indices(len(out_dates), block_size, rng)
        shared = shared % (len(region) - 1)
        src_date_of = pd.Series(region[shared + 1].to_numpy(), index=out_dates)

        for t in alive:
            idx = indices[t]
            out_pos = idx.get_indexer(out_dates)
            keep = out_pos >= 1              # slot i - 1 arrives at idx[i]
            if not keep.any():
                continue
            tgt_pos = out_pos[keep]
            src_dates = src_date_of.to_numpy()[keep]
            src_pos = idx.get_indexer(pd.DatetimeIndex(src_dates))
            ok = src_pos >= 1
            src[t][tgt_pos[ok] - 1] = src_pos[ok] - 1

    if degraded:
        logger.warning(
            "block_bootstrap_ohlcv: %d asset(s) had a stratum with no shared "
            "source window (%s) — cross-asset co-movement is not preserved "
            "there and the null will be easier to beat than it should be",
            len(degraded), ", ".join(degraded[:5]),
        )

    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = ohlcv[ticker]
        index = indices[ticker]
        T = lengths[ticker]
        if T < 2:
            # Nothing to resample; pass it through so the asset still exists
            # and the retention decision downstream is unchanged.
            out[ticker] = df.copy()
            continue

        s_idx = src[ticker]
        # Any slot still unassigned (an asset absent from every stratum pass)
        # falls back to identity so the bar is at least valid.
        missing = s_idx < 0
        if missing.any():
            s_idx = s_idx.copy()
            s_idx[missing] = np.flatnonzero(missing)

        close = df["close"].to_numpy(dtype=np.float64)

        # Log returns, guarded against non-positive prices in the source data.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.diff(np.log(close))
        log_ret = np.nan_to_num(log_ret, nan=0.0, posinf=0.0, neginf=0.0)

        # Rebuild a price path from the resampled returns.
        new_close = np.empty(T, dtype=np.float64)
        new_close[0] = close[0] if np.isfinite(close[0]) and close[0] > 0 else 1.0
        new_close[1:] = new_close[0] * np.exp(np.cumsum(log_ret[s_idx]))

        # Bar geometry travels with the return that was drawn: each surrogate
        # bar reuses a real bar's open/high/low/volume RATIOS to its own close,
        # so the surrogate bars remain internally consistent (low <= close <=
        # high) instead of being synthesized.
        geom_src = np.empty(T, dtype=np.int64)
        geom_src[0] = 0
        geom_src[1:] = s_idx + 1

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
        "block_bootstrap_ohlcv: %d assets, %d strata, block_size=%d",
        len(tickers), len(starts), block_size,
    )
    return out


# ---------------------------------------------------------------------------
# Surrogate fidelity — checked WINDOW BY WINDOW
# ---------------------------------------------------------------------------

# Only statistics the construction actually GUARANTEES window-locally can be
# breach criteria. Drawing one shared source date per bar preserves the
# contemporaneous cross-sectional structure wherever that bar lands, so
# correlation and effective-bet count must match in every window — and they are
# exactly what silently broke: a surrogate whose assets are too independent
# hands the search more bets than the real data has, inflating achievable
# in-sample Sharpe and making the null unbeatable.
_FIDELITY_TOL = {
    "mean_corr": 0.30,     # |surrogate - observed| as a fraction of observed
    "n_eff_bets": 0.50,
}

# Reported for context but NOT breach criteria. A bootstrap resamples from the
# whole eligible pool, so any individual window's return scale legitimately
# differs from the observed window's — that is the surrogate having no signal,
# not a defect. Flagging it would train readers to ignore the check.
_FIDELITY_INFO = ("ew_vol", "mean_abs_ret")


def _window_stats(ohlcv: dict[str, pd.DataFrame], window: pd.DatetimeIndex) -> dict:
    """Cross-sectional and scale statistics of one window."""
    tickers = [t for t in sorted(ohlcv) if window.isin(ohlcv[t].index).all()]
    if len(tickers) < 2:
        return {}
    rets = np.column_stack([
        np.diff(np.log(ohlcv[t].loc[window, "close"].to_numpy(dtype=np.float64)))
        for t in tickers
    ])
    if rets.shape[0] < 3:
        return {}
    corr = np.corrcoef(rets, rowvar=False)
    if not np.isfinite(corr).all():
        return {}
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    eig = np.linalg.eigvalsh(corr)
    return {
        "mean_corr": float(off.mean()),
        "n_eff_bets": float((eig.sum() ** 2) / (eig ** 2).sum()),
        "ew_vol": float(rets.mean(axis=1).std() * np.sqrt(252)),
        "mean_abs_ret": float(np.abs(rets).mean()),
        "n_assets": len(tickers),
    }


def window_fidelity_report(
    observed: dict[str, pd.DataFrame],
    surrogate: dict[str, pd.DataFrame],
    n_windows: int = 8,
) -> list[dict]:
    """Compare observed and surrogate panels window by window.

    A surrogate can match the full sample perfectly and still be wrong inside
    every window — correlation is time-varying, and the window is what the
    model trains on. That is exactly how a surrogate with 0.001 mean pairwise
    correlation in the training window passed a full-sample check showing 69%
    PC1 share. So the comparison here is deliberately LOCAL: the shared
    calendar is cut into `n_windows` contiguous pieces and each is compared on
    its own.

    Returns one dict per window with the observed value, the surrogate value
    and the relative divergence for each statistic, plus a `breaches` list
    naming any statistic outside `_FIDELITY_TOL`.
    """
    tickers = sorted(set(observed) & set(surrogate))
    if not tickers:
        return []
    calendar = None
    for t in tickers:
        idx = observed[t].index
        calendar = idx if calendar is None else calendar.union(idx)
    calendar = calendar.sort_values()
    if len(calendar) < 4 * n_windows:
        n_windows = max(1, len(calendar) // 4)

    edges = np.linspace(0, len(calendar), n_windows + 1).astype(int)
    report: list[dict] = []
    for i in range(n_windows):
        window = calendar[edges[i]:edges[i + 1]]
        obs = _window_stats(observed, window)
        sur = _window_stats(surrogate, window)
        if not obs or not sur:
            continue
        row = {
            "window": f"{window[0].date()}..{window[-1].date()}",
            "n_assets": obs["n_assets"],
            "breaches": [],
        }
        for key in (*_FIDELITY_TOL, *_FIDELITY_INFO):
            o, v = obs[key], sur[key]
            denom = abs(o) if abs(o) > 1e-9 else 1.0
            rel = (v - o) / denom
            row[key] = {"observed": o, "surrogate": v, "rel": float(rel)}
            tol = _FIDELITY_TOL.get(key)
            if tol is not None and abs(rel) > tol:
                row["breaches"].append(key)
        report.append(row)
    return report


def check_surrogate_fidelity(
    observed: dict[str, pd.DataFrame],
    surrogate: dict[str, pd.DataFrame],
    n_windows: int = 8,
) -> list[dict]:
    """Run window_fidelity_report and log any breach loudly.

    Called on the first surrogate of a null control. A breach means the null is
    not the same problem as the observed run, so its p-value cannot be read at
    face value — most often because the surrogate is EASIER, which makes the
    test silently unbeatable rather than conservative.
    """
    report = window_fidelity_report(observed, surrogate, n_windows=n_windows)
    breached = [r for r in report if r["breaches"]]
    if not report:
        logger.warning(
            "check_surrogate_fidelity: no comparable windows — fidelity unverified"
        )
        return report
    if not breached:
        logger.info(
            "Surrogate fidelity OK across %d windows (cross-asset correlation "
            "and effective-bet count within tolerance in every window)",
            len(report),
        )
        return report

    logger.error(
        "SURROGATE FIDELITY BREACH in %d of %d windows — the null control is "
        "not the same problem as the observed run and its p-value is not "
        "trustworthy",
        len(breached), len(report),
    )
    for r in breached:
        for key in r["breaches"]:
            d = r[key]
            logger.error(
                "  %s  %s: observed %.4f vs surrogate %.4f (%+.0f%%)",
                r["window"], key, d["observed"], d["surrogate"], 100 * d["rel"],
            )
    return report


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
    fidelity: list[dict] = field(default_factory=list)

    @property
    def fidelity_breaches(self) -> list[str]:
        """Windows where the surrogate did not match the observed panel."""
        return [f"{r['window']}: {', '.join(r['breaches'])}"
                for r in self.fidelity if r.get("breaches")]

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
        if self.fidelity_breaches:
            lines.append(
                f"  FIDELITY BREACH in {len(self.fidelity_breaches)} window(s) — "
                f"the surrogate is not the same problem as the observed run, so "
                f"this p-value is not trustworthy:"
            )
            lines.extend(f"    {b}" for b in self.fidelity_breaches)
        elif self.fidelity:
            lines.append(
                f"  surrogate fidelity verified across {len(self.fidelity)} windows"
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
    fidelity: list[dict] = []

    for r in range(n_runs):
        rng = np.random.default_rng(seed + r)
        surrogate = block_bootstrap_ohlcv(ohlcv, rng, block_size=block_size)
        if r == 0:
            # Verify the surrogate is the same problem as the observed run
            # before spending 19 experiments comparing against it.
            fidelity = check_surrogate_fidelity(ohlcv, surrogate)
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
        fidelity=fidelity,
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
