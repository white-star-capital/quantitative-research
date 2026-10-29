"""Realized variance and its HAR forecast (RV_FORECAST_IMPLEMENTATION_GUIDE.md §3, §6, §9, §12).

Target: y_d = ln RV_d, RV_d = Σ r² of log close-to-close bar returns over UTC day d (§3). Model
(VRP-A-1): one direct OLS per horizon h = 1..H,

    y_{t+h} = b0 + Σ_l b_l · mean(y_{t−l+1..t}) + c · cal_{t+h} + ε_h,   l ∈ (1, 5, 22)

where cal holds the calendar columns of the *target* day (day-of-week and event-window dummies,
known in advance). The cone uses empirical quantiles of each horizon's in-sample residuals (§12.2).

Missing days are NaN and are never filled. A lag mean averages the non-null days in its window and
is NaN below LAG_MIN_COVERAGE of them (so 4 of 5 and 18 of 22 days); a row with a NaN lag or target
is dropped from the fit, and a forecast from an origin with a NaN lag is NaN. Pure numpy,
deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from terminal_models.core.base import ModelError

DAY_MS = 86_400_000
ANNUAL_DAYS = 365
HAR_LAGS = (1, 5, 22)
LAG_MIN_COVERAGE = 0.8  # share of a lag window's days that must have RV
EWMA_LAMBDA = 0.94
N_DOW = 7


def as_array(values: Sequence[float | None]) -> np.ndarray:
    """None → NaN, so nulls propagate instead of becoming 0."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def to_optional(values: np.ndarray) -> list[float | None]:
    """NaN/inf → None (payloads carry null, never NaN; C-3)."""
    return [float(v) if math.isfinite(v) else None for v in values]


# ---- §3 daily RV from bars ----------------------------------------------------------------


def daily_rv(
    times_ms: Sequence[int],
    closes: Sequence[float],
    as_of_ms: int,
    expected_bars: int,
    min_coverage: float,
    day_boundary_ms: int = 0,
) -> tuple[list[int], list[float | None], list[int]]:
    """Completed UTC days only → (day closes in Unix ms, RV_d, n_bars).

    `times_ms` are bar open times; the return from bar i−1 to bar i belongs to the day holding bar
    i's open time (so the first return of a day spans midnight). A day is complete when its close
    is ≤ `as_of_ms`. n_bars counts the day's returns; below `min_coverage · expected_bars` RV is
    None (VRP-A-8). Days with no bars are listed with None, so `days` is gap-free. A non-positive
    or non-finite close is skipped: the next return runs from the last valid close. A return that
    reaches back past more than one bar into an earlier day is dropped: it would carry a missing
    stretch's variance (a whole missing day, say) into this day.
    """
    if len(times_ms) != len(closes):
        raise ValueError("times_ms and closes differ in length")
    bars = sorted(zip(times_ms, closes, strict=True), key=lambda b: b[0])
    if any(a[0] == b[0] for a, b in zip(bars, bars[1:], strict=False)):
        raise ValueError("duplicate bar time")
    bars = [(t, c) for t, c in bars if c is not None and math.isfinite(c) and c > 0]
    if not bars:
        return [], [], []

    def close_of(t: int) -> int:
        return (t - day_boundary_ms) // DAY_MS * DAY_MS + day_boundary_ms + DAY_MS

    bar_ms = DAY_MS // expected_bars
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for (t_prev, prev), (t, c) in zip(bars, bars[1:], strict=False):
        day = close_of(t)
        if close_of(t_prev) != day and t - t_prev > bar_ms:
            continue
        sums[day] = sums.get(day, 0.0) + math.log(c / prev) ** 2
        counts[day] = counts.get(day, 0) + 1

    first = close_of(bars[0][0])
    last = close_of(as_of_ms) - DAY_MS  # the latest close ≤ as_of
    days = list(range(first, last + 1, DAY_MS))
    need = min_coverage * expected_bars
    n_bars = [counts.get(d, 0) for d in days]
    rv = [sums[d] if n and n >= need else None for d, n in zip(days, n_bars, strict=True)]
    return days, rv, n_bars


def trailing_rv(rv: np.ndarray, window: int, min_days: int) -> np.ndarray:
    """Backward-looking annualized RV at each day: √(365 · mean RV) over the non-null days among
    the `window` ending there; NaN below `min_days` non-null days (§12.4, §12.6)."""
    out = np.full(len(rv), np.nan)
    for t in range(len(rv)):
        tail = rv[max(0, t - window + 1) : t + 1]
        ok = tail[np.isfinite(tail)]
        if len(ok) >= min_days:
            out[t] = math.sqrt(ANNUAL_DAYS * ok.mean())
    return out


# ---- HAR design ---------------------------------------------------------------------------


def har_design(
    y_log: np.ndarray, lags: Sequence[int] = HAR_LAGS, min_coverage: float = LAG_MIN_COVERAGE
) -> np.ndarray:
    """Origin regressors per day t: [1, mean(y_{t−l+1..t}) for l in lags], each mean over the
    window's non-null days. NaN where the window runs off the start or fewer than
    ceil(min_coverage · l) of its days are non-null (so a lag-1 window needs its day)."""
    n = len(y_log)
    out = np.full((n, 1 + len(lags)), np.nan)
    out[:, 0] = 1.0
    missing = ~np.isfinite(y_log)
    csum = np.concatenate([[0.0], np.cumsum(np.where(missing, 0.0, y_log))])
    cmiss = np.concatenate([[0], np.cumsum(missing)])
    for j, lag in enumerate(lags, start=1):
        if lag > n:
            continue
        have = lag - (cmiss[lag:] - cmiss[:-lag])
        need = max(1, math.ceil(min_coverage * lag - 1e-9))
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (csum[lag:] - csum[:-lag]) / have
        out[lag - 1 :, j] = np.where(have < need, np.nan, mean)
    return out


def dow_dummies(weekdays: Sequence[int]) -> np.ndarray:
    """Day-of-week dummies with Monday as the reference level (all zero on Mondays). The
    decomposition never uses that reference as its counterfactual (see `counterfactuals`)."""
    out = np.zeros((len(weekdays), N_DOW - 1))
    for i, wd in enumerate(weekdays):
        if wd:
            out[i, wd - 1] = 1.0
    return out


@dataclass(frozen=True)
class HarFit:
    coef: np.ndarray  # (H, 1 + len(lags) + n_calendar); row h−1 is horizon h
    resid: tuple[np.ndarray, ...]  # in-sample residuals per horizon
    r2: np.ndarray  # (H,) in-sample R², log variance
    n_obs: np.ndarray  # (H,) rows used per horizon
    lags: tuple[int, ...]

    @property
    def horizons(self) -> int:
        return len(self.coef)


def _no_calendar(n: int) -> np.ndarray:
    return np.zeros((n, 0))


def fit_direct(
    y_log: np.ndarray,
    calendar: np.ndarray | None,
    horizons: int,
    lags: Sequence[int] = HAR_LAGS,
    min_obs: int | None = None,
) -> HarFit:
    """One OLS (numpy.linalg.lstsq) per h = 1..horizons on days 0..n−1 only.

    `calendar` has a row per day of `y_log` (later rows, for forecast days, are ignored) and is
    taken at the target day. Rows with a NaN regressor or target are dropped. A column that is zero
    over the sample (an event type that never occurred) gets the minimum-norm coefficient, 0.
    Raises `insufficient_history` when a horizon has fewer than `min_obs` rows (default: one more
    than the number of coefficients).
    """
    n = len(y_log)
    cal = _no_calendar(n) if calendar is None else calendar[:n]
    origin = har_design(y_log, lags)
    p = origin.shape[1] + cal.shape[1]
    need = p + 1 if min_obs is None else max(min_obs, p + 1)
    coefs, resids, r2s, n_obs = [], [], [], []
    for h in range(1, horizons + 1):
        X = np.hstack([origin[: n - h], cal[h:n]])  # noqa: N806
        y = y_log[h:n]
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if ok.sum() < need:
            raise ModelError(
                "insufficient_history", f"h={h}: {int(ok.sum())} usable days, need {need}"
            )
        X, y = X[ok], y[ok]  # noqa: N806
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ b
        sst = float(((y - y.mean()) ** 2).sum())
        coefs.append(b)
        resids.append(e)
        r2s.append(1.0 - float(e @ e) / sst if sst > 0 else np.nan)
        n_obs.append(len(y))
    return HarFit(np.array(coefs), tuple(resids), np.array(r2s), np.array(n_obs), tuple(lags))


def predict_path(fit: HarFit, y_log: np.ndarray, calendar_future: np.ndarray | None) -> np.ndarray:
    """ŷ_k, k = 1..H, from the last day of `y_log` (the origin). `calendar_future` row k−1 holds
    forecast day k's calendar columns. NaN if an origin lag is NaN (see `har_design`)."""
    x0 = har_design(y_log[-max(fit.lags) :], fit.lags)[-1]
    cal = _no_calendar(fit.horizons) if calendar_future is None else calendar_future
    if len(cal) < fit.horizons:
        raise ValueError(f"calendar_future has {len(cal)} rows, need {fit.horizons}")
    X = np.hstack([np.tile(x0, (fit.horizons, 1)), cal[: fit.horizons]])  # noqa: N806
    return (X * fit.coef).sum(axis=1)


# ---- §12.2 cone, aggregation, decomposition -------------------------------------------------


def residual_quantiles(fit: HarFit, quantiles: Sequence[float]) -> np.ndarray:
    """q_k(p): (len(quantiles), H) empirical quantiles of each horizon's residuals (linear)."""
    return np.array([[np.quantile(e, p) for e in fit.resid] for p in quantiles])


def cone_paths(y_hat: np.ndarray, rq: np.ndarray) -> np.ndarray:
    """v̂_k^p = exp(ŷ_k + q_k(p)): one daily-variance path per quantile row."""
    return np.exp(y_hat[None, :] + rq)


def aggregate_rv(path_var: np.ndarray, T: int) -> float:  # noqa: N803
    """RV(T) = √(365/T · Σ_{k≤T} v̂_k), annualized (§3)."""
    if T < 1 or T > len(path_var):
        raise ValueError(f"tenor {T} outside the 1..{len(path_var)} day path")
    return math.sqrt(ANNUAL_DAYS / T * float(path_var[:T].sum()))


def rv_term(path_var: np.ndarray) -> np.ndarray:
    """RV(T) for every T = 1..H (the cone curve)."""
    T = np.arange(1, len(path_var) + 1)  # noqa: N806
    return np.sqrt(ANNUAL_DAYS / T * np.cumsum(path_var))


def counterfactuals(calendar_future: np.ndarray, n_dow: int) -> tuple[np.ndarray, np.ndarray]:
    """(no events, no day-of-week) calendars for `decompose` (§12.2). The first `n_dow` columns
    are day-of-week dummies, the rest event dummies. "No events" zeroes the event columns. "No
    day-of-week" puts every forecast day at the average day (each dummy 1/7, every weekday weighted
    equally), not at the Monday reference, so the seasonal share doesn't depend on the reference
    day (with Monday as the reference, low-vol weekends would pin it at 0)."""
    no_evt, no_dow = calendar_future.copy(), calendar_future.copy()
    no_evt[:, n_dow:] = 0.0
    no_dow[:, :n_dow] = 1.0 / N_DOW
    return no_evt, no_dow


def decompose(
    path_var: np.ndarray,
    path_noevt: np.ndarray,
    path_nodow: np.ndarray,
    T: int,  # noqa: N803
) -> tuple[float, float, float]:
    """(baseline, seasonal, event) variance shares over days 1..T (§12.2).

    s = max(0, 1 − V_counterfactual / V); if the two exceed 1 together they are scaled to sum to 1,
    and baseline is the remainder, so the three sum to 1 and none is negative.
    """
    V = float(path_var[:T].sum())  # noqa: N806
    s_evt = max(0.0, 1.0 - float(path_noevt[:T].sum()) / V)
    s_sea = max(0.0, 1.0 - float(path_nodow[:T].sum()) / V)
    if s_evt + s_sea > 1.0:
        total = s_evt + s_sea
        s_evt, s_sea = s_evt / total, s_sea / total
    return max(0.0, 1.0 - s_evt - s_sea), s_sea, s_evt


# ---- baselines and backtest (§9) ----------------------------------------------------------


def ewma_var(rv: np.ndarray, lam: float = EWMA_LAMBDA) -> np.ndarray:
    """RiskMetrics σ²_t = λ σ²_{t−1} + (1 − λ) RV_t, seeded with the first non-null RV.
    A missing day has no observation, so σ² carries over unchanged; NaN before the seed."""
    out = np.full(len(rv), np.nan)
    s2 = np.nan
    for t, x in enumerate(rv):
        if math.isfinite(x):
            s2 = x if math.isnan(s2) else lam * s2 + (1.0 - lam) * x
        out[t] = s2
    return out


def expanding_forecasts(
    y_log: np.ndarray,
    calendar: np.ndarray | None,
    origins: Sequence[int],
    horizons: int,
    quantile: float | None = 0.5,
    lags: Sequence[int] = HAR_LAGS,
    min_obs: int | None = None,
    refit_every: int = 1,
) -> np.ndarray:
    """Daily-variance paths (len(origins), horizons) made at each origin t with data ≤ t only.

    The model is fit on days 0..t (targets ≤ t, so no look-ahead) every `refit_every` origins,
    counted back from the last one (so the last origin is always fit fresh), and held in between.
    `calendar` must have rows up to max(origins) + horizons. With `quantile`, the path is
    exp(ŷ + q_k(quantile)); with None, exp(ŷ). A row is NaN where the fit lacks data
    (`insufficient_history`) or an origin lag is NaN (`har_design`).
    """
    out = np.full((len(origins), horizons), np.nan)
    fit: HarFit | None = None
    rq: np.ndarray | None = None
    for i, t in enumerate(origins):
        if fit is None or (len(origins) - 1 - i) % refit_every == 0:
            try:
                fit = fit_direct(y_log[: t + 1], calendar, horizons, lags, min_obs)
            except ModelError:
                fit = None
                continue
            rq = None if quantile is None else residual_quantiles(fit, [quantile])[0]
        cal = None if calendar is None else calendar[t + 1 : t + 1 + horizons]
        y_hat = predict_path(fit, y_log[: t + 1], cal)
        out[i] = np.exp(y_hat if rq is None else y_hat + rq)
    return out


def _pairs(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    ok = np.isfinite(a) & np.isfinite(f)
    return a[ok] - f[ok]


def _naive_diffs(insample: np.ndarray, m: int) -> np.ndarray:
    x = np.asarray(insample, float)
    d = x[m:] - x[:-m]
    return d[np.isfinite(d)]


def mase(actual, forecast, insample, m: int = 1) -> float | None:
    """mean|e| / mean|x_t − x_{t−m}| (in-sample naive). None without pairs or with a zero scale."""
    e, d = _pairs(actual, forecast), _naive_diffs(insample, m)
    if not len(e) or not len(d) or not np.abs(d).mean():
        return None
    return float(np.abs(e).mean() / np.abs(d).mean())


def rmsse(actual, forecast, insample, m: int = 1) -> float | None:
    """√(mean e² / mean (x_t − x_{t−m})²). None without pairs or with a zero scale."""
    e, d = _pairs(actual, forecast), _naive_diffs(insample, m)
    if not len(e) or not len(d) or not (d**2).mean():
        return None
    return float(math.sqrt((e**2).mean() / (d**2).mean()))


def trailing_zscore(
    values: np.ndarray, window: int, min_obs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """z_t = (x_t − mean) / sd over the non-null x in the `window` days *before* t (exclusive,
    ddof = 1), §12.6. Returns (z, mean, sd, n); z is NaN where x_t is null, n < min_obs or
    sd = 0."""
    k = len(values)
    z, mu, sd = (np.full(k, np.nan) for _ in range(3))
    n = np.zeros(k, dtype=int)
    for t in range(k):
        tail = values[max(0, t - window) : t]
        ok = tail[np.isfinite(tail)]
        n[t] = len(ok)
        if len(ok) < max(min_obs, 2):
            continue
        mu[t], sd[t] = ok.mean(), ok.std(ddof=1)
        if sd[t] > 0 and math.isfinite(values[t]):
            z[t] = (values[t] - mu[t]) / sd[t]
    return z, mu, sd, n
