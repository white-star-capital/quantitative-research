"""RV target, HAR fit, cone, decomposition, baselines and metrics (RV guide §3, §6, §9, §12.2)
against closed-form answers. Inputs are hand-made test series, not market data."""

import math
from datetime import UTC, datetime

import numpy as np
import pytest

from terminal_models.core.base import ModelError
from terminal_models.core.quant.rv import (
    DAY_MS,
    aggregate_rv,
    as_array,
    cone_paths,
    counterfactuals,
    daily_rv,
    decompose,
    dow_dummies,
    ewma_var,
    expanding_forecasts,
    fit_direct,
    har_design,
    mase,
    predict_path,
    residual_quantiles,
    rmsse,
    rv_term,
    to_optional,
    trailing_rv,
    trailing_zscore,
)

HOUR_MS = 3_600_000
D0 = int(datetime(2026, 10, 1, tzinfo=UTC).timestamp() * 1000)  # a UTC midnight


def ar1(a, b, y0, n):
    """Noise-free y_{t+1} = a + b·y_t: every direct h-step relation holds exactly."""
    y = [y0]
    for _ in range(n - 1):
        y.append(a + b * y[-1])
    return np.array(y)


# ---- daily RV ----------------------------------------------------------------------------


def hourly(days, step=0.01):
    """Hourly bars from D0 with alternating ±step log returns; first bar opens at D0."""
    times = [D0 + i * HOUR_MS for i in range(24 * days)]
    closes = [100.0 * math.exp(step * (i % 2)) for i in range(len(times))]
    return times, closes


def test_daily_rv_completed_days_and_boundary():
    times, closes = hourly(3)
    as_of = D0 + 2 * DAY_MS + 5 * HOUR_MS  # day 3 in progress
    days, rv, n = daily_rv(times, closes, as_of, expected_bars=24, min_coverage=0.9)
    assert days == [D0 + DAY_MS, D0 + 2 * DAY_MS]  # closes, day 3 excluded
    # Day 1 has 23 returns (its first bar has no previous close); day 2's first return spans
    # midnight and belongs to day 2.
    assert n == [23, 24]
    assert rv[0] == pytest.approx(23 * 0.01**2)
    assert rv[1] == pytest.approx(24 * 0.01**2)


def test_daily_rv_coverage_nulls_and_gap_free_days():
    times, closes = hourly(4)
    keep = [
        i for i, t in enumerate(times) if not (D0 + DAY_MS <= t < D0 + 2 * DAY_MS + 12 * HOUR_MS)
    ]
    times, closes = [times[i] for i in keep], [closes[i] for i in keep]
    days, rv, n = daily_rv(times, closes, D0 + 4 * DAY_MS, expected_bars=24, min_coverage=0.9)
    assert days == [D0 + k * DAY_MS for k in (1, 2, 3, 4)]
    # Day 3's first bar (12:00) would reach back to day 1's 23:00 across the gap: dropped.
    assert n == [23, 0, 11, 24]
    assert rv[1] is None and rv[2] is None  # no bars; 11/24 below coverage
    assert rv[0] is not None and rv[3] == pytest.approx(24 * 0.01**2)


def test_daily_rv_never_carries_a_missing_day_into_the_next():
    times, closes = hourly(3)
    keep = [i for i, t in enumerate(times) if not (D0 + DAY_MS <= t < D0 + 2 * DAY_MS)]
    jump = math.exp(0.2)  # the price moves during the missing day
    times = [times[i] for i in keep]
    closes = [closes[i] * (jump if i >= 48 else 1.0) for i in keep]
    days, rv, n = daily_rv(times, closes, D0 + 3 * DAY_MS, expected_bars=24, min_coverage=0.9)
    assert n == [23, 0, 23]  # day 3 loses only the return reaching back across day 2
    assert rv[2] == pytest.approx(23 * 0.01**2)  # the 0.2 move is not day 3's variance


def test_daily_rv_rejects_duplicates_and_skips_bad_closes():
    with pytest.raises(ValueError, match="duplicate"):
        daily_rv([D0, D0], [1.0, 2.0], D0 + DAY_MS, 24, 0.9)
    days, rv, n = daily_rv(
        [D0, D0 + HOUR_MS, D0 + 2 * HOUR_MS], [100.0, -1.0, 110.0], D0 + DAY_MS, 2, 0.5
    )
    assert n == [1] and rv[0] == pytest.approx(math.log(1.1) ** 2)


def test_day_boundary_offset():
    times, closes = hourly(2)
    days, _, n = daily_rv(times, closes, D0 + 2 * DAY_MS, 24, 0.9, day_boundary_ms=8 * HOUR_MS)
    assert days == [D0 + 8 * HOUR_MS, D0 + 8 * HOUR_MS + DAY_MS]
    assert n == [7, 24]


def test_null_round_trip():
    assert np.isnan(as_array([1.0, None])[1])
    assert to_optional(np.array([1.0, np.nan, np.inf])) == [1.0, None, None]


def test_trailing_rv_needs_min_days():
    rv = as_array([0.01, None, 0.03, 0.02])
    out = trailing_rv(rv, window=3, min_days=2)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == pytest.approx(math.sqrt(365 * 0.02))
    assert out[3] == pytest.approx(math.sqrt(365 * 0.025))


# ---- HAR design and fit ------------------------------------------------------------------


def test_har_design_means_and_missing():
    y = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
    X = har_design(y, lags=(1, 2))  # noqa: N806
    assert (X[:, 0] == 1).all()
    assert X[2, 1:].tolist() == [3.0, 2.5]
    assert np.isnan(X[0, 2]) and np.isnan(X[3, 1]) and np.isnan(X[4, 2])
    assert X[5, 1:].tolist() == [6.0, 5.5]


def test_har_design_tolerates_a_few_missing_days():
    y = np.arange(1.0, 23.0)  # 22 days
    y[[3, 20]] = np.nan
    X = har_design(y, lags=(1, 5, 22))  # noqa: N806
    assert X[-1, 1] == 22.0
    assert X[-1, 2] == pytest.approx(np.mean([18.0, 19.0, 20.0, 22.0]))  # 4 of 5
    assert X[-1, 3] == pytest.approx(np.nanmean(y))  # 20 of 22
    y[19] = np.nan  # 3 of 5: too few; 19 of 22 still enough
    X = har_design(y, lags=(1, 5, 22))  # noqa: N806
    assert np.isnan(X[-1, 2]) and np.isfinite(X[-1, 3])
    y[[0, 1]] = np.nan  # 17 of 22
    assert np.isnan(har_design(y, lags=(22,))[-1, 1])


def test_dow_dummies_monday_reference():
    d = dow_dummies([0, 1, 6])
    assert d.shape == (3, 6)
    assert d[0].sum() == 0 and d[1, 0] == 1 and d[2, 5] == 1


def test_ols_matches_closed_form_simple_regression():
    y = np.array([0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0])
    fit = fit_direct(y, None, horizons=1, lags=(1,))
    x, t = y[:-1], y[1:]
    slope = ((x - x.mean()) * (t - t.mean())).sum() / ((x - x.mean()) ** 2).sum()
    assert fit.coef[0, 1] == pytest.approx(slope)
    assert fit.coef[0, 0] == pytest.approx(t.mean() - slope * x.mean())
    e = t - fit.coef[0, 0] - slope * x
    assert fit.resid[0] == pytest.approx(e)
    assert fit.r2[0] == pytest.approx(1 - (e @ e) / ((t - t.mean()) ** 2).sum())
    assert fit.n_obs[0] == 6


def test_direct_horizons_recover_ar1_and_predict():
    a, b = 0.5, 0.8
    y = ar1(a, b, 3.0, 40)
    fit = fit_direct(y, None, horizons=3, lags=(1,))
    for h in (1, 2, 3):
        assert fit.coef[h - 1, 1] == pytest.approx(b**h)
        assert fit.coef[h - 1, 0] == pytest.approx(a * sum(b**i for i in range(h)))
        assert np.abs(fit.resid[h - 1]).max() < 1e-9
    path = predict_path(fit, y, None)
    assert path == pytest.approx(ar1(a, b, y[-1], 4)[1:])


def test_calendar_column_fit_and_zero_column_is_inert():
    # y alternates level by a target-day dummy: y_t = 1 + 2·d_t with lag-1 terms.
    d = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0], dtype=float)
    y = 1.0 + 2.0 * d + 0.1 * np.arange(12)  # trend keeps the lag column informative
    cal = np.column_stack([d, np.zeros(12)])
    fit = fit_direct(y, cal, horizons=1, lags=(1,))
    assert fit.coef.shape == (1, 4)
    assert fit.coef[0, 3] == pytest.approx(0.0, abs=1e-12)  # never active: min-norm 0
    plain = fit_direct(y, cal[:, :1], horizons=1, lags=(1,))
    assert fit.coef[0, :3] == pytest.approx(plain.coef[0])


def test_missing_days_drop_rows_never_fill():
    y = ar1(0.5, 0.8, 3.0, 20)
    holed = y.copy()
    holed[10] = np.nan
    fit = fit_direct(holed, None, horizons=1, lags=(1,))
    assert fit.n_obs[0] == 19 - 2  # the row targeting day 10 and the row lagging it
    assert fit.coef[0, 1] == pytest.approx(0.8)
    origin_missing = holed[:11]
    assert np.isnan(predict_path(fit, origin_missing, None)).all()


def test_insufficient_history_raises():
    with pytest.raises(ModelError) as exc:
        fit_direct(np.array([1.0, 2.0, 3.0]), None, horizons=2, lags=(1,))
    assert exc.value.code == "insufficient_history"


# ---- cone, aggregation, decomposition ----------------------------------------------------


def test_constant_path_cone_identity():
    v = 0.0004
    path = np.full(60, v)
    for T in (1, 7, 30, 60):  # noqa: N806
        assert aggregate_rv(path, T) == pytest.approx(math.sqrt(365 * v))
    assert rv_term(path) == pytest.approx(np.full(60, math.sqrt(365 * v)))
    with pytest.raises(ValueError):
        aggregate_rv(path, 61)


def test_aggregate_rv_sums_variance():
    path = np.array([1.0, 2.0, 3.0]) * 1e-4
    assert aggregate_rv(path, 2) == pytest.approx(math.sqrt(365 / 2 * 3e-4))
    assert rv_term(path)[2] == pytest.approx(aggregate_rv(path, 3))


def test_residual_quantiles_and_cone_order():
    y = np.array([0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 1.0])
    fit = fit_direct(y, None, horizons=2, lags=(1,))
    rq = residual_quantiles(fit, [0.1, 0.5, 0.9])
    assert rq.shape == (3, 2)
    assert rq[0, 0] == pytest.approx(np.quantile(fit.resid[0], 0.1))
    paths = cone_paths(np.array([0.0, 0.0]), rq)
    assert (paths[0] <= paths[1]).all() and (paths[1] <= paths[2]).all()
    exact = fit_direct(ar1(0.5, 0.8, 3.0, 30), None, horizons=2, lags=(1,))
    assert residual_quantiles(exact, [0.5]) == pytest.approx(np.zeros((1, 2)), abs=1e-9)


def test_decomposition_no_events_and_sums_to_one():
    path = np.full(30, 1e-4)
    base, sea, evt = decompose(path, path, path, 30)
    assert (base, sea, evt) == (1.0, 0.0, 0.0)
    noevt, nodow = path * 0.8, path * 0.9
    base, sea, evt = decompose(path, noevt, nodow, 30)
    assert (evt, sea) == (pytest.approx(0.2), pytest.approx(0.1))
    assert base + sea + evt == pytest.approx(1.0, abs=1e-12)


def test_counterfactuals_put_day_of_week_at_the_average_day():
    rng = np.random.default_rng(3)
    n, H = 399, 7  # noqa: N806  the origin (day n−1) is a Sunday: forecast day 1 is a Monday
    wd = [t % 7 for t in range(n + H)]
    weekend = np.array([w >= 5 for w in wd], dtype=float)
    y = -8.0 - 0.5 * weekend[:n] + rng.normal(0.0, 0.3, n)  # low-vol weekends
    cal = np.hstack([dow_dummies(wd), weekend[:, None]])  # one event column
    no_evt, no_dow = counterfactuals(cal[n:], 6)
    assert (no_evt[:, 6] == 0).all() and (no_evt[:, :6] == cal[n:, :6]).all()
    assert no_dow[:, :6] == pytest.approx(np.full((H, 6), 1 / 7))
    assert (no_dow[:, 6] == cal[n:, 6]).all()
    fit = fit_direct(y, cal[:, :6], H, lags=(1,))
    path = np.exp(predict_path(fit, y, cal[n:, :6]))
    nodow = np.exp(predict_path(fit, y, counterfactuals(cal[n:, :6], 6)[1]))
    # Mon–Fri sit above the average day. With Monday as the counterfactual every weekday would
    # read ≈ Monday and the share ≈ 0, so the share would depend on the reference day.
    assert decompose(path, path, nodow, 5)[1] == pytest.approx(1 - math.exp(-1 / 7), abs=0.05)
    assert decompose(path, path, nodow, 7)[1] < 0.05  # a whole week averages out


def test_decomposition_clips_and_normalizes():
    path = np.full(10, 1.0)
    base, sea, evt = decompose(path, path * 1.5, path * 0.5, 10)  # events lower the forecast
    assert (base, sea, evt) == (0.5, 0.5, 0.0)
    base, sea, evt = decompose(path, path * 0.2, path * 0.3, 10)  # 0.8 + 0.7 > 1
    assert evt == pytest.approx(0.8 / 1.5) and sea == pytest.approx(0.7 / 1.5)
    assert base >= 0 and base + sea + evt == pytest.approx(1.0, abs=1e-12)


# ---- baselines, backtest, metrics --------------------------------------------------------


def test_ewma_recursion():
    out = ewma_var(as_array([None, 1.0, 2.0, None, 4.0]), lam=0.5)
    assert np.isnan(out[0])
    assert out[1:].tolist() == [1.0, 1.5, 1.5, 2.75]


def test_expanding_forecasts_no_look_ahead():
    y = ar1(0.5, 0.8, 3.0, 40)
    out = expanding_forecasts(y, None, origins=[5, 20, 39], horizons=2, quantile=None, lags=(1,))
    assert np.isfinite(out[0]).all()  # origin 5: h = 2 needs 3 rows and has 4
    assert out[1] == pytest.approx(np.exp(ar1(0.5, 0.8, y[20], 3)[1:]))
    changed = y.copy()
    changed[21:] += 5.0  # the future must not move the origin-20 forecast
    again = expanding_forecasts(changed, None, origins=[20], horizons=2, quantile=None, lags=(1,))
    assert again[0] == pytest.approx(out[1])


def test_expanding_forecasts_short_history_is_nan_and_refit_every():
    y = ar1(0.5, 0.8, 3.0, 30)
    out = expanding_forecasts(y, None, origins=[1, 2, 25], horizons=3, lags=(1,))
    assert np.isnan(out[0]).all() and np.isnan(out[1]).all() and np.isfinite(out[2]).all()
    held = expanding_forecasts(y, None, [20, 25], 1, None, (1,), refit_every=2)
    fresh = expanding_forecasts(y, None, [20, 25], 1, None, (1,))
    assert held == pytest.approx(fresh)  # exact AR: the held fit predicts the same


def test_expanding_forecasts_refits_land_on_the_last_origin():
    y = ar1(0.5, 0.8, 3.0, 40) + np.random.default_rng(5).normal(0.0, 0.1, 40)
    origins = [30, 31, 32, 33, 34]
    held = expanding_forecasts(y, None, origins, 1, None, (1,), refit_every=2)
    fresh = expanding_forecasts(y, None, origins, 1, None, (1,))
    for i in (0, 2, 4):  # the first (no fit yet), then every 2nd counted back from the last
        assert held[i] == pytest.approx(fresh[i], rel=1e-12)
    assert not np.allclose(held[1], fresh[1]) and not np.allclose(held[3], fresh[3])


def test_expanding_forecasts_uses_future_calendar_rows():
    d = np.array([0, 1] * 10, dtype=float)
    y = 1.0 + 2.0 * d + 0.1 * np.arange(20)
    cal = np.concatenate([d, [0.0, 1.0]])[:, None]  # two forecast days past the data
    out = expanding_forecasts(y, cal, origins=[19], horizons=1, quantile=None, lags=(1,))
    fit = fit_direct(y, cal, 1, (1,))
    assert out[0, 0] == pytest.approx(math.exp(predict_path(fit, y, cal[20:21])[0]))


def test_mase_rmsse_by_hand():
    insample = np.array([1.0, 3.0, 2.0, 4.0])  # naive diffs 2, -1, 2
    actual = np.array([2.0, 5.0, np.nan])
    forecast = np.array([1.0, 3.0, 9.0])  # errors 1, 2 (third pair dropped)
    assert mase(actual, forecast, insample) == pytest.approx(1.5 / (5 / 3))
    assert rmsse(actual, forecast, insample) == pytest.approx(math.sqrt(2.5 / 3))
    assert mase(actual, forecast, np.ones(4)) is None
    assert rmsse([np.nan], [1.0], insample) is None


def test_trailing_zscore_excludes_today():
    x = as_array([1.0, 2.0, 3.0, None, 10.0])
    z, mu, sd, n = trailing_zscore(x, window=4, min_obs=3)
    assert n.tolist() == [0, 1, 2, 3, 3]
    assert np.isnan(z[:4]).all()  # below min_obs, or today null
    assert (mu[4], sd[4]) == (pytest.approx(2.0), pytest.approx(1.0))
    assert z[4] == pytest.approx(8.0)
    flat = trailing_zscore(np.ones(5), window=3, min_obs=2)[0]
    assert np.isnan(flat).all()  # sd = 0
