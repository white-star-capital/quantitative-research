"""Null control tests (vgp/analysis/null_control.py).

The null control is the only check in this codebase that can detect bias shared
by every trial — a lookahead primitive, one training window reused by all seeds,
a survivor-biased universe. DSR is blind to all of it, because such bias moves
every trial together and leaves sigma_SR unchanged.

For the comparison to mean anything the surrogate must be a fair one: same
distributional properties, same cross-asset structure, same bar geometry, with
only the exploitable ordering destroyed. A surrogate that is too easy to beat
manufactures significance; one that is too hard hides real signal. These tests
pin those properties.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

_T = 500
_A = 3


@pytest.fixture(scope="module")
def ohlcv():
    """Synthetic OHLCV with a genuine market factor across assets."""
    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-01-01", periods=_T, freq="D")
    market = rng.standard_normal(_T) * 0.01
    out = {}
    for a in range(_A):
        ret = 0.7 * market + rng.standard_normal(_T) * 0.01
        close = 100.0 * np.exp(np.cumsum(ret))
        out[f"ASSET{a}"] = pd.DataFrame(
            {
                "open": close * (1 + rng.standard_normal(_T) * 0.001),
                "high": close * (1 + np.abs(rng.standard_normal(_T)) * 0.005),
                "low": close * (1 - np.abs(rng.standard_normal(_T)) * 0.005),
                "close": close,
                "volume": np.abs(rng.standard_normal(_T)) * 1e6 + 1e5,
            },
            index=idx,
        )
    return out


def _log_returns(data: dict, ticker: str) -> np.ndarray:
    return np.diff(np.log(data[ticker]["close"].to_numpy(dtype=np.float64)))


# ---------------------------------------------------------------------------
# Surrogate shape and validity
# ---------------------------------------------------------------------------

def test_surrogate_preserves_shape_index_and_columns(ohlcv):
    """The surrogate must be a drop-in replacement for the real data."""
    from vgp.analysis import block_bootstrap_ohlcv

    sur = block_bootstrap_ohlcv(ohlcv, np.random.default_rng(0), block_size=20)

    assert set(sur) == set(ohlcv)
    for t in ohlcv:
        assert sur[t].shape == ohlcv[t].shape
        assert sur[t].index.equals(ohlcv[t].index)
        assert list(sur[t].columns) == list(ohlcv[t].columns)


def test_surrogate_bars_are_internally_consistent(ohlcv):
    """low <= close <= high and all prices positive.

    Bar geometry is carried over as ratios from real bars rather than
    synthesized, so an invalid bar means the geometry indexing is wrong — which
    would feed the FeatureEngine (ATR, normalized close) impossible inputs.
    """
    from vgp.analysis import block_bootstrap_ohlcv

    sur = block_bootstrap_ohlcv(ohlcv, np.random.default_rng(3), block_size=15)

    for t, df in sur.items():
        assert (df[["open", "high", "low", "close"]] > 0).all().all(), f"{t}: non-positive price"
        assert (df["low"] <= df["close"] + 1e-9).all(), f"{t}: low above close"
        assert (df["close"] <= df["high"] + 1e-9).all(), f"{t}: close above high"
        assert (df["low"] <= df["high"] + 1e-9).all(), f"{t}: low above high"
        assert np.isfinite(df.to_numpy(dtype=np.float64)).all(), f"{t}: non-finite value"


def test_surrogate_preserves_return_distribution(ohlcv):
    """Volatility and tail shape must survive — the null must be a fair opponent.

    A surrogate with materially lower volatility would be trivially easy to beat
    and would manufacture significance for any strategy.
    """
    from vgp.analysis import block_bootstrap_ohlcv

    sur = block_bootstrap_ohlcv(ohlcv, np.random.default_rng(5), block_size=20)

    for t in ohlcv:
        real, fake = _log_returns(ohlcv, t), _log_returns(sur, t)
        assert fake.std() == pytest.approx(real.std(), rel=0.20), (
            f"{t}: surrogate volatility {fake.std():.5f} vs real {real.std():.5f}"
        )
        real_kurt = float(pd.Series(real).kurt())
        fake_kurt = float(pd.Series(fake).kurt())
        assert abs(fake_kurt - real_kurt) < 1.5, (
            f"{t}: tail shape changed materially ({real_kurt:.2f} -> {fake_kurt:.2f})"
        )


def test_surrogate_preserves_cross_asset_correlation(ohlcv):
    """Shared block indices must keep the market factor intact.

    Drawing blocks independently per asset would destroy cross-sectional
    correlation and make the null far too easy to beat.
    """
    from vgp.analysis import block_bootstrap_ohlcv

    sur = block_bootstrap_ohlcv(ohlcv, np.random.default_rng(9), block_size=20)

    tickers = list(ohlcv)
    real_corr = np.corrcoef([_log_returns(ohlcv, t) for t in tickers])
    fake_corr = np.corrcoef([_log_returns(sur, t) for t in tickers])

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            assert fake_corr[i, j] > 0.5 * real_corr[i, j], (
                f"cross-correlation collapsed for {tickers[i]}/{tickers[j]}: "
                f"{real_corr[i, j]:.3f} -> {fake_corr[i, j]:.3f}"
            )


def test_surrogate_is_reproducible_and_seed_dependent(ohlcv):
    """Same seed, same surrogate; different seed, different surrogate."""
    from vgp.analysis import block_bootstrap_ohlcv

    a = block_bootstrap_ohlcv(ohlcv, np.random.default_rng(42), block_size=20)
    b = block_bootstrap_ohlcv(ohlcv, np.random.default_rng(42), block_size=20)
    c = block_bootstrap_ohlcv(ohlcv, np.random.default_rng(43), block_size=20)

    t = next(iter(ohlcv))
    np.testing.assert_allclose(a[t].to_numpy(), b[t].to_numpy())
    assert not np.allclose(a[t]["close"].to_numpy(), c[t]["close"].to_numpy())


def test_surrogate_serial_structure_decays_with_block_size(ohlcv):
    """Predictability must fall as the block shrinks, and vanish at block_size=1.

    A block bootstrap preserves autocorrelation WITHIN each block by design and
    only breaks it at block boundaries, so with L-bar blocks roughly 1 in L
    transitions is severed. That is the trade-off block size controls, and it
    has a direct consequence for interpreting the control: with the default
    20-bar blocks, a strategy exploiting only 1-5 day momentum survives into
    the surrogate and the null will NOT flag it. Blocks must be shorter than
    the horizon of the effect being tested.
    """
    from vgp.analysis import block_bootstrap_ohlcv

    rng = np.random.default_rng(2)
    idx = pd.date_range("2024-01-01", periods=_T, freq="D")
    ret = np.zeros(_T)
    for i in range(1, _T):
        ret[i] = 0.85 * ret[i - 1] + rng.standard_normal() * 0.004   # AR(1)
    close = 100.0 * np.exp(np.cumsum(ret))
    trending = {
        "T": pd.DataFrame(
            {"open": close, "high": close * 1.002, "low": close * 0.998,
             "close": close, "volume": np.ones(_T)},
            index=idx,
        )
    }

    def ac1(x):
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    real_ac = ac1(_log_returns(trending, "T"))
    assert real_ac > 0.6, f"fixture is not autocorrelated enough ({real_ac:.3f})"

    acs = {}
    for block in (1, 5, 50):
        sur = block_bootstrap_ohlcv(trending, np.random.default_rng(4), block_size=block)
        acs[block] = ac1(_log_returns(sur, "T"))

    assert acs[1] < 0.15, (
        f"block_size=1 is a full iid shuffle and must destroy serial structure, "
        f"got autocorrelation {acs[1]:.3f}"
    )
    assert acs[1] < acs[5] < acs[50], (
        f"predictability must increase monotonically with block size, got {acs}"
    )
    assert acs[50] < real_ac, (
        f"even long blocks must break some structure ({real_ac:.3f} -> {acs[50]:.3f})"
    )


def test_surrogate_rejects_bad_block_size(ohlcv):
    from vgp.analysis import block_bootstrap_ohlcv

    with pytest.raises(ValueError, match="block_size"):
        block_bootstrap_ohlcv(ohlcv, np.random.default_rng(0), block_size=0)


def test_surrogate_handles_ragged_asset_histories(ohlcv):
    """Staggered listing dates must work and preserve each asset's own span.

    Crypto panels are ragged by nature and the fetcher returns them that way.
    Critically, each surrogate asset must keep its ORIGINAL row count: that is
    what makes FeatureEngine's min_obs_fraction filter reach the same retention
    decision on the surrogate as on the real data, so the null run is computed
    on the same universe as the observed run. If a short-history asset came
    back full-length it would survive into the null universe when it was
    dropped from the real one, and the comparison would be invalid.
    """
    from vgp.analysis import block_bootstrap_ohlcv

    ragged = {k: v.copy() for k, v in ohlcv.items()}
    short = sorted(ragged)[0]
    ragged[short] = ragged[short].iloc[-60:]        # late listing
    tiny = sorted(ragged)[1]
    ragged[tiny] = ragged[tiny].iloc[-1:]           # single bar, degenerate

    sur = block_bootstrap_ohlcv(ragged, np.random.default_rng(0), block_size=20)

    assert set(sur) == set(ragged), "an asset went missing"
    for t in ragged:
        assert len(sur[t]) == len(ragged[t]), (
            f"{t}: surrogate has {len(sur[t])} rows, original had "
            f"{len(ragged[t])} — the retention decision would differ"
        )
        assert sur[t].index.equals(ragged[t].index)
        assert np.isfinite(sur[t].to_numpy(dtype=np.float64)).all(), f"{t}: non-finite"
        assert (sur[t][["open", "high", "low", "close"]] > 0).all().all()

    # The 60-bar asset must still be resampled, not passed through unchanged
    assert not np.allclose(
        sur[short]["close"].to_numpy(), ragged[short]["close"].to_numpy()
    ), f"{short}: short-history asset was not resampled"


def test_surrogate_preserves_ragged_asset_return_distribution(ohlcv):
    """A folded shared draw must still sample from the asset's OWN returns."""
    from vgp.analysis import block_bootstrap_ohlcv

    ragged = {k: v.copy() for k, v in ohlcv.items()}
    short = sorted(ragged)[0]
    ragged[short] = ragged[short].iloc[-150:]

    sur = block_bootstrap_ohlcv(ragged, np.random.default_rng(11), block_size=10)

    real = _log_returns(ragged, short)
    fake = _log_returns(sur, short)
    assert fake.std() == pytest.approx(real.std(), rel=0.35), (
        f"{short}: surrogate volatility {fake.std():.5f} vs real {real.std():.5f} — "
        f"the fold is not drawing from this asset's own history"
    )


def test_surrogate_handles_empty_and_tiny_input():
    from vgp.analysis import block_bootstrap_ohlcv

    assert block_bootstrap_ohlcv({}, np.random.default_rng(0)) == {}

    idx = pd.date_range("2024-01-01", periods=1, freq="D")
    tiny = {"X": pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=idx)}
    out = block_bootstrap_ohlcv(tiny, np.random.default_rng(0), block_size=20)
    assert len(out["X"]) == 1


# ---------------------------------------------------------------------------
# Empirical p-value
# ---------------------------------------------------------------------------

def test_empirical_p_value_never_zero():
    """The (1+k)/(1+n) form must not claim more resolution than n runs support."""
    from vgp.analysis import empirical_p_value

    p = empirical_p_value(100.0, [0.1, 0.2, 0.3])
    assert p == pytest.approx(0.25)
    assert p > 0.0, "a p-value of exactly 0 asserts infinite resolution"


def test_empirical_p_value_bounds_and_ordering():
    from vgp.analysis import empirical_p_value

    null = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert empirical_p_value(-1.0, null) == pytest.approx(1.0)
    strong = empirical_p_value(5.0, null)
    weak = empirical_p_value(1.5, null)
    assert strong < weak, "a better observed value must give a smaller p"
    assert 0.0 < strong <= 1.0 and 0.0 < weak <= 1.0


def test_empirical_p_value_unmeasured_is_nan():
    """NaN means 'not measured' — it must never read as 'not significant'."""
    from vgp.analysis import empirical_p_value

    assert np.isnan(empirical_p_value(float("nan"), [1.0, 2.0]))
    assert np.isnan(empirical_p_value(1.0, []))
    assert np.isnan(empirical_p_value(1.0, [np.nan, np.inf]))


def test_empirical_p_value_ignores_nonfinite_null_samples():
    from vgp.analysis import empirical_p_value

    assert empirical_p_value(2.0, [1.0, 3.0, np.nan, -np.inf]) == pytest.approx(
        empirical_p_value(2.0, [1.0, 3.0])
    )


# ---------------------------------------------------------------------------
# best_sharpes: unmeasured rows are not bad rows
# ---------------------------------------------------------------------------

def test_best_sharpes_skips_unmeasured_rows():
    from vgp.analysis import best_sharpes

    is_sr, oos_sr = best_sharpes([
        {"is_sharpe": 1.0, "oos_sharpe": float("nan")},
        {"is_sharpe": 2.0, "oos_sharpe": 0.4},
        {"is_sharpe": float("nan"), "oos_sharpe": -np.inf},
    ])
    assert is_sr == pytest.approx(2.0)
    assert oos_sr == pytest.approx(0.4)


def test_best_sharpes_all_unmeasured_is_nan():
    from vgp.analysis import best_sharpes

    is_sr, oos_sr = best_sharpes([{"is_sharpe": float("nan"), "oos_sharpe": -np.inf}])
    assert np.isnan(is_sr) and np.isnan(oos_sr)


# ---------------------------------------------------------------------------
# run_null_control orchestration (stubbed experiment — no evolution runs)
# ---------------------------------------------------------------------------

def test_run_null_control_detects_noise_mimicking_the_real_run(ohlcv):
    """When the null matches the observed result, p must be large.

    This is the failure the control exists to catch: the pipeline finding the
    same performance in signal-free data.
    """
    from vgp.analysis import run_null_control

    def stub(fm, close, dates):
        return [{"is_sharpe": 4.0, "oos_sharpe": 1.0}]

    res = run_null_control(
        ohlcv=ohlcv,
        experiment_fn=stub,
        observed_results=[{"is_sharpe": 4.0, "oos_sharpe": 1.0}],
        n_runs=9,
        feature_builder=lambda o: (None, None, None),
    )

    assert res.p_value_is == pytest.approx(1.0)
    assert "unproven" in res.summary()


def test_run_null_control_clears_null_when_observed_is_better(ohlcv):
    """When the observed result beats every surrogate, p hits the resolution floor."""
    from vgp.analysis import run_null_control

    def stub(fm, close, dates):
        return [{"is_sharpe": 0.2, "oos_sharpe": 0.05}]

    res = run_null_control(
        ohlcv=ohlcv,
        experiment_fn=stub,
        observed_results=[{"is_sharpe": 4.0, "oos_sharpe": 1.5}],
        n_runs=19,
        feature_builder=lambda o: (None, None, None),
    )

    assert res.p_value_is == pytest.approx(1 / 20)
    assert res.resolution == pytest.approx(1 / 20)
    assert "outside the null distribution" in res.summary()


def test_run_null_control_reports_resolution_limit(ohlcv):
    """Too few runs cannot resolve p<=0.05, and the summary must say so."""
    from vgp.analysis import run_null_control

    res = run_null_control(
        ohlcv=ohlcv,
        experiment_fn=lambda fm, c, d: [{"is_sharpe": 0.1, "oos_sharpe": 0.0}],
        observed_results=[{"is_sharpe": 4.0, "oos_sharpe": 1.0}],
        n_runs=3,
        feature_builder=lambda o: (None, None, None),
    )

    assert res.resolution == pytest.approx(0.25)
    assert "cannot resolve p below" in res.summary()


def test_run_null_control_survives_a_failing_surrogate(ohlcv):
    """One bad surrogate must not abandon the control, and must be counted."""
    from vgp.analysis import run_null_control

    calls = {"n": 0}

    def flaky(fm, close, dates):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("synthetic failure")
        return [{"is_sharpe": 0.3, "oos_sharpe": 0.1}]

    res = run_null_control(
        ohlcv=ohlcv,
        experiment_fn=flaky,
        observed_results=[{"is_sharpe": 4.0, "oos_sharpe": 1.0}],
        n_runs=5,
        feature_builder=lambda o: (None, None, None),
    )

    assert res.n_runs_failed == 1
    assert res.null_best_is_sharpe.size == 4
    assert "1 failed" in res.summary()


def test_run_null_control_uses_a_fresh_surrogate_each_run(ohlcv):
    """Each run must see different data, or the null distribution is degenerate."""
    from vgp.analysis import run_null_control

    seen = []

    def recorder(fm, close, dates):
        seen.append(float(fm))
        return [{"is_sharpe": fm, "oos_sharpe": 0.0}]

    # feature_builder collapses each surrogate to a single number so we can
    # tell the runs apart
    def builder(surrogate):
        t = next(iter(surrogate))
        return float(surrogate[t]["close"].to_numpy().sum()), None, None

    run_null_control(
        ohlcv=ohlcv, experiment_fn=recorder,
        observed_results=[{"is_sharpe": 1.0, "oos_sharpe": 0.0}],
        n_runs=5, feature_builder=builder,
    )

    assert len(set(seen)) == 5, f"surrogates repeated across runs: {seen}"


def test_run_null_control_rejects_zero_runs(ohlcv):
    from vgp.analysis import run_null_control

    with pytest.raises(ValueError, match="n_runs"):
        run_null_control(
            ohlcv=ohlcv, experiment_fn=lambda *a: [],
            observed_results=[], n_runs=0,
            feature_builder=lambda o: (None, None, None),
        )


# ---------------------------------------------------------------------------
# Cross-asset co-movement on RAGGED panels
#
# The first ragged-panel fix folded the shared block sequence into each shorter
# asset's own range by modulo. That kept row counts correct but broke
# co-movement for the short asset even over the window where it DID co-exist
# with the others, because it was reading from a different position in its own
# array. The shared sequence is now drawn over the intersection window and
# mapped through each asset's own dates, so all assets take their return from
# the same source DATE. These tests pin that.
# ---------------------------------------------------------------------------


def _ragged_panel(seed: int = 3, n_short: int = 200):
    """Four assets on a common market factor; one lists late."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=_T, freq="D")
    market = rng.standard_normal(_T) * 0.012
    panel = {}
    for a in range(4):
        ret = 0.85 * market + rng.standard_normal(_T) * 0.006
        close = 100.0 * np.exp(np.cumsum(ret))
        df = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.004,
                "low": close * 0.996,
                "close": close,
                "volume": np.full(_T, 1e6),
            },
            index=idx,
        )
        panel[f"A{a}"] = df.iloc[-n_short:] if a == 3 else df
    return panel


def test_ragged_surrogate_preserves_comovement_over_the_overlap():
    """A late-listing asset must still co-move with the others where it exists.

    Correlation is measured only over the intersection window — the span
    FeatureEngine actually keeps — between the short asset and a full-history
    one. Under the modulo fold this collapsed toward zero.
    """
    from vgp.analysis import block_bootstrap_ohlcv

    panel = _ragged_panel()
    sur = block_bootstrap_ohlcv(panel, np.random.default_rng(0), block_size=20)

    short, long = "A3", "A0"
    overlap = panel[short].index.intersection(panel[long].index)

    def corr_over(data, i, j):
        a = np.diff(np.log(data[i].loc[overlap, "close"].to_numpy(dtype=np.float64)))
        b = np.diff(np.log(data[j].loc[overlap, "close"].to_numpy(dtype=np.float64)))
        return float(np.corrcoef(a, b)[0, 1])

    real = corr_over(panel, short, long)
    fake = corr_over(sur, short, long)

    assert real > 0.7, f"fixture is not correlated enough ({real:.3f})"
    assert fake > 0.7 * real, (
        f"co-movement over the overlap collapsed for the late-listing asset: "
        f"real {real:.3f} -> surrogate {fake:.3f}. The shared block sequence is "
        f"not reaching this asset through its own dates."
    )


def test_ragged_surrogate_draws_the_same_source_date_for_every_asset():
    """The strongest form of the guarantee, checked directly.

    With identical returns across assets over the overlap, a shared source date
    means identical surrogate returns over the overlap too. Any per-asset
    remapping of the sequence would break this exactly.
    """
    from vgp.analysis import block_bootstrap_ohlcv

    idx = pd.date_range("2024-01-01", periods=300, freq="D")
    rng = np.random.default_rng(5)
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(300) * 0.01))
    base = pd.DataFrame(
        {"open": close, "high": close * 1.002, "low": close * 0.998,
         "close": close, "volume": np.full(300, 1e6)},
        index=idx,
    )
    panel = {"LONG": base, "SHORT": base.iloc[-120:].copy()}

    sur = block_bootstrap_ohlcv(panel, np.random.default_rng(1), block_size=15)

    overlap = panel["SHORT"].index
    long_ret = np.diff(np.log(
        sur["LONG"].loc[overlap, "close"].to_numpy(dtype=np.float64)))
    short_ret = np.diff(np.log(
        sur["SHORT"].loc[overlap, "close"].to_numpy(dtype=np.float64)))

    np.testing.assert_allclose(long_ret, short_ret, atol=1e-9, err_msg=(
        "assets with identical source returns produced different surrogate "
        "returns over the overlap — they are not drawing the same source date"
    ))


def test_ragged_surrogate_keeps_pre_overlap_history_out_of_the_overlap():
    """Warm-up bars must be resampled from warm-up bars, not from the overlap.

    Importing overlap-window returns into the pre-listing warm-up would leak
    the joint regime into a period that had none.
    """
    from vgp.analysis import block_bootstrap_ohlcv

    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    rng = np.random.default_rng(9)
    # Deliberately different scales so the two regimes are distinguishable
    early = rng.standard_normal(200) * 0.001
    late = rng.standard_normal(200) * 0.05
    close = 100.0 * np.exp(np.cumsum(np.concatenate([early, late])))
    long_df = pd.DataFrame(
        {"open": close, "high": close * 1.002, "low": close * 0.998,
         "close": close, "volume": np.full(400, 1e6)},
        index=idx,
    )
    panel = {"LONG": long_df, "SHORT": long_df.iloc[-200:].copy()}

    sur = block_bootstrap_ohlcv(panel, np.random.default_rng(2), block_size=10)

    warmup = sur["LONG"].loc[idx[:200], "close"].to_numpy(dtype=np.float64)
    warmup_ret = np.diff(np.log(warmup))

    # Warm-up must retain the quiet regime's scale, not the volatile one
    assert warmup_ret.std() < 0.01, (
        f"pre-overlap warm-up volatility {warmup_ret.std():.4f} looks like the "
        f"overlap regime (~0.05) — overlap returns leaked into the warm-up"
    )


def test_ragged_surrogate_retention_decision_matches_real_data():
    """The null must be computed on the SAME universe as the observed run.

    If a short-history asset came back full-length it would survive
    min_obs_fraction in the null while being dropped from the real run, and the
    two would not be comparable.
    """
    from vgp.analysis import block_bootstrap_ohlcv
    from vgp.data import FeatureEngine

    panel = _ragged_panel(n_short=60)   # short enough to be dropped
    sur = block_bootstrap_ohlcv(panel, np.random.default_rng(4), block_size=20)

    real_engine, sur_engine = FeatureEngine(), FeatureEngine()
    real_arr = real_engine.fit_transform(panel)
    sur_arr = sur_engine.fit_transform(sur)

    assert real_engine.retained_assets_ == sur_engine.retained_assets_, (
        f"universe differs: real {real_engine.retained_assets_} vs surrogate "
        f"{sur_engine.retained_assets_}"
    )
    assert real_engine.dropped_assets_ == sur_engine.dropped_assets_
    assert real_arr.shape == sur_arr.shape, (
        f"panel shape differs: {real_arr.shape} vs {sur_arr.shape}"
    )


# ---------------------------------------------------------------------------
# Correlation must be preserved WITHIN A WINDOW, not just over the full sample
#
# This is the regression for the bug that made the null control unbeatable
# in-sample. Co-movement was shared only over the intersection of ALL assets;
# with one very late listing that intersection was the final few months, so
# every training window fell outside it and each asset was resampled
# independently there. The full-sample correlation matrix still looked right,
# which is exactly why it went unnoticed — the assertion has to be made on the
# window the GP actually trains on.
#
# Measured on the real 27-asset Binance panel before the fix: mean pairwise
# correlation 0.615 in the real training window against 0.001 in the surrogate,
# 2.4 effective bets against 20. A cross-sectional book with twenty independent
# bets instead of two reaches a far higher in-sample Sharpe, inflating the null.
# ---------------------------------------------------------------------------


def _mean_pairwise_corr(data: dict, tickers, window) -> float:
    rets = np.column_stack([
        np.diff(np.log(data[t].loc[window, "close"].to_numpy(dtype=np.float64)))
        for t in tickers
    ])
    c = np.corrcoef(rets, rowvar=False)
    return float(c[~np.eye(c.shape[0], dtype=bool)].mean())


def _effective_bets(data: dict, tickers, window) -> float:
    """Participation ratio of the correlation spectrum: independent directions."""
    rets = np.column_stack([
        np.diff(np.log(data[t].loc[window, "close"].to_numpy(dtype=np.float64)))
        for t in tickers
    ])
    eig = np.linalg.eigvalsh(np.corrcoef(rets, rowvar=False))
    return float((eig.sum() ** 2) / (eig ** 2).sum())


@pytest.fixture
def late_listing_panel():
    """Correlated assets, plus one that lists only in the final stretch.

    Mirrors the real panel's shape: staggered starts, common end, and one
    asset (EUL, 2025-10-13) whose listing date collapses the all-asset
    intersection to a window that excludes the training periods.
    """
    T = 600
    rng = np.random.default_rng(17)
    idx = pd.date_range("2024-01-01", periods=T, freq="D")
    market = rng.standard_normal(T) * 0.02
    panel = {}
    for a in range(5):
        ret = 0.9 * market + rng.standard_normal(T) * 0.005   # strongly correlated
        close = 100.0 * np.exp(np.cumsum(ret))
        df = pd.DataFrame(
            {"open": close, "high": close * 1.004, "low": close * 0.996,
             "close": close, "volume": np.full(T, 1e6)},
            index=idx,
        )
        panel[f"LONG{a}"] = df
    # The late lister: present only for the last 80 bars
    panel["LATE"] = panel["LONG0"].iloc[-80:].copy()
    return panel, idx


def test_correlation_preserved_in_an_early_window_despite_a_late_listing(
    late_listing_panel,
):
    """The regression: an EARLY window must keep its cross-asset correlation.

    The early window lies entirely outside the all-asset intersection, which is
    precisely where the previous implementation fell back to per-asset draws.
    """
    from vgp.analysis import block_bootstrap_ohlcv

    panel, idx = late_listing_panel
    long_tickers = [t for t in panel if t.startswith("LONG")]
    early = idx[:400]                     # ends long before LATE lists

    sur = block_bootstrap_ohlcv(panel, np.random.default_rng(0), block_size=20)

    real_corr = _mean_pairwise_corr(panel, long_tickers, early)
    sur_corr = _mean_pairwise_corr(sur, long_tickers, early)

    assert real_corr > 0.8, f"fixture is not correlated enough ({real_corr:.3f})"
    assert sur_corr > 0.7 * real_corr, (
        f"cross-asset correlation collapsed in the early window: real "
        f"{real_corr:.3f} -> surrogate {sur_corr:.3f}. The shared source "
        f"sequence is not reaching this window, so the assets are being drawn "
        f"independently and a cross-sectional book gets far too many "
        f"independent bets — which inflates the null's in-sample Sharpe."
    )


def test_effective_bets_preserved_in_an_early_window(late_listing_panel):
    """The quantity that actually drives achievable Sharpe dispersion.

    Correlation is the mechanism; the effective number of independent bets is
    what a 1/N book experiences. Before the fix this went from ~1 to ~5 on this
    fixture (2.4 to 20 on the real panel).
    """
    from vgp.analysis import block_bootstrap_ohlcv

    panel, idx = late_listing_panel
    long_tickers = [t for t in panel if t.startswith("LONG")]
    early = idx[:400]

    sur = block_bootstrap_ohlcv(panel, np.random.default_rng(1), block_size=20)

    real_n = _effective_bets(panel, long_tickers, early)
    sur_n = _effective_bets(sur, long_tickers, early)

    assert sur_n < 2.0 * real_n, (
        f"effective independent bets inflated from {real_n:.2f} to {sur_n:.2f} "
        f"in the early window — the surrogate is an easier problem than the "
        f"real data, so it is not a valid null"
    )


def test_correlation_preserved_across_several_disjoint_windows(late_listing_panel):
    """Every window the walk-forward grid can land on, not just one."""
    from vgp.analysis import block_bootstrap_ohlcv

    panel, idx = late_listing_panel
    long_tickers = [t for t in panel if t.startswith("LONG")]
    sur = block_bootstrap_ohlcv(panel, np.random.default_rng(2), block_size=20)

    for lo, hi in ((0, 150), (150, 300), (300, 450), (450, 600)):
        window = idx[lo:hi]
        real_corr = _mean_pairwise_corr(panel, long_tickers, window)
        sur_corr = _mean_pairwise_corr(sur, long_tickers, window)
        assert sur_corr > 0.7 * real_corr, (
            f"window {lo}:{hi} lost correlation: real {real_corr:.3f} -> "
            f"surrogate {sur_corr:.3f}"
        )


def test_full_sample_correlation_alone_would_not_catch_this(late_listing_panel):
    """Documents why the bug survived: the full-sample check passes either way.

    A surrogate can preserve the unconditional correlation matrix while
    destroying it inside every window, because correlation is time-varying and
    the window is what the model trains on. Any future check must be
    window-local.
    """
    from vgp.analysis import block_bootstrap_ohlcv

    panel, idx = late_listing_panel
    long_tickers = [t for t in panel if t.startswith("LONG")]
    sur = block_bootstrap_ohlcv(panel, np.random.default_rng(3), block_size=20)

    full = idx
    real_full = _mean_pairwise_corr(panel, long_tickers, full)
    sur_full = _mean_pairwise_corr(sur, long_tickers, full)

    # Both are high — this assertion held even when every window was broken
    assert real_full > 0.8 and sur_full > 0.7 * real_full, (
        "full-sample correlation should be preserved; if this fails the "
        "surrogate is broken in a more basic way"
    )


# ---------------------------------------------------------------------------
# Window-local fidelity check — the pipeline must catch an unfaithful surrogate
# itself, rather than relying on someone running the diagnostic script.
# ---------------------------------------------------------------------------

def test_fidelity_report_passes_for_a_faithful_surrogate(late_listing_panel):
    """A correct surrogate must clear the check in every window."""
    from vgp.analysis import block_bootstrap_ohlcv, window_fidelity_report

    panel, _idx = late_listing_panel
    sur = block_bootstrap_ohlcv(panel, np.random.default_rng(0), block_size=20)

    report = window_fidelity_report(panel, sur)

    assert report, "no comparable windows were produced"
    breaches = [r for r in report if r["breaches"]]
    assert not breaches, (
        f"faithful surrogate flagged in {len(breaches)} window(s): "
        f"{[(r['window'], r['breaches']) for r in breaches]}"
    )


def test_fidelity_report_catches_decorrelated_surrogate(late_listing_panel):
    """The check must flag the exact failure that inflated the null.

    Independently shuffling each asset preserves every marginal property and
    destroys only the cross-asset structure — the signature of the bug.
    """
    from vgp.analysis import window_fidelity_report

    panel, _idx = late_listing_panel
    rng = np.random.default_rng(4)
    broken = {}
    for t, df in panel.items():
        close = df["close"].to_numpy(dtype=np.float64)
        r = np.diff(np.log(close))
        rng.shuffle(r)                      # per-asset, independent
        new = np.empty_like(close)
        new[0] = close[0]
        new[1:] = close[0] * np.exp(np.cumsum(r))
        broken[t] = pd.DataFrame(
            {"open": new, "high": new * 1.004, "low": new * 0.996,
             "close": new, "volume": df["volume"].to_numpy()},
            index=df.index,
        )

    report = window_fidelity_report(panel, broken)
    breached = [r for r in report if r["breaches"]]

    assert breached, "a fully decorrelated surrogate was not flagged"
    flagged = {k for r in breached for k in r["breaches"]}
    assert "mean_corr" in flagged or "n_eff_bets" in flagged, (
        f"the cross-sectional statistics did not trip; flagged only {flagged}"
    )


def test_check_surrogate_fidelity_logs_breaches(late_listing_panel, caplog):
    """A breach must be logged at ERROR — it invalidates the p-value."""
    import logging

    from vgp.analysis import check_surrogate_fidelity

    panel, _idx = late_listing_panel
    rng = np.random.default_rng(5)
    broken = {}
    for t, df in panel.items():
        close = df["close"].to_numpy(dtype=np.float64)
        r = np.diff(np.log(close))
        rng.shuffle(r)
        new = np.empty_like(close)
        new[0] = close[0]
        new[1:] = close[0] * np.exp(np.cumsum(r))
        broken[t] = pd.DataFrame(
            {"open": new, "high": new * 1.004, "low": new * 0.996,
             "close": new, "volume": df["volume"].to_numpy()},
            index=df.index,
        )

    with caplog.at_level(logging.ERROR, logger="vgp.analysis.null_control"):
        check_surrogate_fidelity(panel, broken)

    assert any("FIDELITY BREACH" in m for m in caplog.messages), (
        "an unfaithful surrogate did not produce an ERROR-level log"
    )


def test_null_result_summary_surfaces_a_fidelity_breach():
    """The verdict text must not report a clean p-value over a broken null."""
    from vgp.analysis import NullControlResult

    res = NullControlResult(
        n_runs=19, block_size=20,
        observed_best_is_sharpe=3.7, observed_best_oos_sharpe=1.0,
        null_best_is_sharpe=np.full(19, 1.0),
        null_best_oos_sharpe=np.full(19, 0.0),
        fidelity=[{"window": "2024-01-01..2024-06-30",
                   "breaches": ["mean_corr", "n_eff_bets"]}],
    )

    summary = res.summary()
    assert "FIDELITY BREACH" in summary
    assert "not trustworthy" in summary
    assert res.fidelity_breaches == [
        "2024-01-01..2024-06-30: mean_corr, n_eff_bets"
    ]


def test_null_result_summary_confirms_verified_fidelity():
    """Conversely, a verified surrogate should say so."""
    from vgp.analysis import NullControlResult

    res = NullControlResult(
        n_runs=19, block_size=20,
        observed_best_is_sharpe=3.7, observed_best_oos_sharpe=1.0,
        null_best_is_sharpe=np.full(19, 1.0),
        null_best_oos_sharpe=np.full(19, 0.0),
        fidelity=[{"window": "w1", "breaches": []},
                  {"window": "w2", "breaches": []}],
    )

    assert "fidelity verified across 2 windows" in res.summary()
    assert not res.fidelity_breaches
