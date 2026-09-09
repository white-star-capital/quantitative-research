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
