"""The feature diagnostic must separate a static asset property from timing.

The defect these guard against: an earlier diagnostic ranked features by RAW
cross-sectional rank IC and tested each against zero. A raw IC is dominated by
"is this asset persistently more volatile than that one", which is a property
of the assets rather than a prediction, and which any surrogate preserving each
asset's return distribution reproduces in full. Five features cleared a
Bonferroni threshold on that quantity; none of them carried timing information.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "diagnose_feature_ic.py"


@pytest.fixture(scope="module")
def diag():
    spec = importlib.util.spec_from_file_location("diagnose_feature_ic", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_standardising_removes_a_pure_static_level(diag):
    """A feature that only encodes WHICH asset it is must standardise to nothing.

    Build a panel where every asset's feature is a constant that differs across
    assets and never moves in time. That is the pathological case the raw IC
    scores highly and no strategy can trade. After within-asset standardisation
    the cross-section must carry no information at all.
    """
    T, A = 200, 8
    levels = np.linspace(1.0, 9.0, A)
    fm = np.broadcast_to(levels, (T, 1, A)).copy()  # [T, 1, A], constant in t
    assert fm.shape == (T, 1, A)

    out = diag._standardise_within_asset(fm)

    # sd is zero per asset, so every entry is undefined rather than silently 0.
    assert np.isnan(out).all(), "a constant feature must not survive standardisation"


def test_standardising_keeps_within_asset_variation_and_drops_the_level(diag):
    """Two assets with the same shape but different levels must come out identical.

    This is the whole point: the level is the untradeable part, the shape is the
    timing part. The raw feature distinguishes these assets; the standardised
    one must not.
    """
    T = 300
    rng = np.random.default_rng(0)
    shape = rng.normal(size=T)
    fm = np.empty((T, 1, 2))
    fm[:, 0, 0] = shape + 100.0  # same signal, huge level
    fm[:, 0, 1] = shape + 0.0  # same signal, no level

    out = diag._standardise_within_asset(fm)

    assert np.allclose(
        out[:, 0, 0], out[:, 0, 1], atol=1e-12
    ), "standardisation must remove the asset's own level, leaving only its shape"
    # And it must not have flattened the shape away.
    assert np.std(out[:, 0, 0]) == pytest.approx(1.0, abs=1e-9)


def test_a_scaled_asset_is_not_treated_as_a_different_signal(diag):
    """Doubling one asset's feature scale must not change its standardised series.

    An asset quoted in different units, or simply more volatile throughout, is
    not carrying more signal. Dividing by the asset's own sd is what makes the
    cross-section comparable.
    """
    T = 300
    rng = np.random.default_rng(1)
    shape = rng.normal(size=T)
    fm = np.empty((T, 1, 2))
    fm[:, 0, 0] = shape
    fm[:, 0, 1] = 7.5 * shape

    out = diag._standardise_within_asset(fm)

    assert np.allclose(out[:, 0, 0], out[:, 0, 1], atol=1e-12)


def test_the_ic_steps_use_a_block_shorter_than_the_effect(diag):
    """block_size must be below the horizon of the effect being tested.

    block_bootstrap_ohlcv severs serial dependence only at block boundaries, so
    with L-bar blocks an effect shorter than L survives into the surrogate and
    the null cannot flag it. These are 1-day-horizon ICs, so anything above 1
    makes the IC steps unfalsifiable.
    """
    assert diag.IC_BLOCK == 1, (
        "IC steps are 1-day horizon; a block above 1 lets the effect survive "
        "into the null and guarantees a negative result"
    )


def test_surrogate_count_can_resolve_the_bonferroni_threshold(diag):
    """An empirical p cannot resolve below 1/(1+N).

    With 12 features the Bonferroni threshold is 0.05/12 = 0.0042, so a run with
    too few surrogates cannot call anything significant no matter what the data
    says — the same defect as the 19-run null control that made p = 0.05 the
    best attainable result.
    """
    n_features = 12
    threshold = 0.05 / n_features
    floor = 1 / (1 + diag.N_NULL_IC)
    assert floor <= threshold, (
        f"N_NULL_IC={diag.N_NULL_IC} gives a p-value floor of {floor:.4f}, "
        f"above the Bonferroni threshold {threshold:.4f}: nothing could be called significant"
    )
