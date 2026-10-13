"""Trial accounting for the DSR multiple-testing correction (vgp/trials.py).

The trial count is not bookkeeping trivia: it sets the size of the DSR hurdle.
Counting only the per-seed winners (N = 9) instead of every individual the GP
evaluated (N in the thousands) shrinks the hurdle multiplier from ~3.2 to ~1.5,
which is the difference between rejecting a noise-derived strategy and
certifying it. These tests pin the accounting.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# TrialAccumulator: streaming statistics must match the batch computation
# ---------------------------------------------------------------------------


def test_accumulator_matches_numpy_std():
    """Welford's algorithm must agree with numpy to floating-point precision."""
    from vgp.trials import TrialAccumulator

    rng = np.random.default_rng(0)
    xs = rng.standard_normal(10_000) * 2.5 + 1.0

    acc = TrialAccumulator()
    acc.extend(xs)

    assert acc.n_finite == xs.size
    assert acc.sr_std == pytest.approx(float(np.std(xs, ddof=1)), rel=1e-12)
    assert acc.sr_mean == pytest.approx(float(np.mean(xs)), rel=1e-12)


def test_accumulator_excludes_worst_fitness_sentinels():
    """-inf and NaN are searched individuals with no measurable Sharpe.

    They must count toward n_evaluations (they were trials) but never enter the
    statistics, where a single -inf would make sigma_SR meaningless.
    """
    from vgp.trials import TrialAccumulator

    acc = TrialAccumulator()
    acc.extend([1.0, 2.0, 3.0])
    acc.add(-np.inf)
    acc.add(np.inf)
    acc.add(np.nan)

    assert acc.n_finite == 3
    assert acc.n_nonfinite == 3
    assert acc.n_evaluations == 6
    assert math.isfinite(acc.sr_std), "sentinels leaked into sigma_SR"
    assert acc.sr_std == pytest.approx(1.0)


def test_accumulator_merge_is_exact():
    """Merging per-seed accumulators must equal accumulating the whole set.

    attach_dsr() merges one accumulator per (window, seed), so a biased merge
    would mis-size the correction for the whole experiment.
    """
    from vgp.trials import TrialAccumulator

    rng = np.random.default_rng(1)
    xs = rng.standard_normal(5_000) * 3.0

    whole = TrialAccumulator()
    whole.extend(xs)

    parts = TrialAccumulator()
    for chunk in np.array_split(xs, 7):
        sub = TrialAccumulator()
        sub.extend(chunk)
        parts.merge(sub)

    assert parts.n_finite == whole.n_finite
    assert parts.sr_std == pytest.approx(whole.sr_std, rel=1e-10)
    assert parts.sr_mean == pytest.approx(whole.sr_mean, rel=1e-10)


def test_accumulator_merge_with_empty():
    """Merging an empty accumulator must be a no-op on the statistics."""
    from vgp.trials import TrialAccumulator

    acc = TrialAccumulator()
    acc.extend([1.0, 2.0, 3.0])
    before = (acc.n_finite, acc.sr_std)

    empty = TrialAccumulator()
    empty.add(-np.inf)  # nonfinite only
    acc.merge(empty)

    assert (acc.n_finite, acc.sr_std) == pytest.approx(before)
    assert acc.n_nonfinite == 1, "nonfinite count must still carry over"

    # And merging INTO an empty one adopts the other's statistics
    fresh = TrialAccumulator()
    fresh.merge(acc)
    assert fresh.n_finite == acc.n_finite
    assert fresh.sr_std == pytest.approx(acc.sr_std)


def test_accumulator_memory_is_constant():
    """The accumulator must not retain samples — it rides inside dill checkpoints."""
    from vgp.trials import TrialAccumulator

    small, large = TrialAccumulator(), TrialAccumulator()
    small.extend(np.zeros(10) + np.arange(10))
    large.extend(np.arange(100_000, dtype=np.float64))

    # __slots__ with only scalar state: no container can grow with input size
    assert not hasattr(small, "__dict__"), "accumulator must use __slots__"
    for name in TrialAccumulator.__slots__:
        assert isinstance(
            getattr(large, name), (int, float)
        ), f"slot {name!r} is not a scalar — samples are being retained"


# ---------------------------------------------------------------------------
# TrialSet: usability gating
# ---------------------------------------------------------------------------


def test_trial_set_from_sharpes_drops_nonfinite():
    from vgp.trials import TrialSet

    ts = TrialSet.from_sharpes([1.0, 2.0, 3.0, -np.inf, np.nan])
    assert ts.n_trials == 3
    assert ts.sr_std == pytest.approx(1.0)
    assert ts.is_usable


@pytest.mark.parametrize(
    "sharpes,reason",
    [
        ([], "no trials"),
        ([1.0], "one trial has no spread"),
        ([2.0, 2.0, 2.0], "zero spread"),
        ([-np.inf, np.nan], "no finite trials"),
    ],
)
def test_trial_set_unusable_cases(sharpes, reason):
    """A trial set that cannot support a hurdle must say so, not fake one."""
    from vgp.trials import TrialSet

    assert not TrialSet.from_sharpes(sharpes).is_usable, reason


def test_accumulator_to_trial_set_roundtrip():
    from vgp.trials import TrialAccumulator

    acc = TrialAccumulator()
    acc.extend([1.0, 2.0, 3.0, 4.0])
    ts = acc.to_trial_set(label="all_evaluations")

    assert ts.n_trials == 4
    assert ts.label == "all_evaluations"
    assert ts.sr_std == pytest.approx(float(np.std([1.0, 2.0, 3.0, 4.0], ddof=1)))
    assert ts.is_usable
