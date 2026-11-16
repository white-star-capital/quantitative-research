"""Trial bookkeeping for the Deflated Sharpe Ratio.

DEPENDENCY NOTE: numpy only. This module sits below both `vgp.evolution` and
`vgp.analysis` so the evolution loop can record trials without importing
`vgp.analysis` (which would pull vectorbt in through `vgp.analysis.runner` and
break the D-15 invariant).

WHAT COUNTS AS A TRIAL
----------------------
In Bailey & Lopez de Prado (2014), N is the number of configurations the
researcher effectively searched before reporting the best one. For a genetic
program that is the number of individuals EVALUATED — pop_size x generations x
seeds x windows, typically thousands — not the handful of per-seed winners that
end up in the results table.

Counting only the reported winners understates N by three orders of magnitude
and makes the multiple-testing correction almost inert: with N = 9 the hurdle
multiplier is 1.52, with N = 3600 it is 3.24, so the deflation roughly doubles.

The opposite bias is real too, and neither this module nor the DSR formula can
remove it: Proposition 3 assumes INDEPENDENT trials, and GP individuals are
correlated by descent — offspring resemble their parents, and later generations
concentrate in whatever region of strategy space the search already favours. So
the effective number of independent trials is somewhere below the evaluation
count and far above the winner count. Report both bounds rather than one point
estimate; `vgp.analysis.dsr.attach_dsr()` does that.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrialSet:
    """The two properties of a trial set that the DSR hurdle depends on.

    Attributes
    ----------
    n_trials : int
        Number of trials with a finite Sharpe ratio. Worst-fitness sentinels
        are not trials with a measured Sharpe and are excluded upstream.
    sr_std : float
        Standard deviation (ddof=1) of the trial Sharpe ratios, in the same
        units they were supplied in — annualized, by convention here.
    label : str
        Where the set came from, so a reported DSR can be traced back to the
        trial population it was deflated against.
    """

    n_trials: int
    sr_std: float
    label: str = "unknown"

    @property
    def is_usable(self) -> bool:
        """True when the set can support a DSR hurdle at all."""
        return self.n_trials >= 2 and math.isfinite(self.sr_std) and self.sr_std > 0.0

    @classmethod
    def from_sharpes(cls, sharpes, label: str = "sharpes") -> TrialSet:
        """Build from an array of Sharpe ratios, dropping non-finite entries."""
        arr = np.asarray(
            list(sharpes) if not hasattr(sharpes, "dtype") else sharpes, dtype=np.float64
        ).ravel()
        finite = arr[np.isfinite(arr)]
        n = int(finite.size)
        std = float(np.std(finite, ddof=1)) if n >= 2 else float("nan")
        return cls(n_trials=n, sr_std=std, label=label)


class TrialAccumulator:
    """Streaming count and standard deviation of evaluated trial Sharpes.

    Welford's algorithm — O(1) memory regardless of how many individuals are
    evaluated. This matters because the accumulator travels inside the DEAP
    logbook, which is checkpointed with dill: retaining every sample would put
    hundreds of thousands of floats into each checkpoint for no benefit, since
    the DSR hurdle needs only the count and the spread.

    Non-finite values (the (-inf, -inf, -size) worst-fitness sentinel) are
    counted separately and excluded from the statistics — they are individuals
    that were searched but produced no measurable Sharpe.
    """

    __slots__ = ("_n", "_mean", "_m2", "_n_nonfinite")

    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._n_nonfinite = 0

    def add(self, sharpe: float) -> None:
        """Record one evaluated individual's Sharpe ratio."""
        x = float(sharpe)
        if not math.isfinite(x):
            self._n_nonfinite += 1
            return
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (x - self._mean)

    def extend(self, sharpes) -> None:
        """Record many evaluated individuals."""
        for s in sharpes:
            self.add(s)

    def merge(self, other: TrialAccumulator) -> None:
        """Fold another accumulator in (Chan et al. parallel variance)."""
        if other._n == 0:
            self._n_nonfinite += other._n_nonfinite
            return
        if self._n == 0:
            self._n, self._mean, self._m2 = other._n, other._mean, other._m2
            self._n_nonfinite += other._n_nonfinite
            return
        n_ab = self._n + other._n
        delta = other._mean - self._mean
        self._m2 += other._m2 + (delta * delta) * self._n * other._n / n_ab
        self._mean += delta * other._n / n_ab
        self._n = n_ab
        self._n_nonfinite += other._n_nonfinite

    @property
    def n_finite(self) -> int:
        """Evaluated individuals that produced a measurable Sharpe."""
        return self._n

    @property
    def n_nonfinite(self) -> int:
        """Evaluated individuals that returned the worst-fitness sentinel."""
        return self._n_nonfinite

    @property
    def n_evaluations(self) -> int:
        """Total individuals evaluated, measurable or not."""
        return self._n + self._n_nonfinite

    @property
    def sr_std(self) -> float:
        """Sample standard deviation (ddof=1) of the finite Sharpes."""
        if self._n < 2:
            return float("nan")
        return math.sqrt(self._m2 / (self._n - 1))

    @property
    def sr_mean(self) -> float:
        return self._mean if self._n else float("nan")

    def to_trial_set(self, label: str = "evaluations") -> TrialSet:
        return TrialSet(n_trials=self._n, sr_std=self.sr_std, label=label)

    def __repr__(self) -> str:  # pragma: no cover — diagnostics only
        return (
            f"TrialAccumulator(n_evaluations={self.n_evaluations}, "
            f"n_finite={self._n}, n_nonfinite={self._n_nonfinite}, "
            f"sr_mean={self.sr_mean:.4f}, sr_std={self.sr_std:.4f})"
        )
