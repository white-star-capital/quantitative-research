"""Same seed, same answer — across PROCESSES, not just within one.

Every other reproducibility test in this suite runs the thing twice inside a
single interpreter, and that is structurally blind to the defect this module
exists for.

DEAP's `gp.cxOnePoint` chooses the crossover type with::

    common_types = set(types1.keys()).intersection(set(types2.keys()))
    type_ = random.choice(list(common_types))

The keys are type objects (`Vector`, `Scalar`), classes inherit the
address-derived `object.__hash__`, and so the order of that list depends on
where the interpreter loaded them. Addresses are constant for the life of a
process and randomised between processes, so two runs of the same seed consume
the identical RNG draw and index a differently ordered list — picking a
different type and diverging for the rest of the run. Measured here: 63% of
crossover pairs have two candidate types, the order flips in roughly 1 process
in 12, and a 30-individual 5-generation run produced two distinct Pareto fronts
across repeated invocations of one seed.

Two tests, doing different jobs:

* `test_candidate_types_are_ordered_before_the_draw` pins the mechanism. It is
  deterministic and fast, and it is the real guard.
* `test_evolution_is_reproducible_across_processes` is the end-to-end canary.
  It depends on ASLR actually flipping, so it is probabilistic — sized for the
  measured ~25% per-process divergence rate, which it must never see.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_the_ordering_key_actually_drives_the_choice(monkeypatch):
    """Reversing the ordering key reverses the candidate list handed to the RNG.

    The deterministic guard, and the one that cannot miss.

    Asserting merely that the candidates come out sorted is not enough: the
    types here are `Scalar` and `Vector`, set iteration usually yields them in
    that order anyway, and alphabetically that IS sorted. So a broken
    implementation using raw set order passes such a check on most machines —
    which is exactly how the defect survived, and why the first version of this
    test was worthless.

    Instead this swaps the ordering key for its reverse and asserts the
    candidate sequence flips with it. Code that sorts must respond; code that
    returns set iteration order cannot, because set order does not depend on
    the key. It therefore fails on every machine, every run.
    """
    import random as random_mod

    import numpy as np
    from deap import base, gp, tools

    from vgp.gp import variation
    from vgp.gp.gp_types import build_pset, creator
    from vgp.gp.primitives import Scalar, Vector

    def candidate_orders() -> list[list]:
        """Run crossovers and capture every multi-candidate type sequence."""
        random_mod.seed(7)
        np.random.seed(7)
        pset = build_pset()
        tb = base.Toolbox()
        tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4)
        tb.register("individual", tools.initIterate, creator.Individual, tb.expr)
        tb.register("population", tools.initRepeat, list, tb.individual)
        pop = tb.population(n=60)

        seen: list[list] = []
        real_choice = random_mod.choice

        def spy(seq):
            if seq and all(isinstance(x, type) for x in seq):
                seen.append(list(seq))
            return real_choice(seq)

        monkeypatch.setattr(variation.random, "choice", spy)
        for i in range(0, len(pop) - 1, 2):
            variation.cx_one_point(tb.clone(pop[i]), tb.clone(pop[i + 1]))
        monkeypatch.undo()
        return [s for s in seen if len(s) > 1]

    forward = candidate_orders()
    assert forward, (
        "no crossover saw more than one candidate type, so this test proved "
        "nothing — the population is not exercising the vulnerable path"
    )
    assert all(c == sorted(c, key=variation.stable_type_key) for c in forward)

    # Reverse the key. A sorting implementation must now emit the mirror image.
    monkeypatch.setattr(
        variation,
        "stable_type_key",
        lambda t: tuple(-ord(ch) for ch in getattr(t, "__qualname__", repr(t))),
    )
    reversed_ = candidate_orders()
    assert reversed_, "reversed-key pass saw no multi-candidate crossover"

    assert reversed_[0] != forward[0], (
        f"reversing the ordering key left the candidate order unchanged "
        f"({[t.__name__ for t in forward[0]]}) — the candidates are being taken "
        "in set iteration order, which no key can influence and which varies "
        "between processes"
    )
    assert reversed_[0] == list(
        reversed(forward[0])
    ), "reversing the key should mirror the candidate order exactly"
    assert variation.stable_type_key is not None
    assert {Scalar, Vector} == set(forward[0])


_EVOLVE = """
import logging, random
import numpy as np
logging.disable(logging.CRITICAL)
import pandas as pd
from pathlib import Path
from vgp.data import DataLoader, FeatureEngine
from vgp.backtest.runner import EvalConfig
from vgp.evolution.config import EvolutionConfig
from vgp.evolution.loop import run_evolution

loader = DataLoader(cache_dir=Path(r"{data}"))
ohlcv = loader.fetch_ohlcv(start_date="2024-01-01", end_date="2026-04-01",
                           allow_partial=True, min_assets=10)
fe = FeatureEngine(); fm = fe.fit_transform(ohlcv)
close = pd.DataFrame({{t: ohlcv[t]["close"] for t in fe.retained_assets_}}).reindex(fe.dates_)
cfg_eval = EvalConfig(fee_bps=10.0, min_trades=20, close_prices=close.iloc[:300].copy())
cfg = EvolutionConfig(pop_size=30, n_generations=5, seed=0, n_jobs=1, checkpoint_freq=999)
_p, hof, _l = run_evolution(cfg, fm[:300], cfg_eval)
print("|".join(repr(tuple(round(v, 12) for v in i.fitness.values)) for i in hof))
"""


@pytest.mark.slow
def test_evolution_is_reproducible_across_processes(tmp_path):
    """A full evolution at one seed gives one answer, whichever process runs it.

    End-to-end canary. Before the fix this produced two distinct Pareto fronts
    in roughly one run in four, so 10 processes would have caught it ~94% of the
    time. It is inherently probabilistic — the deterministic guard above is the
    one that cannot miss — but it is the only test that exercises the real loop
    the way real results are produced.
    """
    data = Path(__file__).resolve().parent.parent / "data"
    if not data.is_dir():
        pytest.skip("committed dataset not present")

    outputs = []
    for _ in range(10):
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(_EVOLVE.format(data=data))],
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr[-2000:]}"
        outputs.append(proc.stdout.strip())

    distinct = sorted(set(outputs))
    assert len(distinct) == 1, (
        f"{len(distinct)} different Pareto fronts from {len(outputs)} processes "
        f"at seed 0 — evolution is not reproducible across processes.\n"
        + "\n".join(f"  {d[:120]}" for d in distinct)
    )
