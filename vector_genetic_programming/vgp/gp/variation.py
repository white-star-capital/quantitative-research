"""Variation operators whose output depends on the RNG and nothing else.

DEAP's typed one-point crossover picks the crossover type like this
(`deap/gp.py`, `cxOnePoint`)::

    common_types = set(types1.keys()).intersection(set(types2.keys()))
    type_ = random.choice(list(common_types))

The keys are TYPE OBJECTS — `Vector` and `Scalar` here. Classes inherit
`object.__hash__`, which is derived from the object's address, so the iteration
order of a set of classes depends on where the interpreter happened to load
them. Address-space layout randomisation changes that between processes.

Two runs with the same seed therefore reach this line with identical RNG state,
consume the identical draw, and index a DIFFERENTLY ORDERED list — picking a
different type, a different crossover point, and diverging for the rest of the
run. Measured on this project: the order flipped in roughly one process in
four, and a 30-individual, 5-generation run produced two distinct Pareto fronts
across repeated invocations of the same seed.

It is invisible to in-process testing. Addresses are fixed for the life of a
process, so running an evolution twice inside one interpreter — which is what
`test_seed_reproducibility_exp03` and the serial-vs-pooled bit-identity test
both do — is perfectly reproducible. Only a fresh process can see it, and a
fresh process is how every real result is produced.

`cx_one_point` below is DEAP's function with one change: the candidate types
are ordered by a stable key before the choice. Note that it consumes exactly
the same number of RNG draws as the original, so it does not shift the stream —
it only makes the value the draw selects a function of the seed.
"""

from __future__ import annotations

import random
from collections import defaultdict

from deap import gp

__all__ = ["cx_one_point", "stable_type_key"]


def stable_type_key(type_: object) -> tuple[str, str]:
    """Order types by name rather than by address.

    Total and process-independent for anything DEAP puts in a primitive set:
    real classes carry `__module__`/`__qualname__`, and DEAP's untyped
    sentinel (`gp.__type__`) falls back to its repr.
    """
    module = getattr(type_, "__module__", "")
    name = getattr(type_, "__qualname__", None) or getattr(type_, "__name__", None)
    return (str(module), str(name) if name is not None else repr(type_))


def cx_one_point(ind1, ind2):
    """Typed one-point crossover, deterministic given the RNG state.

    Mirrors `deap.gp.cxOnePoint` exactly except that the shared types are
    sorted by `stable_type_key` instead of being taken in set order. See the
    module docstring for why that matters.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        return ind1, ind2

    types1: dict = defaultdict(list)
    types2: dict = defaultdict(list)

    if ind1.root.ret == gp.__type__:
        # Untyped primitive set: every node is interchangeable, so there is
        # only one candidate type and no ordering question to answer.
        types1[gp.__type__] = list(range(1, len(ind1)))
        types2[gp.__type__] = list(range(1, len(ind2)))
        common_types = [gp.__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):
            types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[1:], 1):
            types2[node.ret].append(idx)
        # THE FIX: a sorted list, not set iteration order.
        common_types = sorted(set(types1).intersection(types2), key=stable_type_key)

    if len(common_types) > 0:
        type_ = random.choice(common_types)

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2
