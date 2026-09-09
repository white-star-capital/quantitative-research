"""Smoke tests: verify the installed dependency set is coherent and usable.

This test file is the GATE that must pass before any backtest code is added.
It is run on every push to main via GitHub Actions (see .github/workflows/ci.yml).

Two classes of dependency failure are covered:

1. A version outside what a dependency actually supports (numpy vs numba).
   Asserted against the INSTALLED package's own declared requirement rather
   than a hardcoded bound — an earlier revision hardcoded numpy<2.3, which was
   numba 0.61's ceiling. numba 0.67 raised it to <2.6, so the hardcoded
   assertion outlived the constraint it described and began failing on a
   perfectly valid environment.

2. A pyproject.toml that declares a set no resolver can satisfy. This repo
   shipped `vectorbt==1.0.0` (which caps pandas<3.0) alongside
   `pandas>=3.0.0`, so `pip install -e .` failed outright for every user while
   the test suite — run against a hand-built environment — stayed green.
"""

import numba
import numpy as np
import pandas as pd
import pytest
from packaging.requirements import Requirement
from packaging.version import Version


def _requirements_of(dist: str) -> list[Requirement]:
    """Parse an installed distribution's declared requirements."""
    from importlib.metadata import requires

    out = []
    for raw in requires(dist) or []:
        try:
            req = Requirement(raw)
        except Exception:  # pragma: no cover — malformed metadata
            continue
        # Skip requirements that only apply to an optional extra
        if req.marker is not None and "extra" in str(req.marker):
            continue
        out.append(req)
    return out


def test_numpy_satisfies_installed_numba_requirement():
    """numpy must sit inside the range the INSTALLED numba declares.

    numba pins numpy tightly because it binds numpy's C extension APIs. Which
    range that is depends on the numba version, so read it from numba rather
    than hardcoding a bound that goes stale (see module docstring).
    """
    numpy_reqs = [r for r in _requirements_of("numba") if r.name.lower() == "numpy"]
    assert numpy_reqs, "numba declares no numpy requirement — metadata unexpected"

    for req in numpy_reqs:
        assert req.specifier.contains(np.__version__, prereleases=True), (
            f"numpy {np.__version__} is installed but numba "
            f"{Version(numba.__version__)} requires numpy{req.specifier}. "
            f"These must agree or numba fails at JIT compilation time."
        )


def test_project_dependencies_are_satisfied_by_environment():
    """Every dependency pin in pyproject.toml must be met by what is installed.

    Guards against a pyproject that cannot be installed at all: if the declared
    set is mutually contradictory, no environment can satisfy all of it, and
    this fails wherever the contradiction bites.
    """
    import tomllib
    from importlib.metadata import PackageNotFoundError, version
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.is_file():  # pragma: no cover — source checkouts always have it
        pytest.skip("pyproject.toml not found")

    declared = tomllib.loads(pyproject.read_text())["project"]["dependencies"]

    unmet = []
    for raw in declared:
        req = Requirement(raw)
        try:
            installed = version(req.name)
        except PackageNotFoundError:
            unmet.append(f"{req.name} is declared but not installed")
            continue
        if not req.specifier.contains(installed, prereleases=True):
            unmet.append(f"{req.name} {installed} installed, pyproject requires {req.specifier}")

    assert not unmet, (
        "pyproject.toml dependencies are not satisfied by this environment:\n  "
        + "\n  ".join(unmet)
        + "\nEither the pins are wrong or the environment is stale; run "
          "`pip install -e .` and re-check."
    )


def test_dependency_set_is_mutually_consistent():
    """Each installed project dependency must have ITS OWN requirements met.

    This is the `pip check` invariant, and the one that the original
    vectorbt==1.0.0 / pandas>=3.0.0 contradiction violated: vectorbt requires
    pandas<3.0, so no resolver could honour both. Declaring a pin is not the
    same as that pin being installable alongside the others.
    """
    import tomllib
    from importlib.metadata import PackageNotFoundError, version
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.is_file():  # pragma: no cover
        pytest.skip("pyproject.toml not found")

    declared = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    conflicts = []
    for raw in declared:
        dist = Requirement(raw).name
        try:
            version(dist)
        except PackageNotFoundError:
            continue
        for req in _requirements_of(dist):
            try:
                have = version(req.name)
            except PackageNotFoundError:
                continue  # optional / not part of this project's surface
            if not req.specifier.contains(have, prereleases=True):
                conflicts.append(
                    f"{dist} requires {req.name}{req.specifier} but {req.name} {have} is installed"
                )

    assert not conflicts, (
        "Installed dependencies conflict with each other:\n  "
        + "\n  ".join(conflicts)
    )


def test_numba_jit_compiles():
    """Verify numba can JIT-compile a function against the installed numpy version.

    This catches the most common failure mode: a numpy/numba version mismatch
    that causes an ImportError or RuntimeError at JIT compilation time, not at
    import time.
    """
    @numba.njit
    def _sum(x: np.ndarray) -> float:
        return np.sum(x)

    arr = np.ones(100, dtype=np.float64)
    result = _sum(arr)
    assert result == 100.0, f"numba JIT result was {result}, expected 100.0"


def test_deap_imports():
    """Verify all required DEAP sub-modules import cleanly."""
    import deap  # noqa: F401
    from deap import algorithms, base, creator, gp, tools  # noqa: F401


def test_vectorbt_from_signals():
    """Verify vectorbt 1.0.0 Portfolio.from_signals works end-to-end.

    Uses freq="1D" — required for sharpe_ratio() to return a non-NaN value.
    This is a documented 1.0.0 gotcha: omitting freq causes sharpe_ratio()
    to return NaN silently.
    """
    import vectorbt as vbt

    price = pd.Series([100.0, 102.0, 104.0, 102.0, 100.0])
    entries = pd.Series([True, False, False, False, False])
    exits = pd.Series([False, False, True, False, False])

    pf = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        size=1,
        direction="longonly",
        fees=0.001,
        freq="1D",
        init_cash=10_000.0,
    )

    sharpe = pf.sharpe_ratio()
    total_return = pf.total_return()

    assert sharpe is not None, "sharpe_ratio() returned None — is freq='1D' set?"
    assert not np.isnan(float(sharpe)), (
        "sharpe_ratio() returned NaN — this is the silent failure mode when freq= is missing. "
        "Check that freq='1D' is passed to Portfolio.from_signals."
    )
    assert total_return is not None, "total_return() returned None"


def test_pandas_idioms():
    """Verify pandas 3.0 idioms work correctly (CoW semantics, no .values usage).

    pandas 3.0 makes Copy-on-Write the default. Code written for pandas 2.x
    that uses chained assignment or .values will break silently or raise.
    This test confirms the project idioms work in the installed version.
    """
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

    # CORRECT: use .to_numpy(), not .values
    arr = df["a"].to_numpy()
    assert isinstance(arr, np.ndarray), ".to_numpy() must return np.ndarray"
    assert arr.tolist() == [1.0, 2.0, 3.0]

    # CORRECT: use .loc[] for assignment, not chained indexing
    df2 = df.copy()
    df2.loc[df2["a"] > 1.5, "b"] = 99.0
    assert df2.loc[1, "b"] == 99.0
    assert df2.loc[0, "b"] == 4.0  # unchanged
