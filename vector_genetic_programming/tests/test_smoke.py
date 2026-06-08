"""Smoke tests: verify numba/numpy/vectorbt compatibility.

This test file is the GATE that must pass before any backtest code is added.
It is run on every push to main via GitHub Actions (see .github/workflows/ci.yml).

CRITICAL: test_numpy_version_below_2_3 will FAIL if numpy>=2.3 is installed.
NumPy 2.3 hard-breaks numba. The pyproject.toml pin (numpy>=2.0.0,<2.3) prevents
this from happening in normal installs, but this test provides a runtime safety net.
"""

import numpy as np
import numba
import pandas as pd
import pytest
from packaging.version import Version


def test_numpy_version_below_2_3():
    """Assert numpy is <2.3 — required for numba compatibility.

    NumPy 2.3 (mid-2026) breaks numba's internal C extension APIs.
    numba>=0.61.2 supports numpy 2.0, 2.1, 2.2 — but NOT 2.3+.
    If this test fails, downgrade numpy or wait for a numba release that
    supports the newer numpy version.

    Uses packaging.version.Version for correct semantic comparison — string
    comparison fails for numpy 2.10+ ("2.10" < "2.3" is True lexicographically).
    """
    version = np.__version__
    assert Version(version) < Version("2.3"), (
        f"numpy {version} is installed but numpy<2.3 is required for numba compatibility. "
        f"Run: pip install 'numpy>=2.0.0,<2.3'"
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
