---
phase: 01-foundation-environment
reviewed: 2026-06-08T00:00:00Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - pyproject.toml
  - vgp/__init__.py
  - vgp/data/__init__.py
  - vgp/gp/__init__.py
  - vgp/evolution/__init__.py
  - vgp/backtest/__init__.py
  - vgp/analysis/__init__.py
  - tests/__init__.py
  - tests/test_smoke.py
  - .github/workflows/ci.yml
findings:
  critical: 1
  warning: 3
  info: 1
  total: 5
status: issues_found
---

# Phase 1: Code Review Report

**Reviewed:** 2026-06-08T00:00:00Z
**Depth:** standard
**Files Reviewed:** 10
**Status:** issues_found

## Summary

Phase 1 delivers the project skeleton, dependency manifest, smoke tests, and CI workflow.
The dependency pins in `pyproject.toml` correctly match every constraint in CLAUDE.md
(numpy<2.3, deap==1.4.4, vectorbt==1.0.0, numba>=0.61.2, pandas>=3.0.0<4.0).
The architecture guard comments in the package `__init__.py` files are present and accurate.

One critical defect will prevent the package from installing at all: an invalid
`build-backend` path in `pyproject.toml`. Three warnings affect correctness of the
safety checks that are the primary deliverable of this phase: a string-comparison
version check that silently passes invalid numpy versions at 2.10+, a NaN-blind
sharpe assertion, and a CI trigger scope that skips all feature branches.

---

## Critical Issues

### CR-01: Invalid `build-backend` in `pyproject.toml`

**File:** `pyproject.toml:3`

**Issue:** The build backend is set to `"setuptools.backends.legacy:build"`. This path
does not exist in any released version of setuptools. The correct entry point is
`"setuptools.build_meta"`. As written, `pip install -e .` and `pip install -e ".[dev]"`
will both raise `ModuleNotFoundError: No module named 'setuptools.backends'` (or an
equivalent build-backend resolution error), making the package completely uninstallable
and the CI gate non-functional.

**Fix:**
```toml
[build-system]
requires = ["setuptools>=70.0", "wheel"]
build-backend = "setuptools.build_meta"
```

---

## Warnings

### WR-01: String comparison makes numpy version guard unreliable from numpy 2.10 onward

**File:** `tests/test_smoke.py:26`

**Issue:** `assert version < "2.3"` compares version strings lexicographically. This is
correct today (numpy 2.0–2.2 all sort below "2.3"). It will silently fail when numpy
releases 2.10: `"2.10" < "2.3"` is `True` (Python compares character by character:
`"1"` < `"3"`), so the test will **pass** with an incompatible numpy version. The
identical problem exists in the inline CI check at `.github/workflows/ci.yml:42`.

**Fix — `tests/test_smoke.py:26`:**
```python
from packaging.version import Version

def test_numpy_version_below_2_3():
    version = np.__version__
    assert Version(version) < Version("2.3"), (
        f"numpy {version} is installed but numpy<2.3 is required for numba compatibility. "
        f"Run: pip install 'numpy>=2.0.0,<2.3'"
    )
```

**Fix — `.github/workflows/ci.yml:39-44`:**
```yaml
- name: Verify numpy version constraint
  run: |
    python -c "
    from packaging.version import Version
    import numpy as np
    v = Version(np.__version__)
    assert v < Version('2.3'), f'numpy {np.__version__} violates <2.3 constraint'
    print(f'numpy {np.__version__} — OK (< 2.3)')
    "
```

`packaging` is a direct dependency of pip and is always available in any pip-managed
environment; it does not need to be added to `pyproject.toml`.

---

### WR-02: `sharpe_ratio()` NaN not caught by smoke test assertion

**File:** `tests/test_smoke.py:81`

**Issue:** The docstring at line 57 explicitly states that omitting `freq` causes
`sharpe_ratio()` to return `NaN` silently. The assertion `assert sharpe is not None`
does not catch this: `float("nan") is not None` evaluates to `True`, so a NaN sharpe
would pass the test. If the vectorbt 1.0.0 API changes `freq` handling, or if the test
is reused as a template without `freq="1D"`, the smoke test will give a false green.

**Fix:**
```python
sharpe = pf.sharpe_ratio()
assert sharpe is not None, "sharpe_ratio() returned None"
assert not (isinstance(sharpe, float) and np.isnan(sharpe)), (
    "sharpe_ratio() returned NaN — is freq='1D' set? "
    "vectorbt 1.0.0 requires freq for annualisation."
)
```

---

### WR-03: CI smoke gate does not run on feature branches

**File:** `.github/workflows/ci.yml:3-6`

**Issue:** The workflow triggers only on pushes to `main` and PRs targeting `main`.
Feature branch pushes (e.g., `vector-genetic-programming`) never run the smoke tests.
The smoke test is described in the code as "the GATE that must pass before any backtest
code is added." A gate that only fires at merge time is too late to catch breakage
during active development.

**Fix:**
```yaml
on:
  push:
    branches: ["**"]   # all branches
  pull_request:
    branches: [main]
```

Or, to limit noise while still catching feature branch breakage:
```yaml
on:
  push:
    branches: [main, "feature/**", "vector-**"]
  pull_request:
    branches: [main]
```

---

## Info

### IN-01: `pytest-benchmark` declared in dev deps with no benchmark tests yet

**File:** `pyproject.toml:55`

**Issue:** `pytest-benchmark>=4.0` is listed as a dev dependency but no `test_bench_*`
files or `benchmark` fixture usages exist. This adds install time to the dev environment
without providing value in Phase 1. It is appropriate for Phase 4 (evolution engine) but
premature here.

**Fix:** Remove from `[project.optional-dependencies].dev` for now and re-add when
the first benchmark test is written (expected in Phase 4). This is a minor cleanliness
issue and does not block the phase.

---

_Reviewed: 2026-06-08T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
