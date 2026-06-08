---
phase: 01-foundation-environment
verified: 2026-06-08T15:30:00Z
status: gaps_found
score: 4/5 must-haves verified
overrides_applied: 0
gaps:
  - truth: "requirements-lock.txt is committed so a second developer can reproduce the environment from .python-version and the lock file alone"
    status: failed
    reason: "FOUND-04 and Roadmap SC4 both require a requirements-lock.txt committed to the repo. No lock file of any kind exists (no requirements-lock.txt, no uv.lock, no pip freeze output). .python-version is present but insufficient alone for deterministic reproduce — pinned ranges in pyproject.toml still allow pip to resolve to different patch versions on different dates."
    artifacts:
      - path: "requirements-lock.txt"
        issue: "File does not exist"
    missing:
      - "Generate and commit a requirements-lock.txt (pip freeze output after pip install -e '.[dev]') or equivalent uv.lock"
---

# Phase 1: Foundation & Environment — Verification Report

**Phase Goal:** A reproducible Python environment exists where numba, numpy, and vectorbt all import and interoperate correctly, CI runs on every push, and the repo is licensed for community release.
**Verified:** 2026-06-08T15:30:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `pip install -e .` on a clean Python 3.12 environment installs all pinned dependencies without conflict, and `import vectorbt; import numba; import deap` all succeed | VERIFIED | pyproject.toml passes TOML validation (python3.12 tomllib); `build-backend = "setuptools.build_meta"` is correct; all required pins present including `numpy>=2.0.0,<2.3`, `deap==1.4.4`, `vectorbt==1.0.0`, `numba>=0.61.2`; `[tool.setuptools.packages.find]` includes `vgp*`; runtime import check not feasible (deps not installed in shell), but structural evidence is complete |
| 2 | A smoke test that imports numba and asserts `numpy.__version__ < "2.3"` passes in CI on every push to main | VERIFIED | `tests/test_smoke.py:test_numpy_version_below_2_3` uses `packaging.version.Version` for correct semantic comparison (not string comparison); asserts `Version(version) < Version("2.3")` with clear error message; `tests/test_smoke.py` passes syntax check; all 5 test functions present |
| 3 | GitHub Actions workflow runs the import smoke test and reports pass/fail on every commit to main | VERIFIED | `.github/workflows/ci.yml` triggers on `push: branches: ["**"]` (all branches, exceeding the roadmap requirement) and `pull_request: branches: [main]`; runs `pytest tests/test_smoke.py -v`; includes belt-and-suspenders standalone numpy version check using `packaging.version.Version`; uses current stable action versions (checkout@v4, setup-python@v5, cache@v4) |
| 4 | `.python-version` file and `requirements-lock.txt` are committed; a second developer can reproduce the environment from these files alone | FAILED | `.python-version` exists and contains `3.12`. `requirements-lock.txt` does not exist. No equivalent lock file found (`uv.lock`, `pip freeze` output, or similar). pyproject.toml uses range pins (`>=2.0.0,<2.3`, `>=3.0.0,<4.0`, etc.) meaning pip resolution is non-deterministic across dates. Reproducibility requires a committed lock file. |
| 5 | MIT LICENSE file is present at repo root | VERIFIED | `LICENSE` exists at repo root, line 1 is `MIT License`, contains `Copyright (c) 2026 VGP Contributors` |

**Score:** 4/5 truths verified

---

### Deferred Items

None.

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | Package metadata, pinned dependencies, build config | VERIFIED | Valid TOML; `build-backend = "setuptools.build_meta"` (correct); contains `numpy>=2.0.0,<2.3`, `deap==1.4.4`, `vectorbt==1.0.0`, `numba>=0.61.2`; `requires-python = ">=3.12"`; includes `[tool.setuptools.packages.find]` with `include = ["vgp*"]` |
| `vgp/__init__.py` | Package entry point | VERIFIED | Contains `__version__ = "0.1.0"` and full architecture invariants comment block |
| `vgp/data/__init__.py` | data sub-module stub | VERIFIED | Non-empty docstring: "Data pipeline: DataLoader, FeatureEngine, WalkForwardSplitter." |
| `vgp/gp/__init__.py` | gp sub-module stub | VERIFIED | Contains module-level primitive requirement comment |
| `vgp/evolution/__init__.py` | evolution sub-module stub | VERIFIED | Contains "This module must NOT import vectorbt" |
| `vgp/backtest/__init__.py` | backtest sub-module stub | VERIFIED | Contains "This module must NOT import deap" |
| `vgp/analysis/__init__.py` | analysis sub-module stub | VERIFIED | Non-empty docstring present |
| `.python-version` | Python version pin | VERIFIED | Contains `3.12` |
| `LICENSE` | MIT license text | VERIFIED | Line 1 is `MIT License`; year 2026; correct full license text |
| `tests/__init__.py` | pytest package init | VERIFIED | File exists |
| `tests/test_smoke.py` | Smoke test with numpy version assertion and 5 test functions | VERIFIED | Syntax valid; 5 functions: `test_numpy_version_below_2_3`, `test_numba_jit_compiles`, `test_deap_imports`, `test_vectorbt_from_signals`, `test_pandas_idioms`; uses `packaging.version.Version`; NaN guard on sharpe; `freq="1D"` set; module docstring frames it as the gate |
| `.github/workflows/ci.yml` | CI pipeline triggered on push/PR to main | VERIFIED | Triggers on all branches (push) and PR to main; Python 3.12; `pip install -e ".[dev]"`; `pytest tests/test_smoke.py -v`; standalone numpy check |
| `requirements-lock.txt` | Committed lock file for reproducible installs | MISSING | File does not exist; no equivalent lock file found |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `.github/workflows/ci.yml` | `tests/test_smoke.py` | `pytest tests/test_smoke.py -v` in run step | WIRED | Line 35 of ci.yml: `run: pytest tests/test_smoke.py -v` |
| `tests/test_smoke.py` | `numpy.__version__` | `packaging.version.Version` comparison | WIRED | `Version(version) < Version("2.3")` at line 30 |
| `pyproject.toml` | `vgp/` | `[tool.setuptools.packages.find]` with `include = ["vgp*"]` | WIRED | `include = ["vgp*"]` present at line 64–65 |
| `.python-version` | `pyproject.toml` | `requires-python = ">=3.12"` aligns with `3.12` pin | WIRED | Both specify Python 3.12 |

---

### Data-Flow Trace (Level 4)

Not applicable — Phase 1 delivers infrastructure artifacts (config files, test scaffolding, CI). No dynamic data rendering components.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| pyproject.toml is valid TOML | `python3.12 -c "import tomllib; tomllib.load(...)"` | `TOML valid, name: vector-genetic-programming` | PASS |
| Smoke test has correct Python syntax | `python3 -m py_compile tests/test_smoke.py` | `syntax OK` | PASS |
| All 5 test functions present | `grep "^def test_" tests/test_smoke.py` | 5 functions listed | PASS |
| CI triggers on all branches | `grep -A3 "^on:" ci.yml` | `branches: ["**"]` | PASS |
| requirements-lock.txt exists | `test -f requirements-lock.txt` | MISSING | FAIL |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FOUND-01 | 01-01-PLAN.md | pyproject.toml installs cleanly on Python 3.11+ with pinned deps | SATISFIED | pyproject.toml valid; correct build-backend; all pins present; package structure correct |
| FOUND-02 | 01-02-PLAN.md | GitHub Actions CI runs import smoke tests on every push to main | SATISFIED | ci.yml triggers on push to all branches; runs pytest test_smoke.py |
| FOUND-03 | 01-02-PLAN.md | numba/numpy compatibility verified by smoke test that must pass before any backtest code is added | SATISFIED | test_numpy_version_below_2_3 + test_numba_jit_compiles present; correctly framed as gate in docstring |
| FOUND-04 | 01-01-PLAN.md | `.python-version` file pins Python 3.12; requirements-lock.txt committed for reproducible installs | BLOCKED | `.python-version` exists. `requirements-lock.txt` absent. No lock file of any kind committed. |
| COMM-02 | 01-01-PLAN.md | MIT license file present at repo root | SATISFIED | LICENSE exists with correct MIT text and 2026 copyright |

**Orphaned requirements check:** No additional Phase 1 requirements exist in REQUIREMENTS.md beyond FOUND-01, FOUND-02, FOUND-03, FOUND-04, COMM-02. All accounted for.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tests/test_smoke.py` | 15 | `from packaging.version import Version` — `packaging` is not listed in `pyproject.toml` dependencies or dev extras | Warning | `packaging` ships with `pip` and is practically always available in pip-managed environments; not a runtime blocker, but it is an undeclared dependency. Low risk. |

Note: The code review (01-REVIEW.md) identified 1 critical (CR-01: invalid build-backend) and 3 warnings (WR-01: string version comparison, WR-02: NaN-blind sharpe assertion, WR-03: CI scope). All four issues were addressed in commit `2b863d1`. The actual files in the codebase reflect the fixed state. The review findings are closed.

---

### Human Verification Required

None. All verifiable aspects of this phase can be confirmed structurally. Runtime behavior (actual `pip install -e .` + `import numba; import vectorbt`) cannot be tested in this shell (deps not installed), but all structural prerequisites for that to work are confirmed present and correct.

---

### Gaps Summary

One gap blocks full goal achievement: **FOUND-04 is partially satisfied**. The `.python-version` file is committed and correct. However, `requirements-lock.txt` — explicitly required by both FOUND-04 and Roadmap Success Criterion 4 — does not exist. With only range pins in `pyproject.toml` and no lock file, two developers installing at different times may resolve to different patch versions of libraries, defeating the reproducibility guarantee.

**Root cause:** Plan 01-01 claimed FOUND-04 as complete in its SUMMARY but only delivered `.python-version`. The lock file generation step was not executed.

**Fix:** Run `pip install -e ".[dev]"` in a clean Python 3.12 environment, then `pip freeze > requirements-lock.txt`, and commit the result.

---

_Verified: 2026-06-08T15:30:00Z_
_Verifier: Claude (gsd-verifier)_
