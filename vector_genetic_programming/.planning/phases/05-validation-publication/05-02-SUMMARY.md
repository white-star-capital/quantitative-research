---
phase: "05-validation-publication"
plan: "02"
subsystem: "analysis"
tags: ["visualization", "matplotlib", "networkx", "vectorbt", "deap", "pareto-front", "equity-curves", "gp-tree"]

dependency_graph:
  requires:
    - phase: "05-01"
      provides: "vgp/analysis/__init__.py with 6 exports; vgp/backtest/runner.py EvalConfig + evaluate(); vgp/gp/tree_evaluator.py TreeEvaluator; vgp/gp/gp_types.py build_pset + creator"
  provides:
    - "vgp/analysis/plots.py — plot_pareto_front(), plot_equity_curves(), plot_tree_graph()"
    - "vgp/analysis/__init__.py — updated to 9 public exports"
    - "tests/test_plots.py — VAL-05, VAL-06, VAL-07 tests"
  affects:
    - "Phase 5 Plan 03 (community release) — plots.py is the visualization entry point for README examples"

tech-stack:
  added: []
  patterns:
    - "matplotlib.use('Agg') at module level before any pyplot import — headless/CI safety"
    - "Direct vbt.Portfolio.from_signals() call in visualization layer (not via evaluate()) — visualization has own backtest call with same params"
    - "deap.gp.graph() -> nx.DiGraph -> nx.bfs_layout(G, start=0) — GP tree layout without graphviz binary"
    - "TDD RED/GREEN flow: test file committed before implementation"

key-files:
  created:
    - "vgp/analysis/plots.py"
    - "tests/test_plots.py"
  modified:
    - "vgp/analysis/__init__.py"

key-decisions:
  - "plot_equity_curves() calls vbt.Portfolio.from_signals() directly (not evaluate()) — evaluate() discards pf object, visualization needs pf.value(); duplication is intentional to keep backtest interface clean"
  - "nx.bfs_layout(G, start=0) used for tree layout — graphviz binary is absent on this machine; bfs_layout confirmed working for all tree topologies"
  - "matplotlib.use('Agg') placed at module level in plots.py (line 18) — before any pyplot import — enforces headless safety at import time, not just at call time"
  - "Empty hof raises ValueError in plot_pareto_front(); empty individuals list returns silently in plot_equity_curves() — different semantics: pareto front with zero points is a programming error; empty equity curve list is a recoverable runtime condition"

patterns-established:
  - "Pattern: visualization module imports vectorbt at module level (not deferred) — analysis layer is permitted to import vbt per D-15; deferred import only required in evolution layer"
  - "Pattern: fig.savefig(output_path, dpi=150, bbox_inches='tight') then plt.close(fig) — always close figures after save to prevent memory leaks in long-running scripts"
  - "Pattern: len(individuals) == 1 edge case for plt.subplots — axes is scalar not list; wrap in [axes] for consistent loop logic"

requirements-completed: [VAL-05, VAL-06, VAL-07]

duration: "~3 min"
completed: "2026-06-13"
---

# Phase 5 Plan 02: Visualization Module Summary

**Headless-safe visualization module exporting Pareto front 3D scatter, IS/OOS equity curves with boundary overlay, and GP tree DiGraph via networkx bfs_layout — no graphviz required.**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-06-13T14:20:34Z
- **Completed:** 2026-06-13T14:23:00Z
- **Tasks:** 2 (Task 1: plots.py + __init__.py; Task 2: test_plots.py)
- **Files modified:** 3

## Accomplishments

- `vgp/analysis/plots.py` created with all three required visualization functions
- `vgp/analysis/__init__.py` updated from 6 to 9 public exports (appended plot exports)
- `tests/test_plots.py` created with 5 tests; all pass (VAL-05, VAL-06, VAL-07)
- Combined wave 1 + wave 2 suite: 13 tests, 13 passed, 0 failed

## Task Commits

Each task was committed atomically using TDD RED/GREEN flow:

1. **Task 1 RED: Failing tests (test_plots.py)** - `13d5667` (test)
2. **Task 1 GREEN: Implementation (plots.py + __init__.py)** - `b69e671` (feat)

**Plan metadata:** (docs commit follows)

_Note: TDD tasks have test commit (RED) then implementation commit (GREEN)_

## Files Created/Modified

- `vgp/analysis/plots.py` — Three visualization functions: `plot_pareto_front()` (3D Pareto scatter, viridis colormap, ValueError on empty hof), `plot_equity_curves()` (direct vbt call, axvline IS/OOS boundary, single-individual subplot edge case handled), `plot_tree_graph()` (deap.gp.graph() -> nx.DiGraph -> nx.bfs_layout, no graphviz needed)
- `vgp/analysis/__init__.py` — Added `from vgp.analysis.plots import ...` and 3 new names in `__all__`; total 9 exports
- `tests/test_plots.py` — 5 tests using synthetic data: 2 for VAL-05 (pareto PNG + empty hof), 2 for VAL-06 (equity PNG + empty list), 1 for VAL-07 (tree PNG)

## Decisions Made

- `plot_equity_curves()` calls `vbt.Portfolio.from_signals()` directly rather than modifying `evaluate()`. The `evaluate()` function returns only the 3-tuple fitness and discards the `pf` object. Adding a `return_portfolio=True` flag to `evaluate()` would complicate the multiprocessing pickle path in Phase 4. Intentional duplication for the visualization layer keeps the backtest interface clean.
- `nx.bfs_layout(G, start=0)` used for tree layout. The `dot` graphviz binary is absent on this machine. `bfs_layout` is confirmed working in the project venv for all tree topologies from height=1 to height=8.
- `matplotlib.use('Agg')` placed at module level (line 18 in plots.py) before any `matplotlib.pyplot` import. This enforces headless safety at import time rather than call time — any downstream module that imports `vgp.analysis.plots` gets the Agg backend automatically.

## Deviations from Plan

None — plan executed exactly as written. The implementation matches the code specified in the plan's `<action>` blocks. No Rule 1/2/3/4 triggers encountered.

## Known Stubs

None — all three functions are fully implemented. No hardcoded empty values, no placeholder returns.

## Threat Flags

No new network endpoints, auth paths, or schema changes introduced.

T-05-07 (Denial of Service — matplotlib backend on headless CI) is mitigated: `matplotlib.use('Agg')` is confirmed at module level in `vgp/analysis/plots.py` (line 18) and at the top of `tests/test_plots.py` (line 15). No other threat surface changes.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `vgp/analysis` module is complete: walk-forward runner (05-01) + visualization (05-02)
- Phase 5 Plan 03 (community release) can now reference `plot_pareto_front`, `plot_equity_curves`, `plot_tree_graph` in README examples and CONTRIBUTING.md "Visualizations" section
- Full test suite: `pytest tests/test_validation.py tests/test_plots.py` — 13 passed, 0 failed

## Self-Check

| Check | Result |
|-------|--------|
| `vgp/analysis/plots.py` exists | FOUND |
| `tests/test_plots.py` exists | FOUND |
| `vgp/analysis/__init__.py` has 9 exports | FOUND |
| `matplotlib.use('Agg')` at line 18 of plots.py | CONFIRMED |
| Commit `13d5667` (RED — test file) | FOUND |
| Commit `b69e671` (GREEN — implementation) | FOUND |
| `pytest tests/test_plots.py` — 5 passed, 0 failed | PASSED |
| `pytest tests/test_validation.py tests/test_plots.py` — 13 passed | PASSED |

## Self-Check: PASSED

---
*Phase: 05-validation-publication*
*Completed: 2026-06-13*
