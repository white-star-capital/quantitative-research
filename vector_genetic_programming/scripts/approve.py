#!/usr/bin/env python3
"""Human verification scenarios for VGP milestone phases.

Usage
-----
    python scripts/approve.py           # run all scenarios for the current phase
    python scripts/approve.py --phase 4 # run phase 4 scenarios only
    make approve                        # same as the first form

Each phase registers its scenarios below.  When a new phase is planned,
add a new `@scenario(phase=N)` block.  Nothing else changes.

Exit codes
----------
    0 — all scenarios passed
    1 — one or more scenarios failed
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Minimal scenario registry
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    phase: int
    name: str
    description: str
    fn: Callable[[], None]


_registry: list[Scenario] = []


def scenario(phase: int, name: str, description: str):
    """Decorator that registers a verification scenario for a phase."""
    def decorator(fn: Callable[[], None]) -> Callable[[], None]:
        _registry.append(Scenario(phase=phase, name=name, description=description, fn=fn))
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Phase 4 scenarios
# ---------------------------------------------------------------------------

@scenario(
    phase=4,
    name="parallel_eval_no_errors",
    description=(
        "Run evolution with n_jobs=2 (spawn pool + _jit_warmup) on synthetic data.\n"
        "Verifies: workers start, JIT warmup fires, evolution returns (pop, hof, logbook)."
    ),
)
def verify_phase4_parallel_eval() -> None:
    import numpy as np
    import pandas as pd
    from vgp.backtest.runner import EvalConfig
    from vgp.evolution.config import EvolutionConfig
    from vgp.evolution.loop import run_evolution

    T, F, A = 300, 12, 2
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=T, freq="D")
    close = pd.DataFrame(
        (100.0 * np.exp(np.cumsum(rng.standard_normal((T, A)) * 0.01, axis=0))).astype(np.float64),
        index=dates,
        columns=[f"asset_{i}" for i in range(A)],
    )
    fm = rng.standard_normal((T, F, A)).astype(np.float32)
    eval_cfg = EvalConfig(close_prices=close, min_trades=1)

    cfg_serial = EvolutionConfig(pop_size=20, n_generations=3, seed=42, n_jobs=1, checkpoint_freq=999)
    cfg_parallel = EvolutionConfig(pop_size=20, n_generations=3, seed=42, n_jobs=2, checkpoint_freq=999)

    t0 = time.perf_counter()
    pop_s, hof_s, _ = run_evolution(cfg_serial, fm, eval_cfg)
    serial_time = time.perf_counter() - t0
    assert len(pop_s) == 20, f"serial: expected pop_size=20, got {len(pop_s)}"
    assert len(hof_s) > 0, "serial: hof is empty"

    t0 = time.perf_counter()
    pop_p, hof_p, _ = run_evolution(cfg_parallel, fm, eval_cfg)
    parallel_time = time.perf_counter() - t0
    assert len(pop_p) == 20, f"parallel: expected pop_size=20, got {len(pop_p)}"
    assert len(hof_p) > 0, "parallel: hof is empty after parallel run"

    print(f"    serial  n_jobs=1: {serial_time:.1f}s")
    print(f"    parallel n_jobs=2: {parallel_time:.1f}s")
    print(f"    hof sizes: serial={len(hof_s)}, parallel={len(hof_p)}")


# ---------------------------------------------------------------------------
# Phase 5 scenarios (add here when Phase 5 is implemented)
# ---------------------------------------------------------------------------

# @scenario(
#     phase=5,
#     name="walk_forward_oos_split",
#     description="Walk-forward splitter enforces non-overlapping OOS windows.",
# )
# def verify_phase5_walk_forward() -> None:
#     ...


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _current_phase() -> int | None:
    """Read current phase from STATE.md, return None if unreadable."""
    from pathlib import Path
    state = Path(__file__).parent.parent / ".planning" / "STATE.md"
    if not state.exists():
        return None
    for line in state.read_text().splitlines():
        line = line.strip()
        if line.startswith("Phase:") or line.startswith("phase:"):
            try:
                return int(line.split(":")[1].split()[0])
            except (IndexError, ValueError):
                pass
    return None


def run_scenarios(phase_filter: int | None) -> bool:
    scenarios = [s for s in _registry if phase_filter is None or s.phase == phase_filter]
    if not scenarios:
        target = f"phase {phase_filter}" if phase_filter else "any phase"
        print(f"No scenarios registered for {target}.")
        return True

    print(f"\nRunning {len(scenarios)} scenario(s)" +
          (f" for phase {phase_filter}" if phase_filter else "") + ":\n")

    passed = failed = 0
    for s in scenarios:
        label = f"[phase {s.phase}] {s.name}"
        print(f"  ▶ {label}")
        for line in s.description.strip().splitlines():
            print(f"    {line}")
        try:
            t0 = time.perf_counter()
            s.fn()
            elapsed = time.perf_counter() - t0
            print(f"  ✓ PASSED ({elapsed:.1f}s)\n")
            passed += 1
        except Exception:
            print(f"  ✗ FAILED\n")
            traceback.print_exc()
            print()
            failed += 1

    width = 56
    print("─" * width)
    print(f"  Results: {passed} passed, {failed} failed")
    print("─" * width)

    if failed == 0:
        print("\nAll scenarios passed.")
        print('Type "approved" in your GSD session to advance to the next phase.\n')
    else:
        print(f"\n{failed} scenario(s) failed. Fix the issues above, then re-run.\n")

    return failed == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run human verification scenarios.")
    parser.add_argument(
        "--phase", type=int, default=None,
        help="Phase number to verify (default: current phase from STATE.md)",
    )
    args = parser.parse_args()

    phase = args.phase if args.phase is not None else _current_phase()
    ok = run_scenarios(phase)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
