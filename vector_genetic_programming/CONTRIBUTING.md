# Contributing to VGP

VGP is a research framework for evolving trading strategies using genetic programming. Contributions are welcome. This guide covers the three most common contributor tasks.

## Adding a GP Primitive

Primitives live in `vgp/gp/` (to be populated in Phase 3). Before adding a primitive, read the architecture constraints in `CLAUDE.md` carefully — violating any of the three rules below will cause silent failures in multiprocessing workers.

**Three mandatory rules:**

1. All primitives MUST be module-level functions (not lambdas). `multiprocessing.Pool` pickles worker functions by name; lambdas and closures defined inside other functions are not picklable and cause `AttributeError` in workers with no obvious error message.

2. All primitives MUST accept and return `np.ndarray`. No pandas objects inside primitives. The GP evaluation loop passes raw numpy arrays — introducing a DataFrame conversion inside a primitive creates a serialization boundary and kills performance.

3. Primitives operate on vectors: shape `[T]` (time series) or `[T, F]` (feature window), depending on the primitive type. `PrimitiveSetTyped` distinguishes `Vector` (array) and `Scalar` (float) types — only matching types can be composed. Both type categories will be documented here once Phase 3 is complete.

See `vgp/gp/primitives.py` for examples (added in Phase 3).

## Running an Experiment

The experiment entry point will be documented here once Phase 4 (Evolution Engine) is complete. The planned interface is:

```
python -m vgp.evolution.run --config config.yaml
```

Note: the full CLI including config schema, population size, and NSGA-II parameters is added in Phase 4.

See `vgp/evolution/` for the NSGA-II loop (added in Phase 4).

## Updating the Dependency Lock File

VGP uses `pip-tools` to maintain a reproducible lock file.

1. Install pip-tools:
   ```
   pip install pip-tools
   ```

2. Compile the lock file from pyproject.toml:
   ```
   pip-compile pyproject.toml -o requirements-lock.txt
   ```

3. Commit the updated `requirements-lock.txt`.

**Important:** The `numpy<2.3` upper bound in `pyproject.toml` must remain. NumPy 2.3 hard-breaks numba's internal C extension APIs, which makes the entire evolution engine inoperable. Do not remove or relax this pin. If numba adds NumPy 2.3 support in a future release, update the pin in `pyproject.toml` and re-run the smoke tests (`test_numba_jit_compiles`) before committing.

## Code Conventions

**pandas 3.0 idioms** — pandas 3.0 makes Copy-on-Write the default and removes several deprecated patterns. Follow these rules in all DataFrame code:
- Use `.to_numpy()` (not `.values`) to extract numpy arrays from DataFrames
- Use explicit `.copy()` before mutating a DataFrame slice
- Use `.loc[]` for all index-based assignment (no chained indexing)

**Future annotations** — All new Python files must start with:
```python
from __future__ import annotations
```
This enables PEP 563 postponed evaluation for forward references in type hints and is required for consistency across the codebase.

**Module-level loggers** — Declare loggers at module level, not inside functions:
```python
import logging
logger = logging.getLogger(__name__)
```

**Run tests:**
```
python -m pytest tests/ -v
```
