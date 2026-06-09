"""GP core: PrimitiveSetTyped, primitives, tree evaluation, signal generation.

IMPORTANT: All primitive functions must be defined at module level (not inside
functions or lambdas) for multiprocessing.Pool pickling compatibility.
All primitives accept and return np.ndarray — no pandas objects inside primitives.
"""

from .primitives import Vector, Scalar
from .gp_types import build_pset

try:
    from .tree_evaluator import TreeEvaluator
except ImportError:
    TreeEvaluator = None  # type: ignore[assignment,misc]

__all__ = [
    "Vector",
    "Scalar",
    "build_pset",
    "TreeEvaluator",
]
