"""GP core: PrimitiveSetTyped, primitives, tree evaluation, signal generation.

IMPORTANT: All primitive functions must be defined at module level (not inside
functions or lambdas) for multiprocessing.Pool pickling compatibility.
All primitives accept and return np.ndarray — no pandas objects inside primitives.
"""
