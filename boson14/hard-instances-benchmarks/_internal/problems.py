"""Motzkin-Strauss problem helpers used by the classical solvers.

Kept minimal — only the two functions the solvers call. The full scaled-MS
formulation lives in the parent `boson14_bench.core` package.
"""

import numpy as np


def objective(x: np.ndarray, A: np.ndarray) -> float:
    """Motzkin-Strauss objective: 0.5 * x^T A x."""
    return float(0.5 * (x @ A @ x))


def objective_to_omega(f_star: float) -> int:
    """omega = round(1 / (1 - 2*f*))  for x on the UNIT simplex (sum=1)."""
    denom = 1.0 - 2.0 * f_star
    if abs(denom) < 1e-12:
        return 1
    return round(1.0 / denom)
