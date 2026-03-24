"""SLSQP classical baseline for the Motzkin-Straus quadratic program.

Uses scipy.optimize.minimize with method='SLSQP' and multi-restart
from random Dirichlet-distributed starting points on the simplex.
"""

import time

import numpy as np
from scipy.optimize import minimize

from dirac_bench.problems import objective, objective_to_omega


def solve_slsqp(
    A: np.ndarray,
    num_restarts: int = 10,
    seed: int = 42,
) -> dict:
    """Solve the Motzkin-Straus QP using SLSQP.

    Args:
        A: Adjacency matrix (n x n, float64).
        num_restarts: Number of random restarts.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: omega, best_objective, solution, solve_time, all_objectives.
    """
    n = A.shape[0]
    rng = np.random.default_rng(seed)

    # SLSQP minimises, so we minimise -0.5 * x^T A x
    def neg_objective(x):
        return -0.5 * (x @ A @ x)

    def neg_jacobian(x):
        return -(A @ x)

    constraint = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
    bounds = [(0.0, None)] * n

    t0 = time.time()

    best_obj = -np.inf
    best_x = None
    all_objectives = []

    for i in range(num_restarts):
        # Start from random point on the simplex
        x0 = rng.dirichlet(np.ones(n))

        result = minimize(
            neg_objective,
            x0,
            method="SLSQP",
            jac=neg_jacobian,
            bounds=bounds,
            constraints=constraint,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        obj = objective(result.x, A)
        all_objectives.append(obj)

        if obj > best_obj:
            best_obj = obj
            best_x = result.x.copy()

    solve_time = time.time() - t0
    omega = objective_to_omega(best_obj)

    print(f"  SLSQP: best objective = {best_obj:.6f}  =>  omega = {omega}  ({solve_time:.1f}s, {num_restarts} restarts)")

    return {
        "omega": omega,
        "best_objective": best_obj,
        "solution": best_x,
        "solve_time": solve_time,
        "all_objectives": all_objectives,
    }
