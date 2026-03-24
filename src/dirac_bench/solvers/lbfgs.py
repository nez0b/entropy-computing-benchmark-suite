"""L-BFGS-B classical baseline using softmax reparameterization.

L-BFGS-B only supports box constraints, not the simplex constraint sum(x)=1.
We use the softmax trick: optimise unconstrained z in R^n, then map to the
simplex via x = softmax(z) = exp(z) / sum(exp(z)).

The chain rule gives:  df/dz_i = x_i * (df/dx_i - <df/dx, x>)
"""

import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from dirac_bench.problems import objective, objective_to_omega


def solve_lbfgs(
    A: np.ndarray,
    num_restarts: int = 10,
    seed: int = 42,
) -> dict:
    """Solve the Motzkin-Straus QP using L-BFGS-B with softmax parameterization.

    Args:
        A: Adjacency matrix (n x n, float64).
        num_restarts: Number of random restarts.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: omega, best_objective, solution, solve_time, all_objectives.
    """
    n = A.shape[0]
    rng = np.random.default_rng(seed)

    def neg_objective_and_grad(z):
        x = softmax(z)
        obj = 0.5 * (x @ A @ x)

        # Gradient in x-space: df/dx = A @ x
        grad_x = A @ x

        # Chain rule through softmax: df/dz_i = x_i * (grad_x_i - <grad_x, x>)
        grad_z = x * (grad_x - np.dot(grad_x, x))

        # We minimise the negative objective
        return -obj, -grad_z

    t0 = time.time()

    best_obj = -np.inf
    best_x = None
    all_objectives = []

    for i in range(num_restarts):
        # Random starting point in z-space
        z0 = rng.standard_normal(n)

        result = minimize(
            neg_objective_and_grad,
            z0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 1000, "ftol": 1e-15},
        )

        x = softmax(result.x)
        obj = objective(x, A)
        all_objectives.append(obj)

        if obj > best_obj:
            best_obj = obj
            best_x = x.copy()

    solve_time = time.time() - t0
    omega = objective_to_omega(best_obj)

    print(f"  L-BFGS-B: best objective = {best_obj:.6f}  =>  omega = {omega}  ({solve_time:.1f}s, {num_restarts} restarts)")

    return {
        "omega": omega,
        "best_objective": best_obj,
        "solution": best_x,
        "solve_time": solve_time,
        "all_objectives": all_objectives,
    }
