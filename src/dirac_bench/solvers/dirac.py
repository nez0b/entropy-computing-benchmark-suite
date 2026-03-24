"""Dirac-3 continuous cloud solver for the Motzkin-Straus quadratic program.

Solves:  max  0.5 * x^T A x   s.t.  sum(x)=1, x>=0
via the QCI Dirac-3 cloud API (minimization form: J = -0.5*A).

Requires env vars: QCI_API_URL, QCI_TOKEN (loaded before import).
"""

import time

import numpy as np
from eqc_models.solvers import Dirac3ContinuousCloudSolver
from eqc_models.base import QuadraticModel

from dirac_bench.problems import objective, objective_to_omega


def solve_dirac(
    A: np.ndarray,
    num_samples: int = 100,
    relaxation_schedule: int = 2,
    sum_constraint: int = 1,
) -> dict:
    """Solve the Motzkin-Straus QP using Dirac-3 cloud API.

    Args:
        A: Adjacency matrix (n x n, float64).
        num_samples: Number of Dirac-3 samples (1-100).
        relaxation_schedule: Schedule parameter (1-4).
        sum_constraint: Sum constraint value (default 1 for simplex).

    Returns:
        Dict with keys: omega, best_objective, all_objectives,
        raw_response, solve_time.
    """
    n = A.shape[0]

    # Dirac-3 minimises x^T J x; we want to maximise 0.5 * x^T A x
    C = np.zeros(n, dtype=np.float64)
    J = -0.5 * A

    model = QuadraticModel(C, J)
    model.upper_bound = np.ones(n, dtype=np.float64)

    solver = Dirac3ContinuousCloudSolver()

    print(
        f"  Submitting {n}-variable QP to Dirac-3 cloud "
        f"(samples={num_samples}, schedule={relaxation_schedule})"
    )

    t0 = time.time()
    response = solver.solve(
        model,
        sum_constraint=sum_constraint,
        num_samples=num_samples,
        relaxation_schedule=relaxation_schedule,
    )
    solve_time = time.time() - t0

    # Extract solutions and compute objectives
    solutions = response.get("results", {}).get("solutions", [])
    if not solutions:
        raise RuntimeError("Dirac-3 returned no solutions")

    all_objectives = []
    for sol in solutions:
        x = np.array(sol, dtype=np.float64)
        obj = objective(x, A)
        if np.isfinite(obj):
            all_objectives.append(obj)

    if not all_objectives:
        raise RuntimeError("All Dirac-3 solutions produced non-finite objectives")

    best_obj = max(all_objectives)
    omega = objective_to_omega(best_obj)

    print(f"  Best objective = {best_obj:.6f}  =>  omega = {omega}  ({solve_time:.1f}s)")

    return {
        "omega": omega,
        "best_objective": best_obj,
        "all_objectives": all_objectives,
        "raw_response": response,
        "solve_time": solve_time,
    }
