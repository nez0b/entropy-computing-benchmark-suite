"""Dirac-3 direct hardware solver for the Motzkin-Straus quadratic program.

Solves:  max  0.5 * x^T A x   s.t.  sum(x)=1, x>=0
via direct gRPC connection to Dirac-3 hardware (minimization form: J = -0.5*A).

Requires:  eqc-direct >= 1.0.7
"""

import time

import numpy as np
from eqc_models.solvers import Dirac3DirectSolver
from eqc_models.base import QuadraticModel

from dirac_bench.problems import objective, objective_to_omega


DEFAULT_IP = "172.18.41.79"
DEFAULT_PORT = "50051"


def solve_dirac_direct(
    A: np.ndarray,
    ip_address: str = DEFAULT_IP,
    port: str = DEFAULT_PORT,
    num_samples: int = 100,
    relaxation_schedule: int = 2,
) -> dict:
    """Solve the Motzkin-Straus QP using Dirac-3 direct hardware.

    Args:
        A: Adjacency matrix (n x n, float64).
        ip_address: Dirac-3 hardware IP address.
        port: gRPC port.
        num_samples: Number of Dirac-3 samples (1-100).
        relaxation_schedule: Schedule parameter (1-4).

    Returns:
        Dict with keys: omega, best_objective, all_objectives,
        device_energies, raw_response, solve_time.
    """
    n = A.shape[0]

    C = np.zeros(n, dtype=np.float64)
    J = -0.5 * A

    model = QuadraticModel(C, J)
    model.upper_bound = np.ones(n, dtype=np.float64)

    solver = Dirac3DirectSolver()

    print(f"  Connecting to Dirac-3 at {ip_address}:{port}")
    solver.connect(ip_address, port)

    print(
        f"  Submitting {n}-variable QP to Dirac-3 (direct) "
        f"(samples={num_samples}, schedule={relaxation_schedule})"
    )

    t0 = time.time()
    response = solver.solve(
        model,
        sum_constraint=1,
        num_samples=num_samples,
        relaxation_schedule=relaxation_schedule,
    )
    solve_time = time.time() - t0

    # Direct solver returns 2D numpy array
    solutions = np.array(response["solution"])
    if solutions.ndim == 1:
        solutions = solutions.reshape(1, -1)

    device_energies = np.array(response["energy"])

    all_objectives = []
    for x in solutions:
        obj = objective(x, A)
        if np.isfinite(obj):
            all_objectives.append(obj)

    if not all_objectives:
        raise RuntimeError("All Dirac-3 solutions produced non-finite objectives")

    best_obj = max(all_objectives)
    omega = objective_to_omega(best_obj)

    hw_runtime = response.get("runtime", "N/A")
    print(
        f"  Best objective = {best_obj:.6f}  =>  omega = {omega}  "
        f"({solve_time:.1f}s wall, hw_runtime={hw_runtime})"
    )

    return {
        "omega": omega,
        "best_objective": best_obj,
        "all_objectives": all_objectives,
        "device_energies": device_energies.tolist(),
        "raw_response": response,
        "solve_time": solve_time,
    }
