"""Dirac-3 continuous cloud solver wrapper.

Uses J = -0.5*A (standard MS formulation) with sum_constraint=R. Returns the
full set of solution vectors and objectives for distribution plots.

Lazily imports `eqc_models` and loads `.env` at call time so this module can be
imported without Dirac credentials. The caller must provide QCI_API_URL and
QCI_TOKEN in the environment (or in a .env file next to the pipeline).
"""

import time
from pathlib import Path

import numpy as np


def solve_dirac_cloud(
    A: np.ndarray,
    R: int = 100,
    num_samples: int = 100,
    relaxation_schedule: int = 2,
    env_path: Path | None = None,
) -> dict:
    """Submit Motzkin-Strauss QP to Dirac-3 cloud (J = -0.5*A, sum=R).

    Returns:
        all_solutions:     np.ndarray shape (num_finite, n)  — raw y vectors
        all_objectives:    list[float]  — g(y) = 0.5*y^T A y per sample
        best_objective:    float
        best_omega:        int          — via MS inversion
        solve_time:        float
        num_finite:        int
        num_samples:       int  (requested)
    """
    from dotenv import load_dotenv  # noqa: F401
    if env_path is None:
        env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    from eqc_models.base import QuadraticModel
    from eqc_models.solvers import Dirac3ContinuousCloudSolver

    n = A.shape[0]
    C = np.zeros(n, dtype=np.float64)
    J = -0.5 * A
    model = QuadraticModel(C, J)
    model.upper_bound = R * np.ones(n, dtype=np.float64)

    solver = Dirac3ContinuousCloudSolver()
    print(f"    Dirac-3 cloud: submitting {n}-var QP "
          f"(R={R}, samples={num_samples}, schedule={relaxation_schedule})")
    t0 = time.time()
    response = solver.solve(
        model,
        sum_constraint=R,
        num_samples=num_samples,
        relaxation_schedule=relaxation_schedule,
    )
    solve_time = time.time() - t0

    solutions = response.get("results", {}).get("solutions", [])
    if not solutions:
        raise RuntimeError("Dirac-3 returned no solutions")

    y_vectors: list[np.ndarray] = []
    all_objectives: list[float] = []
    for sol in solutions:
        y = np.array(sol, dtype=np.float64)
        g = float(0.5 * (y @ A @ y))
        if np.isfinite(g):
            y_vectors.append(y)
            all_objectives.append(g)

    if not all_objectives:
        raise RuntimeError("All Dirac-3 solutions produced non-finite objectives")

    best_idx = int(np.argmax(all_objectives))
    best_obj = all_objectives[best_idx]
    denom = R * R - 2.0 * best_obj
    best_omega = round(R * R / denom) if abs(denom) > 1e-12 else 1

    return {
        "all_solutions": np.array(y_vectors),
        "all_objectives": all_objectives,
        "best_objective": best_obj,
        "best_omega": int(best_omega),
        "solve_time": solve_time,
        "num_finite": len(all_objectives),
        "num_samples": num_samples,
        "relaxation_schedule": relaxation_schedule,
        "R": R,
    }
