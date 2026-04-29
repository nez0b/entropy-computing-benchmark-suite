"""Classical baseline solvers for the Motzkin-Strauss QP.

Three solvers — Greedy (combinatorial), PGD (projected gradient), SLSQP
(constrained NLP) — each multi-restart. All operate on the UNIT simplex
(sum(x) = 1). Callers that want to compare with scaled (sum=R) Boson14 / Dirac
samples should multiply the returned solution arrays by R before analysis.

All solvers return a common dict:
    best_omega:       int   — MS-inverted omega from best objective
    best_objective:   float — max over all restarts
    best_solution:    np.ndarray shape (n,)  — argmax vector
    all_objectives:   list[float]  — one per restart
    all_solutions:    np.ndarray shape (num_restarts, n)  — one row per restart
    solve_time:       float
    hit_rate:         float — fraction of restarts whose omega ≥ true_omega (if
                              true_omega is passed; else 0.0)
    num_restarts:     int
"""

import time

import numpy as np
from scipy.optimize import minimize

from .problems import objective, objective_to_omega


# ---------------------------------------------------------------------------
# Greedy (combinatorial, no continuous optimisation)
# ---------------------------------------------------------------------------

def solve_greedy_degree(
    A: np.ndarray,
    num_restarts: int = 30,
    seed: int = 42,
    true_omega: int | None = None,
) -> dict:
    """Greedy clique construction: highest-degree-first + random orderings."""
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    degrees = A.sum(axis=1)
    t0 = time.time()

    all_sizes: list[int] = []
    all_cliques: list[list[int]] = []

    for trial in range(num_restarts):
        order = np.argsort(-degrees) if trial == 0 else rng.permutation(n)
        clique: list[int] = []
        for v in order:
            if all(A[v, u] == 1 for u in clique):
                clique.append(int(v))
        all_sizes.append(len(clique))
        all_cliques.append(clique)

    solve_time = time.time() - t0
    best_idx = int(np.argmax(all_sizes))
    best_size = all_sizes[best_idx]
    best_clique = all_cliques[best_idx]

    # Synthesise an equal-weight y-vector (unit simplex) from each clique.
    # Per-restart "solution" = characteristic vector 1_S / |S|.
    all_solutions = np.zeros((num_restarts, n), dtype=np.float64)
    all_objectives: list[float] = []
    for i, cq in enumerate(all_cliques):
        if cq:
            x = np.zeros(n)
            x[cq] = 1.0 / len(cq)
            all_solutions[i] = x
            all_objectives.append(float(0.5 * (x @ A @ x)))
        else:
            all_objectives.append(0.0)

    best_objective = float(0.5 * (1.0 - 1.0 / best_size)) if best_size > 1 else 0.0
    best_solution = all_solutions[best_idx]
    best_omega = best_size  # greedy knows the exact clique it built
    hit_rate = (
        sum(1 for s in all_sizes if s >= true_omega) / num_restarts
        if true_omega else 0.0
    )

    return {
        "best_omega": best_omega,
        "best_objective": best_objective,
        "best_solution": best_solution,
        "all_objectives": all_objectives,
        "all_solutions": all_solutions,
        "solve_time": solve_time,
        "hit_rate": hit_rate,
        "num_restarts": num_restarts,
    }


# ---------------------------------------------------------------------------
# Projected Gradient Descent on the simplex
# ---------------------------------------------------------------------------

def _project_simplex(z: np.ndarray) -> np.ndarray:
    """Project z onto {x >= 0, sum(x) = 1}  (Duchi et al. 2008, O(n log n))."""
    n = len(z)
    u = np.sort(z)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(z - theta, 0)


def solve_pgd(
    A: np.ndarray,
    num_restarts: int = 30,
    max_iters: int = 2000,
    lr: float = 1.0,
    seed: int = 42,
    true_omega: int | None = None,
) -> dict:
    """PGD on max 0.5*x^T A x, s.t. x in simplex, with backtracking step size."""
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    t0 = time.time()

    all_objectives: list[float] = []
    all_solutions = np.zeros((num_restarts, n), dtype=np.float64)

    for r in range(num_restarts):
        x = rng.dirichlet(np.ones(n))
        for _ in range(max_iters):
            grad = A @ x
            step = lr
            for _ in range(10):
                x_new = _project_simplex(x + step * grad)
                if objective(x_new, A) >= objective(x, A) - 1e-12:
                    break
                step *= 0.5
            if np.allclose(x_new, x, atol=1e-10):
                break
            x = x_new
        all_solutions[r] = x
        all_objectives.append(float(objective(x, A)))

    solve_time = time.time() - t0
    best_idx = int(np.argmax(all_objectives))
    best_obj = all_objectives[best_idx]
    best_omega = objective_to_omega(best_obj)
    hit_rate = (
        sum(1 for o in all_objectives if objective_to_omega(o) >= true_omega) / num_restarts
        if true_omega else 0.0
    )

    return {
        "best_omega": best_omega,
        "best_objective": best_obj,
        "best_solution": all_solutions[best_idx],
        "all_objectives": all_objectives,
        "all_solutions": all_solutions,
        "solve_time": solve_time,
        "hit_rate": hit_rate,
        "num_restarts": num_restarts,
    }


# ---------------------------------------------------------------------------
# SLSQP (scipy sequential-least-squares QP)
# ---------------------------------------------------------------------------

def solve_slsqp(
    A: np.ndarray,
    num_restarts: int = 30,
    seed: int = 42,
    true_omega: int | None = None,
) -> dict:
    """SLSQP with Dirichlet restarts on the simplex."""
    n = A.shape[0]
    rng = np.random.default_rng(seed)

    def neg_obj(x): return -0.5 * (x @ A @ x)
    def neg_jac(x): return -(A @ x)

    constraint = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
    bounds = [(0.0, None)] * n
    t0 = time.time()

    all_objectives: list[float] = []
    all_solutions = np.zeros((num_restarts, n), dtype=np.float64)

    for r in range(num_restarts):
        x0 = rng.dirichlet(np.ones(n))
        res = minimize(
            neg_obj, x0, method="SLSQP", jac=neg_jac,
            bounds=bounds, constraints=constraint,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        all_solutions[r] = res.x
        all_objectives.append(float(objective(res.x, A)))

    solve_time = time.time() - t0
    best_idx = int(np.argmax(all_objectives))
    best_obj = all_objectives[best_idx]
    best_omega = objective_to_omega(best_obj)
    hit_rate = (
        sum(1 for o in all_objectives if objective_to_omega(o) >= true_omega) / num_restarts
        if true_omega else 0.0
    )

    return {
        "best_omega": best_omega,
        "best_objective": best_obj,
        "best_solution": all_solutions[best_idx],
        "all_objectives": all_objectives,
        "all_solutions": all_solutions,
        "solve_time": solve_time,
        "hit_rate": hit_rate,
        "num_restarts": num_restarts,
    }
