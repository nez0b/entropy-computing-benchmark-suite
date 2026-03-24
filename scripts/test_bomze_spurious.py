#!/usr/bin/env python3
"""Integration test: demonstrate that Bomze regularization eliminates spurious solutions.

For each test graph, runs SLSQP with many restarts on both standard A and
regularized A_bar, then reports how many per-restart solutions have support
that forms a clique in the original graph.

Usage:
    uv run python scripts/test_bomze_spurious.py
"""

import time

import numpy as np
from scipy.optimize import minimize

from dirac_bench.problems import (
    bomze_regularize,
    extract_support,
    is_clique,
)
from dirac_bench.test_graphs import make_overlapping_k4, make_erdos_renyi


# ── Per-restart solver ───────────────────────────────────────────────────

def solve_all_restarts(
    A: np.ndarray,
    n_restarts: int = 100,
    seed: int = 42,
) -> list[np.ndarray]:
    """Run SLSQP n_restarts times and return ALL solution vectors."""
    n = A.shape[0]
    rng = np.random.default_rng(seed)

    def neg_objective(x):
        return -0.5 * (x @ A @ x)

    def neg_jacobian(x):
        return -(A @ x)

    constraint = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
    bounds = [(0.0, None)] * n

    solutions = []
    for _ in range(n_restarts):
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
        solutions.append(result.x)

    return solutions


# ── Analysis ─────────────────────────────────────────────────────────────

def analyze_solutions(
    solutions: list[np.ndarray],
    A_original: np.ndarray,
) -> tuple[int, int]:
    """Count how many solutions have clique support vs spurious."""
    clique_count = 0
    spurious_count = 0
    for x in solutions:
        support = extract_support(x)
        if is_clique(support, A_original):
            clique_count += 1
        else:
            spurious_count += 1
    return clique_count, spurious_count


def run_comparison(
    A: np.ndarray,
    name: str,
    known_omega: int | None,
    n_restarts: int = 100,
    seed: int = 42,
) -> None:
    """Run standard vs Bomze comparison on one graph."""
    n = A.shape[0]
    omega_str = f"known_omega={known_omega}" if known_omega else "omega=?"
    print(f"\nGraph: {name} ({n} vertices)  {omega_str}")

    # Standard formulation
    t0 = time.time()
    std_solutions = solve_all_restarts(A, n_restarts=n_restarts, seed=seed)
    std_time = time.time() - t0
    std_clique, std_spurious = analyze_solutions(std_solutions, A)
    pct_spurious = 100.0 * std_spurious / n_restarts
    print(
        f"  Standard:  {n_restarts} restarts → "
        f"{std_clique} clique, {std_spurious} spurious ({pct_spurious:.1f}%)  "
        f"[{std_time:.2f}s]"
    )

    # Bomze formulation
    A_bar = bomze_regularize(A)
    t0 = time.time()
    bomze_solutions = solve_all_restarts(A_bar, n_restarts=n_restarts, seed=seed)
    bomze_time = time.time() - t0
    bomze_clique, bomze_spurious = analyze_solutions(bomze_solutions, A)
    pct_spurious_b = 100.0 * bomze_spurious / n_restarts
    print(
        f"  Bomze:     {n_restarts} restarts → "
        f"{bomze_clique} clique, {bomze_spurious} spurious ({pct_spurious_b:.1f}%)  "
        f"[{bomze_time:.2f}s]"
    )

    if bomze_spurious == 0 and std_spurious > 0:
        print("  ✓ Bomze eliminated all spurious solutions!")
    elif bomze_spurious == 0:
        print("  ✓ Bomze: zero spurious (standard also had none on this graph)")
    else:
        print(f"  ⚠ Bomze still has {bomze_spurious} spurious — check thresholds")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Bomze Regularization: Spurious Solution Elimination Demo")
    print("=" * 70)

    n_restarts = 100

    # 1. Overlapping K4s — the classic spurious-solution generator
    A, omega, name = make_overlapping_k4()
    run_comparison(A, name, known_omega=omega, n_restarts=n_restarts)

    # 2. Random Erdős–Rényi graphs
    for n, p in [(20, 0.7), (30, 0.5), (50, 0.9)]:
        A, _omega, name = make_erdos_renyi(n, p, seed=42)
        run_comparison(A, name, known_omega=None, n_restarts=n_restarts, seed=42)

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
