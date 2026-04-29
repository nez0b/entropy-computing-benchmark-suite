#!/usr/bin/env python3
"""Generate planted max-clique instances in DIMACS format.

Usage:
    python scripts/generate_planted.py --suite all
    python scripts/generate_planted.py --n 100 --k 15 --p 0.5 --seed 42
    python scripts/generate_planted.py --n 100 --k 15 --verify
"""

import argparse
from pathlib import Path

import numpy as np

from dirac_bench.planted_clique import (
    generate_planted_clique,
    instance_name,
    planted_clique_info,
    write_planted_dimacs,
    write_planted_metadata,
)
from dirac_bench.problems import motzkin_straus_adjacency, objective, objective_to_omega

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "problems" / "planted"

# Preset suites: (n, k, p, seed, label)
SUITES = {
    "easy": [
        (100, 20, 0.5, 42, "easy"),
        (200, 25, 0.5, 42, "easy"),
        (500, 30, 0.5, 42, "easy"),
    ],
    "moderate": [],
    "hard": [
        (100, 15, 0.5, 42, "hard"),
        (200, 18, 0.5, 42, "hard"),
        (500, 22, 0.5, 42, "hard"),
    ],
    "all": [
        (100, 20, 0.5, 42, "easy"),
        (100, 15, 0.5, 42, "hard"),
        (200, 25, 0.5, 42, "easy"),
        (200, 18, 0.5, 42, "hard"),
        (500, 30, 0.5, 42, "easy"),
        (500, 22, 0.5, 42, "hard"),
    ],
}


def generate_instance(
    n: int, k: int, p: float, seed: int, output_dir: Path, verify: bool = False
) -> None:
    """Generate a single planted clique instance."""
    name = instance_name(n, k, p)
    clq_path = output_dir / f"{name}.clq"
    json_path = output_dir / f"{name}.json"

    info = planted_clique_info(n, k, p)
    print(f"\n{name} (n={n}, k={k}, p={p})")
    print(f"  Natural omega: ~{info['natural_omega']}")
    print(f"  Planted objective: {info['planted_objective']:.6f}")
    print(f"  Natural objective: {info['natural_objective']:.6f}")
    print(f"  Gap: {info['gap']:.6f}")
    print(f"  Difficulty: {info['difficulty']}")

    G, planted_nodes = generate_planted_clique(n, k, p, seed)
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"  Planted nodes: {planted_nodes[:10]}{'...' if len(planted_nodes) > 10 else ''}")

    write_planted_dimacs(G, planted_nodes, clq_path, n, k, p, seed)
    write_planted_metadata(json_path, planted_nodes, info, seed)
    print(f"  Written: {clq_path.name}, {json_path.name}")

    if verify:
        verify_instance(G, planted_nodes, k, info)


def verify_instance(
    G, planted_nodes: list[int], k: int, info: dict, num_restarts: int = 100
) -> None:
    """Run SLSQP to verify the planted clique is findable and measure difficulty."""
    from dirac_bench.solvers.slsqp import solve_slsqp

    A = motzkin_straus_adjacency(G)
    n = A.shape[0]

    # Verify planted clique objective
    planted_0based = [v - 1 for v in planted_nodes]
    x_planted = np.zeros(n)
    for v in planted_0based:
        x_planted[v] = 1.0 / k
    planted_obj = objective(x_planted, A)
    expected_obj = 0.5 * (1 - 1 / k)
    print(f"  Planted objective (computed): {planted_obj:.6f} (expected: {expected_obj:.6f})")

    # Run SLSQP
    print(f"  Running SLSQP with {num_restarts} restarts...")
    result = solve_slsqp(A, num_restarts=num_restarts, seed=42)

    # Count how many restarts found the planted clique
    found_planted = sum(
        1 for obj in result["all_objectives"] if objective_to_omega(obj) >= k
    )
    omegas = [objective_to_omega(obj) for obj in result["all_objectives"]]
    median_omega = int(np.median(omegas))

    pct = 100 * found_planted / num_restarts
    print(f"  Found planted clique: {found_planted}/{num_restarts} ({pct:.1f}%)")
    print(f"  Best omega found: {result['omega']}")
    print(f"  Median omega: {median_omega}")
    print(f"  Objective range: [{min(result['all_objectives']):.4f}, {max(result['all_objectives']):.4f}]")

    if pct < 5:
        verdict = "HARD"
    elif pct < 50:
        verdict = "MODERATE"
    else:
        verdict = "EASY"
    print(f"  Verdict: {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate planted max-clique instances in DIMACS format."
    )
    parser.add_argument(
        "--suite",
        choices=["easy", "moderate", "hard", "all"],
        help="Preset difficulty suite (default: generate single instance with --n/--k)",
    )
    parser.add_argument("--n", type=int, help="Graph size")
    parser.add_argument("--k", type=int, help="Planted clique size")
    parser.add_argument("--p", type=float, default=0.5, help="Edge probability (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Run SLSQP to verify difficulty"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    if args.suite:
        instances = SUITES[args.suite]
        if not instances:
            print(f"No instances defined for suite '{args.suite}'")
            return
        print(f"Generating {args.suite} suite ({len(instances)} instances)...")
        for n, k, p, seed, _label in instances:
            generate_instance(n, k, p, seed, args.output_dir, verify=args.verify)
    elif args.n and args.k:
        generate_instance(args.n, args.k, args.p, args.seed, args.output_dir, verify=args.verify)
    else:
        parser.error("Specify either --suite or both --n and --k")

    print(f"\nAll instances written to {args.output_dir}")


if __name__ == "__main__":
    main()
