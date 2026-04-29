#!/usr/bin/env python3
"""Measure difficulty of planted clique instances using SLSQP multi-restart.

Usage:
    python scripts/test_planted_hardness.py --all
    python scripts/test_planted_hardness.py --graph problems/planted/planted_100_k15_p05.clq
    python scripts/test_planted_hardness.py --all --restarts 500
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dirac_bench.io import read_dimacs_graph
from dirac_bench.problems import motzkin_straus_adjacency, objective, objective_to_omega
from dirac_bench.solvers.slsqp import solve_slsqp

DEFAULT_PLANTED_DIR = Path(__file__).resolve().parent.parent / "problems" / "planted"


def test_instance(
    graph_path: Path,
    metadata_path: Path,
    num_restarts: int,
    seed: int,
) -> dict:
    """Test a single planted clique instance for hardness.

    Returns dict with results summary.
    """
    # Load graph and metadata
    G = read_dimacs_graph(graph_path)
    with open(metadata_path) as f:
        meta = json.load(f)

    planted_nodes = meta["planted_nodes"]
    k = meta["k"]
    n = meta["n"]
    natural_omega = meta["natural_omega"]

    A = motzkin_straus_adjacency(G)

    # Verify planted clique objective
    planted_0based = [v - 1 for v in planted_nodes]
    x_planted = np.zeros(A.shape[0])
    for v in planted_0based:
        x_planted[v] = 1.0 / k
    planted_obj = objective(x_planted, A)

    # Run SLSQP
    result = solve_slsqp(A, num_restarts=num_restarts, seed=seed)

    # Analyze results
    omegas = [objective_to_omega(obj) for obj in result["all_objectives"]]
    found_planted = sum(1 for omega in omegas if omega >= k)
    pct = 100 * found_planted / num_restarts
    median_omega = int(np.median(omegas))
    obj_min = min(result["all_objectives"])
    obj_max = max(result["all_objectives"])

    if pct < 5:
        verdict = "HARD"
    elif pct < 50:
        verdict = "MODERATE"
    else:
        verdict = "EASY"

    return {
        "name": graph_path.stem,
        "n": n,
        "k": k,
        "natural_omega": natural_omega,
        "planted_objective": planted_obj,
        "num_restarts": num_restarts,
        "found_planted": found_planted,
        "found_pct": pct,
        "best_omega": result["omega"],
        "median_omega": median_omega,
        "obj_min": obj_min,
        "obj_max": obj_max,
        "solve_time": result["solve_time"],
        "verdict": verdict,
    }


def print_result(r: dict) -> None:
    """Pretty-print a single instance result."""
    print(f"\n{r['name']} ({r['n']} nodes, k={r['k']}, omega_natural~{r['natural_omega']})")
    print(f"  SLSQP ({r['num_restarts']} restarts):")
    print(f"    Found planted clique:  {r['found_planted']}/{r['num_restarts']} ({r['found_pct']:.1f}%)")
    print(f"    Best omega found:      {r['best_omega']}")
    print(f"    Median omega:          {r['median_omega']}")
    print(f"    Objective range:       [{r['obj_min']:.4f}, {r['obj_max']:.4f}]")
    print(f"    Solve time:            {r['solve_time']:.1f}s")
    print(f"  Verdict: {r['verdict']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test planted clique instances for hardness using SLSQP."
    )
    parser.add_argument("--graph", type=Path, help="Single .clq graph file to test")
    parser.add_argument("--metadata", type=Path, help="Metadata .json file (inferred from --graph if omitted)")
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Test all instances in {DEFAULT_PLANTED_DIR}",
    )
    parser.add_argument(
        "--restarts", type=int, default=500, help="Number of SLSQP restarts (default: 500)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    results = []

    if args.all:
        clq_files = sorted(DEFAULT_PLANTED_DIR.glob("planted_*.clq"))
        if not clq_files:
            print(f"No planted instances found in {DEFAULT_PLANTED_DIR}")
            print("Run: python scripts/generate_planted.py --suite all")
            return
        print(f"Testing {len(clq_files)} instances with {args.restarts} restarts each...")

        for clq_path in clq_files:
            json_path = clq_path.with_suffix(".json")
            if not json_path.exists():
                print(f"\nSkipping {clq_path.name}: no metadata file found")
                continue
            r = test_instance(clq_path, json_path, args.restarts, args.seed)
            print_result(r)
            results.append(r)

    elif args.graph:
        if not args.graph.exists():
            print(f"Graph file not found: {args.graph}")
            return
        metadata_path = args.metadata or args.graph.with_suffix(".json")
        if not metadata_path.exists():
            print(f"Metadata file not found: {metadata_path}")
            return
        r = test_instance(args.graph, metadata_path, args.restarts, args.seed)
        print_result(r)
        results.append(r)

    else:
        parser.error("Specify either --all or --graph FILE")

    # Summary table
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"{'Instance':<35} {'k':>3} {'omega':>5} {'Found':>8} {'Best':>5} {'Verdict':<10}")
        print("-" * 80)
        for r in results:
            found_str = f"{r['found_planted']}/{r['num_restarts']}"
            print(
                f"{r['name']:<35} {r['k']:>3} {r['natural_omega']:>5} "
                f"{found_str:>8} {r['best_omega']:>5} {r['verdict']:<10}"
            )


if __name__ == "__main__":
    main()
