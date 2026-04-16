#!/usr/bin/env python3
"""CLI for generating Boson14 hardware benchmark instances.

Usage examples:
    python boson14/generate.py --n 50 --k 12 --seed 42
    python boson14/generate.py --n 50 --k 12 --planted-indices 1,2,3,4,5,6,7,8,9,10,11,12
    python boson14/generate.py --n 50 --k 10 --num-cliques 3 --scramble
    python boson14/generate.py --n 40 --k 15 --plot-distribution
    python boson14/generate.py --n 30 --k 12 --density dense --verify
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np

from boson14_bench.core import (
    build_integer_qp,
    compute_all_max_clique_solutions,
    compute_clique_distribution,
    energy_to_omega,
    generate_degenerate_planted,
    hardware_energy,
    omega_to_scaled_objective,
    plot_clique_distribution,
    scaled_objective,
    scaled_objective_to_omega,
    scaled_optimal_y,
    scramble_graph,
    to_polynomial_json,
)
from boson14_bench.planted_clique import generate_planted_clique
from boson14_bench.problems import motzkin_straus_adjacency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Boson14 hardware benchmark instances"
    )
    parser.add_argument("--n", type=int, required=True, help="Graph size")
    parser.add_argument("--k", type=int, required=True, help="Planted clique size")
    parser.add_argument("--p", type=float, default=0.5, help="Edge probability")
    parser.add_argument(
        "--density",
        choices=["dense", "sparse"],
        default=None,
        help="Shortcut: dense->p=0.9, sparse->p=0.3 (overrides --p)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--R", type=int, default=100, help="Sum constraint")
    parser.add_argument(
        "--planted-indices",
        type=str,
        default=None,
        help="Comma-separated 1-based vertices for planted clique",
    )
    parser.add_argument(
        "--num-cliques",
        type=int,
        default=1,
        help="Number of planted cliques (degenerate case)",
    )
    parser.add_argument(
        "--scramble", action="store_true", help="Apply random vertex permutation"
    )
    parser.add_argument(
        "--plot-distribution", action="store_true", help="Plot clique objective distribution"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Output directory (default: boson14/output)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Brute-force verify with nx.find_cliques"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "csv", "both"],
        default="both",
        help="Output format for StQP problem file (default: both)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve density shortcut
    p = args.p
    if args.density == "dense":
        p = 0.9
    elif args.density == "sparse":
        p = 0.3

    # Parse planted indices
    planted_indices = None
    if args.planted_indices:
        planted_indices = [int(x.strip()) for x in args.planted_indices.split(",")]

    R = args.R
    n, k = args.n, args.k

    # Generate graph
    print(f"Generating: n={n}, k={k}, p={p}, seed={args.seed}, R={R}")

    if args.num_cliques > 1:
        # Degenerate case
        vertex_sets = None
        if planted_indices:
            # Split into groups of k
            if len(planted_indices) != args.num_cliques * k:
                raise ValueError(
                    f"planted-indices has {len(planted_indices)} vertices "
                    f"but num-cliques*k = {args.num_cliques * k}"
                )
            vertex_sets = [
                planted_indices[i * k : (i + 1) * k]
                for i in range(args.num_cliques)
            ]

        G, planted_sets = generate_degenerate_planted(
            n, k, args.num_cliques, p=p, seed=args.seed, vertex_sets=vertex_sets
        )
        planted_nodes = planted_sets[0]  # primary set for solution
        all_planted = planted_sets
        print(f"  Degenerate: {args.num_cliques} planted {k}-cliques")
        for i, vs in enumerate(planted_sets):
            print(f"  Set {i+1}: {vs}")
    else:
        G, planted_nodes = generate_planted_clique(
            n, k, p=p, seed=args.seed, vertices=planted_indices
        )
        all_planted = [planted_nodes]
        print(f"  Planted nodes: {planted_nodes}")

    # Scramble if requested
    forward_perm = None
    inverse_perm = None
    if args.scramble:
        G, forward_perm, inverse_perm = scramble_graph(G, seed=args.seed + 1000)
        print(f"  Scrambled vertex labels (seed={args.seed + 1000})")
        # Map planted nodes through permutation
        scrambled_planted = [[forward_perm[v] for v in vs] for vs in all_planted]
        print(f"  Scrambled planted sets: {scrambled_planted}")

    # Compute theoretical values
    A = motzkin_straus_adjacency(G)
    g_star = omega_to_scaled_objective(k, R)
    E_star = -2.0 * g_star

    print(f"\nTheoretical values (R={R}, w={k}):")
    print(f"  g* = {g_star:.2f}")
    print(f"  E* = {E_star:.2f}")
    print(f"  y_i = R/w = {R}/{k} = {R/k:.4f} for planted vertices")

    # Compute optimal y (in original ordering)
    planted_0 = [v - 1 for v in planted_nodes]
    y = scaled_optimal_y(planted_0, n, k, R)
    g_actual = scaled_objective(y, A)
    E_actual = hardware_energy(y, A)
    print(f"\nComputed from optimal y:")
    print(f"  g(y) = {g_actual:.2f}")
    print(f"  E(y) = {E_actual:.2f}")
    print(f"  w(g) = {scaled_objective_to_omega(g_actual, R)}")
    print(f"  w(E) = {energy_to_omega(E_actual, R)}")

    # Integer QP
    C, J = build_integer_qp(A)
    print(f"\nInteger QP: J has entries in {{{int(J.min())}, {int(J.max())}}}")

    # Verify
    if args.verify:
        print("\nBrute-force verification:")
        cliques = list(nx.find_cliques(G))
        omega_bf = max(len(c) for c in cliques)
        max_cliques = [sorted(c) for c in cliques if len(c) == omega_bf]
        print(f"  Clique number (brute-force): w = {omega_bf}")
        print(f"  Number of maximum cliques: {len(max_cliques)}")
        if omega_bf == k:
            print(f"  CONFIRMED: w = k = {k}")
        else:
            print(f"  WARNING: w = {omega_bf} != k = {k}")

    # Compute instance name and output directory
    instance_name = f"n{n}_k{k}_p{p}_s{args.seed}"
    if args.num_cliques > 1:
        instance_name += f"_nc{args.num_cliques}"
    if args.scramble:
        instance_name += "_scrambled"

    instance_dir = args.output_dir / instance_name
    instance_dir.mkdir(parents=True, exist_ok=True)

    # Plot distribution
    if args.plot_distribution:
        dist = compute_clique_distribution(G, R)
        plot_clique_distribution(
            dist, instance_name, R=R, known_omega=k,
            save_path=str(instance_dir),
        )
        print(f"\n  Clique distribution: {dist['size_counts']}")

    # Save metadata JSON

    metadata = {
        "n": n,
        "k": k,
        "p": p,
        "seed": args.seed,
        "R": R,
        "num_cliques": args.num_cliques,
        "planted_sets": all_planted,
        "g_star": g_star,
        "E_star": E_star,
        "y_planted": R / k,
    }
    if args.scramble:
        metadata["forward_perm"] = {str(k): int(v) for k, v in forward_perm.items()}
        metadata["inverse_perm"] = {str(k): int(v) for k, v in inverse_perm.items()}

    meta_path = instance_dir / f"{instance_name}_meta.json"

    # Compute and save all maximum clique solutions as NPZ
    y_solutions, omega_verified = compute_all_max_clique_solutions(G, R)
    npz_path = instance_dir / f"{instance_name}_solutions.npz"
    np.savez(npz_path, y_solutions=y_solutions)
    print(f"\nSaved solutions -> {npz_path}  (shape {y_solutions.shape})")

    metadata["omega_verified"] = omega_verified
    metadata["num_max_cliques"] = y_solutions.shape[0]

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print(f"Saved metadata -> {meta_path}")

    # Save Boson14 JSON (maximization convention)
    # Polynomial represents +A (val = +2.0 per edge, symmetrized upper-tri).
    # Objective: maximize g(y) = 0.5 * y^T A y  s.t. sum(y) = R.
    # For Dirac-3 submission, negate to get J = -A (hardware minimises y^T J y).
    num_edges = G.number_of_edges()
    max_edges = n * (n - 1) / 2 if n > 1 else 1
    C_zero = np.zeros(n, dtype=np.float64)

    if args.output_format in ("json", "both"):
        poly_json = to_polynomial_json(C_zero, A)
        boson14_payload = {
            "file": poly_json,
            "job_params": {
                "device_type": "dirac-3",
                "num_samples": 100,
                "relaxation_schedule": 2,
                "sum_constraint": R,
            },
            "graph_info": {
                "name": instance_name,
                "n": n,
                "k": k,
                "p": p,
                "seed": args.seed,
                "R": R,
                "num_cliques": args.num_cliques,
                "scrambled": args.scramble,
                "edges": num_edges,
                "density": round(num_edges / max_edges, 4) if max_edges > 0 else 0.0,
                "known_omega": k,
                "g_star": g_star,
                "E_star": E_star,
            },
        }
        boson14_json_path = instance_dir / f"{instance_name}_boson14.json"
        with open(boson14_json_path, "w") as f:
            json.dump(boson14_payload, f, indent=2)
            f.write("\n")
        num_terms = len(poly_json["file_config"]["polynomial"]["data"])
        print(f"Saved Boson14 JSON -> {boson14_json_path}  ({num_terms} terms, maximization)")

    if args.output_format in ("csv", "both"):
        csv_data = np.column_stack([C_zero, A])  # shape (n, n+1)
        csv_path = instance_dir / f"{instance_name}_boson14.csv"
        np.savetxt(csv_path, csv_data, delimiter=",", fmt="%g")
        print(f"Saved Boson14 CSV -> {csv_path}  ({n}x{n+1}, maximization)")


if __name__ == "__main__":
    main()
