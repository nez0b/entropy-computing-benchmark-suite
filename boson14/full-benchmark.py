#!/usr/bin/env python3
"""Full benchmark: generate 3 permutation variants of a planted-clique graph
and optionally submit to Dirac-3 to test position-dependent performance.

Tests whether the position of the planted clique in the adjacency matrix
affects Dirac-3 hardware performance. Three variants are generated from the
same base graph: clique at front, end, and random positions.

Usage:
    uv run python full-benchmark.py --n 30 --k 10 --seed 42 --verify
    uv run python full-benchmark.py --n 30 --k 10 --run-dirac --plot
    uv run python full-benchmark.py --n 30 --k 10 --analyze --plot
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np

from boson14_bench.core import (
    compute_all_max_clique_solutions,
    create_targeted_permutation,
    omega_to_scaled_objective,
    plot_scaled_dirac_histogram,
    scaled_objective,
    scaled_objective_to_omega,
    to_polynomial_json,
)
from boson14_bench.planted_clique import generate_planted_clique
from boson14_bench.problems import motzkin_straus_adjacency

VARIANT_NAMES = ["front", "end", "random"]


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------

def compute_variant_targets(
    planted_nodes: list[int],
    n: int,
    k: int,
    seed: int,
) -> dict[str, list[int]]:
    """Compute target positions for the 3 variants."""
    front_targets = list(range(1, k + 1))
    end_targets = list(range(n - k + 1, n + 1))

    # Random: ensure different from original planted positions
    rng = np.random.default_rng(seed + 2000)
    planted_set = set(planted_nodes)
    for _ in range(100):
        random_targets = sorted(int(x) for x in rng.choice(n, size=k, replace=False) + 1)
        if set(random_targets) != planted_set:
            break

    return {
        "front": front_targets,
        "end": end_targets,
        "random": random_targets,
    }


def generate_variant(
    G: nx.Graph,
    planted_nodes: list[int],
    n: int,
    k: int,
    target_positions: list[int],
    variant_name: str,
) -> tuple[nx.Graph, np.ndarray, dict[int, int], dict[int, int]]:
    """Create a permuted variant where the planted clique sits at target_positions.

    Includes sanity check: verifies the permuted adjacency matrix has a
    complete subgraph at the target positions.
    """
    forward_perm, inverse_perm = create_targeted_permutation(
        planted_nodes, n, target_positions,
    )
    G_variant = nx.relabel_nodes(G, forward_perm)
    A_variant = motzkin_straus_adjacency(G_variant)

    # Sanity check: submatrix at target positions must be all-ones (minus diagonal)
    target_0based = sorted(t - 1 for t in target_positions)
    submatrix = A_variant[np.ix_(target_0based, target_0based)]
    expected = np.ones((k, k)) - np.eye(k)
    if not np.allclose(submatrix, expected):
        raise RuntimeError(
            f"Variant '{variant_name}': submatrix at target positions is NOT a "
            f"complete subgraph. Permutation is incorrect."
        )
    print(f"  [{variant_name}] Sanity check PASSED: clique at {target_positions}")

    return G_variant, A_variant, forward_perm, inverse_perm


def save_variant(
    variant_name: str,
    variant_dir: Path,
    G_variant: nx.Graph,
    A_variant: np.ndarray,
    forward_perm: dict[int, int],
    inverse_perm: dict[int, int],
    target_positions: list[int],
    base_meta: dict,
    R: int,
    k: int,
    n: int,
    output_format: str = "both",
) -> None:
    """Save variant artifacts: adjacency (JSON/CSV), metadata, theoretical solutions."""
    variant_dir.mkdir(parents=True, exist_ok=True)
    instance_name = base_meta["instance_name"]

    meta = {
        **base_meta,
        "variant": variant_name,
        "target_positions": target_positions,
        "forward_perm": {str(key): int(val) for key, val in forward_perm.items()},
        "inverse_perm": {str(key): int(val) for key, val in inverse_perm.items()},
    }

    y_solutions, omega_verified = compute_all_max_clique_solutions(G_variant, R)
    npz_path = variant_dir / f"{variant_name}_solutions.npz"
    np.savez(npz_path, y_solutions=y_solutions)
    meta["omega_verified"] = omega_verified
    meta["num_max_cliques"] = int(y_solutions.shape[0])
    print(f"  [{variant_name}] Saved solutions -> {npz_path}  (shape {y_solutions.shape})")

    meta_path = variant_dir / f"{variant_name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"  [{variant_name}] Saved metadata -> {meta_path}")

    C_zero = np.zeros(n, dtype=np.float64)

    if output_format in ("json", "both"):
        poly_json = to_polynomial_json(C_zero, A_variant)
        num_edges = G_variant.number_of_edges()
        max_edges = n * (n - 1) / 2 if n > 1 else 1
        payload = {
            "file": poly_json,
            "job_params": {
                "device_type": "dirac-3",
                "num_samples": 100,
                "relaxation_schedule": 2,
                "sum_constraint": R,
            },
            "graph_info": {
                "name": f"{instance_name}_{variant_name}",
                "variant": variant_name,
                "target_positions": target_positions,
                "n": n, "k": k, "p": base_meta["p"], "seed": base_meta["seed"],
                "R": R,
                "edges": num_edges,
                "density": round(num_edges / max_edges, 4) if max_edges > 0 else 0.0,
                "known_omega": k,
                "g_star": base_meta["g_star"],
            },
        }
        json_path = variant_dir / f"{variant_name}_boson14.json"
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"  [{variant_name}] Saved Boson14 JSON -> {json_path}")

    if output_format in ("csv", "both"):
        csv_data = np.column_stack([C_zero, A_variant])
        csv_path = variant_dir / f"{variant_name}_boson14.csv"
        np.savetxt(csv_path, csv_data, delimiter=",", fmt="%g")
        print(f"  [{variant_name}] Saved Boson14 CSV -> {csv_path}")


# ---------------------------------------------------------------------------
# Dirac-3 solver (standard formulation: J = -0.5*A)
# ---------------------------------------------------------------------------

def solve_dirac_standard(
    A: np.ndarray,
    R: int = 100,
    num_samples: int = 100,
    relaxation_schedule: int = 2,
) -> dict:
    """Submit Motzkin-Strauss QP to Dirac-3 (J = -0.5*A, standard formulation)."""
    import time
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
    from eqc_models.solvers import Dirac3ContinuousCloudSolver
    from eqc_models.base import QuadraticModel

    n = A.shape[0]
    C = np.zeros(n, dtype=np.float64)
    J = -0.5 * A

    model = QuadraticModel(C, J)
    model.upper_bound = R * np.ones(n, dtype=np.float64)

    solver = Dirac3ContinuousCloudSolver()
    print(
        f"    Submitting {n}-variable QP to Dirac-3 "
        f"(R={R}, samples={num_samples}, schedule={relaxation_schedule})"
    )

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

    y_vectors, objectives, omegas = [], [], []
    for sol in solutions:
        y = np.array(sol, dtype=np.float64)
        g = scaled_objective(y, A)
        if np.isfinite(g):
            y_vectors.append(y)
            objectives.append(g)
            omegas.append(scaled_objective_to_omega(g, R))

    if not objectives:
        raise RuntimeError("All Dirac-3 solutions produced non-finite objectives")

    best_idx = int(np.argmax(objectives))
    print(f"    Best g = {objectives[best_idx]:.4f}  =>  omega = {omegas[best_idx]}  ({solve_time:.1f}s)")

    return {
        "y_vectors": np.array(y_vectors),
        "objectives": objectives,
        "omegas": omegas,
        "best_omega": omegas[best_idx],
        "best_objective": float(objectives[best_idx]),
        "mean_omega": float(np.mean(omegas)),
        "omega_counts": dict(sorted(Counter(omegas).items())),
        "solve_time": solve_time,
    }


# ---------------------------------------------------------------------------
# Analysis skeleton
# ---------------------------------------------------------------------------

def analyze_variants(
    experiment_dir: Path,
    n: int,
    k: int,
    R: int,
    plot: bool = False,
) -> dict:
    """Load Dirac NPZ results for each variant and compare.

    Each NPZ file contains y_vectors of shape (num_samples, n).
    Computes g(y) = 0.5*y^T*A*y and omega for each sample.
    """
    summary = {}

    for variant in VARIANT_NAMES:
        variant_dir = experiment_dir / variant
        npz_path = variant_dir / f"{variant}_dirac_solutions.npz"
        csv_path = variant_dir / f"{variant}_boson14.csv"

        if not npz_path.exists():
            print(f"  [{variant}] No Dirac results at {npz_path}, skipping")
            continue
        if not csv_path.exists():
            print(f"  [{variant}] No CSV at {csv_path}, cannot compute objectives")
            continue

        data = np.load(npz_path)
        y_vectors = data["y_vectors"]

        raw = np.loadtxt(csv_path, delimiter=",")
        A = raw[:, 1:]

        objectives, omegas = [], []
        for y in y_vectors:
            g = scaled_objective(y, A)
            if np.isfinite(g):
                objectives.append(g)
                omegas.append(scaled_objective_to_omega(g, R))

        if not objectives:
            print(f"  [{variant}] All solutions non-finite, skipping")
            continue

        best_idx = int(np.argmax(objectives))
        omega_counts = dict(sorted(Counter(omegas).items()))

        summary[variant] = {
            "num_samples": len(objectives),
            "best_omega": omegas[best_idx],
            "best_objective": objectives[best_idx],
            "mean_omega": float(np.mean(omegas)),
            "omega_distribution": {str(k_): v for k_, v in omega_counts.items()},
        }
        print(
            f"  [{variant}] {len(objectives)} samples, "
            f"best omega={omegas[best_idx]}, mean={float(np.mean(omegas)):.1f}"
        )

    if summary:
        comparison_dir = experiment_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        summary_path = comparison_dir / "comparison_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
            f.write("\n")
        print(f"  Saved comparison -> {summary_path}")

    if plot and summary:
        plot_variant_comparison(summary, experiment_dir, k, R)

    return summary


def plot_variant_comparison(
    variant_results: dict[str, dict],
    experiment_dir: Path,
    k: int,
    R: int,
) -> Path | None:
    """Overlay omega histograms for all variants.

    TODO: implement full visualization when Dirac results are available.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = sorted(variant_results.keys())
    if not variants:
        return None

    comparison_dir = experiment_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder: print what would be plotted
    print(f"  [plot] Comparison histogram across {variants}: TODO (skeleton)")
    print(f"  [plot] Theoretical g*(w={k}, R={R}) = {omega_to_scaled_objective(k, R):.4f}")

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full benchmark: 3 permutation variants of a planted-clique graph",
    )
    parser.add_argument("--n", type=int, required=True, help="Graph size")
    parser.add_argument("--k", type=int, required=True, help="Planted clique size")
    parser.add_argument("--p", type=float, default=0.5, help="Edge probability (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--R", type=int, default=100, help="Sum constraint (default: 100)")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Output directory (default: boson14/output)",
    )
    parser.add_argument(
        "--output-format", choices=["json", "csv", "both"], default="both",
        help="Output format for StQP problem file (default: both)",
    )
    parser.add_argument("--verify", action="store_true", help="Brute-force verify omega for each variant")
    parser.add_argument("--run-dirac", action="store_true", help="Submit all 3 variants to Dirac-3")
    parser.add_argument("--num-samples", type=int, default=100, help="Dirac-3 samples (default: 100)")
    parser.add_argument(
        "--relaxation-schedule", type=int, default=2,
        help="Relaxation schedule 1-4 (default: 2)",
    )
    parser.add_argument("--analyze", action="store_true", help="Run analysis on existing NPZ files")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-sample details")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    n, k, R = args.n, args.k, args.R
    g_star = omega_to_scaled_objective(k, R)

    experiment_name = f"benchmark_n{n}_k{k}_p{args.p}_s{args.seed}"
    experiment_dir = args.output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Full Benchmark: n={n}, k={k}, p={args.p}, seed={args.seed}, R={R}")
    print(f"{'='*60}")
    print(f"  Theoretical: g* = {g_star:.2f}, y_i = R/k = {R/k:.4f}")

    # Step 1: Generate base graph
    print("\n--- Generate base graph ---")
    G, planted_nodes = generate_planted_clique(n, k, p=args.p, seed=args.seed)
    print(f"  Planted nodes (original): {planted_nodes}")

    base_meta = {
        "instance_name": experiment_name,
        "n": n, "k": k, "p": args.p, "seed": args.seed, "R": R,
        "planted_nodes": planted_nodes,
        "g_star": g_star,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    base_meta_path = experiment_dir / "base_meta.json"
    with open(base_meta_path, "w") as f:
        json.dump(base_meta, f, indent=2)
        f.write("\n")
    print(f"  Saved base metadata -> {base_meta_path}")

    # Step 2: Compute target positions for 3 variants
    print("\n--- Compute variant targets ---")
    targets = compute_variant_targets(planted_nodes, n, k, args.seed)
    for name, tgt in targets.items():
        print(f"  {name:>6s}: {tgt}")

    if 2 * k > n:
        print(f"  WARNING: k={k} > n/2={n/2}, front and end target positions overlap")

    # Step 3: Generate and save each variant
    print("\n--- Generate variants ---")
    variant_data = {}
    for name in VARIANT_NAMES:
        G_var, A_var, fwd, inv = generate_variant(
            G, planted_nodes, n, k, targets[name], name,
        )
        save_variant(
            name, experiment_dir / name, G_var, A_var, fwd, inv,
            targets[name], base_meta, R, k, n, args.output_format,
        )
        variant_data[name] = (G_var, A_var, fwd, inv)

        if args.verify:
            cliques = list(nx.find_cliques(G_var))
            omega_bf = max(len(c) for c in cliques)
            status = "CONFIRMED" if omega_bf == k else f"WARNING: w={omega_bf} != k={k}"
            print(f"  [{name}] Brute-force omega = {omega_bf}: {status}")

    # Step 4: Optional Dirac-3 submission
    if args.run_dirac:
        print(f"\n--- Dirac-3 Submission ---")
        for name in VARIANT_NAMES:
            _, A_var, _, _ = variant_data[name]
            variant_dir = experiment_dir / name

            print(f"\n  [{name}]")
            try:
                result = solve_dirac_standard(
                    A_var, R=R,
                    num_samples=args.num_samples,
                    relaxation_schedule=args.relaxation_schedule,
                )
            except RuntimeError as e:
                print(f"    ERROR: {e}")
                continue

            npz_path = variant_dir / f"{name}_dirac_solutions.npz"
            np.savez(npz_path, y_vectors=result["y_vectors"])
            print(f"    Saved Dirac solutions -> {npz_path}")

            dirac_meta = {
                "variant": name,
                "n": n, "k": k, "R": R,
                "num_samples": args.num_samples,
                "relaxation_schedule": args.relaxation_schedule,
                "best_omega": result["best_omega"],
                "best_objective": result["best_objective"],
                "mean_omega": result["mean_omega"],
                "omega_distribution": {str(k_): v for k_, v in result["omega_counts"].items()},
                "solve_time": result["solve_time"],
                "num_finite_samples": len(result["objectives"]),
                "formulation": "J = -0.5*A (standard Dirac-3)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            meta_path = variant_dir / f"{name}_dirac_meta.json"
            with open(meta_path, "w") as f:
                json.dump(dirac_meta, f, indent=2)
                f.write("\n")
            print(f"    Saved Dirac metadata -> {meta_path}")

            if args.plot:
                plot_scaled_dirac_histogram(
                    result["objectives"],
                    f"{experiment_name}_{name}",
                    computed_omega=result["best_omega"],
                    R=R,
                    known_omega=k,
                    save_path=str(variant_dir),
                )

    # Step 5: Analysis
    if args.analyze or args.run_dirac:
        print(f"\n--- Analysis ---")
        analyze_variants(experiment_dir, n, k, R, plot=args.plot)

    # Step 6: Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Graph: n={n}, k={k}, p={args.p}, seed={args.seed}, R={R}")
    print(f"  Planted nodes (base): {planted_nodes}")
    for name in VARIANT_NAMES:
        print(f"  {name:>6s} variant: clique at {targets[name]}")
    print(f"  Output: {experiment_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
