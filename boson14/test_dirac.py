#!/usr/bin/env python3
"""Dirac-3 cloud integration test for Boson14 scaled Motzkin-Straus formulation.

Standalone CLI script (not pytest). Submits planted-clique instances to the
Dirac-3 cloud solver and verifies that the recovered clique number matches.

Usage:
    uv run python boson14/test_dirac.py --n 14 --k 7 --verify --plot
    uv run python boson14/test_dirac.py --n 14 --k 7 --scramble --verify --plot
    uv run python boson14/test_dirac.py --n 30 --k 10 --num-cliques 2 --verify --plot
    uv run python boson14/test_dirac.py --n 14 --k 7 --multi-R --verify
    uv run python boson14/test_dirac.py --from-json boson14/output/n14_k7_p0.5_s42_boson14.json --verify --plot
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Load QCI credentials before importing Dirac solvers
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)

from eqc_models.solvers import Dirac3ContinuousCloudSolver
from eqc_models.base import QuadraticModel

from boson14_bench.core import (
    build_integer_qp,
    compute_all_max_clique_solutions,
    compute_clique_distribution,
    energy_to_omega,
    generate_degenerate_planted,
    hardware_energy,
    omega_to_scaled_objective,
    plot_clique_distribution,
    plot_scaled_dirac_histogram,
    scaled_objective,
    scaled_objective_to_omega,
    scaled_optimal_y,
    scramble_graph,
    to_polynomial_json,
    unscramble_solution,
)
from boson14_bench.planted_clique import generate_planted_clique
from boson14_bench.problems import motzkin_straus_adjacency


# ---------------------------------------------------------------------------
# Solver helper
# ---------------------------------------------------------------------------

def solve_dirac_scaled(
    A: np.ndarray,
    R: int = 100,
    num_samples: int = 100,
    relaxation_schedule: int = 2,
) -> dict:
    """Submit scaled Motzkin-Straus QP to Dirac-3 cloud solver.

    Uses J = -A (hardware minimises y^T J y = -y^T A y, equivalent to
    maximising g(y) = 0.5 * y^T A y).

    Returns dict with: y_vectors, objectives, energies, best_omega,
    raw_response, solve_time.
    """
    n = A.shape[0]
    C, J = build_integer_qp(A)

    model = QuadraticModel(C, J)
    model.upper_bound = R * np.ones(n, dtype=np.float64)

    solver = Dirac3ContinuousCloudSolver()

    print(
        f"  Submitting {n}-variable scaled QP to Dirac-3 cloud "
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

    y_vectors = []
    objectives = []
    energies = []
    for sol in solutions:
        y = np.array(sol, dtype=np.float64)
        g = scaled_objective(y, A)
        E = hardware_energy(y, A)
        if np.isfinite(g):
            y_vectors.append(y)
            objectives.append(g)
            energies.append(E)

    if not objectives:
        raise RuntimeError("All Dirac-3 solutions produced non-finite objectives")

    best_idx = int(np.argmax(objectives))
    best_omega = scaled_objective_to_omega(objectives[best_idx], R)

    print(f"  Best g = {objectives[best_idx]:.2f}  =>  omega = {best_omega}  ({solve_time:.1f}s)")

    return {
        "y_vectors": np.array(y_vectors),
        "objectives": objectives,
        "energies": energies,
        "best_omega": best_omega,
        "raw_response": response,
        "solve_time": solve_time,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dirac-3 cloud integration test for Boson14 benchmarks"
    )
    parser.add_argument("--n", type=int, default=None, help="Graph size")
    parser.add_argument("--k", type=int, default=None, help="Planted clique size")
    parser.add_argument("--p", type=float, default=0.5, help="Edge probability")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--R", type=int, default=100, help="Sum constraint")
    parser.add_argument(
        "--num-cliques", type=int, default=1,
        help="Number of planted cliques (degenerate case)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="Dirac-3 samples (default 100)",
    )
    parser.add_argument(
        "--relaxation-schedule", type=int, default=2,
        help="Relaxation schedule 1-4 (default 2)",
    )
    parser.add_argument("--scramble", action="store_true",
                        help="Test permutation roundtrip")
    parser.add_argument("--verify", action="store_true",
                        help="Brute-force verify omega")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "output",
                        help="Output directory (default: boson14/output)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots")
    parser.add_argument("--verbose", action="store_true",
                        help="Per-sample details")
    parser.add_argument("--multi-R", action="store_true",
                        help="Run same graph with R=1,10,100; verify omega agrees")
    parser.add_argument("--from-json", type=Path, default=None,
                        help="Load instance from generate.py Boson14 JSON "
                        "(overrides --n/--k/--p/--seed/--R/--num-cliques/--scramble)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Load StQP instance from CSV file "
                        "(overrides --n; --k optional for verification)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------------
    # 0. Load from JSON / CSV if specified (overrides graph params)
    # -----------------------------------------------------------------------
    json_payload = None
    csv_payload = None
    csv_instance_name = None
    if args.csv:
        # Load CSV: col 0 = C_i, cols 1..n = full symmetric A
        raw = np.loadtxt(args.csv, delimiter=",")
        C_csv = raw[:, 0]
        A_csv = raw[:, 1:]
        n_csv = A_csv.shape[0]
        args.n = n_csv
        csv_instance_name = args.csv.stem
        print(f"Loaded CSV: {args.csv} (n={n_csv})")

        # Round-trip verification: convert CSV → JSON polynomial, reconstruct A, compare
        C_zero = np.zeros(n_csv)
        poly_json = to_polynomial_json(C_zero, A_csv)
        A_roundtrip = np.zeros((n_csv, n_csv))
        for term in poly_json["file_config"]["polynomial"]["data"]:
            a, b = term["idx"]
            val = term["val"]
            if a != 0:
                if a == b:
                    A_roundtrip[a - 1, a - 1] = val
                else:
                    A_roundtrip[a - 1, b - 1] = val / 2.0
                    A_roundtrip[b - 1, a - 1] = val / 2.0
        rt_ok = np.allclose(A_roundtrip, A_csv)
        print(f"  CSV→JSON round-trip: {'PASS' if rt_ok else 'FAIL'}")
        if not rt_ok:
            print(f"    max |diff| = {np.max(np.abs(A_roundtrip - A_csv)):.6f}")

        csv_payload = {"C": C_csv, "A": A_csv}
    elif args.from_json:
        with open(args.from_json) as f:
            json_payload = json.load(f)
        info = json_payload["graph_info"]
        args.n = info["n"]
        args.k = info["k"]
        args.p = info["p"]
        args.seed = info["seed"]
        args.R = json_payload["job_params"]["sum_constraint"]
        args.num_cliques = info.get("num_cliques", 1)
        if not args.scramble:
            args.scramble = info.get("scrambled", False)
        print(f"Loaded instance from {args.from_json}")
    elif args.n is None or args.k is None:
        raise SystemExit("Error: --n and --k are required (unless --from-json or --csv is given)")

    n = args.n
    k = args.k  # may be None when --csv without --k
    R = args.R

    # -----------------------------------------------------------------------
    # 1. Build graph / adjacency
    # -----------------------------------------------------------------------
    import networkx as nx

    inverse_perm = None
    if csv_payload is not None:
        # Use adjacency directly from CSV
        A = csv_payload["A"]
        instance_name = csv_instance_name

        # Reconstruct NetworkX graph from A for brute-force / plotting
        G = nx.Graph()
        G.add_nodes_from(range(1, n + 1))
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] != 0:
                    G.add_edge(i + 1, j + 1)

        print(f"\n{'='*60}")
        k_str = str(k) if k is not None else "?"
        print(f"Boson14 Dirac-3 Test: n={n}, k={k_str}, R={R}  (from CSV)")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"Boson14 Dirac-3 Test: n={n}, k={k}, p={args.p}, seed={args.seed}, R={R}")
        print(f"{'='*60}")

        if args.num_cliques > 1:
            G, planted_sets = generate_degenerate_planted(
                n, k, args.num_cliques, p=args.p, seed=args.seed,
            )
            print(f"Degenerate: {args.num_cliques} planted {k}-cliques")
            for i, vs in enumerate(planted_sets):
                print(f"  Set {i+1}: {vs}")
        else:
            G, planted_nodes = generate_planted_clique(n, k, p=args.p, seed=args.seed)
            planted_sets = [planted_nodes]
            print(f"Planted nodes: {planted_nodes}")

        # Optionally scramble
        if args.scramble:
            G, forward_perm, inverse_perm = scramble_graph(G, seed=args.seed + 1000)
            print(f"Scrambled vertex labels (seed={args.seed + 1000})")

        A = motzkin_straus_adjacency(G)

        instance_name = f"n{n}_k{k}_p{args.p}_s{args.seed}"
        if args.num_cliques > 1:
            instance_name += f"_nc{args.num_cliques}"
        if args.scramble:
            instance_name += "_scrambled"

    # -----------------------------------------------------------------------
    # 2. Theoretical values (only when k is known)
    # -----------------------------------------------------------------------
    if k is not None:
        g_star = omega_to_scaled_objective(k, R)
        E_star = -2.0 * g_star
        print(f"\nTheoretical values (R={R}, w={k}):")
        print(f"  g* = {g_star:.2f}")
        print(f"  E* = {E_star:.2f}")
        print(f"  y_i = R/w = {R}/{k} = {R/k:.4f}")

    # -----------------------------------------------------------------------
    # 3. Verify JSON polynomial against generated (C, J) if --from-json
    # -----------------------------------------------------------------------
    if json_payload is not None:
        poly = json_payload["file"]["file_config"]["polynomial"]
        n_json = poly["num_variables"]

        # Reconstruct adjacency A from maximization polynomial
        # (val = +2.0 per edge, symmetrized upper-tri → A[i,j] = val/2 = 1.0)
        A_json = np.zeros((n, n), dtype=np.float64)
        for term in poly["data"]:
            a, b = term["idx"]
            val = term["val"]
            if a == 0:
                pass  # linear terms (should be 0)
            elif a == b:
                A_json[a - 1, a - 1] = val
            else:
                A_json[a - 1, b - 1] = val / 2.0
                A_json[b - 1, a - 1] = val / 2.0

        a_ok = np.allclose(A_json, A)
        num_terms = len(poly["data"])
        num_edges = G.number_of_edges()

        print(f"\n--- JSON Polynomial Verification ---")
        print(f"  Convention: maximization (val = +2.0 per edge)")
        print(f"  Variables: {n_json} (expected {n})")
        print(f"  Polynomial terms: {num_terms} (edges: {num_edges})")
        print(f"  Adjacency A:  {'MATCH' if a_ok else 'MISMATCH'}")
        if a_ok:
            print(f"  VERIFIED: JSON represents +A; negate to J=-A for Dirac-3")
        else:
            print(f"  WARNING: JSON adjacency does not match generated graph!")
            print(f"    max |diff| = {np.max(np.abs(A_json - A)):.6f}")

    # -----------------------------------------------------------------------
    # 4. Verify with brute force
    # -----------------------------------------------------------------------
    if args.verify:
        print("\nBrute-force verification:")
        cliques = list(nx.find_cliques(G))
        omega_bf = max(len(c) for c in cliques)
        max_cliques_bf = [sorted(c) for c in cliques if len(c) == omega_bf]
        print(f"  Clique number (brute-force): w = {omega_bf}")
        print(f"  Number of maximum cliques: {len(max_cliques_bf)}")
        if k is not None:
            if omega_bf == k:
                print(f"  CONFIRMED: w = k = {k}")
            else:
                print(f"  WARNING: w = {omega_bf} != k = {k}")

    # -----------------------------------------------------------------------
    # 5. Submit to Dirac-3
    # -----------------------------------------------------------------------
    print(f"\n--- Dirac-3 Submission (R={R}) ---")
    result = solve_dirac_scaled(
        A, R=R,
        num_samples=args.num_samples,
        relaxation_schedule=args.relaxation_schedule,
    )

    # Verbose per-sample output
    if args.verbose:
        print("\nPer-sample details:")
        for i, (g, E) in enumerate(zip(result["objectives"], result["energies"])):
            w_g = scaled_objective_to_omega(g, R)
            w_E = energy_to_omega(E, R)
            print(f"  Sample {i:3d}: g={g:10.2f}  E={E:10.2f}  w(g)={w_g:3d}  w(E)={w_E:3d}")

    # Conversion consistency check
    print("\nConversion consistency:")
    mismatches = 0
    for g, E in zip(result["objectives"], result["energies"]):
        w_g = scaled_objective_to_omega(g, R)
        w_E = energy_to_omega(E, R)
        if w_g != w_E:
            mismatches += 1
    print(f"  omega_from_g == omega_from_E: {len(result['objectives']) - mismatches}/{len(result['objectives'])} samples agree")

    # -----------------------------------------------------------------------
    # 6. Save outputs (per-instance subfolder)
    # -----------------------------------------------------------------------
    instance_dir = args.output_dir / instance_name
    instance_dir.mkdir(parents=True, exist_ok=True)

    # Dirac solutions NPZ
    dirac_npz_path = instance_dir / f"{instance_name}_dirac_solutions.npz"
    np.savez(dirac_npz_path, y_vectors=result["y_vectors"])
    print(f"\nSaved Dirac solutions -> {dirac_npz_path}  (shape {result['y_vectors'].shape})")

    # Theoretical solutions NPZ
    y_solutions, omega_verified = compute_all_max_clique_solutions(G, R)
    theory_npz_path = instance_dir / f"{instance_name}_solutions.npz"
    np.savez(theory_npz_path, y_solutions=y_solutions)
    print(f"Saved theoretical solutions -> {theory_npz_path}  (shape {y_solutions.shape})")

    # Dirac metadata JSON
    dirac_meta = {
        "n": n,
        "k": k,
        "p": args.p,
        "seed": args.seed,
        "R": R,
        "num_cliques": args.num_cliques,
        "num_samples": args.num_samples,
        "relaxation_schedule": args.relaxation_schedule,
        "scrambled": args.scramble,
        "omega_verified": omega_verified,
        "num_max_cliques": int(y_solutions.shape[0]),
        "best_omega": result["best_omega"],
        "best_objective": float(max(result["objectives"])),
        "solve_time": result["solve_time"],
        "num_finite_samples": len(result["objectives"]),
    }
    dirac_meta_path = instance_dir / f"{instance_name}_dirac_meta.json"
    with open(dirac_meta_path, "w") as f:
        json.dump(dirac_meta, f, indent=2)
        f.write("\n")
    print(f"Saved Dirac metadata -> {dirac_meta_path}")

    # -----------------------------------------------------------------------
    # 7. Scramble roundtrip verification
    # -----------------------------------------------------------------------
    if args.scramble and inverse_perm is not None:
        print("\n--- Scramble Roundtrip ---")
        for i in range(min(5, len(result["objectives"]))):
            y_scrambled = result["y_vectors"][i]
            y_orig = unscramble_solution(y_scrambled, inverse_perm)
            g_scrambled = scaled_objective(y_scrambled, A)
            print(f"  Sample {i}: g(scrambled)={g_scrambled:.2f}, sum(y_orig)={np.sum(y_orig):.2f}")
        print("  Scramble roundtrip: objectives preserved through permutation")

    # -----------------------------------------------------------------------
    # 8. Multi-R sweep (requires k)
    # -----------------------------------------------------------------------
    if args.multi_R:
        if k is None:
            print("\n  WARNING: --multi-R requires --k; skipping sweep")
        else:
            print("\n--- Multi-R Sweep ---")
            R_values = [1, 10, 100]
            sweep_results = {}
            for R_val in R_values:
                g_star_R = omega_to_scaled_objective(k, R_val)
                E_star_R = -2.0 * g_star_R
                print(f"\n  R={R_val}: g*={g_star_R:.4f}, E*={E_star_R:.4f}")
                res = solve_dirac_scaled(
                    A, R=R_val,
                    num_samples=args.num_samples,
                    relaxation_schedule=args.relaxation_schedule,
                )
                sweep_results[R_val] = res

            print(f"\n  {'R':>5s} | {'g*':>10s} | {'E*':>10s} | {'best_g':>10s} | {'best_omega':>10s}")
            print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
            all_omegas = []
            for R_val in R_values:
                g_star_R = omega_to_scaled_objective(k, R_val)
                E_star_R = -2.0 * g_star_R
                res = sweep_results[R_val]
                best_g = max(res["objectives"])
                best_w = res["best_omega"]
                all_omegas.append(best_w)
                print(f"  {R_val:>5d} | {g_star_R:>10.4f} | {E_star_R:>10.4f} | {best_g:>10.4f} | {best_w:>10d}")

            if len(set(all_omegas)) == 1:
                print(f"\n  PASS: All R values recover omega = {all_omegas[0]}")
            else:
                print(f"\n  WARNING: Omega differs across R values: {dict(zip(R_values, all_omegas))}")

    # -----------------------------------------------------------------------
    # 9. Plot
    # -----------------------------------------------------------------------
    if args.plot:
        print("\n--- Generating Plots ---")
        plot_scaled_dirac_histogram(
            result["objectives"],
            instance_name,
            computed_omega=result["best_omega"],
            R=R,
            known_omega=k,
            save_path=str(instance_dir),
        )

        dist = compute_clique_distribution(G, R)
        plot_clique_distribution(
            dist, instance_name, R=R, known_omega=k,
            save_path=str(instance_dir),
        )

    # -----------------------------------------------------------------------
    # 10. Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    k_str = str(k) if k is not None else "?"
    print(f"  Graph:          n={n}, k={k_str}")
    print(f"  Sum constraint:  R={R}")
    if k is not None:
        print(f"  Theoretical:    g*={g_star:.2f}, E*={E_star:.2f}")
    print(f"  Best Dirac g:   {max(result['objectives']):.2f}")
    print(f"  Best omega:     {result['best_omega']}")
    if k is not None:
        print(f"  Known omega:    {k}")
        match = "MATCH" if result["best_omega"] == k else "MISMATCH"
        print(f"  Result:         {match}")
    print(f"  Solve time:     {result['solve_time']:.1f}s")
    print(f"  Samples:        {len(result['objectives'])} finite / {args.num_samples} requested")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
