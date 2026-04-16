#!/usr/bin/env python3
"""Dirac-3 R-scaling experiment on DIMACS graphs.

Runs the standard Motzkin-Strauss max-clique formulation (J = -0.5*A) with
different sum-constraint values R to test whether the scaling parameter
affects solution quality on real-world benchmark instances.

Usage:
    uv run python run_R_sweep.py
    uv run python run_R_sweep.py --dimacs inputs/C500.9.clq --known-omega 57
    uv run python run_R_sweep.py --R-values 1,10,50,100 --num-samples 50
"""

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Load QCI credentials before importing Dirac solvers (no-ops if .env missing)
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eqc_models.solvers import Dirac3ContinuousCloudSolver
from eqc_models.base import QuadraticModel

from boson14_bench.io import read_dimacs_graph
from boson14_bench.problems import motzkin_straus_adjacency
from boson14_bench.core import (
    scaled_objective,
    scaled_objective_to_omega,
    omega_to_scaled_objective,
    plot_scaled_dirac_histogram,
)


def solve_dirac_R(
    A: np.ndarray,
    R: int = 1,
    num_samples: int = 100,
    relaxation_schedule: int = 2,
) -> dict:
    """Submit Motzkin-Strauss QP to Dirac-3 with sum_constraint=R.

    Uses J = -0.5*A (standard formulation). The solver minimises
    x^T J x = -0.5 x^T A x, equivalent to maximising g(x) = 0.5 x^T A x.
    """
    n = A.shape[0]
    C = np.zeros(n, dtype=np.float64)
    J = -0.5 * A

    model = QuadraticModel(C, J)
    model.upper_bound = R * np.ones(n, dtype=np.float64)

    solver = Dirac3ContinuousCloudSolver()

    print(
        f"  Submitting {n}-variable QP to Dirac-3 "
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
    omegas = []
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
    best_objective = objectives[best_idx]
    best_omega = omegas[best_idx]

    print(
        f"  Best g = {best_objective:.4f}  =>  omega = {best_omega}  "
        f"({solve_time:.1f}s, {len(objectives)} finite samples)"
    )

    return {
        "y_vectors": np.array(y_vectors),
        "objectives": objectives,
        "omegas": omegas,
        "best_omega": best_omega,
        "best_objective": best_objective,
        "mean_omega": float(np.mean(omegas)),
        "omega_counts": dict(sorted(Counter(omegas).items())),
        "solve_time": solve_time,
    }


def result_to_metadata(result: dict) -> dict:
    """Extract JSON-serializable per-R stats from a solve result."""
    return {
        "best_omega": result["best_omega"],
        "best_objective": float(result["best_objective"]),
        "mean_omega": result["mean_omega"],
        "omega_distribution": {str(k): v for k, v in result["omega_counts"].items()},
        "solve_time": result["solve_time"],
        "num_finite_samples": len(result["objectives"]),
    }


def plot_R_sweep_omega_histograms(
    all_results: dict[int, dict],
    graph_name: str,
    known_omega: int | None,
    save_path: Path,
) -> Path:
    """Stacked omega histograms — one subplot per R, shared x-axis."""
    R_values = sorted(all_results.keys())
    fig, axes = plt.subplots(
        len(R_values), 1, figsize=(10, 4 * len(R_values)), sharex=True,
    )
    if len(R_values) == 1:
        axes = [axes]

    all_omegas = [w for R in R_values for w in all_results[R]["omegas"]]
    bins = range(min(all_omegas) - 1, max(all_omegas) + 2)

    for ax, R in zip(axes, R_values):
        res = all_results[R]
        omegas = res["omegas"]
        ax.hist(
            omegas, bins=bins, color="#4C72B0", edgecolor="black",
            alpha=0.7, align="left",
        )
        if known_omega is not None:
            ax.axvline(
                x=known_omega, color="#2ca02c", linestyle="-",
                linewidth=2, label=f"Known w={known_omega}",
            )
        ax.set_ylabel("Count")
        ax.set_title(
            f"R = {R}  (best w={res['best_omega']}, "
            f"mean={res['mean_omega']:.1f}, n={len(omegas)} samples)"
        )
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Recovered clique number w")
    fig.suptitle(
        f"{graph_name}: Omega Distribution by R Value\n"
        f"(Dirac-3, J = -0.5A, schedule 2)",
        fontsize=14,
    )
    plt.tight_layout()
    fname = save_path / f"{graph_name}_R_sweep_omega_hist.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved omega histogram -> {fname}")
    return fname


def plot_R_sweep_summary_bars(
    all_results: dict[int, dict],
    graph_name: str,
    known_omega: int | None,
    save_path: Path,
) -> Path:
    """Summary bar chart: best/mean omega and hit fraction per R."""
    R_values = sorted(all_results.keys())
    best_omegas = [all_results[R]["best_omega"] for R in R_values]
    mean_omegas = [all_results[R]["mean_omega"] for R in R_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(R_values))
    width = 0.35
    ax1.bar(x - width / 2, best_omegas, width, label="Best w", color="#4C72B0")
    ax1.bar(x + width / 2, mean_omegas, width, label="Mean w", color="#ff7f0e")
    if known_omega is not None:
        ax1.axhline(
            y=known_omega, color="#2ca02c", linestyle="--",
            linewidth=2, label=f"Known w={known_omega}",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(R) for R in R_values])
    ax1.set_xlabel("R value")
    ax1.set_ylabel("Clique number w")
    ax1.set_title("Best and Mean Omega")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    if known_omega is not None:
        hit_fracs = [
            sum(1 for w in all_results[R]["omegas"] if w >= known_omega)
            / len(all_results[R]["omegas"])
            for R in R_values
        ]
        ax2.bar(x, hit_fracs, color="#2ca02c", alpha=0.7, edgecolor="black")
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(R) for R in R_values])
        ax2.set_xlabel("R value")
        ax2.set_ylabel("Fraction of samples")
        ax2.set_title(f"Fraction with w >= {known_omega}")
        ax2.set_ylim(0, 1.05)
        for i, frac in enumerate(hit_fracs):
            ax2.text(i, frac + 0.02, f"{frac:.0%}", ha="center", fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(
            0.5, 0.5, "Known omega\nnot provided",
            transform=ax2.transAxes, ha="center", va="center", fontsize=12,
        )

    fig.suptitle(f"{graph_name}: R-Scaling Summary", fontsize=14)
    plt.tight_layout()
    fname = save_path / f"{graph_name}_R_sweep_summary.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary bar chart -> {fname}")
    return fname


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dirac-3 R-scaling experiment on DIMACS graphs",
    )
    parser.add_argument(
        "--dimacs", type=Path,
        default=Path(__file__).resolve().parent / "inputs" / "C250.9.clq",
        help="Path to DIMACS .clq file (default: boson14/inputs/C250.9.clq)",
    )
    parser.add_argument(
        "--known-omega", type=int, default=44,
        help="Known clique number for reference lines (default: 44 for C250.9)",
    )
    parser.add_argument(
        "--R-values", type=str, default="1,10,100",
        help="Comma-separated R values to sweep (default: 1,10,100)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="Dirac-3 samples per R value (default: 100)",
    )
    parser.add_argument(
        "--relaxation-schedule", type=int, default=2,
        help="Relaxation schedule 1-4 (default: 2)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Output directory (default: boson14/output)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    parser.add_argument("--verbose", action="store_true", help="Per-sample details")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    R_values = [int(x.strip()) for x in args.R_values.split(",")]

    G = read_dimacs_graph(args.dimacs)
    graph_name = args.dimacs.stem
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = m / (n * (n - 1) / 2)
    A = motzkin_straus_adjacency(G)

    out_dir = args.output_dir / f"{graph_name}_R_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    g_star_by_R = {R: omega_to_scaled_objective(args.known_omega, R) for R in R_values}

    print(f"\n{'='*60}")
    print(f"R-Scaling Experiment: {graph_name}")
    print(f"{'='*60}")
    print(f"  |V|={n}, |E|={m}, density={density:.4f}")
    print(f"  Known omega = {args.known_omega}")
    print(f"  R values = {R_values}")
    print(f"  Formulation: J = -0.5*A (standard Dirac-3)")
    print(f"  Samples per R = {args.num_samples}")
    print(f"  Relaxation schedule = {args.relaxation_schedule}")
    if args.known_omega:
        print(f"\n  Theoretical g*(w={args.known_omega}) by R:")
        for R in R_values:
            print(f"    R={R:>5d}:  g* = {g_star_by_R[R]:.6f}")

    graph_info = {
        "graph": graph_name,
        "n": n,
        "edges": m,
        "density": round(density, 4),
        "known_omega": args.known_omega,
        "formulation": "J = -0.5*A (standard Dirac-3)",
        "num_samples": args.num_samples,
        "relaxation_schedule": args.relaxation_schedule,
    }

    all_results: dict[int, dict] = {}
    for R in R_values:
        print(f"\n{'='*60}")
        print(f"R = {R}")
        print(f"{'='*60}")

        try:
            result = solve_dirac_R(
                A, R=R,
                num_samples=args.num_samples,
                relaxation_schedule=args.relaxation_schedule,
            )
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            continue

        all_results[R] = result

        if args.verbose:
            print("\n  Per-sample details:")
            for i, (g, w) in enumerate(zip(result["objectives"], result["omegas"])):
                print(f"    Sample {i:3d}: g={g:12.4f}  w={w:3d}")

        npz_path = out_dir / f"{graph_name}_R{R}_dirac_solutions.npz"
        np.savez(npz_path, y_vectors=result["y_vectors"])
        print(f"  Saved solutions -> {npz_path}  (shape {result['y_vectors'].shape})")

        meta = {
            **graph_info,
            "R": R,
            **result_to_metadata(result),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = out_dir / f"{graph_name}_R{R}_dirac_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
        print(f"  Saved metadata -> {meta_path}")

    if not all_results:
        raise SystemExit("All R values failed — no results to report.")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"  {'R':>5s} | {'g*(w)':>12s} | {'best_g':>12s} | {'best_w':>7s} | {'mean_w':>7s} | {'hit_rate':>8s} | {'time':>7s}"
    print(header)
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}")
    for R in sorted(all_results.keys()):
        res = all_results[R]
        g_star = g_star_by_R[R] if args.known_omega else float("nan")
        hits = sum(1 for w in res["omegas"] if w >= args.known_omega) if args.known_omega else 0
        hit_str = f"{hits}/{len(res['omegas'])}" if args.known_omega else "n/a"
        print(
            f"  {R:>5d} | {g_star:>12.4f} | {res['best_objective']:>12.4f} | "
            f"{res['best_omega']:>7d} | {res['mean_omega']:>7.1f} | "
            f"{hit_str:>8s} | {res['solve_time']:>6.1f}s"
        )

    summary = {
        **graph_info,
        "R_values": sorted(all_results.keys()),
        "results": {str(R): result_to_metadata(res) for R, res in sorted(all_results.items())},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = out_dir / f"{graph_name}_R_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\nSaved sweep summary -> {summary_path}")

    if not args.no_plot:
        print(f"\n--- Generating Plots ---")
        plot_R_sweep_omega_histograms(all_results, graph_name, args.known_omega, out_dir)
        plot_R_sweep_summary_bars(all_results, graph_name, args.known_omega, out_dir)
        for R in sorted(all_results.keys()):
            res = all_results[R]
            plot_scaled_dirac_histogram(
                res["objectives"],
                f"{graph_name}_R{R}",
                computed_omega=res["best_omega"],
                R=R,
                known_omega=args.known_omega,
                save_path=str(out_dir),
            )

    print(f"\n{'='*60}")
    print(f"All outputs saved to {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
