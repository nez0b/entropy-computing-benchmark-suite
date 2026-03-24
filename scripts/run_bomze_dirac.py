#!/usr/bin/env python3
"""Compare standard vs Bomze Motzkin-Straus formulations on Dirac-3 hardware.

For each test graph, submits both the standard adjacency matrix A and the
Bomze-regularized A_bar = A + 0.5*I to the Dirac-3 solver, then extracts
per-sample x-vectors to verify that Bomze eliminates spurious solutions.

Usage:
    uv run python scripts/run_bomze_dirac.py --backend cloud
    uv run python scripts/run_bomze_dirac.py --backend direct
    uv run python scripts/run_bomze_dirac.py --backend both
    uv run python scripts/run_bomze_dirac.py --backend cloud --graph C125.9
    uv run python scripts/run_bomze_dirac.py --backend cloud --num-samples 50 --relaxation-schedule 3
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Load QCI credentials before importing Dirac solvers
_env_path = Path(__file__).resolve().parent.parent / "qci-eqc-models" / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)

from dirac_bench.problems import (  # noqa: E402
    objective,
    bomze_regularize,
    bomze_objective_to_omega,
    objective_to_omega,
    extract_support,
    is_clique,
    motzkin_straus_adjacency,
)
from dirac_bench.benchmark import KNOWN_OMEGA  # noqa: E402
from dirac_bench.io import read_dimacs_graph, get_graph_info  # noqa: E402
from dirac_bench.solvers.dirac_direct import DEFAULT_IP, DEFAULT_PORT  # noqa: E402
from dirac_bench.test_graphs import make_overlapping_k4, make_erdos_renyi  # noqa: E402
from dirac_bench.utils import _numpy_converter  # noqa: E402

BACKEND_CLOUD = "cloud"
BACKEND_DIRECT = "direct"


# ── Graph selection ──────────────────────────────────────────────────────

def build_graph_list(args) -> list[tuple[np.ndarray, int | None, str]]:
    """Build list of (adjacency_matrix, known_omega_or_None, name) tuples."""
    graphs = []

    if args.graph:
        # DIMACS graph
        dimacs_dir = Path(args.dimacs_dir)
        clq_path = dimacs_dir / f"{args.graph}.clq"
        if not clq_path.exists():
            print(f"ERROR: DIMACS file not found: {clq_path}")
            sys.exit(1)
        graph = read_dimacs_graph(str(clq_path))
        info = get_graph_info(graph)
        if args.max_nodes and info["nodes"] > args.max_nodes:
            print(f"Skipping {args.graph}: {info['nodes']} nodes > {args.max_nodes}")
        else:
            A = motzkin_straus_adjacency(graph)
            graphs.append((A, KNOWN_OMEGA.get(args.graph), args.graph))
    else:
        # Inline graphs
        inline_all = [
            make_overlapping_k4(),
            make_erdos_renyi(20, 0.7, seed=42),
            make_erdos_renyi(30, 0.5, seed=42),
            make_erdos_renyi(50, 0.9, seed=42),
        ]
        if args.inline_graph:
            matched = [g for g in inline_all if g[2] == args.inline_graph]
            if not matched:
                names = [g[2] for g in inline_all]
                print(f"ERROR: inline graph '{args.inline_graph}' not found. Available: {names}")
                sys.exit(1)
            graphs.extend(matched)
        else:
            graphs.extend(inline_all)

    return graphs


# ── Per-sample extraction ────────────────────────────────────────────────

def extract_solutions_from_response(result: dict, backend: str) -> list[np.ndarray]:
    """Normalize per-sample x-vectors from either backend's raw_response."""
    raw = result["raw_response"]

    if backend == BACKEND_CLOUD:
        solutions_raw = raw.get("results", {}).get("solutions", [])
        if not solutions_raw:
            raise RuntimeError("Cloud backend returned no solutions")
        return [np.array(s, dtype=np.float64) for s in solutions_raw]
    elif backend == BACKEND_DIRECT:
        solutions = np.array(raw["solution"])
        if solutions.ndim == 1:
            solutions = solutions.reshape(1, -1)
        return list(solutions)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── Per-sample analysis ─────────────────────────────────────────────────

def analyze_samples(
    solutions: list[np.ndarray],
    A: np.ndarray,
    A_bar: np.ndarray,
    threshold: float = 1e-4,
) -> tuple[list[dict], dict]:
    """Analyze each sample and return (per-sample list, aggregate summary)."""
    samples = []
    clique_count = 0
    for idx, x in enumerate(solutions):
        support = extract_support(x, threshold=threshold)
        clique = is_clique(support, A)
        if clique:
            clique_count += 1
        samples.append({
            "sample_index": idx,
            "x": x.tolist(),
            "support": support,
            "support_size": len(support),
            "is_clique": clique,
            "obj_standard": round(objective(x, A), 6),
            "obj_bomze": round(objective(x, A_bar), 6),
        })

    total = len(samples)
    spurious_count = total - clique_count
    support_sizes = [s["support_size"] for s in samples]
    std_objs = [s["obj_standard"] for s in samples]
    bomze_objs = [s["obj_bomze"] for s in samples]

    summary = {
        "total_samples": total,
        "clique_count": clique_count,
        "spurious_count": spurious_count,
        "spurious_pct": round(100.0 * spurious_count / total, 1) if total else 0.0,
        "best_standard_obj": max(std_objs) if std_objs else 0.0,
        "best_bomze_obj": max(bomze_objs) if bomze_objs else 0.0,
        "omega_standard": objective_to_omega(max(std_objs)) if std_objs else 0,
        "omega_bomze": bomze_objective_to_omega(max(bomze_objs)) if bomze_objs else 0,
        "support_size_min": min(support_sizes) if support_sizes else 0,
        "support_size_max": max(support_sizes) if support_sizes else 0,
        "support_size_mean": round(float(np.mean(support_sizes)), 1) if support_sizes else 0.0,
    }
    return samples, summary


# ── Solver dispatch ──────────────────────────────────────────────────────

def run_dirac_config(
    Q: np.ndarray,
    backend: str,
    args,
) -> dict:
    """Call the appropriate Dirac solver with matrix Q."""
    if backend == BACKEND_CLOUD:
        from dirac_bench.solvers.dirac import solve_dirac
        return solve_dirac(
            Q,
            num_samples=args.num_samples,
            relaxation_schedule=args.relaxation_schedule,
        )
    elif backend == BACKEND_DIRECT:
        from dirac_bench.solvers.dirac_direct import solve_dirac_direct
        return solve_dirac_direct(
            Q,
            ip_address=args.ip,
            port=args.port,
            num_samples=args.num_samples,
            relaxation_schedule=args.relaxation_schedule,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── Output ───────────────────────────────────────────────────────────────

def save_analysis(
    graph_name: str,
    num_nodes: int,
    backend: str,
    formulation: str,
    args,
    samples: list[dict],
    summary: dict,
    solve_time: float,
    run_timestamp: str,
) -> Path | None:
    """Write per-sample JSON analysis to results directory."""
    if args.no_save:
        return None

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "graph_name": graph_name,
        "num_nodes": num_nodes,
        "backend": backend,
        "formulation": formulation,
        "num_samples_requested": args.num_samples,
        "relaxation_schedule": args.relaxation_schedule,
        "timestamp": run_timestamp,
        "solve_time": round(solve_time, 2),
        "summary": summary,
        "samples": samples,
    }

    safe_name = graph_name.replace("(", "").replace(")", "").replace(",", "_")
    filename = results_dir / f"{formulation}_{backend}_{safe_name}_{run_timestamp}.json"

    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=_numpy_converter)

    print(f"    Saved -> {filename}")
    return filename


def print_comparison_table(summary_rows: list[dict]) -> None:
    """Print formatted ASCII summary table."""
    if not summary_rows:
        return

    print(f"\n{'=' * 25} Summary {'=' * 25}")
    header = (
        f"{'Graph':<17} {'Backend':<8} {'Formulation':<12} "
        f"{'Samples':>7} {'Cliques':>8} {'Spurious':>8} {'%':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in summary_rows:
        print(
            f"{r['graph']:<17} {r['backend']:<8} {r['formulation']:<12} "
            f"{r['samples']:>7} {r['cliques']:>8} {r['spurious']:>8} "
            f"{r['spurious_pct']:>5.1f}"
        )


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare standard vs Bomze Motzkin-Straus on Dirac-3 hardware"
    )

    graph_group = parser.add_mutually_exclusive_group()
    graph_group.add_argument(
        "--graph", type=str, default=None,
        help="Run a specific DIMACS graph by stem name (e.g. C125.9)",
    )
    graph_group.add_argument(
        "--inline-graph", type=str, default=None,
        help="Run a specific inline graph by name (e.g. 'ER(20,0.7)')",
    )

    parser.add_argument(
        "--backend", type=str, default="cloud",
        choices=[BACKEND_CLOUD, BACKEND_DIRECT, "both"],
        help="Dirac backend: cloud (default), direct, or both",
    )
    parser.add_argument(
        "--dimacs-dir", type=str, default="problems/dimacs",
        help="Directory containing .clq files (default: problems/dimacs)",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=None,
        help="Skip graphs with more than this many nodes",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="Number of Dirac-3 samples per run (1-100, default: 100)",
    )
    parser.add_argument(
        "--relaxation-schedule", type=int, default=2,
        help="Dirac-3 relaxation schedule (1-4, default: 2)",
    )
    parser.add_argument(
        "--ip", type=str, default=DEFAULT_IP,
        help=f"Dirac-3 hardware IP address (default: {DEFAULT_IP})",
    )
    parser.add_argument(
        "--port", type=str, default=DEFAULT_PORT,
        help=f"Dirac-3 gRPC port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--threshold", type=float, default=1e-4,
        help="Support extraction threshold (default: 1e-4)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/bomze_dirac",
        help="Directory for output JSON files (default: results/bomze_dirac)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip writing JSON output files",
    )

    args = parser.parse_args()

    backends = [BACKEND_CLOUD, BACKEND_DIRECT] if args.backend == "both" else [args.backend]

    print("=" * 70)
    print("Bomze vs Standard Motzkin-Straus on Dirac-3")
    print("=" * 70)
    print(f"Backends: {', '.join(backends)}    Samples: {args.num_samples}    Schedule: {args.relaxation_schedule}")

    graphs = build_graph_list(args)
    if not graphs:
        print("No graphs to run.")
        return 1

    run_timestamp = time.strftime("%Y%m%dT%H%M%S")
    summary_rows = []

    for A, known_omega, graph_name in graphs:
        n = A.shape[0]
        omega_str = f"known_omega={known_omega}" if known_omega else "omega=?"
        print(f"\nGraph: {graph_name} ({n} nodes)  {omega_str}")

        A_bar = bomze_regularize(A)

        # Build all (backend, formulation, Q) jobs for this graph
        jobs = []
        for backend in backends:
            for formulation in ["standard", "bomze"]:
                Q = A if formulation == "standard" else A_bar
                jobs.append((backend, formulation, Q))

        # Run solver jobs concurrently (they are I/O-bound API calls)
        results_map = {}
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = {
                pool.submit(run_dirac_config, Q, backend, args): (backend, formulation)
                for backend, formulation, Q in jobs
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results_map[key] = future.result()
                except Exception as e:
                    results_map[key] = e

        # Process results in deterministic order
        for backend, formulation, _Q in jobs:
            key = (backend, formulation)
            label = f"[{backend}/{formulation}]"
            result = results_map[key]

            if isinstance(result, Exception):
                print(f"  {label:<22} ERROR: {result}")
                summary_rows.append({
                    "graph": graph_name, "backend": backend,
                    "formulation": formulation, "samples": 0,
                    "cliques": 0, "spurious": 0, "spurious_pct": 0.0,
                })
                continue

            try:
                solutions = extract_solutions_from_response(result, backend)
                samples, summary = analyze_samples(solutions, A, A_bar, threshold=args.threshold)

                print(
                    f"  {label:<22} {summary['total_samples']} samples: "
                    f"{summary['clique_count']} clique, {summary['spurious_count']} spurious "
                    f"({summary['spurious_pct']:.1f}%)"
                )

                save_analysis(
                    graph_name, n, backend, formulation, args,
                    samples, summary, result["solve_time"], run_timestamp,
                )

                summary_rows.append({
                    "graph": graph_name, "backend": backend,
                    "formulation": formulation,
                    "samples": summary["total_samples"],
                    "cliques": summary["clique_count"],
                    "spurious": summary["spurious_count"],
                    "spurious_pct": summary["spurious_pct"],
                })

            except Exception as e:
                print(f"  {label:<22} ERROR: {e}")
                summary_rows.append({
                    "graph": graph_name, "backend": backend,
                    "formulation": formulation, "samples": 0,
                    "cliques": 0, "spurious": 0, "spurious_pct": 0.0,
                })

    print_comparison_table(summary_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
