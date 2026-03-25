#!/usr/bin/env python3
"""Collect raw x-vectors from Dirac-3 hardware across all config combinations.

Saves results as .npz files so that extraction analysis can be done offline
without re-running hardware.

Config matrix:
    (C125.9, C250.9) × (cloud, direct) × (standard, bomze) × (schedule 2, 3, 4)
    = 2 × 2 × 2 × 3 = 24 runs

Usage:
    uv run python scripts/collect_raw_vectors.py
    uv run python scripts/collect_raw_vectors.py --graphs C125.9 --backends cloud --schedules 2
    uv run python scripts/collect_raw_vectors.py --force   # re-run even if .npz exists
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
    motzkin_straus_adjacency,
    bomze_regularize,
    objective,
    objective_to_omega,
)
from dirac_bench.benchmark import KNOWN_OMEGA  # noqa: E402
from dirac_bench.io import read_dimacs_graph  # noqa: E402
from dirac_bench.solvers.dirac_direct import DEFAULT_IP, DEFAULT_PORT  # noqa: E402

BACKEND_CLOUD = "cloud"
BACKEND_DIRECT = "direct"


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


def run_dirac_config(
    Q: np.ndarray, backend: str, args, schedule: int | None = None,
) -> dict:
    """Call the appropriate Dirac solver.

    Args:
        schedule: Override relaxation_schedule (thread-safe vs mutating args).
    """
    sched = schedule if schedule is not None else args.relaxation_schedule
    if backend == BACKEND_CLOUD:
        from dirac_bench.solvers.dirac import solve_dirac
        return solve_dirac(
            Q,
            num_samples=args.num_samples,
            relaxation_schedule=sched,
        )
    elif backend == BACKEND_DIRECT:
        from dirac_bench.solvers.dirac_direct import solve_dirac_direct
        return solve_dirac_direct(
            Q,
            ip_address=args.ip,
            port=args.port,
            num_samples=args.num_samples,
            relaxation_schedule=sched,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def output_filename(graph: str, backend: str, formulation: str, schedule: int) -> str:
    """Build the .npz filename for a config."""
    return f"{graph}_{backend}_{formulation}_s{schedule}.npz"


def collect_one(
    graph_name: str,
    A: np.ndarray,
    backend: str,
    formulation: str,
    schedule: int,
    args,
) -> dict:
    """Run one (graph, backend, formulation, schedule) config and save .npz.

    Returns a summary dict for the console table.
    """
    fname = output_filename(graph_name, backend, formulation, schedule)
    out_path = Path(args.output_dir) / fname

    if out_path.exists() and not args.force:
        print(f"  SKIP {fname} (exists, use --force to overwrite)")
        return {
            "graph": graph_name, "backend": backend, "formulation": formulation,
            "schedule": schedule, "status": "skipped",
        }

    # Build the Q matrix
    Q = bomze_regularize(A) if formulation == "bomze" else A

    label = f"{graph_name}/{backend}/{formulation}/s{schedule}"
    print(f"  Running {label} ...")

    try:
        t0 = time.time()
        result = run_dirac_config(Q, backend, args, schedule=schedule)
        solve_time = time.time() - t0

        solutions = extract_solutions_from_response(result, backend)
        x_vectors = np.array(solutions, dtype=np.float64)

        # Compute objectives against ORIGINAL A (not Q)
        objectives = np.array([objective(x, A) for x in solutions], dtype=np.float64)

        best_obj = float(objectives.max())
        omega = objective_to_omega(best_obj)

        # Metadata
        metadata = json.dumps({
            "graph": graph_name,
            "backend": backend,
            "formulation": formulation,
            "schedule": schedule,
            "num_samples": len(solutions),
            "solve_time": round(solve_time, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "known_omega": KNOWN_OMEGA.get(graph_name),
            "omega": omega,
        })

        # Save
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            x_vectors=x_vectors,
            objectives=objectives,
            metadata=np.array(metadata),  # store as 0-d string array
        )
        print(f"  DONE {fname}: {len(solutions)} samples, best_obj={best_obj:.6f}, omega={omega} ({solve_time:.1f}s)")

        return {
            "graph": graph_name, "backend": backend, "formulation": formulation,
            "schedule": schedule, "status": "ok",
            "samples": len(solutions), "best_obj": round(best_obj, 6),
            "omega": omega, "solve_time": round(solve_time, 1),
        }

    except Exception as e:
        print(f"  ERROR {fname}: {e}")
        return {
            "graph": graph_name, "backend": backend, "formulation": formulation,
            "schedule": schedule, "status": f"error: {e}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect raw x-vectors from Dirac-3 across config combinations"
    )
    parser.add_argument(
        "--graphs", nargs="+", default=["C125.9", "C250.9"],
        help="DIMACS graph stems (default: C125.9 C250.9)",
    )
    parser.add_argument(
        "--backends", nargs="+", default=["cloud", "direct"],
        choices=["cloud", "direct"],
        help="Dirac backends (default: cloud direct)",
    )
    parser.add_argument(
        "--formulations", nargs="+", default=["standard", "bomze"],
        choices=["standard", "bomze"],
        help="Formulations (default: standard bomze)",
    )
    parser.add_argument(
        "--schedules", nargs="+", type=int, default=[2, 3, 4],
        help="Relaxation schedules (default: 2 3 4)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="Samples per run (default: 100)",
    )
    parser.add_argument(
        "--ip", type=str, default=DEFAULT_IP,
        help=f"Dirac-3 hardware IP (default: {DEFAULT_IP})",
    )
    parser.add_argument(
        "--port", type=str, default=DEFAULT_PORT,
        help=f"Dirac-3 gRPC port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/raw_vectors",
        help="Output directory (default: results/raw_vectors)",
    )
    parser.add_argument(
        "--dimacs-dir", type=str, default="problems/dimacs",
        help="DIMACS graph directory (default: problems/dimacs)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing .npz files",
    )

    args = parser.parse_args()
    # Store a default relaxation_schedule (will be overridden per-config)
    args.relaxation_schedule = 2

    print("=" * 70)
    print("Raw Vector Collection")
    print("=" * 70)

    # Load graphs
    dimacs_dir = Path(args.dimacs_dir)
    graph_data: dict[str, np.ndarray] = {}
    for name in args.graphs:
        clq_path = dimacs_dir / f"{name}.clq"
        if not clq_path.exists():
            print(f"ERROR: {clq_path} not found, skipping {name}")
            continue
        G = read_dimacs_graph(str(clq_path))
        A = motzkin_straus_adjacency(G)
        graph_data[name] = A
        omega_str = KNOWN_OMEGA.get(name, "?")
        print(f"  Loaded {name}: {A.shape[0]} nodes, known omega={omega_str}")

    if not graph_data:
        print("No graphs loaded.")
        return 1

    # Build job list
    jobs = []
    for graph_name in args.graphs:
        if graph_name not in graph_data:
            continue
        for backend in args.backends:
            for formulation in args.formulations:
                for schedule in args.schedules:
                    jobs.append((graph_name, backend, formulation, schedule))

    total = len(jobs)
    print(f"\nTotal configs: {total}")
    print(f"Output dir: {args.output_dir}")
    print()

    summary = []

    # Cloud configs can run in parallel (API); direct must serialize (single device)
    cloud_jobs = [(g, b, f, s) for g, b, f, s in jobs if b == BACKEND_CLOUD]
    direct_jobs = [(g, b, f, s) for g, b, f, s in jobs if b == BACKEND_DIRECT]

    # Run cloud jobs with thread pool
    if cloud_jobs:
        print(f"── Cloud configs ({len(cloud_jobs)} jobs, parallel) ──")
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {}
            for graph_name, backend, formulation, schedule in cloud_jobs:
                A = graph_data[graph_name]
                fut = pool.submit(
                    collect_one, graph_name, A, backend, formulation, schedule, args
                )
                futures[fut] = (graph_name, backend, formulation, schedule)

            for fut in as_completed(futures):
                result = fut.result()
                summary.append(result)

    # Run direct jobs serially
    if direct_jobs:
        print(f"\n── Direct configs ({len(direct_jobs)} jobs, serial) ──")
        for graph_name, backend, formulation, schedule in direct_jobs:
            A = graph_data[graph_name]
            result = collect_one(graph_name, A, backend, formulation, schedule, args)
            summary.append(result)

    # Print summary table
    print(f"\n{'=' * 70}")
    print("Collection Summary")
    print(f"{'=' * 70}")
    header = f"{'Graph':<10} {'Backend':<8} {'Form':<10} {'Sched':>5} {'Status':<10} {'Samples':>7} {'BestObj':>10} {'Omega':>7} {'Time':>6}"
    print(header)
    print("-" * len(header))
    for r in sorted(summary, key=lambda r: (r["graph"], r["backend"], r["formulation"], r["schedule"])):
        print(
            f"{r['graph']:<10} {r['backend']:<8} {r['formulation']:<10} "
            f"{r['schedule']:>5} {r['status']:<10} "
            f"{r.get('samples', ''):>7} {str(r.get('best_obj', '')):>10} "
            f"{str(r.get('omega', '')):>7} {str(r.get('solve_time', '')):>6}"
        )

    ok_count = sum(1 for r in summary if r["status"] == "ok")
    skip_count = sum(1 for r in summary if r["status"] == "skipped")
    err_count = sum(1 for r in summary if r["status"].startswith("error"))
    print(f"\n{ok_count} completed, {skip_count} skipped, {err_count} errors")

    return 0


if __name__ == "__main__":
    sys.exit(main())
