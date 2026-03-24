#!/usr/bin/env python3
"""Run the Dirac-3 benchmark suite on DIMACS max-clique graphs.

Usage:
    uv run python scripts/run_benchmark.py                          # all solvers, all graphs
    uv run python scripts/run_benchmark.py --graph C125.9           # single graph
    uv run python scripts/run_benchmark.py --skip-dirac             # classical only
    uv run python scripts/run_benchmark.py --num-samples 50         # fewer Dirac samples
    uv run python scripts/run_benchmark.py --backend direct --ip 172.18.41.79  # direct hw
    uv run python scripts/run_benchmark.py --backend both           # cloud + direct
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load QCI credentials before importing Dirac solvers
_env_path = Path(__file__).resolve().parent.parent / "qci-eqc-models" / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)

from dirac_bench.benchmark import run_benchmark, print_results_table
from dirac_bench.utils import save_results_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dirac-3 benchmark suite for Motzkin-Straus max-clique problems"
    )

    # Graph selection
    parser.add_argument(
        "--dimacs-dir", type=str, default="problems/dimacs",
        help="Directory containing .clq files (default: problems/dimacs)",
    )
    parser.add_argument(
        "--graph", type=str, default=None,
        help="Run a single graph by stem name (e.g. C125.9)",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=None,
        help="Skip graphs with more than this many nodes",
    )

    # Solver selection
    parser.add_argument(
        "--skip-dirac", action="store_true",
        help="Skip Dirac solvers (run classical baselines only)",
    )
    parser.add_argument(
        "--backend", type=str, default="cloud", choices=["cloud", "direct", "both"],
        help="Dirac backend: cloud (default), direct, or both",
    )

    # Dirac options
    parser.add_argument(
        "--ip", type=str, default="172.18.41.79",
        help="Dirac-3 hardware IP address (default: 172.18.41.79)",
    )
    parser.add_argument(
        "--port", type=str, default="50051",
        help="Dirac-3 gRPC port (default: 50051)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="Number of Dirac-3 samples per graph (1-100, default: 100)",
    )
    parser.add_argument(
        "--relaxation-schedule", type=int, default=2,
        help="Dirac-3 relaxation schedule (1-4, default: 2)",
    )

    # Classical solver options
    parser.add_argument(
        "--slsqp-restarts", type=int, default=10,
        help="SLSQP random restarts (default: 10)",
    )
    parser.add_argument(
        "--lbfgs-restarts", type=int, default=10,
        help="L-BFGS-B random restarts (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Output options
    parser.add_argument(
        "--results-dir", type=str, default="results/raw",
        help="Directory for raw JSON output (default: results/raw)",
    )
    parser.add_argument(
        "--plots-dir", type=str, default="plots",
        help="Directory for histogram PNGs (default: plots)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip histogram plots")

    args = parser.parse_args()

    config = {
        "dimacs_dir": args.dimacs_dir,
        "graph": args.graph,
        "max_nodes": args.max_nodes,
        "skip_dirac": args.skip_dirac,
        "backend": args.backend,
        "ip": args.ip,
        "port": args.port,
        "num_samples": args.num_samples,
        "relaxation_schedule": args.relaxation_schedule,
        "slsqp_restarts": args.slsqp_restarts,
        "lbfgs_restarts": args.lbfgs_restarts,
        "seed": args.seed,
        "results_dir": args.results_dir,
        "plots_dir": args.plots_dir,
        "no_plot": args.no_plot,
    }

    rows = run_benchmark(config)

    if rows:
        print()
        print_results_table(rows)
        save_results_csv(rows, "results/benchmark_results.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
