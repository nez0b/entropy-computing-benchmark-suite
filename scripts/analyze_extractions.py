#!/usr/bin/env python3
"""Offline extraction analysis on saved raw x-vectors.

Loads .npz files from collect_raw_vectors.py, applies all extraction methods
from clique_extraction.py, and generates comparison tables/reports.

Usage:
    uv run python scripts/analyze_extractions.py
    uv run python scripts/analyze_extractions.py --input-dir results/raw_vectors
    uv run python scripts/analyze_extractions.py --graphs C125.9
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from dirac_bench.problems import (
    motzkin_straus_adjacency,
    is_clique,
)
from dirac_bench.benchmark import KNOWN_OMEGA
from dirac_bench.io import read_dimacs_graph
from dirac_bench.clique_extraction import run_all_extractions
from dirac_bench.utils import _numpy_converter

# Method display names (short) in column order
METHOD_COLUMNS = [
    ("greedy_desc", "g_desc"),
    ("greedy_asc", "g_asc"),
    ("greedy_random", "g_rand"),
    ("threshold_sweep", "thresh"),
    ("top_k", "top_k"),
    ("randomized_rounding", "rround"),
    ("local_search_1swap", "ls1"),
    ("local_search_2swap", "ls2"),
    ("cluster", "cluster"),
]


def parse_npz_filename(path: Path) -> dict | None:
    """Extract (graph, backend, formulation, schedule) from filename.

    Expected format: {graph}_{backend}_{formulation}_s{schedule}.npz
    """
    stem = path.stem  # e.g. "C125.9_cloud_standard_s2"
    parts = stem.rsplit("_", 3)
    if len(parts) < 4:
        return None
    # Handle graph names with underscores by taking the last 3 parts
    schedule_str = parts[-1]  # "s2"
    formulation = parts[-2]   # "standard" or "bomze"
    backend = parts[-3]       # "cloud" or "direct"
    graph = "_".join(parts[:-3])  # everything before

    if not schedule_str.startswith("s"):
        return None
    try:
        schedule = int(schedule_str[1:])
    except ValueError:
        return None

    return {
        "graph": graph,
        "backend": backend,
        "formulation": formulation,
        "schedule": schedule,
    }


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load x_vectors, objectives, and metadata from .npz file."""
    data = np.load(path, allow_pickle=True)
    x_vectors = data["x_vectors"]
    objectives = data["objectives"]
    metadata = json.loads(str(data["metadata"]))
    return x_vectors, objectives, metadata


def analyze_one(
    npz_path: Path,
    A: np.ndarray,
    known_omega: int | None,
    seed: int = 42,
) -> dict:
    """Run all extraction methods on one .npz file.

    Returns a result dict with method sizes, best clique, etc.
    """
    config = parse_npz_filename(npz_path)
    x_vectors, objectives, metadata = load_npz(npz_path)

    results = run_all_extractions(x_vectors, A, known_omega=known_omega, seed=seed)

    # Build row
    row = {
        "graph": config["graph"],
        "backend": config["backend"],
        "formulation": config["formulation"],
        "schedule": config["schedule"],
        "num_samples": len(x_vectors),
        "best_obj": round(float(objectives.max()), 6),
    }

    # Per-method sizes
    best_method = None
    best_size = 0
    for method_key, _ in METHOD_COLUMNS:
        r = results.get(method_key, {"size": 0, "valid": False})
        row[method_key] = r["size"]
        row[f"{method_key}_valid"] = r["valid"]
        if r["size"] > best_size and r["valid"]:
            best_size = r["size"]
            best_method = method_key

    row["best_method"] = best_method
    row["best_size"] = best_size
    row["known_omega"] = known_omega
    row["match"] = best_size == known_omega if known_omega else None

    # Full results for detailed JSON
    row["_full_results"] = results

    return row


def save_detailed_json(row: dict, output_dir: Path) -> Path:
    """Save detailed per-config analysis as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_name = f"{row['graph']}_{row['backend']}_{row['formulation']}_s{row['schedule']}"
    path = output_dir / f"{config_name}_analysis.json"

    # Build detailed output
    full_results = row.pop("_full_results", {})
    detail = {**row}
    detail["methods"] = {}
    for method_key, _ in METHOD_COLUMNS:
        r = full_results.get(method_key, {})
        detail["methods"][method_key] = {
            "clique": r.get("clique", []),
            "size": r.get("size", 0),
            "valid": r.get("valid", False),
        }

    with open(path, "w") as f:
        json.dump(detail, f, indent=2, default=_numpy_converter)

    return path


def save_summary_csv(rows: list[dict], output_dir: Path) -> Path:
    """Save summary CSV with one row per config."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.csv"

    fieldnames = [
        "graph", "backend", "formulation", "schedule", "num_samples", "best_obj",
    ] + [mk for mk, _ in METHOD_COLUMNS] + [
        "best_method", "best_size", "known_omega", "match",
    ]

    # Remove internal keys
    clean_rows = []
    for row in rows:
        clean = {k: v for k, v in row.items() if not k.startswith("_") and k in fieldnames}
        clean_rows.append(clean)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean_rows)

    return path


def print_comparison_table(rows: list[dict]) -> None:
    """Print console comparison table."""
    if not rows:
        print("No results to display.")
        return

    # Column widths
    method_cols = [(mk, short) for mk, short in METHOD_COLUMNS]
    col_w = 7

    # Header
    header = (
        f"{'Graph':<10} {'Backend':<8} {'Form':<8} {'Sched':>5}"
    )
    for _, short in method_cols:
        header += f"  {short:>{col_w}}"
    header += f"  {'BEST':>{col_w}}  {'Known':>{col_w}}"
    print()
    print("=" * len(header))
    print("Extraction Method Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for row in sorted(rows, key=lambda r: (r["graph"], r["backend"], r["formulation"], r["schedule"])):
        line = (
            f"{row['graph']:<10} {row['backend']:<8} {row['formulation']:<8} "
            f"{row['schedule']:>5}"
        )
        for mk, _ in method_cols:
            val = row.get(mk, 0)
            valid = row.get(f"{mk}_valid", False)
            marker = "" if valid else "!"
            line += f"  {str(val) + marker:>{col_w}}"

        best = row.get("best_size", 0)
        known = row.get("known_omega", "?")
        match_str = ""
        if row.get("match") is True:
            match_str = " *"
        elif row.get("match") is False:
            match_str = ""
        line += f"  {best:>{col_w}}{match_str}  {str(known):>{col_w}}"
        print(line)

    print("-" * len(header))
    print("  ! = invalid clique   * = matches known omega")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline extraction analysis on saved raw x-vectors"
    )
    parser.add_argument(
        "--input-dir", type=str, default="results/raw_vectors",
        help="Directory containing .npz files (default: results/raw_vectors)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/extraction_analysis",
        help="Output directory for analysis results (default: results/extraction_analysis)",
    )
    parser.add_argument(
        "--dimacs-dir", type=str, default="problems/dimacs",
        help="DIMACS graph directory (default: problems/dimacs)",
    )
    parser.add_argument(
        "--graphs", nargs="+", default=None,
        help="Filter to specific graphs (default: all found in input-dir)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    dimacs_dir = Path(args.dimacs_dir)

    print("=" * 70)
    print("Extraction Analysis")
    print("=" * 70)

    # Find .npz files
    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return 1

    # Filter by graph if specified
    if args.graphs:
        npz_files = [
            f for f in npz_files
            if parse_npz_filename(f) and parse_npz_filename(f)["graph"] in args.graphs
        ]

    print(f"Found {len(npz_files)} .npz files in {input_dir}")

    # Load adjacency matrices (cache to avoid re-reading)
    graph_cache: dict[str, np.ndarray] = {}

    rows = []
    for npz_path in npz_files:
        config = parse_npz_filename(npz_path)
        if config is None:
            print(f"  SKIP {npz_path.name} (unrecognized filename format)")
            continue

        graph_name = config["graph"]

        # Load adjacency matrix if not cached
        if graph_name not in graph_cache:
            clq_path = dimacs_dir / f"{graph_name}.clq"
            if not clq_path.exists():
                print(f"  SKIP {npz_path.name}: DIMACS file {clq_path} not found")
                continue
            G = read_dimacs_graph(str(clq_path))
            graph_cache[graph_name] = motzkin_straus_adjacency(G)

        A = graph_cache[graph_name]
        known_omega = KNOWN_OMEGA.get(graph_name)

        label = f"{graph_name}/{config['backend']}/{config['formulation']}/s{config['schedule']}"
        print(f"  Analyzing {label} ...")

        row = analyze_one(npz_path, A, known_omega, seed=args.seed)

        # Save detailed JSON
        json_path = save_detailed_json(row, output_dir)
        print(f"    -> {json_path}")

        rows.append(row)

    if not rows:
        print("No configs analyzed.")
        return 1

    # Save summary CSV
    csv_path = save_summary_csv(rows, output_dir)
    print(f"\nSummary CSV -> {csv_path}")

    # Print console table
    print_comparison_table(rows)

    # Print highlights
    if rows:
        best_row = max(rows, key=lambda r: r["best_size"])
        print(f"Best overall: {best_row['graph']}/{best_row['backend']}/{best_row['formulation']}/s{best_row['schedule']}")
        print(f"  Method: {best_row['best_method']}  Size: {best_row['best_size']}  Known: {best_row.get('known_omega', '?')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
