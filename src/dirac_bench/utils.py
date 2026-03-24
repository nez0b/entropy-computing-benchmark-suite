"""Utility functions for JSON serialization and CSV output."""

import csv
import json
import time
from pathlib import Path

import numpy as np


def _numpy_converter(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def save_raw_response(response: dict, graph_name: str, solver_name: str, save_dir: str = "results/raw", hash_id: str | None = None) -> Path:
    """Save a raw solver response as JSON.

    Args:
        response: The raw response dict from a solver.
        graph_name: Graph name for the filename.
        solver_name: Solver name (e.g., "dirac", "dirac_direct").
        save_dir: Output directory.
        hash_id: Optional short hash for unique filename identification.

    Returns:
        Path to the saved file.
    """
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    if hash_id:
        filename = out / f"{solver_name}_{graph_name}_{hash_id}.json"
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = out / f"{solver_name}_{graph_name}_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(response, f, indent=2, default=_numpy_converter)

    print(f"  Saved raw JSON -> {filename}")
    return filename


def save_results_csv(rows: list[dict], path: str | Path) -> None:
    """Save benchmark summary rows to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV -> {path}")
