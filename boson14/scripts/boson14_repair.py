"""Repair analysis for boson14 samples.

For every boson14 sample y (sigma_array row) we compute three views:

  raw    — objective on the raw continuous vector: g = 0.5 * y^T A y
  eq-w   — if threshold-support S is a clique, project to the
           MS-optimal equal-weight embedding on S:  g = (R^2/2)(1 - 1/|S|)
           (undefined otherwise)
  1-opt  — run a greedy 1-opt local search seeded at S:
             (a) iteratively remove the in-S vertex with the fewest
                 intra-S neighbours until S is a clique,
             (b) greedily add any v outside S adjacent to all of S.
           Report g from the equal-weight embedding on the resulting
           maximal clique.

Outputs:
  scripts/cache/boson14_repair.csv      — one row per sample
  scripts/cache/boson14_repair.json     — per-run aggregate stats
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from _common import BOSON14_DIR, CACHE_DIR, iter_runs, load_meta

sys.path.insert(0, str(BOSON14_DIR))
from boson14_bench.core import scaled_objective  # noqa: E402


def threshold_support(y: np.ndarray, R: int, k: int, factor: float = 2.0) -> list[int]:
    """Return 0-based indices where y > R/(factor*k)."""
    thr = R / (factor * k)
    return [int(i) for i in np.where(y > thr)[0]]


def is_clique_subset(G: nx.Graph, support: list[int]) -> bool:
    """True if every pair in support is an edge of G (1-based nodes)."""
    # G uses 1-based labels; support is 0-based.
    nodes = [i + 1 for i in support]
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            if not G.has_edge(nodes[a], nodes[b]):
                return False
    return True


def repair_1opt(G: nx.Graph, seed_support: list[int]) -> list[int]:
    """Greedy 1-opt from a seed (0-based indices), returning a maximal clique
    (still 0-based)."""
    # Work in 1-based labels, convert back at the end.
    S = set(i + 1 for i in seed_support)

    # (a) Remove worst vertex until S is a clique.
    while len(S) > 1:
        intra = {v: sum(1 for u in S if u != v and G.has_edge(u, v)) for v in S}
        worst = min(intra, key=intra.get)
        if intra[worst] == len(S) - 1:
            break  # everyone connected to all others — already a clique
        S.remove(worst)

    # (b) Greedy extension with vertices adjacent to all of S.
    # Precompute candidate set as intersection of neighborhoods.
    if S:
        common_nbrs = set(G.nodes) - S
        for v in S:
            common_nbrs &= set(G.neighbors(v))
        while common_nbrs:
            # Add the candidate with the most connections to other candidates
            # (tie-breaker for faster extension). Plain pick also works.
            v = next(iter(common_nbrs))
            S.add(v)
            common_nbrs &= set(G.neighbors(v))

    return sorted(i - 1 for i in S)


def equal_weight_g(size: int, R: int) -> float:
    """MS-optimal objective for an equal-weight embedding on a size-s clique."""
    if size < 1:
        return float("nan")
    if size == 1:
        return 0.0
    return (R ** 2 / 2.0) * (1.0 - 1.0 / size)


def graph_from_A(A: np.ndarray) -> nx.Graph:
    """Build a 1-based networkx graph from the integer adjacency A."""
    n = A.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    # A is the permuted adjacency for the variant.
    iu, ju = np.triu_indices(n, k=1)
    mask = A[iu, ju] > 0
    for i, j in zip(iu[mask], ju[mask]):
        G.add_edge(int(i) + 1, int(j) + 1)
    return G


def process_run(run, rows: list[dict], summaries: list[dict]) -> None:
    meta = load_meta(run)
    k = int(meta["k"])
    n = int(meta["n"])
    R = int(meta["R"])
    g_star = float(meta["g_star"])

    data = np.load(run.npz_path, allow_pickle=True)
    sigma = data["sigma_array"]                         # (100, n)
    A = data["quadratic_term"].astype(np.float64)       # +A (verified)
    G = graph_from_A(A)

    # Per-run counters
    n_supp_valid = 0
    n_supp_eq_k = 0
    size_counts_raw: Counter = Counter()
    size_counts_repaired: Counter = Counter()

    for i in range(sigma.shape[0]):
        y = sigma[i]
        g_raw = float(scaled_objective(y, A))

        supp = threshold_support(y, R, k)
        supp_size = len(supp)
        supp_valid = is_clique_subset(G, supp)
        g_eq = equal_weight_g(supp_size, R) if supp_valid else float("nan")

        repaired = repair_1opt(G, supp)
        rep_size = len(repaired)
        g_rep = equal_weight_g(rep_size, R)

        if supp_valid:
            n_supp_valid += 1
            if supp_size == k:
                n_supp_eq_k += 1
        size_counts_raw[supp_size] += 1
        size_counts_repaired[rep_size] += 1

        rows.append({
            "instance": run.instance,
            "variant": run.variant,
            "amplitude": run.amplitude,
            "timestamp": run.timestamp,
            "sample_idx": i,
            "n": n, "k": k, "R": R, "g_star": g_star,
            "g_raw": g_raw,
            "supp_size": supp_size,
            "supp_is_clique": bool(supp_valid),
            "g_equal_weight": g_eq,
            "repaired_size": rep_size,
            "g_repaired": g_rep,
        })

    summaries.append({
        "instance": run.instance,
        "variant": run.variant,
        "amplitude": run.amplitude,
        "k": k,
        "g_star": g_star,
        "num_samples": int(sigma.shape[0]),
        "frac_supp_valid_clique": n_supp_valid / sigma.shape[0],
        "frac_supp_size_equal_k": n_supp_eq_k / sigma.shape[0],
        "supp_size_distribution": dict(sorted(size_counts_raw.items())),
        "repaired_size_distribution": dict(sorted(size_counts_repaired.items())),
    })


def main() -> int:
    runs = iter_runs()
    if not runs:
        print("No runs found.")
        return 1

    rows: list[dict] = []
    summaries: list[dict] = []
    for idx, run in enumerate(runs, start=1):
        process_run(run, rows, summaries)
        last = summaries[-1]
        print(
            f"  [{idx:2d}/{len(runs)}] {run.instance}/{run.variant:>6s} "
            f"a{run.amplitude}  valid-supp: {last['frac_supp_valid_clique']:.0%}  "
            f"supp==k: {last['frac_supp_size_equal_k']:.0%}  "
            f"repaired sizes: {last['repaired_size_distribution']}"
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CACHE_DIR / "boson14_repair.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nWrote {len(rows)} sample rows -> {csv_path}")

    json_path = CACHE_DIR / "boson14_repair.json"
    with json_path.open("w") as f:
        json.dump(summaries, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(summaries)} run summaries -> {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
