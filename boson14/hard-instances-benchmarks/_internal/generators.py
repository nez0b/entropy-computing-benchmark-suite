"""Hard max-clique instance generators (Strategies A–E).

All generators return (A, metadata) where A is an n×n float64 adjacency matrix
and metadata is a dict with at least 'name', 'n', 'strategy', and 'planted_omega'
(None if unknown/unplanted).
"""

import math

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Strategy A: Near-threshold planted clique
# ---------------------------------------------------------------------------

def strategy_a_near_threshold(
    n: int = 60,
    p: float = 0.5,
    seed: int = 42,
    k_multiplier: float = 2.5,
) -> tuple[np.ndarray, dict]:
    """Planted clique barely above natural clique number.

    k ≈ k_multiplier * log2(n). At this size the planted clique is near the
    detection threshold and the Motzkin-Strauss landscape has many competing
    local optima of similar depth.
    """
    k = max(3, int(math.ceil(k_multiplier * math.log2(n))))
    if k > n:
        k = n

    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    planted = sorted(int(v) for v in rng.choice(n, size=k, replace=False))

    for i in range(len(planted)):
        for j in range(i + 1, len(planted)):
            G.add_edge(planted[i], planted[j])

    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).toarray().astype(np.float64)
    natural_omega = round(2 * math.log(n) / math.log(1 / p)) if p < 1 else n

    return A, {
        "name": f"near_threshold_n{n}_k{k}_p{p}_s{seed}",
        "strategy": "A_near_threshold",
        "n": n, "k": k, "p": p, "seed": seed,
        "planted_nodes": planted,
        "planted_omega": k,
        "natural_omega_estimate": natural_omega,
        "k_over_sqrt_n": round(k / math.sqrt(n), 2),
    }


# ---------------------------------------------------------------------------
# Strategy B: High-density random (C-family style)
# ---------------------------------------------------------------------------

def strategy_b_dense_random(
    n: int = 80,
    p: float = 0.9,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Dense random graph G(n, 0.9) — no planted clique.

    At density 0.9 there are exponentially many near-optimal cliques, creating
    a rugged landscape with many competing local optima.
    """
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).toarray().astype(np.float64)

    return A, {
        "name": f"dense_random_n{n}_p{p}_s{seed}",
        "strategy": "B_dense_random",
        "n": n, "p": p, "seed": seed,
        "planted_omega": None,
    }


# ---------------------------------------------------------------------------
# Strategy C: Degenerate planted cliques (multiple competing basins)
# ---------------------------------------------------------------------------

def strategy_c_degenerate(
    n: int = 60,
    k: int = 10,
    num_cliques: int = 3,
    p: float = 0.5,
    seed: int = 42,
    sever_cross: bool = True,
) -> tuple[np.ndarray, dict]:
    """Multiple planted cliques of the same size.

    Creates num_cliques disjoint k-cliques, producing competing global optima.
    Cross-edges between planted groups are severed to prevent accidental
    larger cliques.
    """
    if num_cliques * k > n:
        raise ValueError(f"num_cliques*k={num_cliques * k} > n={n}")

    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n, p, seed=seed)

    pool = list(rng.permutation(n))
    planted_sets = []
    for i in range(num_cliques):
        group = sorted(int(v) for v in pool[i * k : (i + 1) * k])
        planted_sets.append(group)
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                G.add_edge(group[a], group[b])

    if sever_cross:
        all_planted = set()
        group_of = {}
        for g_idx, group in enumerate(planted_sets):
            for v in group:
                all_planted.add(v)
                group_of[v] = g_idx
        edges_to_remove = [
            (u, v) for u, v in G.edges()
            if u in all_planted and v in all_planted and group_of[u] != group_of[v]
        ]
        G.remove_edges_from(edges_to_remove)

    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).toarray().astype(np.float64)

    return A, {
        "name": f"degenerate_n{n}_k{k}_nc{num_cliques}_p{p}_s{seed}",
        "strategy": "C_degenerate",
        "n": n, "k": k, "num_cliques": num_cliques, "p": p, "seed": seed,
        "planted_sets": planted_sets,
        "planted_omega": k,
    }


# ---------------------------------------------------------------------------
# Strategy D: Degree-camouflaged planted clique (brock-style)
# ---------------------------------------------------------------------------

def strategy_d_camouflage(
    n: int = 60,
    k: int = 12,
    p: float = 0.5,
    seed: int = 42,
    removal_frac: float = 0.4,
) -> tuple[np.ndarray, dict]:
    """Planted clique with suppressed degrees on clique vertices.

    After planting a k-clique, remove a fraction of non-clique edges incident
    to clique vertices. This hides the clique from degree-based heuristics
    while preserving the clique structure.
    """
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    planted = sorted(int(v) for v in rng.choice(n, size=k, replace=False))
    planted_set = set(planted)

    for i in range(len(planted)):
        for j in range(i + 1, len(planted)):
            G.add_edge(planted[i], planted[j])

    # Remove non-clique edges incident to clique vertices
    removable = [
        (u, v) for u, v in G.edges()
        if (u in planted_set) != (v in planted_set)  # exactly one endpoint in clique
    ]
    rng2 = np.random.default_rng(seed + 500)
    n_remove = int(len(removable) * removal_frac)
    if n_remove > 0:
        to_remove = rng2.choice(len(removable), size=n_remove, replace=False)
        G.remove_edges_from([removable[i] for i in to_remove])

    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).toarray().astype(np.float64)

    # Compute degree stats for planted vs non-planted
    degrees = dict(G.degree())
    planted_avg_deg = np.mean([degrees[v] for v in planted])
    non_planted_avg_deg = np.mean([degrees[v] for v in range(n) if v not in planted_set])

    return A, {
        "name": f"camouflage_n{n}_k{k}_rf{removal_frac}_p{p}_s{seed}",
        "strategy": "D_camouflage",
        "n": n, "k": k, "p": p, "seed": seed,
        "removal_frac": removal_frac,
        "planted_nodes": planted,
        "planted_omega": k,
        "planted_avg_degree": round(planted_avg_deg, 1),
        "non_planted_avg_degree": round(non_planted_avg_deg, 1),
    }


# ---------------------------------------------------------------------------
# Strategy E: Overlapping near-cliques (spurious solution traps)
# ---------------------------------------------------------------------------

def strategy_e_overlap(
    n: int = 40,
    k: int = 10,
    overlap: int = 8,
    p: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Two near-cliques sharing 'overlap' vertices — a spurious-solution trap.

    Creates a k-clique S and a (k-1)-clique T where |S ∩ T| = overlap.
    Without Bomze regularization, the Motzkin-Strauss formulation produces
    spurious solutions supported on S ∪ T.
    """
    if overlap >= k:
        raise ValueError(f"overlap={overlap} must be < k={k}")
    second_k = k - 1
    total_clique_vertices = k + second_k - overlap
    if total_clique_vertices > n:
        raise ValueError(f"Need at least {total_clique_vertices} vertices, have n={n}")

    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n, p, seed=seed)

    pool = list(rng.permutation(n))
    clique_s = sorted(int(v) for v in pool[:k])
    # T shares 'overlap' vertices with S and has (second_k - overlap) new ones
    shared = clique_s[:overlap]
    new_t = sorted(int(v) for v in pool[k : k + second_k - overlap])
    clique_t = sorted(shared + new_t)

    for clique in [clique_s, clique_t]:
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                G.add_edge(clique[i], clique[j])

    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).toarray().astype(np.float64)

    return A, {
        "name": f"overlap_n{n}_k{k}_ov{overlap}_p{p}_s{seed}",
        "strategy": "E_overlap",
        "n": n, "k": k, "second_k": second_k, "overlap": overlap,
        "p": p, "seed": seed,
        "clique_s": clique_s,
        "clique_t": clique_t,
        "planted_omega": k,
        "union_size": len(set(clique_s) | set(clique_t)),
    }


# ---------------------------------------------------------------------------
# Generate a suite of hard instances
# ---------------------------------------------------------------------------

def generate_suite(seed: int = 42) -> list[tuple[np.ndarray, dict]]:
    """Generate one instance per strategy with default parameters."""
    return [
        strategy_a_near_threshold(n=80, seed=seed),
        strategy_b_dense_random(n=80, p=0.9, seed=seed),
        strategy_c_degenerate(n=60, k=10, num_cliques=3, seed=seed),
        strategy_d_camouflage(n=60, k=12, removal_frac=0.4, seed=seed),
        strategy_e_overlap(n=40, k=10, overlap=8, seed=seed),
    ]


def generate_sweep(n: int = 100, seeds: list[int] | None = None) -> list[tuple[np.ndarray, dict]]:
    """Generate multiple instances per strategy at fixed n, varying parameters.

    Returns a list of (A, metadata) tuples covering a range of difficulty
    settings for each of the 5 strategies.
    """
    if seeds is None:
        seeds = [42, 123, 7]
    instances = []

    # Strategy A: vary k_multiplier (how close to detection threshold)
    for mult in [2.0, 2.5, 3.0]:
        for s in seeds:
            instances.append(strategy_a_near_threshold(n=n, k_multiplier=mult, seed=s))

    # Strategy B: vary density
    for p in [0.8, 0.9]:
        for s in seeds:
            instances.append(strategy_b_dense_random(n=n, p=p, seed=s))

    # Strategy C: vary num_cliques and k
    for nc, k in [(2, 12), (3, 10), (5, 8)]:
        if nc * k <= n:
            for s in seeds:
                instances.append(strategy_c_degenerate(n=n, k=k, num_cliques=nc, seed=s))

    # Strategy D: vary removal_frac and k
    for rf in [0.3, 0.5]:
        for k in [12, 16]:
            for s in seeds:
                instances.append(strategy_d_camouflage(n=n, k=k, removal_frac=rf, seed=s))

    # Strategy E: vary k and overlap
    for k, ov in [(10, 8), (12, 10), (15, 12)]:
        for s in seeds:
            instances.append(strategy_e_overlap(n=n, k=k, overlap=ov, seed=s))

    return instances
