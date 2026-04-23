"""Exact max-clique via networkx Bron-Kerbosch.

Extracted from hard-instances/evaluate.py. Uses a generator to avoid
materializing all maximal cliques in memory (dense graphs can have billions).
Returns a sample maximum clique (1-based nodes) alongside the stats.
"""

import time

import networkx as nx
import numpy as np


def solve_bruteforce(A: np.ndarray, timeout: float = 120.0) -> dict:
    """Exact clique number + a representative max clique.

    Returns dict with:
        omega: clique number
        max_clique: one max clique as a sorted list of 1-based node IDs
        num_max_cliques, num_maximal_cliques, degeneracy, clique_core_gap
        solve_time, timed_out
    """
    n = A.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0:
                G.add_edge(i, j)

    degeneracy = max(nx.core_number(G).values()) if G.number_of_edges() else 0

    t0 = time.time()
    omega = 0
    num_max_cliques = 0
    num_maximal = 0
    sample_max_clique: list[int] = []
    timed_out = False

    for clique in nx.find_cliques(G):
        size = len(clique)
        num_maximal += 1
        if size > omega:
            omega = size
            num_max_cliques = 1
            sample_max_clique = sorted(int(v) for v in clique)
        elif size == omega:
            num_max_cliques += 1

        if time.time() - t0 > timeout:
            timed_out = True
            break

    # Convert 0-based (from numpy adjacency indexing) to 1-based node IDs
    # for consistency with other strategies (which produce 1-based planted_nodes).
    sample_max_clique_1based = sorted(v + 1 for v in sample_max_clique)

    return {
        "omega": omega,
        "max_clique": sample_max_clique_1based,
        "num_max_cliques": num_max_cliques,
        "num_maximal_cliques": num_maximal,
        "degeneracy": degeneracy,
        "clique_core_gap": degeneracy + 1 - omega,
        "solve_time": time.time() - t0,
        "timed_out": timed_out,
    }
