"""Reusable inline test graph builders for benchmarks and tests.

Each builder returns (adjacency_matrix, known_omega_or_None, name).
"""

import numpy as np
import networkx as nx


def make_overlapping_k4() -> tuple[np.ndarray, int, str]:
    """Two K4s sharing 2 vertices: {0,1,2,3} and {2,3,4,5}. omega=4."""
    G = nx.Graph()
    G.add_nodes_from(range(6))
    for i in range(4):
        for j in range(i + 1, 4):
            G.add_edge(i, j)
    for i in range(2, 6):
        for j in range(i + 1, 6):
            G.add_edge(i, j)
    A = nx.adjacency_matrix(G).toarray().astype(np.float64)
    return A, 4, "overlap_2xK4"


def make_erdos_renyi(
    n: int, p: float, seed: int = 42
) -> tuple[np.ndarray, int | None, str]:
    """Random Erdos-Renyi graph. omega unknown."""
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    A = nx.adjacency_matrix(G).toarray().astype(np.float64)
    name = f"ER({n},{p})"
    return A, None, name
