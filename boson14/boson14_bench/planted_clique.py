"""Planted max-clique graph generator.

Generates G(n, p) random graphs with a planted k-clique of known size.
"""

import networkx as nx
import numpy as np


def generate_planted_clique(
    n: int,
    k: int,
    p: float = 0.5,
    seed: int = 42,
    vertices: list[int] | None = None,
) -> tuple[nx.Graph, list[int]]:
    """Generate G(n, p) with a k-clique planted on random or specified vertices.

    Returns (graph, planted_nodes) where planted_nodes are 1-based.
    """
    if vertices is not None:
        if len(vertices) != len(set(vertices)):
            raise ValueError("vertices must not contain duplicates")
        if not all(1 <= v <= n for v in vertices):
            raise ValueError(f"All vertices must be in [1, {n}]")
        if len(vertices) != k:
            raise ValueError(f"len(vertices)={len(vertices)} does not match k={k}")

    if k > n:
        raise ValueError(f"Clique size k={k} exceeds graph size n={n}")
    if k < 2:
        raise ValueError(f"Clique size k={k} must be at least 2")

    G = nx.erdos_renyi_graph(n, p, seed=seed)

    if vertices is not None:
        planted_0based = sorted(v - 1 for v in vertices)
    else:
        rng = np.random.default_rng(seed)
        planted_0based = sorted(int(v) for v in rng.choice(n, size=k, replace=False))

    for i in range(len(planted_0based)):
        for j in range(i + 1, len(planted_0based)):
            G.add_edge(planted_0based[i], planted_0based[j])

    G = nx.relabel_nodes(G, {i: i + 1 for i in range(n)})
    planted_nodes = [v + 1 for v in planted_0based]

    return G, planted_nodes
