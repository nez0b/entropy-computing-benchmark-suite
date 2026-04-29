"""Planted max-clique problem generator.

Generates G(n, p) random graphs with a planted k-clique of known size.
The planted clique becomes the maximum clique, providing benchmark instances
with known optimal solutions and tunable difficulty.

Key references:
- Bomze (1997): Motzkin-Straus landscape has local max at every maximal clique
- Chen-Moitra-Rohatgi (2023): gradient descent on Motzkin-Straus fails for k = o(n)
"""

import json
import math
from pathlib import Path

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

    Args:
        n: Number of vertices.
        k: Size of planted clique.
        p: Edge probability for the base Erdos-Renyi graph.
        seed: Random seed for reproducibility.
        vertices: Optional 1-based vertex indices for the planted clique.
            If None, k random vertices are chosen. If provided, k is
            validated against len(vertices).

    Returns:
        (graph, planted_nodes) where planted_nodes are 1-based.
    """
    if vertices is not None:
        if len(vertices) != len(set(vertices)):
            raise ValueError("vertices must not contain duplicates")
        if not all(1 <= v <= n for v in vertices):
            raise ValueError(f"All vertices must be in [1, {n}]")
        if len(vertices) != k:
            raise ValueError(
                f"len(vertices)={len(vertices)} does not match k={k}"
            )

    if k > n:
        raise ValueError(f"Clique size k={k} exceeds graph size n={n}")
    if k < 2:
        raise ValueError(f"Clique size k={k} must be at least 2")

    # 1. Generate G(n, p)
    G = nx.erdos_renyi_graph(n, p, seed=seed)

    # 2. Pick k vertices or use explicit vertices (converted to 0-based)
    if vertices is not None:
        planted_0based = sorted(v - 1 for v in vertices)
    else:
        rng = np.random.default_rng(seed)
        planted_0based = sorted(int(v) for v in rng.choice(n, size=k, replace=False))

    # 3. Add all missing edges between planted vertices
    for i in range(len(planted_0based)):
        for j in range(i + 1, len(planted_0based)):
            G.add_edge(planted_0based[i], planted_0based[j])

    # 4. Relabel to 1-based for DIMACS compatibility
    G = nx.relabel_nodes(G, {i: i + 1 for i in range(n)})
    planted_nodes = [v + 1 for v in planted_0based]

    return G, planted_nodes


def planted_clique_info(n: int, k: int, p: float = 0.5) -> dict:
    """Return metadata about a planted clique instance.

    Includes theoretical natural omega, objective values, gap,
    and expected difficulty classification.
    """
    # Natural clique number of G(n, p): omega ~ 2 * log_b(n) where b = 1/p
    natural_omega = round(2 * math.log(n) / math.log(1 / p))

    planted_objective = 0.5 * (1 - 1 / k)
    natural_objective = 0.5 * (1 - 1 / natural_omega) if natural_omega > 1 else 0.0
    gap = planted_objective - natural_objective

    # Classify difficulty based on k relative to natural omega.
    # Hard: k just above omega (solver trapped in natural cliques).
    # Easy: k well above omega (planted clique dominates landscape).
    sqrt_n = math.sqrt(n)
    if k <= natural_omega:
        difficulty = "invisible"
    elif k <= 1.3 * natural_omega:
        difficulty = "hard"
    elif k <= 1.5 * natural_omega:
        difficulty = "moderate"
    else:
        difficulty = "easy"

    return {
        "n": n,
        "k": k,
        "p": p,
        "natural_omega": natural_omega,
        "planted_objective": round(planted_objective, 6),
        "natural_objective": round(natural_objective, 6),
        "gap": round(gap, 6),
        "sqrt_n": round(sqrt_n, 2),
        "difficulty": difficulty,
    }


def write_planted_dimacs(
    graph: nx.Graph,
    planted_nodes: list[int],
    file_path: str | Path,
    n: int,
    k: int,
    p: float,
    seed: int,
) -> None:
    """Write a planted clique graph to DIMACS format with metadata comments."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    with open(path, "w") as f:
        f.write(f"c Planted clique instance: G({n}, {p}) + k={k} clique\n")
        f.write(f"c Planted nodes: {' '.join(str(v) for v in planted_nodes)}\n")
        f.write(f"c Seed: {seed}\n")
        f.write(f"p edge {num_nodes} {num_edges}\n")
        for u, v in sorted(graph.edges()):
            f.write(f"e {u} {v}\n")


def write_planted_metadata(
    file_path: str | Path,
    planted_nodes: list[int],
    info: dict,
    seed: int,
) -> None:
    """Write metadata JSON for a planted clique instance."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "planted_nodes": planted_nodes,
        "seed": seed,
        **info,
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def instance_name(n: int, k: int, p: float) -> str:
    """Return canonical instance name: planted_{n}_k{k}_p{p_str}."""
    p_str = str(p).replace(".", "")
    return f"planted_{n}_k{k}_p{p_str}"
