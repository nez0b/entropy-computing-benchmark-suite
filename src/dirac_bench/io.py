"""DIMACS graph file reader/writer.

Handles both 'p edge N M' and 'p col N M' format variants.
"""

from pathlib import Path

import networkx as nx


def read_dimacs_graph(file_path: str | Path) -> nx.Graph:
    """Read a graph from a DIMACS format file.

    Args:
        file_path: Path to DIMACS .clq file.

    Returns:
        NetworkX graph with 1-based node indexing.
    """
    graph = nx.Graph()

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue

            if line.startswith("p "):
                parts = line.split()
                if len(parts) >= 4 and parts[1] in ("edge", "col"):
                    num_nodes = int(parts[2])
                    graph.add_nodes_from(range(1, num_nodes + 1))
            elif line.startswith("e "):
                parts = line.split()
                if len(parts) >= 3:
                    u, v = int(parts[1]), int(parts[2])
                    graph.add_edge(u, v)

    return graph


def write_dimacs_graph(graph: nx.Graph, file_path: str | Path, comment: str = "") -> None:
    """Write a graph to DIMACS format.

    Args:
        graph: NetworkX graph.
        file_path: Output path for .clq file.
        comment: Optional comment line for the file header.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    with open(path, "w") as f:
        if comment:
            f.write(f"c {comment}\n")
        f.write(f"p edge {n} {m}\n")
        for u, v in sorted(graph.edges()):
            f.write(f"e {u} {v}\n")


def get_graph_info(graph: nx.Graph) -> dict:
    """Return summary information about a graph."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    max_edges = n * (n - 1) / 2 if n > 1 else 1
    return {
        "nodes": n,
        "edges": m,
        "density": m / max_edges if max_edges > 0 else 0.0,
    }
