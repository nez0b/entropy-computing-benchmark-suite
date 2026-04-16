"""DIMACS graph file reader.

Handles both 'p edge N M' and 'p col N M' format variants.
"""

from pathlib import Path

import networkx as nx


def read_dimacs_graph(file_path: str | Path) -> nx.Graph:
    """Read a graph from a DIMACS .clq file. Returns a NetworkX graph with 1-based nodes."""
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
