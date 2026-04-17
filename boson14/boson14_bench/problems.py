"""Adjacency matrix builder for the Motzkin-Strauss QP."""

import networkx as nx
import numpy as np


def motzkin_straus_adjacency(graph: nx.Graph) -> np.ndarray:
    """Adjacency matrix A as float64 ndarray, rows/cols in sorted node order."""
    nodelist = sorted(graph.nodes())
    return nx.adjacency_matrix(graph, nodelist=nodelist).toarray().astype(np.float64)
