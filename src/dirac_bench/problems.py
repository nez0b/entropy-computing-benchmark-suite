"""Motzkin-Straus quadratic program formulation for max-clique.

The Motzkin-Straus theorem states that for a graph G with clique number omega:

    max  0.5 * x^T A x   s.t.  sum(x) = 1, x >= 0

achieves optimum  f*(omega) = 0.5 * (1 - 1/omega).

Inverting:  omega = round(1 / (1 - 2*f*))
"""

import numpy as np
import networkx as nx


def motzkin_straus_adjacency(graph: nx.Graph) -> np.ndarray:
    """Build the adjacency matrix for the Motzkin-Straus QP.

    Args:
        graph: NetworkX graph.

    Returns:
        Adjacency matrix A as float64 ndarray.
    """
    return nx.adjacency_matrix(graph).toarray().astype(np.float64)


def objective(x: np.ndarray, A: np.ndarray) -> float:
    """Compute the Motzkin-Straus objective: 0.5 * x^T A x."""
    return float(0.5 * (x @ A @ x))


def gradient(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Gradient of the objective: A @ x (for maximization)."""
    return A @ x


def objective_to_omega(f_star: float) -> int:
    """Convert optimal objective value to clique number.

    omega = round(1 / (1 - 2*f*))
    """
    denom = 1.0 - 2.0 * f_star
    if abs(denom) < 1e-12:
        return 1
    return round(1.0 / denom)


def omega_to_theoretical_objective(omega: int) -> float:
    """Theoretical Motzkin-Straus objective for a given clique number.

    f*(omega) = 0.5 * (1 - 1/omega)
    """
    if omega <= 1:
        return 0.0
    return 0.5 * (1.0 - 1.0 / omega)


# ---------------------------------------------------------------------------
# Bomze regularization  (Bomze 1997)
#
# Replaces A with A_bar = A + 0.5*I so that every local/global maximum of
# 0.5 * x^T A_bar x on the simplex corresponds to a maximal/maximum clique.
#
# Optimal value:  f*(omega) = 0.5 * (1 - 1/(2*omega))
# Inversion:      omega = round(1 / (2*(1 - 2*f*)))
# ---------------------------------------------------------------------------


def bomze_regularize(A: np.ndarray) -> np.ndarray:
    """Bomze-regularize an adjacency matrix: A_bar = A + 0.5 * I."""
    return A + 0.5 * np.eye(A.shape[0])


def bomze_objective_to_omega(f_star: float) -> int:
    """Convert Bomze objective value to clique number.

    omega = round(1 / (2 * (1 - 2*f*)))
    """
    denom = 2.0 * (1.0 - 2.0 * f_star)
    if abs(denom) < 1e-12:
        return 1
    return round(1.0 / denom)


def bomze_theoretical_objective(omega: int) -> float:
    """Theoretical Bomze objective for a given clique number.

    f*(omega) = 0.5 * (1 - 1/(2*omega))
    """
    if omega <= 1:
        return 0.0
    return 0.5 * (1.0 - 1.0 / (2.0 * omega))


def extract_support(x: np.ndarray, threshold: float = 1e-4) -> list[int]:
    """Return sorted 0-based indices where x_i > threshold."""
    return sorted(int(i) for i in np.where(x > threshold)[0])


def is_clique(support: list[int], A: np.ndarray) -> bool:
    """Check if all pairs in support are adjacent in A (zero-diagonal)."""
    for i in range(len(support)):
        for j in range(i + 1, len(support)):
            if A[support[i], support[j]] == 0:
                return False
    return True
