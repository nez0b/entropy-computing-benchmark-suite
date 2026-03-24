"""Tests for the Motzkin-Straus problem formulation.

Uses complete graphs K4 and K5 where the clique number is known exactly.
"""

import numpy as np
import networkx as nx
import pytest

from dirac_bench.problems import (
    motzkin_straus_adjacency,
    objective,
    gradient,
    objective_to_omega,
    omega_to_theoretical_objective,
)


def _make_complete_graph(n: int) -> nx.Graph:
    """Create K_n with 1-based node labels (like DIMACS)."""
    return nx.relabel_nodes(nx.complete_graph(n), {i: i + 1 for i in range(n)})


class TestAdjacencyMatrix:
    def test_k4_shape(self):
        G = _make_complete_graph(4)
        A = motzkin_straus_adjacency(G)
        assert A.shape == (4, 4)

    def test_k4_symmetric(self):
        G = _make_complete_graph(4)
        A = motzkin_straus_adjacency(G)
        np.testing.assert_array_equal(A, A.T)

    def test_k4_diagonal_zero(self):
        G = _make_complete_graph(4)
        A = motzkin_straus_adjacency(G)
        np.testing.assert_array_equal(np.diag(A), np.zeros(4))


class TestObjective:
    def test_uniform_on_k4(self):
        """On K4, the uniform distribution x = [1/4]*4 gives f* = 3/8."""
        G = _make_complete_graph(4)
        A = motzkin_straus_adjacency(G)
        x = np.ones(4) / 4
        obj = objective(x, A)
        assert pytest.approx(obj, abs=1e-10) == 0.375  # 0.5 * (1 - 1/4)

    def test_uniform_on_k5(self):
        """On K5, the uniform distribution x = [1/5]*5 gives f* = 2/5."""
        G = _make_complete_graph(5)
        A = motzkin_straus_adjacency(G)
        x = np.ones(5) / 5
        obj = objective(x, A)
        assert pytest.approx(obj, abs=1e-10) == 0.4  # 0.5 * (1 - 1/5)


class TestGradient:
    def test_gradient_shape(self):
        G = _make_complete_graph(4)
        A = motzkin_straus_adjacency(G)
        x = np.ones(4) / 4
        g = gradient(x, A)
        assert g.shape == (4,)

    def test_gradient_on_complete_graph(self):
        """On K4, gradient at uniform = A @ x = (n-1)/n * ones."""
        G = _make_complete_graph(4)
        A = motzkin_straus_adjacency(G)
        x = np.ones(4) / 4
        g = gradient(x, A)
        # Each row of K4 adjacency sums to 3, so A @ (1/4 * ones) = 3/4 * ones
        np.testing.assert_allclose(g, 0.75 * np.ones(4))


class TestOmegaConversion:
    @pytest.mark.parametrize("omega", [2, 3, 4, 5, 10, 34, 100])
    def test_roundtrip(self, omega):
        """objective_to_omega(omega_to_theoretical_objective(omega)) == omega."""
        f_star = omega_to_theoretical_objective(omega)
        recovered = objective_to_omega(f_star)
        assert recovered == omega

    def test_omega_1(self):
        assert omega_to_theoretical_objective(1) == 0.0

    def test_omega_2(self):
        assert pytest.approx(omega_to_theoretical_objective(2)) == 0.25

    def test_omega_34(self):
        """C125.9 has omega=34, theoretical f* = 0.5*(1-1/34)."""
        expected = 0.5 * (1 - 1 / 34)
        assert pytest.approx(omega_to_theoretical_objective(34)) == expected


class TestPetersenGraph:
    def test_petersen_omega_2(self):
        """Petersen graph has clique number 2 (no triangles)."""
        G = nx.petersen_graph()
        # Relabel to 1-based
        G = nx.relabel_nodes(G, {i: i + 1 for i in range(10)})
        A = motzkin_straus_adjacency(G)

        # Optimal: put weight 0.5 on two adjacent nodes
        # f* = 0.5 * 0.5 * 1 * 0.5 = 0.125 but let's compute via formula
        f_star = omega_to_theoretical_objective(2)
        assert pytest.approx(f_star) == 0.25

        # Verify: place 0.5 on nodes 1 and 2 (which are adjacent in Petersen)
        x = np.zeros(10)
        x[0] = 0.5
        x[1] = 0.5
        obj = objective(x, A)
        assert pytest.approx(obj) == 0.25
