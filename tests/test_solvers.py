"""Tests for classical solvers (SLSQP and L-BFGS-B) on small known graphs."""

import numpy as np
import networkx as nx
import pytest

from dirac_bench.problems import motzkin_straus_adjacency
from dirac_bench.solvers.slsqp import solve_slsqp
from dirac_bench.solvers.lbfgs import solve_lbfgs


def _complete_graph_adjacency(n: int) -> np.ndarray:
    G = nx.relabel_nodes(nx.complete_graph(n), {i: i + 1 for i in range(n)})
    return motzkin_straus_adjacency(G)


def _petersen_adjacency() -> np.ndarray:
    G = nx.relabel_nodes(nx.petersen_graph(), {i: i + 1 for i in range(10)})
    return motzkin_straus_adjacency(G)


class TestSLSQP:
    def test_k4_omega(self):
        A = _complete_graph_adjacency(4)
        result = solve_slsqp(A, num_restarts=5)
        assert result["omega"] == 4

    def test_k5_omega(self):
        A = _complete_graph_adjacency(5)
        result = solve_slsqp(A, num_restarts=5)
        assert result["omega"] == 5

    def test_petersen_omega(self):
        A = _petersen_adjacency()
        result = solve_slsqp(A, num_restarts=5)
        assert result["omega"] == 2

    def test_result_keys(self):
        A = _complete_graph_adjacency(4)
        result = solve_slsqp(A, num_restarts=3)
        assert "omega" in result
        assert "best_objective" in result
        assert "solution" in result
        assert "solve_time" in result
        assert "all_objectives" in result

    def test_solution_on_simplex(self):
        A = _complete_graph_adjacency(4)
        result = solve_slsqp(A, num_restarts=3)
        x = result["solution"]
        assert pytest.approx(np.sum(x), abs=1e-6) == 1.0
        assert np.all(x >= -1e-8)


class TestLBFGS:
    def test_k4_omega(self):
        A = _complete_graph_adjacency(4)
        result = solve_lbfgs(A, num_restarts=5)
        assert result["omega"] == 4

    def test_k5_omega(self):
        A = _complete_graph_adjacency(5)
        result = solve_lbfgs(A, num_restarts=5)
        assert result["omega"] == 5

    def test_petersen_omega(self):
        A = _petersen_adjacency()
        result = solve_lbfgs(A, num_restarts=5)
        assert result["omega"] == 2

    def test_result_keys(self):
        A = _complete_graph_adjacency(4)
        result = solve_lbfgs(A, num_restarts=3)
        assert "omega" in result
        assert "best_objective" in result
        assert "solution" in result
        assert "solve_time" in result
        assert "all_objectives" in result

    def test_solution_on_simplex(self):
        """Softmax output should always be on the simplex."""
        A = _complete_graph_adjacency(4)
        result = solve_lbfgs(A, num_restarts=3)
        x = result["solution"]
        assert pytest.approx(np.sum(x), abs=1e-6) == 1.0
        assert np.all(x >= 0)
