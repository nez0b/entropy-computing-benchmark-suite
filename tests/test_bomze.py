"""Tests for Bomze regularization helpers and solver wrappers."""

import numpy as np
import networkx as nx
import pytest

from dirac_bench.problems import (
    motzkin_straus_adjacency,
    objective,
    bomze_regularize,
    bomze_objective_to_omega,
    bomze_theoretical_objective,
    extract_support,
    is_clique,
)
from dirac_bench.solvers.bomze import solve_bomze_slsqp, solve_bomze_lbfgs


# ── helpers ──────────────────────────────────────────────────────────────

def _complete_graph_adjacency(n: int) -> np.ndarray:
    G = nx.relabel_nodes(nx.complete_graph(n), {i: i + 1 for i in range(n)})
    return motzkin_straus_adjacency(G)


def _petersen_adjacency() -> np.ndarray:
    G = nx.relabel_nodes(nx.petersen_graph(), {i: i + 1 for i in range(10)})
    return motzkin_straus_adjacency(G)


def _overlapping_k4_adjacency() -> np.ndarray:
    """Two K4s sharing 2 vertices: {0,1,2,3} and {2,3,4,5}."""
    from dirac_bench.test_graphs import make_overlapping_k4
    A, _omega, _name = make_overlapping_k4()
    return A


# ── TestBomzeRegularize ─────────────────────────────────────────────────

class TestBomzeRegularize:
    def test_diagonal_is_half(self):
        A = _complete_graph_adjacency(4)
        A_bar = bomze_regularize(A)
        np.testing.assert_array_equal(np.diag(A_bar), 0.5 * np.ones(4))

    def test_off_diagonal_unchanged(self):
        A = _complete_graph_adjacency(4)
        A_bar = bomze_regularize(A)
        mask = ~np.eye(4, dtype=bool)
        np.testing.assert_array_equal(A_bar[mask], A[mask])

    def test_symmetric(self):
        A = _complete_graph_adjacency(4)
        A_bar = bomze_regularize(A)
        np.testing.assert_array_equal(A_bar, A_bar.T)

    def test_shape_preserved(self):
        A = _complete_graph_adjacency(5)
        A_bar = bomze_regularize(A)
        assert A_bar.shape == A.shape


# ── TestBomzeOmegaConversion ────────────────────────────────────────────

class TestBomzeOmegaConversion:
    @pytest.mark.parametrize("omega", [2, 3, 4, 5, 10, 34, 100])
    def test_roundtrip(self, omega):
        """bomze_objective_to_omega(bomze_theoretical_objective(omega)) == omega."""
        f_star = bomze_theoretical_objective(omega)
        recovered = bomze_objective_to_omega(f_star)
        assert recovered == omega

    def test_omega_1(self):
        assert bomze_theoretical_objective(1) == 0.0

    def test_omega_2(self):
        # 0.5 * (1 - 1/4) = 0.375
        assert pytest.approx(bomze_theoretical_objective(2)) == 0.375

    def test_omega_4(self):
        # 0.5 * (1 - 1/8) = 0.4375
        assert pytest.approx(bomze_theoretical_objective(4)) == 0.4375


# ── TestBomzeObjectiveOnKn ──────────────────────────────────────────────

class TestBomzeObjectiveOnKn:
    def test_k4_uniform(self):
        """On K4 with A_bar, uniform x gives 0.5*(1 - 1/8) = 0.4375."""
        A = _complete_graph_adjacency(4)
        A_bar = bomze_regularize(A)
        x = np.ones(4) / 4
        obj = objective(x, A_bar)
        assert pytest.approx(obj, abs=1e-10) == 0.4375

    def test_k5_uniform(self):
        """On K5 with A_bar, uniform x gives 0.5*(1 - 1/10) = 0.45."""
        A = _complete_graph_adjacency(5)
        A_bar = bomze_regularize(A)
        x = np.ones(5) / 5
        obj = objective(x, A_bar)
        assert pytest.approx(obj, abs=1e-10) == 0.45


# ── TestExtractSupport ──────────────────────────────────────────────────

class TestExtractSupport:
    def test_uniform_k4(self):
        x = np.ones(4) / 4
        support = extract_support(x)
        assert support == [0, 1, 2, 3]

    def test_sparse(self):
        x = np.array([0.5, 0.0, 0.5, 0.0, 0.0])
        support = extract_support(x)
        assert support == [0, 2]

    def test_threshold(self):
        x = np.array([0.5, 1e-5, 0.5 - 1e-5, 0.0])
        # Default threshold 1e-4: only indices 0, 2
        support = extract_support(x, threshold=1e-4)
        assert support == [0, 2]

    def test_all_zero(self):
        x = np.zeros(5)
        assert extract_support(x) == []

    def test_custom_threshold(self):
        x = np.array([0.5, 0.001, 0.499, 0.0])
        support = extract_support(x, threshold=0.01)
        assert support == [0, 2]


# ── TestIsClique ────────────────────────────────────────────────────────

class TestIsClique:
    def test_k4_full(self):
        A = _complete_graph_adjacency(4)
        assert is_clique([0, 1, 2, 3], A)

    def test_k4_subset(self):
        A = _complete_graph_adjacency(4)
        assert is_clique([0, 1], A)
        assert is_clique([1, 2, 3], A)

    def test_single_vertex(self):
        A = _complete_graph_adjacency(4)
        assert is_clique([0], A)

    def test_empty(self):
        A = _complete_graph_adjacency(4)
        assert is_clique([], A)

    def test_petersen_edge(self):
        A = _petersen_adjacency()
        # Nodes 0 and 1 are adjacent in Petersen graph
        assert is_clique([0, 1], A)

    def test_petersen_no_triangle(self):
        """Petersen graph has no triangles, so any 3 vertices fail."""
        A = _petersen_adjacency()
        # Try several triples — none should be a clique
        assert not is_clique([0, 1, 2], A)

    def test_overlapping_non_clique(self):
        """Vertices spanning both K4s but not forming a clique."""
        A = _overlapping_k4_adjacency()
        # {0, 1, 4, 5}: 0-4 not adjacent
        assert not is_clique([0, 1, 4, 5], A)


# ── TestBomzeSolverWrappers ─────────────────────────────────────────────

class TestBomzeSolverWrappers:
    def test_slsqp_k4_omega(self):
        A = _complete_graph_adjacency(4)
        result = solve_bomze_slsqp(A, num_restarts=5)
        assert result["omega"] == 4
        assert result["formulation"] == "bomze"

    def test_slsqp_k5_omega(self):
        A = _complete_graph_adjacency(5)
        result = solve_bomze_slsqp(A, num_restarts=5)
        assert result["omega"] == 5

    def test_slsqp_petersen_omega(self):
        A = _petersen_adjacency()
        result = solve_bomze_slsqp(A, num_restarts=5)
        assert result["omega"] == 2

    def test_lbfgs_k4_omega(self):
        A = _complete_graph_adjacency(4)
        result = solve_bomze_lbfgs(A, num_restarts=5)
        assert result["omega"] == 4
        assert result["formulation"] == "bomze"

    def test_lbfgs_k5_omega(self):
        A = _complete_graph_adjacency(5)
        result = solve_bomze_lbfgs(A, num_restarts=5)
        assert result["omega"] == 5

    def test_lbfgs_petersen_omega(self):
        A = _petersen_adjacency()
        result = solve_bomze_lbfgs(A, num_restarts=5)
        assert result["omega"] == 2

    def test_solution_on_simplex(self):
        A = _complete_graph_adjacency(4)
        result = solve_bomze_slsqp(A, num_restarts=3)
        x = result["solution"]
        assert pytest.approx(np.sum(x), abs=1e-6) == 1.0
        assert np.all(x >= -1e-8)

    def test_support_is_clique(self):
        """The Bomze solution's support should be a clique in the original graph."""
        A = _complete_graph_adjacency(4)
        result = solve_bomze_slsqp(A, num_restarts=5)
        support = extract_support(result["solution"])
        assert is_clique(support, A)


# ── TestSpuriousSolutionElimination ─────────────────────────────────────

class TestSpuriousSolutionElimination:
    """Verify Bomze regularization eliminates spurious solutions on
    overlapping-cliques graphs."""

    def test_overlapping_k4_bomze_all_clique(self):
        """Every Bomze restart on overlapping K4s should land on a clique."""
        A = _overlapping_k4_adjacency()
        n_restarts = 20

        for seed in range(n_restarts):
            result = solve_bomze_slsqp(A, num_restarts=1, seed=seed)
            support = extract_support(result["solution"])
            assert is_clique(support, A), (
                f"Bomze seed={seed}: support {support} is NOT a clique"
            )

    def test_overlapping_k4_bomze_omega(self):
        """Bomze on overlapping K4s should find omega=4."""
        A = _overlapping_k4_adjacency()
        result = solve_bomze_slsqp(A, num_restarts=20)
        assert result["omega"] == 4
