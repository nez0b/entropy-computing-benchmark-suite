"""Tests for clique extraction and refinement methods."""

import numpy as np
import networkx as nx
import pytest

from dirac_bench.problems import is_clique, motzkin_straus_adjacency
from dirac_bench.test_graphs import make_overlapping_k4
from dirac_bench.clique_extraction import (
    _greedy_prune_to_clique,
    greedy_clique_desc,
    greedy_clique_asc,
    greedy_clique_random,
    threshold_sweep_extract,
    top_k_extract,
    randomized_rounding_extract,
    local_search_1swap,
    local_search_2swap,
    cluster_and_extract,
    run_all_extractions,
)


# ── Test graph helpers ───────────────────────────────────────────────────


def _k4_adjacency() -> np.ndarray:
    """Complete graph K4, 0-indexed."""
    G = nx.complete_graph(4)
    return nx.adjacency_matrix(G).toarray().astype(np.float64)


def _k5_adjacency() -> np.ndarray:
    """Complete graph K5, 0-indexed."""
    G = nx.complete_graph(5)
    return nx.adjacency_matrix(G).toarray().astype(np.float64)


def _petersen_adjacency() -> np.ndarray:
    """Petersen graph, 0-indexed. omega=2, no triangles."""
    G = nx.petersen_graph()
    return nx.adjacency_matrix(G).toarray().astype(np.float64)


def _overlapping_k4() -> tuple[np.ndarray, int]:
    """Two K4s sharing 2 vertices. omega=4."""
    A, omega, _ = make_overlapping_k4()
    return A, omega


# ── TestGreedyPrune ──────────────────────────────────────────────────────


class TestGreedyPrune:
    def test_k4_all_candidates(self):
        A = _k4_adjacency()
        clique = _greedy_prune_to_clique([0, 1, 2, 3], A)
        assert len(clique) == 4
        assert is_clique(clique, A)

    def test_empty_candidates(self):
        A = _k4_adjacency()
        assert _greedy_prune_to_clique([], A) == []

    def test_single_candidate(self):
        A = _k4_adjacency()
        assert _greedy_prune_to_clique([2], A) == [2]

    def test_petersen_pair(self):
        A = _petersen_adjacency()
        # In Petersen graph, max clique is 2
        clique = _greedy_prune_to_clique(list(range(10)), A)
        assert len(clique) <= 2
        assert is_clique(clique, A)

    def test_order_matters(self):
        """Different orderings can produce different cliques."""
        A, _ = _overlapping_k4()
        # Starting with node 0 favors first K4 {0,1,2,3}
        c1 = _greedy_prune_to_clique([0, 1, 2, 3, 4, 5], A)
        # Starting with node 5 favors second K4 {2,3,4,5}
        c2 = _greedy_prune_to_clique([5, 4, 3, 2, 1, 0], A)
        assert is_clique(c1, A)
        assert is_clique(c2, A)


# ── TestGreedyDesc ───────────────────────────────────────────────────────


class TestGreedyDesc:
    def test_k4_uniform_finds_full_clique(self):
        A = _k4_adjacency()
        x = np.ones(4) / 4  # uniform on K4
        clique = greedy_clique_desc(x, A)
        assert len(clique) == 4
        assert is_clique(clique, A)

    def test_output_is_valid_clique(self):
        A, _ = _overlapping_k4()
        x = np.array([0.3, 0.25, 0.2, 0.15, 0.05, 0.05])
        clique = greedy_clique_desc(x, A)
        assert is_clique(clique, A)
        assert len(clique) >= 1

    def test_zero_vector(self):
        A = _k4_adjacency()
        x = np.zeros(4)
        assert greedy_clique_desc(x, A) == []


# ── TestGreedyAsc ────────────────────────────────────────────────────────


class TestGreedyAsc:
    def test_k4_uniform(self):
        A = _k4_adjacency()
        x = np.ones(4) / 4
        clique = greedy_clique_asc(x, A)
        assert len(clique) == 4
        assert is_clique(clique, A)

    def test_valid_clique(self):
        A, _ = _overlapping_k4()
        x = np.array([0.3, 0.25, 0.2, 0.15, 0.05, 0.05])
        clique = greedy_clique_asc(x, A)
        assert is_clique(clique, A)


# ── TestGreedyRandom ─────────────────────────────────────────────────────


class TestGreedyRandom:
    def test_k4_finds_full(self):
        A = _k4_adjacency()
        x = np.ones(4) / 4
        clique = greedy_clique_random(x, A, n_trials=10, seed=0)
        assert len(clique) == 4

    def test_deterministic_with_seed(self):
        A, _ = _overlapping_k4()
        x = np.array([0.2, 0.2, 0.2, 0.15, 0.15, 0.1])
        c1 = greedy_clique_random(x, A, seed=42)
        c2 = greedy_clique_random(x, A, seed=42)
        assert c1 == c2

    def test_empty_support(self):
        A = _k4_adjacency()
        x = np.zeros(4)
        assert greedy_clique_random(x, A) == []


# ── TestThresholdSweep ───────────────────────────────────────────────────


class TestThresholdSweep:
    def test_returns_dict(self):
        A = _k4_adjacency()
        x = np.ones(4) / 4
        result = threshold_sweep_extract(x, A)
        assert isinstance(result, dict)
        assert len(result) == 6  # default 6 thresholds

    def test_all_valid_cliques(self):
        A, _ = _overlapping_k4()
        x = np.array([0.3, 0.25, 0.2, 0.15, 0.05, 0.05])
        result = threshold_sweep_extract(x, A)
        for t, clique in result.items():
            assert is_clique(clique, A), f"threshold={t} produced invalid clique"

    def test_higher_threshold_smaller_support(self):
        A = _k5_adjacency()
        x = np.array([0.4, 0.3, 0.2, 0.08, 0.02])
        result = threshold_sweep_extract(x, A, thresholds=[1e-4, 0.1, 0.3])
        # Higher thresholds exclude more nodes, so clique ≤ lower threshold
        assert len(result[0.3]) <= len(result[1e-4])


# ── TestTopK ─────────────────────────────────────────────────────────────


class TestTopK:
    def test_k4_with_known_omega(self):
        A = _k4_adjacency()
        x = np.array([0.4, 0.3, 0.2, 0.1])
        result = top_k_extract(x, A, known_omega=4)
        assert isinstance(result, dict)
        # At k=4, should find full clique
        assert len(result[4]) == 4

    def test_all_valid(self):
        A, _ = _overlapping_k4()
        x = np.array([0.3, 0.25, 0.2, 0.15, 0.05, 0.05])
        result = top_k_extract(x, A, known_omega=4)
        for k, clique in result.items():
            assert is_clique(clique, A), f"k={k} produced invalid clique"


# ── TestRandomizedRounding ───────────────────────────────────────────────


class TestRandomizedRounding:
    def test_k4_finds_clique(self):
        A = _k4_adjacency()
        x = np.ones(4) / 4
        clique = randomized_rounding_extract(x, A, n_trials=100, seed=0)
        assert is_clique(clique, A)
        assert len(clique) >= 2  # should find something

    def test_deterministic_with_seed(self):
        A, _ = _overlapping_k4()
        x = np.array([0.2, 0.2, 0.2, 0.15, 0.15, 0.1])
        c1 = randomized_rounding_extract(x, A, seed=42)
        c2 = randomized_rounding_extract(x, A, seed=42)
        assert c1 == c2

    def test_zero_vector(self):
        A = _k4_adjacency()
        x = np.zeros(4)
        assert randomized_rounding_extract(x, A) == []


# ── TestLocalSearch1Swap ─────────────────────────────────────────────────


class TestLocalSearch1Swap:
    def test_does_not_shrink(self):
        A = _k4_adjacency()
        seed_clique = [0, 1]
        result = local_search_1swap(seed_clique, A)
        assert len(result) >= len(seed_clique)
        assert is_clique(result, A)

    def test_grows_edge_to_k4(self):
        """Starting from an edge in K4, LS should grow to full K4."""
        A = _k4_adjacency()
        result = local_search_1swap([0, 1], A)
        assert len(result) == 4
        assert is_clique(result, A)

    def test_preserves_valid_clique(self):
        A, _ = _overlapping_k4()
        seed = [0, 1, 2, 3]  # valid K4
        result = local_search_1swap(seed, A)
        assert len(result) >= 4
        assert is_clique(result, A)

    def test_single_vertex_seed(self):
        A = _k4_adjacency()
        result = local_search_1swap([0], A)
        assert len(result) >= 1
        assert is_clique(result, A)

    def test_respects_weights(self):
        A, _ = _overlapping_k4()
        # Weight heavily toward second K4
        weights = np.array([0.01, 0.01, 0.2, 0.3, 0.3, 0.2])
        result = local_search_1swap([2], A, x_weights=weights)
        assert is_clique(result, A)


# ── TestLocalSearch2Swap ─────────────────────────────────────────────────


class TestLocalSearch2Swap:
    def test_does_not_shrink(self):
        A = _k4_adjacency()
        seed_clique = [0, 1, 2]
        result = local_search_2swap(seed_clique, A)
        assert len(result) >= len(seed_clique)
        assert is_clique(result, A)

    def test_k4_stays_full(self):
        A = _k4_adjacency()
        result = local_search_2swap([0, 1, 2, 3], A)
        assert len(result) == 4
        assert is_clique(result, A)

    def test_valid_output(self):
        A, _ = _overlapping_k4()
        result = local_search_2swap([0, 1], A)
        assert is_clique(result, A)


# ── TestClusterAndExtract ────────────────────────────────────────────────


class TestClusterAndExtract:
    def test_k4_single_vector(self):
        A = _k4_adjacency()
        x_vectors = np.ones((1, 4)) / 4
        clique = cluster_and_extract(x_vectors, A, n_clusters=1, seed=0)
        assert is_clique(clique, A)

    def test_multiple_vectors(self):
        A = _k4_adjacency()
        rng = np.random.default_rng(42)
        # 10 random simplex vectors
        raw = rng.dirichlet(np.ones(4), size=10)
        clique = cluster_and_extract(raw, A, n_clusters=3, seed=42)
        assert is_clique(clique, A)

    def test_1d_input(self):
        A = _k4_adjacency()
        x = np.ones(4) / 4
        clique = cluster_and_extract(x, A, n_clusters=1)
        assert is_clique(clique, A)


# ── TestRunAllExtractions ────────────────────────────────────────────────


class TestRunAllExtractions:
    def test_k4_all_methods_valid(self):
        """Every method should return valid cliques on K4."""
        A = _k4_adjacency()
        rng = np.random.default_rng(42)
        x_vectors = rng.dirichlet(np.ones(4), size=5)
        results = run_all_extractions(x_vectors, A, known_omega=4, seed=42)

        expected_methods = {
            "greedy_desc", "greedy_asc", "greedy_random",
            "threshold_sweep", "top_k", "randomized_rounding",
            "cluster", "local_search_1swap", "local_search_2swap",
        }
        assert expected_methods.issubset(results.keys())

        for name, r in results.items():
            assert r["valid"], f"{name} produced invalid clique: {r['clique']}"
            assert r["size"] == len(r["clique"])

    def test_k4_finds_omega(self):
        """At least one method should find the full K4 clique."""
        A = _k4_adjacency()
        # Uniform simplex vectors on K4 — optimal
        x_vectors = np.ones((10, 4)) / 4
        results = run_all_extractions(x_vectors, A, known_omega=4, seed=42)

        best_size = max(r["size"] for r in results.values())
        assert best_size == 4

    def test_overlapping_k4_finds_omega(self):
        A, omega = _overlapping_k4()
        # Vectors concentrated on first K4
        x_vectors = np.zeros((5, 6))
        x_vectors[:, :4] = 0.25
        results = run_all_extractions(x_vectors, A, known_omega=omega, seed=42)

        best_size = max(r["size"] for r in results.values())
        assert best_size == omega

    def test_single_vector_input(self):
        A = _k4_adjacency()
        x = np.ones(4) / 4
        results = run_all_extractions(x, A, seed=0)
        assert all(r["valid"] for r in results.values())

    def test_petersen_max_2(self):
        """On Petersen graph, no method should find clique > 2."""
        A = _petersen_adjacency()
        rng = np.random.default_rng(42)
        x_vectors = rng.dirichlet(np.ones(10), size=10)
        results = run_all_extractions(x_vectors, A, known_omega=2, seed=42)

        for name, r in results.items():
            assert r["valid"], f"{name} invalid on Petersen"
            assert r["size"] <= 2, f"{name} found clique of size {r['size']} on Petersen"

    def test_deterministic_with_seed(self):
        A = _k4_adjacency()
        x_vectors = np.ones((3, 4)) / 4
        r1 = run_all_extractions(x_vectors, A, seed=42)
        r2 = run_all_extractions(x_vectors, A, seed=42)
        for name in r1:
            assert r1[name]["clique"] == r2[name]["clique"], f"{name} not deterministic"
