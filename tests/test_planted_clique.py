"""Tests for planted max-clique generator."""

import json
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from dirac_bench.planted_clique import (
    generate_planted_clique,
    instance_name,
    planted_clique_info,
    write_planted_dimacs,
    write_planted_metadata,
)
from dirac_bench.io import read_dimacs_graph
from dirac_bench.problems import (
    motzkin_straus_adjacency,
    objective,
    objective_to_omega,
    is_clique,
)


class TestGeneratePlantedClique:
    def test_basic_generation(self):
        G, planted = generate_planted_clique(50, 10, p=0.5, seed=42)
        assert G.number_of_nodes() == 50
        assert len(planted) == 10

    def test_nodes_are_1_based(self):
        G, planted = generate_planted_clique(50, 10, p=0.5, seed=42)
        assert min(G.nodes()) == 1
        assert max(G.nodes()) == 50
        assert all(1 <= v <= 50 for v in planted)

    def test_planted_nodes_form_clique(self):
        G, planted = generate_planted_clique(50, 10, p=0.5, seed=42)
        # Every pair of planted nodes must be connected
        for i in range(len(planted)):
            for j in range(i + 1, len(planted)):
                assert G.has_edge(planted[i], planted[j]), (
                    f"Missing edge between planted nodes {planted[i]} and {planted[j]}"
                )

    def test_planted_clique_in_adjacency(self):
        """Planted nodes form a clique in the adjacency matrix."""
        G, planted = generate_planted_clique(50, 10, p=0.5, seed=42)
        A = motzkin_straus_adjacency(G)
        planted_0based = [v - 1 for v in planted]
        assert is_clique(planted_0based, A)

    def test_edge_count_reasonable(self):
        """Edge count should be between G(n,p) edges and G(n,p) + k*(k-1)/2."""
        n, k, p = 100, 15, 0.5
        G, _ = generate_planted_clique(n, k, p=p, seed=42)
        max_er_edges = n * (n - 1) // 2  # complete graph
        clique_edges = k * (k - 1) // 2
        assert G.number_of_edges() >= clique_edges
        assert G.number_of_edges() <= max_er_edges

    def test_deterministic_with_seed(self):
        G1, p1 = generate_planted_clique(50, 10, seed=42)
        G2, p2 = generate_planted_clique(50, 10, seed=42)
        assert p1 == p2
        assert set(G1.edges()) == set(G2.edges())

    def test_different_seeds_differ(self):
        _, p1 = generate_planted_clique(50, 10, seed=42)
        _, p2 = generate_planted_clique(50, 10, seed=99)
        assert p1 != p2

    def test_invalid_k_too_large(self):
        with pytest.raises(ValueError, match="exceeds"):
            generate_planted_clique(10, 15)

    def test_invalid_k_too_small(self):
        with pytest.raises(ValueError, match="at least 2"):
            generate_planted_clique(10, 1)


class TestPlantedObjective:
    def test_planted_objective_matches_theory(self):
        """Uniform weight on planted clique gives f* = 0.5*(1 - 1/k)."""
        n, k = 50, 10
        G, planted = generate_planted_clique(n, k, p=0.5, seed=42)
        A = motzkin_straus_adjacency(G)

        x = np.zeros(n)
        for v in planted:
            x[v - 1] = 1.0 / k
        obj = objective(x, A)

        expected = 0.5 * (1 - 1 / k)
        assert pytest.approx(obj, abs=1e-10) == expected

    def test_planted_is_max_clique_easy(self):
        """On an easy instance, planted clique should be the max clique."""
        n, k = 30, 12  # k >> sqrt(30) ~ 5.5, very easy
        G, planted = generate_planted_clique(n, k, p=0.5, seed=42)
        A = motzkin_straus_adjacency(G)

        x_planted = np.zeros(n)
        for v in planted:
            x_planted[v - 1] = 1.0 / k
        planted_obj = objective(x_planted, A)
        planted_omega = objective_to_omega(planted_obj)
        assert planted_omega == k


class TestPlantedCliqueInfo:
    def test_info_keys(self):
        info = planted_clique_info(100, 15, 0.5)
        expected_keys = {
            "n", "k", "p", "natural_omega", "planted_objective",
            "natural_objective", "gap", "sqrt_n", "difficulty",
        }
        assert set(info.keys()) == expected_keys

    def test_natural_omega_n100(self):
        info = planted_clique_info(100, 15, 0.5)
        # omega(G(100, 0.5)) ~ 2*log2(100) ~ 13.3 -> 13
        assert info["natural_omega"] == 13

    def test_difficulty_easy(self):
        # k=20 > 2*sqrt(100)=20, borderline easy
        info = planted_clique_info(100, 25, 0.5)
        assert info["difficulty"] == "easy"

    def test_difficulty_hard(self):
        # k=15, sqrt(100)=10, natural_omega~13; 13 < 15 < 10 -> hard
        info = planted_clique_info(100, 15, 0.5)
        assert info["difficulty"] == "hard"

    def test_gap_positive(self):
        info = planted_clique_info(100, 15, 0.5)
        assert info["gap"] > 0


class TestSLSQPFindsEasyInstance:
    def test_slsqp_finds_planted_easy(self):
        """SLSQP should reliably find the planted clique on an easy instance."""
        n, k = 30, 12
        G, planted = generate_planted_clique(n, k, p=0.5, seed=42)
        A = motzkin_straus_adjacency(G)

        from dirac_bench.solvers.slsqp import solve_slsqp

        result = solve_slsqp(A, num_restarts=20, seed=42)
        assert result["omega"] >= k


class TestDIMACSRoundTrip:
    def test_write_and_read(self):
        """DIMACS write + read preserves graph structure."""
        G, planted = generate_planted_clique(30, 8, p=0.5, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            clq_path = Path(tmpdir) / "test.clq"
            write_planted_dimacs(G, planted, clq_path, 30, 8, 0.5, 42)

            G2 = read_dimacs_graph(clq_path)
            assert G2.number_of_nodes() == G.number_of_nodes()
            assert G2.number_of_edges() == G.number_of_edges()
            assert set(G2.edges()) == set(G.edges())


class TestMetadataRoundTrip:
    def test_json_roundtrip(self):
        """Metadata JSON writes and reads back correctly."""
        info = planted_clique_info(100, 15, 0.5)
        planted = [3, 17, 22, 45, 67, 78, 82, 85, 90, 91, 92, 94, 96, 98, 100]

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test.json"
            write_planted_metadata(json_path, planted, info, seed=42)

            with open(json_path) as f:
                loaded = json.load(f)

            assert loaded["planted_nodes"] == planted
            assert loaded["seed"] == 42
            assert loaded["k"] == 15
            assert loaded["n"] == 100
            assert loaded["difficulty"] == "hard"


class TestExplicitVertices:
    def test_explicit_vertices_form_clique(self):
        verts = [1, 2, 3, 4, 5]
        G, planted = generate_planted_clique(12, 5, seed=42, vertices=verts)
        assert planted == verts
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                assert G.has_edge(verts[i], verts[j])

    def test_explicit_vertices_graph_size(self):
        G, _ = generate_planted_clique(15, 4, seed=0, vertices=[3, 7, 11, 15])
        assert G.number_of_nodes() == 15
        assert min(G.nodes()) == 1
        assert max(G.nodes()) == 15

    def test_explicit_vertices_deterministic(self):
        verts = [2, 5, 8]
        G1, p1 = generate_planted_clique(10, 3, seed=42, vertices=verts)
        G2, p2 = generate_planted_clique(10, 3, seed=42, vertices=verts)
        assert p1 == p2
        assert set(G1.edges()) == set(G2.edges())

    def test_explicit_vertices_invalid_out_of_range(self):
        with pytest.raises(ValueError, match="must be in"):
            generate_planted_clique(10, 3, vertices=[0, 5, 8])

    def test_explicit_vertices_invalid_duplicates(self):
        with pytest.raises(ValueError, match="duplicates"):
            generate_planted_clique(10, 3, vertices=[5, 5, 8])

    def test_explicit_vertices_invalid_k_mismatch(self):
        with pytest.raises(ValueError, match="does not match"):
            generate_planted_clique(10, 4, vertices=[1, 2, 3])


class TestBruteForceMaxClique:
    """Brute-force verification using nx.find_cliques (Bron-Kerbosch).

    All (n, k) combos are chosen so k exceeds the natural clique number
    of G(n, 0.5) across many seeds, ensuring the planted clique dominates.
    """

    # --- Random-vertex planted cliques ---

    @pytest.mark.parametrize(
        "n, k, seed",
        [
            # Small graphs (n <= 15)
            (10, 5, 42),
            (12, 6, 42),
            (15, 7, 42),
            (15, 9, 99),
            # Medium graphs (n = 20..35)
            (20, 10, 0),
            (20, 10, 42),
            (25, 12, 7),
            (25, 12, 42),
            (30, 12, 0),
            (30, 12, 99),
            (35, 14, 42),
            (35, 11, 7),
            # Larger graphs (n = 40..50)
            (40, 15, 0),
            (40, 12, 42),
            (45, 15, 7),
            (45, 12, 99),
            (50, 15, 0),
            (50, 15, 42),
        ],
    )
    def test_planted_is_maximum_clique_random(self, n, k, seed):
        """Planted clique (random vertices) is the maximum clique."""
        G, planted = generate_planted_clique(n, k, p=0.5, seed=seed)
        planted_set = set(planted)

        all_cliques = list(nx.find_cliques(G))
        max_clique_size = max(len(c) for c in all_cliques)
        assert max_clique_size == k, (
            f"n={n}, k={k}, seed={seed}: expected max clique {k}, got {max_clique_size}"
        )

        max_cliques = [set(c) for c in all_cliques if len(c) == max_clique_size]
        assert planted_set in max_cliques, (
            f"n={n}, k={k}, seed={seed}: planted vertices not among maximum cliques"
        )

    # --- Explicit-vertex planted cliques ---

    @pytest.mark.parametrize(
        "n, k, vertices, seed",
        [
            # Small: contiguous block
            (12, 5, [1, 2, 3, 4, 5], 42),
            (15, 8, [1, 3, 5, 7, 9, 11, 13, 15], 42),
            (15, 6, [1, 5, 8, 10, 13, 15], 42),
            # Medium: even-spaced vertices
            (20, 10, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 0),
            (20, 10, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 42),
            # Medium: contiguous low / high blocks
            (30, 12, list(range(1, 13)), 0),
            (30, 12, list(range(19, 31)), 99),
            # Larger: first/last k vertices
            (40, 15, list(range(1, 16)), 42),
            (40, 15, list(range(26, 41)), 0),
            # Larger: scattered vertices
            (50, 15, list(range(1, 30, 2)), 42),
            (50, 15, [1, 2, 3, 4, 5, 6, 10, 15, 20, 25, 30, 35, 40, 45, 50], 99),
            # Different seeds for same layout
            (30, 12, list(range(1, 13)), 7),
            (30, 12, list(range(1, 13)), 42),
        ],
    )
    def test_planted_is_maximum_clique_explicit(self, n, k, vertices, seed):
        """Planted clique (explicit vertices) is the maximum clique."""
        G, planted = generate_planted_clique(n, k, p=0.5, seed=seed, vertices=vertices)
        assert planted == sorted(vertices)

        all_cliques = list(nx.find_cliques(G))
        max_clique_size = max(len(c) for c in all_cliques)
        assert max_clique_size == k, (
            f"n={n}, k={k}, seed={seed}: expected max clique {k}, got {max_clique_size}"
        )

        planted_set = set(planted)
        max_cliques = [set(c) for c in all_cliques if len(c) == max_clique_size]
        assert planted_set in max_cliques, (
            f"n={n}, k={k}, seed={seed}: planted vertices not among maximum cliques"
        )

    # --- Motzkin-Straus objective verification ---

    @pytest.mark.parametrize(
        "n, k, seed",
        [
            (10, 5, 42),
            (12, 6, 42),
            (15, 7, 42),
            (20, 10, 0),
            (30, 12, 42),
            (50, 15, 42),
        ],
    )
    def test_motzkin_straus_optimum_at_planted(self, n, k, seed):
        """Uniform weight on planted clique achieves the Motzkin-Straus optimum."""
        G, planted = generate_planted_clique(n, k, p=0.5, seed=seed)
        A = motzkin_straus_adjacency(G)

        x = np.zeros(n)
        for v in planted:
            x[v - 1] = 1.0 / k
        obj = objective(x, A)

        max_clique_size = max(len(c) for c in nx.find_cliques(G))
        expected_obj = 0.5 * (1 - 1 / max_clique_size)
        assert pytest.approx(obj, abs=1e-10) == expected_obj

    @pytest.mark.parametrize(
        "n, k, vertices",
        [
            (20, 10, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]),
            (30, 12, list(range(1, 13))),
            (40, 15, list(range(1, 16))),
            (50, 15, list(range(1, 30, 2))),
        ],
    )
    def test_motzkin_straus_support_at_explicit_indices(self, n, k, vertices):
        """Motzkin-Straus optimum support is exactly at the specified indices."""
        G, planted = generate_planted_clique(n, k, p=0.5, seed=42, vertices=vertices)
        A = motzkin_straus_adjacency(G)

        # Build optimal x with support on planted vertices
        x = np.zeros(n)
        for v in planted:
            x[v - 1] = 1.0 / k
        obj = objective(x, A)
        omega = objective_to_omega(obj)

        assert omega == k
        # Support should be exactly the planted vertices
        support = {i + 1 for i, xi in enumerate(x) if xi > 0}
        assert support == set(vertices)

    # --- Uniqueness: planted clique is the ONLY max clique ---

    @pytest.mark.parametrize(
        "n, k, seed",
        [
            (20, 10, 42),
            (30, 12, 42),
            (40, 15, 42),
            (50, 15, 42),
        ],
    )
    def test_planted_is_unique_maximum_clique(self, n, k, seed):
        """When k is large enough, the planted clique should be the only max clique."""
        G, planted = generate_planted_clique(n, k, p=0.5, seed=seed)
        planted_set = set(planted)

        max_cliques = [set(c) for c in nx.find_cliques(G) if len(c) == k]
        assert len(max_cliques) == 1, (
            f"Expected unique max clique, found {len(max_cliques)}"
        )
        assert max_cliques[0] == planted_set


class TestInstanceName:
    def test_p05(self):
        assert instance_name(100, 15, 0.5) == "planted_100_k15_p05"

    def test_p09(self):
        assert instance_name(200, 25, 0.9) == "planted_200_k25_p09"
