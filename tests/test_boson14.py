"""Tests for Boson14 scaled Motzkin-Straus benchmark suite."""

import networkx as nx
import numpy as np
import pytest

from dirac_bench.boson14 import (
    build_integer_qp,
    compute_all_max_clique_solutions,
    compute_clique_distribution,
    energy_to_omega,
    generate_degenerate_planted,
    hardware_energy,
    omega_to_scaled_objective,
    plot_clique_distribution,
    scaled_objective,
    scaled_objective_to_omega,
    scaled_optimal_y,
    scramble_graph,
    unscramble_solution,
)
from dirac_bench.planted_clique import generate_planted_clique
from dirac_bench.problems import motzkin_straus_adjacency


class TestScaledObjective:
    """Roundtrip and consistency tests for scaled objective helpers."""

    @pytest.mark.parametrize("omega", [2, 3, 5, 10, 34, 100])
    def test_roundtrip_omega(self, omega):
        g_star = omega_to_scaled_objective(omega)
        recovered = scaled_objective_to_omega(g_star)
        assert recovered == omega

    @pytest.mark.parametrize("omega", [2, 3, 5, 10, 34, 100])
    def test_consistency_with_standard(self, omega):
        """g*(w) = R^2 * f*(w) where f*(w) = 0.5*(1-1/w)."""
        R = 100
        f_star = 0.5 * (1.0 - 1.0 / omega)
        g_star = omega_to_scaled_objective(omega, R)
        assert g_star == pytest.approx(R * R * f_star)

    @pytest.mark.parametrize("omega", [2, 3, 5, 10, 34, 100])
    def test_energy_relationship(self, omega):
        """E* = -2g*."""
        R = 100
        g_star = omega_to_scaled_objective(omega, R)
        E_star = -2.0 * g_star
        recovered = energy_to_omega(E_star, R)
        assert recovered == omega

    @pytest.mark.parametrize("R", [1, 10, 100, 1000])
    def test_different_R_values(self, R):
        omega = 5
        g_star = omega_to_scaled_objective(omega, R)
        assert g_star == pytest.approx((R * R / 2.0) * (1.0 - 1.0 / omega))
        recovered = scaled_objective_to_omega(g_star, R)
        assert recovered == omega


class TestBuildIntegerQP:
    def test_J_is_neg_A(self):
        G, _ = generate_planted_clique(20, 5, seed=42)
        A = motzkin_straus_adjacency(G)
        C, J = build_integer_qp(A)
        np.testing.assert_array_equal(J, -A)

    def test_C_is_zeros(self):
        G, _ = generate_planted_clique(20, 5, seed=42)
        A = motzkin_straus_adjacency(G)
        C, J = build_integer_qp(A)
        np.testing.assert_array_equal(C, np.zeros(20))

    def test_J_entries(self):
        G, _ = generate_planted_clique(20, 5, seed=42)
        A = motzkin_straus_adjacency(G)
        _, J = build_integer_qp(A)
        unique = set(np.unique(J))
        assert unique <= {0.0, -1.0}

    def test_J_is_integer(self):
        G, _ = generate_planted_clique(20, 5, seed=42)
        A = motzkin_straus_adjacency(G)
        _, J = build_integer_qp(A)
        np.testing.assert_array_equal(J, np.round(J))


class TestScaledOptimalY:
    @pytest.mark.parametrize("n, k", [(10, 5), (20, 10), (30, 12), (50, 15)])
    def test_achieves_theoretical_objective(self, n, k):
        G, planted = generate_planted_clique(n, k, seed=42)
        A = motzkin_straus_adjacency(G)
        planted_0 = [v - 1 for v in planted]
        R = 100
        y = scaled_optimal_y(planted_0, n, k, R)
        g = scaled_objective(y, A)
        g_theory = omega_to_scaled_objective(k, R)
        assert g == pytest.approx(g_theory)

    @pytest.mark.parametrize("n, k", [(10, 5), (20, 10), (30, 12), (50, 15)])
    def test_sum_equals_R(self, n, k):
        planted_0 = list(range(k))
        R = 100
        y = scaled_optimal_y(planted_0, n, k, R)
        assert np.sum(y) == pytest.approx(R)

    @pytest.mark.parametrize("n, k", [(10, 5), (20, 10), (30, 12), (50, 15)])
    def test_support_matches_planted(self, n, k):
        planted_0 = list(range(k))
        R = 100
        y = scaled_optimal_y(planted_0, n, k, R)
        support = sorted(int(i) for i in np.where(y > 1e-10)[0])
        assert support == planted_0


class TestDegeneratePlanted:
    @pytest.mark.parametrize(
        "n, k, nc",
        [(30, 10, 2), (40, 10, 3), (50, 10, 4), (50, 12, 2), (50, 15, 3)],
    )
    def test_each_set_is_clique(self, n, k, nc):
        G, sets = generate_degenerate_planted(n, k, nc, seed=42)
        for vs in sets:
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    assert G.has_edge(vs[i], vs[j])

    @pytest.mark.parametrize(
        "n, k, nc",
        [(30, 10, 2), (40, 10, 3), (50, 10, 4), (50, 12, 2), (50, 15, 3)],
    )
    def test_disjoint_sets(self, n, k, nc):
        _, sets = generate_degenerate_planted(n, k, nc, seed=42)
        all_verts = [v for vs in sets for v in vs]
        assert len(all_verts) == len(set(all_verts))

    @pytest.mark.parametrize(
        "n, k, nc",
        [(30, 10, 2), (40, 10, 3), (50, 10, 4), (50, 12, 2), (50, 15, 3)],
    )
    def test_omega_equals_k(self, n, k, nc):
        G, _ = generate_degenerate_planted(n, k, nc, seed=42)
        max_clique_size = max(len(c) for c in nx.find_cliques(G))
        assert max_clique_size == k

    @pytest.mark.parametrize(
        "n, k, nc",
        [(30, 10, 2), (40, 10, 3), (50, 10, 4), (50, 12, 2), (50, 15, 3)],
    )
    def test_all_planted_achieve_same_objective(self, n, k, nc):
        G, sets = generate_degenerate_planted(n, k, nc, seed=42)
        A = motzkin_straus_adjacency(G)
        R = 100
        objs = []
        for vs in sets:
            planted_0 = [v - 1 for v in vs]
            y = scaled_optimal_y(planted_0, n, k, R)
            objs.append(scaled_objective(y, A))
        for obj in objs:
            assert obj == pytest.approx(objs[0])

    @pytest.mark.parametrize(
        "n, k, nc",
        [(30, 10, 2), (40, 10, 3), (50, 10, 4), (50, 12, 2), (50, 15, 3)],
    )
    def test_no_cross_edges(self, n, k, nc):
        G, sets = generate_degenerate_planted(n, k, nc, seed=42)
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                for u in sets[i]:
                    for v in sets[j]:
                        assert not G.has_edge(u, v)

    def test_too_many_planted_raises(self):
        with pytest.raises(ValueError, match="exceeds"):
            generate_degenerate_planted(10, 5, 3)

    def test_explicit_vertex_sets(self):
        vertex_sets = [list(range(1, 11)), list(range(11, 21))]
        G, sets = generate_degenerate_planted(
            30, 10, 2, seed=42, vertex_sets=vertex_sets
        )
        assert sets == vertex_sets
        for vs in sets:
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    assert G.has_edge(vs[i], vs[j])


class TestScramble:
    @pytest.mark.parametrize("n, k", [(10, 5), (20, 10), (30, 12)])
    def test_isomorphic(self, n, k):
        G, _ = generate_planted_clique(n, k, seed=42)
        G_s, _, _ = scramble_graph(G, seed=99)
        assert nx.is_isomorphic(G, G_s)

    @pytest.mark.parametrize("n, k", [(10, 5), (20, 10), (30, 12)])
    def test_perm_inverses(self, n, k):
        G, _ = generate_planted_clique(n, k, seed=42)
        _, fwd, inv = scramble_graph(G, seed=99)
        for old, new in fwd.items():
            assert inv[new] == old

    @pytest.mark.parametrize("n, k", [(10, 5), (20, 10), (30, 12)])
    def test_unscramble_roundtrip(self, n, k):
        G, planted = generate_planted_clique(n, k, seed=42)
        A = motzkin_straus_adjacency(G)
        planted_0 = [v - 1 for v in planted]
        y_orig = scaled_optimal_y(planted_0, n, k)

        _, fwd, inv = scramble_graph(G, seed=99)
        # Build scrambled y: map original positions to scrambled positions
        y_scrambled = np.zeros(n)
        for old_1 in range(1, n + 1):
            new_1 = fwd[old_1]
            y_scrambled[new_1 - 1] = y_orig[old_1 - 1]

        y_recovered = unscramble_solution(y_scrambled, inv)
        np.testing.assert_array_almost_equal(y_recovered, y_orig)

    @pytest.mark.parametrize("n, k", [(10, 5), (20, 10), (30, 12)])
    def test_omega_unchanged(self, n, k):
        G, _ = generate_planted_clique(n, k, seed=42)
        G_s, _, _ = scramble_graph(G, seed=99)
        omega_orig = max(len(c) for c in nx.find_cliques(G))
        omega_scram = max(len(c) for c in nx.find_cliques(G_s))
        assert omega_orig == omega_scram

    def test_deterministic(self):
        G, _ = generate_planted_clique(20, 10, seed=42)
        G1, fwd1, _ = scramble_graph(G, seed=99)
        G2, fwd2, _ = scramble_graph(G, seed=99)
        assert fwd1 == fwd2
        assert set(G1.edges()) == set(G2.edges())


class TestCliqueDistribution:
    def test_output_keys(self):
        G, _ = generate_planted_clique(30, 12, seed=42)
        dist = compute_clique_distribution(G)
        expected_keys = {
            "clique_sizes", "objectives", "max_clique_size",
            "num_maximal_cliques", "size_counts",
        }
        assert set(dist.keys()) == expected_keys

    def test_max_clique_size(self):
        G, _ = generate_planted_clique(30, 12, seed=42)
        dist = compute_clique_distribution(G)
        assert dist["max_clique_size"] == 12

    def test_monotonicity(self):
        """Larger cliques have strictly larger objectives."""
        G, _ = generate_planted_clique(30, 10, seed=42)
        dist = compute_clique_distribution(G)
        sizes = sorted(dist["size_counts"].keys())
        for i in range(len(sizes) - 1):
            g1 = omega_to_scaled_objective(sizes[i])
            g2 = omega_to_scaled_objective(sizes[i + 1])
            assert g2 > g1


class TestBruteForceBoson14:
    """Integration tests: generate planted graph, verify scaled solution."""

    @pytest.mark.parametrize(
        "n, k, seed",
        [
            (10, 5, 1),
            (15, 7, 2),
            (20, 10, 3),
            (30, 12, 4),
            (40, 15, 5),
            (50, 15, 6),
        ],
    )
    def test_planted_solution(self, n, k, seed):
        R = 100
        G, planted = generate_planted_clique(n, k, seed=seed)
        A = motzkin_straus_adjacency(G)
        planted_0 = [v - 1 for v in planted]

        # Verify optimal y
        y = scaled_optimal_y(planted_0, n, k, R)
        g = scaled_objective(y, A)
        g_theory = omega_to_scaled_objective(k, R)
        assert g == pytest.approx(g_theory)

        # Verify energy
        E = hardware_energy(y, A)
        assert E == pytest.approx(-2.0 * g_theory)

        # Brute-force clique number
        omega_bf = max(len(c) for c in nx.find_cliques(G))
        assert omega_bf == k

        # Planted vertices appear in at least one max clique
        max_cliques = [c for c in nx.find_cliques(G) if len(c) == k]
        planted_set = set(planted)
        assert any(planted_set == set(c) for c in max_cliques)

    @pytest.mark.parametrize(
        "n, k, seed",
        [
            (10, 5, 1),
            (15, 7, 2),
            (20, 10, 3),
            (30, 12, 4),
            (40, 15, 5),
            (50, 15, 6),
        ],
    )
    def test_omega_recovery(self, n, k, seed):
        R = 100
        G, planted = generate_planted_clique(n, k, seed=seed)
        A = motzkin_straus_adjacency(G)
        planted_0 = [v - 1 for v in planted]
        y = scaled_optimal_y(planted_0, n, k, R)

        # From scaled objective
        g = scaled_objective(y, A)
        assert scaled_objective_to_omega(g, R) == k

        # From hardware energy
        E = hardware_energy(y, A)
        assert energy_to_omega(E, R) == k


class TestComputeAllMaxCliqueSolutions:
    """Tests for compute_all_max_clique_solutions."""

    @pytest.mark.parametrize(
        "n, k, seed",
        [(10, 5, 1), (15, 7, 2), (20, 10, 3), (30, 12, 4)],
    )
    def test_shape_and_omega(self, n, k, seed):
        G, _ = generate_planted_clique(n, k, seed=seed)
        y_solutions, omega = compute_all_max_clique_solutions(G)
        assert omega == k
        assert y_solutions.ndim == 2
        assert y_solutions.shape[0] >= 1
        assert y_solutions.shape[1] == n

    @pytest.mark.parametrize(
        "n, k, seed",
        [(10, 5, 1), (15, 7, 2), (20, 10, 3), (30, 12, 4)],
    )
    def test_rows_sum_to_R(self, n, k, seed):
        R = 100
        G, _ = generate_planted_clique(n, k, seed=seed)
        y_solutions, _ = compute_all_max_clique_solutions(G, R=R)
        for i in range(y_solutions.shape[0]):
            assert np.sum(y_solutions[i]) == pytest.approx(R)

    @pytest.mark.parametrize(
        "n, k, seed",
        [(10, 5, 1), (15, 7, 2), (20, 10, 3), (30, 12, 4)],
    )
    def test_each_row_achieves_optimal_objective(self, n, k, seed):
        R = 100
        G, _ = generate_planted_clique(n, k, seed=seed)
        A = motzkin_straus_adjacency(G)
        y_solutions, omega = compute_all_max_clique_solutions(G, R=R)
        g_star = omega_to_scaled_objective(omega, R)
        for i in range(y_solutions.shape[0]):
            g = scaled_objective(y_solutions[i], A)
            assert g == pytest.approx(g_star)

    @pytest.mark.parametrize(
        "n, k, seed",
        [(10, 5, 1), (15, 7, 2), (20, 10, 3), (30, 12, 4)],
    )
    def test_each_row_support_is_clique(self, n, k, seed):
        G, _ = generate_planted_clique(n, k, seed=seed)
        y_solutions, _ = compute_all_max_clique_solutions(G)
        for i in range(y_solutions.shape[0]):
            support = [int(j) + 1 for j in np.where(y_solutions[i] > 1e-10)[0]]
            # Every pair in support must be an edge
            for a in range(len(support)):
                for b in range(a + 1, len(support)):
                    assert G.has_edge(support[a], support[b])

    @pytest.mark.parametrize(
        "n, k, nc",
        [(30, 10, 2), (40, 10, 3), (50, 12, 2)],
    )
    def test_degenerate_has_at_least_num_cliques_rows(self, n, k, nc):
        G, _ = generate_degenerate_planted(n, k, nc, seed=42)
        y_solutions, omega = compute_all_max_clique_solutions(G)
        assert omega == k
        assert y_solutions.shape[0] >= nc
