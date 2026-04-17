"""Scaled Motzkin-Straus formulation for Boson14 hardware benchmarks.

The standard Motzkin-Straus formulation uses sum(x)=1 and J=-0.5*A.
For hardware testing we scale to sum(y)=R (default R=100) so the coupling
matrix J=-A has integer entries {0, -1}.

Substitution y = R*x:
    g(y)  = 0.5 * y^T A y = R^2 * f(x)
    g*    = (R^2/2)(1 - 1/w)
    E*    = -R^2 (1 - 1/w)          (hardware minimises y^T J y with J=-A)
    w     = R^2 / (R^2 - 2*g*)  =  R^2 / (R^2 + E*)

This module also provides degenerate planted cliques, index scrambling,
and clique-size distribution analysis.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Scaled Motzkin-Straus objective helpers
# ---------------------------------------------------------------------------

def scaled_objective(y: np.ndarray, A: np.ndarray) -> float:
    """g(y) = 0.5 * y^T A y"""
    return float(0.5 * (y @ A @ y))


def scaled_objective_to_omega(g_star: float, R: int = 100) -> int:
    """w = round(R^2 / (R^2 - 2*g*))"""
    denom = R * R - 2.0 * g_star
    if abs(denom) < 1e-12:
        return 1
    return round(R * R / denom)


def omega_to_scaled_objective(omega: int, R: int = 100) -> float:
    """g* = (R^2/2)(1 - 1/w)"""
    if omega <= 1:
        return 0.0
    return (R * R / 2.0) * (1.0 - 1.0 / omega)


def scaled_optimal_y(
    planted_nodes_0based: list[int],
    n: int,
    omega: int,
    R: int = 100,
) -> np.ndarray:
    """y_i = R/w for planted vertices, 0 otherwise."""
    y = np.zeros(n)
    for i in planted_nodes_0based:
        y[i] = R / omega
    return y


def hardware_energy(y: np.ndarray, A: np.ndarray) -> float:
    """E = -y^T A y = -2g(y)  (what hardware minimises with J=-A)."""
    return float(-(y @ A @ y))


def energy_to_omega(E: float, R: int = 100) -> int:
    """w = round(R^2 / (R^2 + E))"""
    denom = R * R + E
    if abs(denom) < 1e-12:
        return 1
    return round(R * R / denom)


def build_integer_qp(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (C, J) where C=zeros(n), J=-A (integer-valued)."""
    n = A.shape[0]
    C = np.zeros(n)
    J = -A.copy()
    return C, J


def to_polynomial_json(C: np.ndarray, J: np.ndarray) -> dict:
    """Convert (C, J) to eqc_models polynomial file_config format.

    Matches QuadraticModel.sparse convention:
    - Linear terms: index [0, i+1] for C[i] != 0
    - Quadratic terms: symmetrize J to upper-tri, then [i+1, j+1] for val != 0
    - All indices are 1-based; values are Python float

    For boson14 J = -A the symmetrized upper-tri values are -2.0 per edge
    (vs -1.0 in hw-benchmark-toolkit where J = -0.5*A).
    """
    n = len(np.squeeze(C))
    data = []

    # Linear terms
    for i in range(n):
        if C[i] != 0:
            data.append({"idx": [0, i + 1], "val": float(C[i])})

    # Quadratic terms — symmetrize J to upper triangular
    J_upper = np.triu(J) + np.tril(J, -1).T
    for i in range(n):
        for j in range(i, n):
            val = J_upper[i, j]
            if val != 0:
                data.append({"idx": [i + 1, j + 1], "val": float(val)})

    min_degree = 2
    max_degree = 2
    if any(d["idx"][0] == 0 for d in data):
        min_degree = 1

    return {
        "file_name": "QuadraticModel",
        "file_config": {
            "polynomial": {
                "num_variables": n,
                "max_degree": max_degree,
                "min_degree": min_degree,
                "data": data,
            }
        },
    }


def compute_all_max_clique_solutions(
    G: nx.Graph,
    R: int = 100,
) -> tuple[np.ndarray, int]:
    """Compute optimal scaled y vectors for all maximum cliques.

    Returns (y_solutions, omega) where y_solutions.shape == (num_max_cliques, n).
    """
    n = G.number_of_nodes()
    cliques = list(nx.find_cliques(G))
    omega = max(len(c) for c in cliques)
    max_cliques = [c for c in cliques if len(c) == omega]

    rows = []
    for clique in max_cliques:
        # Graph nodes are 1-based; convert to 0-based indices
        nodes_0 = sorted(v - 1 for v in clique)
        y = scaled_optimal_y(nodes_0, n, omega, R)
        rows.append(y)

    return np.array(rows), omega


# ---------------------------------------------------------------------------
# Degenerate planted cliques
# ---------------------------------------------------------------------------

def generate_degenerate_planted(
    n: int,
    k: int,
    num_cliques: int,
    p: float = 0.5,
    seed: int = 42,
    vertex_sets: list[list[int]] | None = None,
    sever_cross_edges: bool = True,
) -> tuple[nx.Graph, list[list[int]]]:
    """Generate G(n, p) with multiple planted k-cliques.

    Args:
        n: Number of vertices.
        k: Size of each planted clique.
        num_cliques: Number of planted cliques.
        p: Edge probability for base Erdos-Renyi graph.
        seed: Random seed.
        vertex_sets: Optional list of 1-based vertex groups.
            Each group must have exactly k vertices.
        sever_cross_edges: If True, remove all ER edges between vertices
            of different planted sets (prevents accidental larger cliques).

    Returns:
        (graph, planted_sets) where each set is a list of 1-based nodes.
    """
    total_planted = num_cliques * k
    if total_planted > n:
        raise ValueError(
            f"num_cliques*k = {total_planted} exceeds n = {n}"
        )

    if vertex_sets is not None:
        if len(vertex_sets) != num_cliques:
            raise ValueError(
                f"len(vertex_sets)={len(vertex_sets)} != num_cliques={num_cliques}"
            )
        all_verts = []
        for vs in vertex_sets:
            if len(vs) != k:
                raise ValueError(
                    f"Each vertex set must have {k} vertices, got {len(vs)}"
                )
            all_verts.extend(vs)
        if len(all_verts) != len(set(all_verts)):
            raise ValueError("Vertex sets must not contain duplicates")
        if not all(1 <= v <= n for v in all_verts):
            raise ValueError(f"All vertices must be in [1, {n}]")

    # 1. Generate G(n, p)
    G = nx.erdos_renyi_graph(n, p, seed=seed)

    # 2. Determine vertex sets (0-based internally)
    if vertex_sets is not None:
        sets_0based = [sorted(v - 1 for v in vs) for vs in vertex_sets]
    else:
        rng = np.random.default_rng(seed)
        pool = list(rng.permutation(n))
        sets_0based = []
        for i in range(num_cliques):
            group = sorted(int(v) for v in pool[i * k : (i + 1) * k])
            sets_0based.append(group)

    # 3. Plant cliques — add all pairwise edges within each group
    for group in sets_0based:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                G.add_edge(group[i], group[j])

    # 4. Sever cross-edges between different planted groups
    if sever_cross_edges:
        all_planted = set()
        group_of = {}
        for g_idx, group in enumerate(sets_0based):
            for v in group:
                all_planted.add(v)
                group_of[v] = g_idx
        edges_to_remove = []
        for u, v in G.edges():
            if u in all_planted and v in all_planted:
                if group_of[u] != group_of[v]:
                    edges_to_remove.append((u, v))
        G.remove_edges_from(edges_to_remove)

    # 5. Relabel to 1-based
    G = nx.relabel_nodes(G, {i: i + 1 for i in range(n)})
    planted_sets = [[v + 1 for v in group] for group in sets_0based]

    return G, planted_sets


# ---------------------------------------------------------------------------
# Index scrambling
# ---------------------------------------------------------------------------

def scramble_graph(
    G: nx.Graph,
    seed: int = 42,
) -> tuple[nx.Graph, dict[int, int], dict[int, int]]:
    """Randomly permute vertex labels.

    Returns:
        (G_scrambled, forward_perm, inverse_perm)
        forward_perm[old_1based] = new_1based
        inverse_perm[new_1based] = old_1based
    """
    nodes = sorted(G.nodes())
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(len(nodes)))
    # Map old 1-based -> new 1-based
    forward_perm = {nodes[i]: shuffled[i] + 1 for i in range(len(nodes))}
    inverse_perm = {v: k for k, v in forward_perm.items()}
    G_scrambled = nx.relabel_nodes(G, forward_perm)
    return G_scrambled, forward_perm, inverse_perm


def create_targeted_permutation(
    planted_nodes: list[int],
    n: int,
    target_positions: list[int],
    remaining_seed: int | None = None,
) -> tuple[dict[int, int], dict[int, int]]:
    """Build a bijective permutation mapping planted nodes to target positions.

    Args:
        planted_nodes: 1-based vertices of the planted clique.
        n: Total number of vertices (graph has nodes 1..n).
        target_positions: 1-based target positions for the planted clique.
        remaining_seed: If set, randomly shuffle non-planted vertex mapping.

    Returns:
        (forward_perm, inverse_perm) where
        forward_perm[old_1based] = new_1based.
    """
    if len(planted_nodes) != len(target_positions):
        raise ValueError(
            f"len(planted_nodes)={len(planted_nodes)} != "
            f"len(target_positions)={len(target_positions)}"
        )
    if len(set(target_positions)) != len(target_positions):
        raise ValueError("target_positions must not contain duplicates")
    if not all(1 <= t <= n for t in target_positions):
        raise ValueError(f"All target_positions must be in [1, {n}]")

    planted_set = set(planted_nodes)
    target_set = set(target_positions)

    non_planted = sorted(v for v in range(1, n + 1) if v not in planted_set)
    remaining_positions = sorted(v for v in range(1, n + 1) if v not in target_set)

    if remaining_seed is not None:
        rng = np.random.default_rng(remaining_seed)
        remaining_positions = [int(x) for x in rng.permutation(remaining_positions)]

    forward_perm = {}
    for old, new in zip(sorted(planted_nodes), sorted(target_positions)):
        forward_perm[old] = new
    for old, new in zip(non_planted, remaining_positions):
        forward_perm[old] = new

    inverse_perm = {v: k for k, v in forward_perm.items()}
    return forward_perm, inverse_perm


def unscramble_solution(
    y: np.ndarray,
    inverse_perm: dict[int, int],
) -> np.ndarray:
    """Map solution from scrambled to original vertex ordering.

    inverse_perm[new_1based] = old_1based.
    y is indexed 0-based (position i corresponds to 1-based vertex i+1).
    """
    n = len(y)
    y_orig = np.zeros(n)
    for new_1based in range(1, n + 1):
        old_1based = inverse_perm[new_1based]
        y_orig[old_1based - 1] = y[new_1based - 1]
    return y_orig


# ---------------------------------------------------------------------------
# Clique distribution analysis
# ---------------------------------------------------------------------------

def compute_clique_distribution(
    G: nx.Graph,
    R: int = 100,
) -> dict:
    """Enumerate all maximal cliques and compute distribution statistics.

    Returns dict with keys:
        clique_sizes: list of sizes
        objectives: list of scaled objectives g*(w) for each clique's size
        max_clique_size: maximum clique size found
        num_maximal_cliques: total count of maximal cliques
        size_counts: dict mapping size -> count
    """
    cliques = list(nx.find_cliques(G))
    sizes = [len(c) for c in cliques]
    objectives = [omega_to_scaled_objective(s, R) for s in sizes]

    size_counts: dict[int, int] = {}
    for s in sizes:
        size_counts[s] = size_counts.get(s, 0) + 1

    return {
        "clique_sizes": sizes,
        "objectives": objectives,
        "max_clique_size": max(sizes) if sizes else 0,
        "num_maximal_cliques": len(cliques),
        "size_counts": dict(sorted(size_counts.items())),
    }


def plot_clique_distribution(
    distribution: dict,
    graph_name: str,
    R: int = 100,
    known_omega: int | None = None,
    save_path: str | None = None,
    show: bool = False,
) -> Path | None:
    """Two-panel plot: clique-size histogram and objective histogram.

    Top panel: bar chart of maximal clique counts by size.
    Bottom panel: histogram of scaled objectives with vertical lines at
    g*(w) for each integer clique size w observed in the graph.

    Follows project plotting style (Agg backend, figsize=(10,10), #4C72B0,
    wheat annotation box, 150 dpi).
    """
    size_counts = distribution["size_counts"]
    sizes = sorted(size_counts.keys())
    counts = [size_counts[s] for s in sizes]
    objectives = distribution["objectives"]

    fig, (ax_size, ax_obj) = plt.subplots(2, 1, figsize=(10, 10))

    # --- Top panel: clique size histogram ---
    ax_size.bar(sizes, counts, color="#4C72B0", edgecolor="black", alpha=0.7,
                label="Maximal cliques")
    ax_size.set_xlabel("Clique size")
    ax_size.set_ylabel("Count")
    title = f"{graph_name}: Clique Size Distribution"
    subtitle = f"Maximal cliques: {distribution['num_maximal_cliques']}, " \
               f"Max size: {distribution['max_clique_size']}"
    if known_omega is not None:
        subtitle += f", Known w={known_omega}"
    ax_size.set_title(f"{title}\n{subtitle}")
    ax_size.legend(loc="upper right", fontsize=8)
    ax_size.grid(True, alpha=0.3)

    # --- Bottom panel: objective histogram with omega lines ---
    ax_obj.hist(objectives, bins=25, edgecolor="black", alpha=0.7,
                color="#4C72B0", label="Maximal clique objectives")
    ax_obj.set_xlabel(rf"Scaled objective $g = \frac{{1}}{{2}} y^\top A\, y$")
    ax_obj.set_ylabel("Frequency")

    # Vertical lines at g*(w) for each integer omega observed
    y_max = ax_obj.get_ylim()[1]
    for w in sizes:
        g_w = omega_to_scaled_objective(w, R)
        is_max = (w == distribution["max_clique_size"])
        is_known = (known_omega is not None and w == known_omega)
        if is_known:
            color = "#2ca02c"
            style = "-"
        elif is_max:
            color = "#d62728"
            style = "-"
        else:
            color = "#7f7f7f"
            style = "--"
        ax_obj.axvline(x=g_w, color=color, linestyle=style,
                       alpha=0.8, linewidth=1.5)
        ax_obj.text(g_w, y_max * 0.92, f" w={w}",
                    rotation=90, va="bottom", color=color,
                    fontsize=9, fontweight="bold")

    # Formula annotation
    formula = r"$g^*(\omega) = \frac{R^2}{2}\left(1 - \frac{1}{\omega}\right)$"
    ax_obj.text(0.05, 0.95, formula, transform=ax_obj.transAxes,
                fontsize=11, va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))

    ax_obj.set_title(f"{graph_name}: Objective Distribution (R={R})")
    ax_obj.legend(loc="upper left", fontsize=8)
    ax_obj.grid(True, alpha=0.3)

    plt.tight_layout()

    saved_path = None
    if save_path is not None:
        out_dir = Path(save_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"clique_dist_{graph_name}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved distribution plot -> {fname}")
        saved_path = fname

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


# ---------------------------------------------------------------------------
# Scaled Dirac histogram
# ---------------------------------------------------------------------------

def plot_scaled_dirac_histogram(
    objectives: list[float],
    graph_name: str,
    computed_omega: int,
    R: int = 100,
    known_omega: int | None = None,
    save_path: str | None = None,
    show: bool = False,
) -> Path | None:
    """Histogram of scaled objectives g(y) from Dirac-3 samples.

    Vertical lines at g*(w) = (R^2/2)(1 - 1/w) for nearby omega values.
    Style matches project conventions: #4C72B0, wheat box, 150 dpi.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(objectives, bins=25, edgecolor="black", alpha=0.7, color="#4C72B0",
            label=f"Dirac samples (n={len(objectives)})")

    best_obj = max(objectives)
    ax.axvline(x=best_obj, color="red", linestyle="--", linewidth=2,
               label=f"Best Dirac: {best_obj:.2f}")

    # Theoretical omega lines
    omegas_to_show = set()
    omegas_to_show.add(computed_omega)
    if known_omega is not None:
        omegas_to_show.add(known_omega)
    for delta in (-2, -1, 1, 2):
        omegas_to_show.add(computed_omega + delta)
    omegas_to_show = sorted(o for o in omegas_to_show if o >= 2)

    y_max = ax.get_ylim()[1]

    for omega in omegas_to_show:
        g_w = omega_to_scaled_objective(omega, R)
        style = "-" if omega == computed_omega else "--"
        color = "#2ca02c" if omega == known_omega else "#7f7f7f"
        if omega == computed_omega and omega == known_omega:
            color = "#2ca02c"
        elif omega == computed_omega:
            color = "#d62728"

        ax.axvline(x=g_w, color=color, linestyle=style,
                   alpha=0.8, linewidth=1.5)
        ax.text(g_w, y_max * 0.92, f" w={omega}",
                rotation=90, va="bottom", color=color,
                fontsize=9, fontweight="bold")

    # Formula annotation
    formula = r"$g^*(\omega) = \frac{R^2}{2}\left(1 - \frac{1}{\omega}\right)$"
    ax.text(0.05, 0.95, formula, transform=ax.transAxes,
            fontsize=11, va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))

    title = f"{graph_name}: Dirac-3 Scaled Objective Distribution"
    subtitle = f"Samples: {len(objectives)}, R={R}, Computed w={computed_omega}"
    if known_omega is not None:
        subtitle += f", Known w={known_omega}"
        match_str = "MATCH" if computed_omega == known_omega else "MISMATCH"
        subtitle += f" [{match_str}]"
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel(rf"Scaled objective $g = \frac{{1}}{{2}} y^\top A\, y$")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    saved_path = None
    if save_path is not None:
        out_dir = Path(save_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"scaled_histogram_{graph_name}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved scaled histogram -> {fname}")
        saved_path = fname

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path
