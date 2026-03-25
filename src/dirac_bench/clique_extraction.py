"""Clique extraction and refinement methods for Motzkin-Straus x-vectors.

Given continuous x-vectors from a Motzkin-Straus QP solver (e.g. Dirac-3),
these methods extract discrete cliques using various strategies:
greedy pruning, threshold sweeping, randomized rounding, local search,
and clustering.

All is_clique checks use the ORIGINAL adjacency matrix A, not the
Bomze-regularized A_bar.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2

from dirac_bench.problems import extract_support, is_clique


# ── Core helper ──────────────────────────────────────────────────────────


def _greedy_prune_to_clique(candidates: list[int], A: np.ndarray) -> list[int]:
    """Greedily build a clique from an ordered candidate list.

    Process candidates in order; add a node only if it is adjacent to
    every node already in the clique.

    Args:
        candidates: Ordered list of node indices (0-based).
        A: Original adjacency matrix (n x n).

    Returns:
        List of node indices forming a clique.
    """
    clique: list[int] = []
    for v in candidates:
        if all(A[v, u] != 0 for u in clique):
            clique.append(v)
    return clique


# ── Greedy methods ───────────────────────────────────────────────────────


def greedy_clique_desc(
    x: np.ndarray,
    A: np.ndarray,
    threshold: float = 1e-4,
) -> list[int]:
    """Extract clique by processing support nodes in descending x_i order."""
    support = extract_support(x, threshold=threshold)
    candidates = sorted(support, key=lambda i: x[i], reverse=True)
    return _greedy_prune_to_clique(candidates, A)


def greedy_clique_asc(
    x: np.ndarray,
    A: np.ndarray,
    threshold: float = 1e-4,
) -> list[int]:
    """Extract clique by processing support nodes in ascending x_i order."""
    support = extract_support(x, threshold=threshold)
    candidates = sorted(support, key=lambda i: x[i])
    return _greedy_prune_to_clique(candidates, A)


def greedy_clique_random(
    x: np.ndarray,
    A: np.ndarray,
    threshold: float = 1e-4,
    n_trials: int = 50,
    seed: int = 42,
) -> list[int]:
    """Extract clique via multiple random-order greedy trials.

    Returns the largest clique found across all trials.
    """
    rng = np.random.default_rng(seed)
    support = extract_support(x, threshold=threshold)
    if not support:
        return []

    best: list[int] = []
    arr = np.array(support)
    for _ in range(n_trials):
        perm = rng.permutation(arr)
        clique = _greedy_prune_to_clique(perm.tolist(), A)
        if len(clique) > len(best):
            best = clique
    return best


# ── Threshold sweep ──────────────────────────────────────────────────────


def threshold_sweep_extract(
    x: np.ndarray,
    A: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict[float, list[int]]:
    """Run greedy-desc extraction at multiple thresholds.

    Returns:
        Dict mapping threshold -> best clique found at that threshold.
    """
    if thresholds is None:
        thresholds = [1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2]

    results: dict[float, list[int]] = {}
    for t in thresholds:
        results[t] = greedy_clique_desc(x, A, threshold=t)
    return results


# ── Top-K extraction ─────────────────────────────────────────────────────


def top_k_extract(
    x: np.ndarray,
    A: np.ndarray,
    k_range: list[int] | None = None,
    known_omega: int | None = None,
) -> dict[int, list[int]]:
    """Extract cliques by taking top-k vertices by x_i value.

    Args:
        x: Continuous solution vector.
        A: Original adjacency matrix.
        k_range: List of k values to try. If None, auto-generated from
            known_omega or based on support size.
        known_omega: Known clique number (used to set k range).

    Returns:
        Dict mapping k -> clique extracted from top-k vertices.
    """
    n = len(x)
    if k_range is None:
        if known_omega is not None:
            base = known_omega
            k_range = list(range(max(2, base - 5), min(n + 1, base + 15)))
        else:
            support_size = len(extract_support(x, threshold=1e-4))
            k_range = list(range(2, min(n + 1, max(support_size + 5, 20))))

    ranked = np.argsort(x)[::-1]  # descending
    results: dict[int, list[int]] = {}
    for k in k_range:
        top_k = ranked[:k].tolist()
        results[k] = _greedy_prune_to_clique(top_k, A)
    return results


# ── Randomized rounding ──────────────────────────────────────────────────


def randomized_rounding_extract(
    x: np.ndarray,
    A: np.ndarray,
    n_trials: int = 200,
    seed: int = 42,
) -> list[int]:
    """Extract clique via randomized rounding: sample nodes with Bernoulli(α·x_i).

    The scaling α is chosen so that the expected number of selected nodes
    equals the support size.
    """
    rng = np.random.default_rng(seed)
    support = extract_support(x, threshold=1e-4)
    if not support:
        return []

    # Scale so expected selections ≈ support size
    x_max = x.max()
    if x_max <= 0:
        return []
    alpha = min(1.0 / x_max, len(support))

    best: list[int] = []
    for _ in range(n_trials):
        probs = np.clip(alpha * x, 0.0, 1.0)
        selected = np.where(rng.random(len(x)) < probs)[0]
        # Sort selected by x_i descending for greedy
        selected_sorted = sorted(selected.tolist(), key=lambda i: x[i], reverse=True)
        clique = _greedy_prune_to_clique(selected_sorted, A)
        if len(clique) > len(best):
            best = clique
    return best


# ── Local search: 1-swap ─────────────────────────────────────────────────


def local_search_1swap(
    seed_clique: list[int],
    A: np.ndarray,
    x_weights: np.ndarray | None = None,
    max_iters: int = 1000,
) -> list[int]:
    """Improve a clique via grow-then-swap local search.

    Phase 1 (grow): Try to add vertices adjacent to all clique members,
    preferring high x_weight.
    Phase 2 (swap): For each non-clique vertex v, check if removing one
    clique member and adding v yields a net improvement (by x_weight sum).

    Args:
        seed_clique: Initial clique (must be valid).
        A: Original adjacency matrix.
        x_weights: Weight per vertex (e.g. x_i values). If None, uniform.
        max_iters: Maximum swap iterations.

    Returns:
        Improved clique.
    """
    n = A.shape[0]
    if x_weights is None:
        x_weights = np.ones(n)

    clique = list(seed_clique)
    in_clique = set(clique)

    # Phase 1: greedy grow
    # Candidates sorted by x_weight descending
    candidates = sorted(
        [v for v in range(n) if v not in in_clique],
        key=lambda v: x_weights[v],
        reverse=True,
    )
    for v in candidates:
        if all(A[v, u] != 0 for u in clique):
            clique.append(v)
            in_clique.add(v)

    # Phase 2: 1-swap improvement
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for v in range(n):
            if v in in_clique:
                continue
            # Find which clique members v is NOT adjacent to
            non_adj = [u for u in clique if A[v, u] == 0]
            if len(non_adj) == 1:
                u = non_adj[0]
                # Swap u -> v if v has higher weight
                if x_weights[v] > x_weights[u]:
                    clique.remove(u)
                    in_clique.discard(u)
                    clique.append(v)
                    in_clique.add(v)
                    improved = True
                    break  # restart scan

    return clique


# ── Local search: 2-swap ─────────────────────────────────────────────────


def local_search_2swap(
    seed_clique: list[int],
    A: np.ndarray,
    x_weights: np.ndarray | None = None,
    max_iters: int = 500,
) -> list[int]:
    """Improve a clique via 2-swap local search.

    For each pair of clique members, try removing both and greedily
    refilling from remaining candidates (sorted by x_weight desc).
    Accept if the new clique is strictly larger.

    Args:
        seed_clique: Initial clique (must be valid).
        A: Original adjacency matrix.
        x_weights: Weight per vertex. If None, uniform.
        max_iters: Maximum outer iterations.

    Returns:
        Improved clique.
    """
    n = A.shape[0]
    if x_weights is None:
        x_weights = np.ones(n)

    clique = list(seed_clique)
    best_size = len(clique)

    for _ in range(max_iters):
        found_improvement = False
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                # Remove pair (clique[i], clique[j])
                removed = {clique[i], clique[j]}
                base = [v for v in clique if v not in removed]

                # Greedily refill from all non-base vertices
                candidates = sorted(
                    [v for v in range(n) if v not in set(base)],
                    key=lambda v: x_weights[v],
                    reverse=True,
                )
                new_clique = _greedy_prune_to_clique(base + candidates, A)

                if len(new_clique) > best_size:
                    clique = new_clique
                    best_size = len(new_clique)
                    found_improvement = True
                    break
            if found_improvement:
                break
        if not found_improvement:
            break

    return clique


# ── Clustering ───────────────────────────────────────────────────────────


def cluster_and_extract(
    x_vectors: np.ndarray,
    A: np.ndarray,
    n_clusters: int = 5,
    seed: int = 42,
) -> list[int]:
    """Cluster x-vectors with k-means, then extract cliques from each cluster.

    For each cluster, compute the centroid and the element-wise max, then
    run greedy-desc on both. Returns the largest clique found.

    Args:
        x_vectors: Array of shape (num_samples, n).
        A: Original adjacency matrix.
        n_clusters: Number of k-means clusters.
        seed: Random seed.

    Returns:
        Best clique found across all clusters.
    """
    if x_vectors.ndim == 1:
        x_vectors = x_vectors.reshape(1, -1)

    k = min(n_clusters, len(x_vectors))
    if k < 1:
        return []

    centroids, labels = kmeans2(x_vectors, k, minit="points", seed=seed)

    best: list[int] = []

    for c in range(k):
        mask = labels == c
        if not mask.any():
            continue

        cluster_vecs = x_vectors[mask]

        # Try centroid
        centroid = centroids[c]
        clique = greedy_clique_desc(centroid, A, threshold=1e-4)
        if len(clique) > len(best):
            best = clique

        # Try element-wise max across cluster
        elem_max = cluster_vecs.max(axis=0)
        clique = greedy_clique_desc(elem_max, A, threshold=1e-4)
        if len(clique) > len(best):
            best = clique

        # Try the sample with highest L1-norm (most "peaked")
        norms = np.sum(cluster_vecs ** 2, axis=1)
        best_idx = np.argmax(norms)
        clique = greedy_clique_desc(cluster_vecs[best_idx], A, threshold=1e-4)
        if len(clique) > len(best):
            best = clique

    return best


# ── Master orchestrator ──────────────────────────────────────────────────


def run_all_extractions(
    x_vectors: np.ndarray,
    A: np.ndarray,
    known_omega: int | None = None,
    seed: int = 42,
) -> dict:
    """Run all extraction methods on a set of x-vectors.

    Args:
        x_vectors: Array of shape (num_samples, n) from a QP solver.
        A: Original adjacency matrix (NOT Bomze-regularized).
        known_omega: Known clique number (optional, helps Top-K).
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping method name to result dict with keys:
        - clique: list[int] — best clique found
        - size: int — clique size
        - valid: bool — is_clique check on original A
    """
    if x_vectors.ndim == 1:
        x_vectors = x_vectors.reshape(1, -1)

    num_samples, n = x_vectors.shape
    results: dict = {}

    def _record(name: str, clique: list[int]) -> None:
        results[name] = {
            "clique": sorted(clique),
            "size": len(clique),
            "valid": is_clique(clique, A),
        }

    # ── Per-sample methods: pick the best across all samples ──

    best_by_method: dict[str, list[int]] = {
        "greedy_desc": [],
        "greedy_asc": [],
        "greedy_random": [],
        "randomized_rounding": [],
    }
    # threshold_sweep: track best per threshold, then best overall
    best_thresh: dict[float, list[int]] = {}
    # top_k: track best per k, then best overall
    best_topk: dict[int, list[int]] = {}

    for idx in range(num_samples):
        x = x_vectors[idx]

        # Greedy desc
        c = greedy_clique_desc(x, A)
        if len(c) > len(best_by_method["greedy_desc"]):
            best_by_method["greedy_desc"] = c

        # Greedy asc
        c = greedy_clique_asc(x, A)
        if len(c) > len(best_by_method["greedy_asc"]):
            best_by_method["greedy_asc"] = c

        # Greedy random
        c = greedy_clique_random(x, A, seed=seed + idx)
        if len(c) > len(best_by_method["greedy_random"]):
            best_by_method["greedy_random"] = c

        # Threshold sweep
        sweep = threshold_sweep_extract(x, A)
        for t, c in sweep.items():
            if t not in best_thresh or len(c) > len(best_thresh[t]):
                best_thresh[t] = c

        # Top-K
        topk = top_k_extract(x, A, known_omega=known_omega)
        for k, c in topk.items():
            if k not in best_topk or len(c) > len(best_topk[k]):
                best_topk[k] = c

        # Randomized rounding
        c = randomized_rounding_extract(x, A, seed=seed + idx)
        if len(c) > len(best_by_method["randomized_rounding"]):
            best_by_method["randomized_rounding"] = c

    # Record per-sample method results
    for name, clique in best_by_method.items():
        _record(name, clique)

    # Threshold sweep: best across all thresholds
    if best_thresh:
        best_t_clique = max(best_thresh.values(), key=len)
        _record("threshold_sweep", best_t_clique)
    else:
        _record("threshold_sweep", [])

    # Top-K: best across all k values
    if best_topk:
        best_k_clique = max(best_topk.values(), key=len)
        _record("top_k", best_k_clique)
    else:
        _record("top_k", [])

    # Clustering (uses all x_vectors together)
    c = cluster_and_extract(x_vectors, A, seed=seed)
    _record("cluster", c)

    # ── Local search refinement on best initial seeds ──

    # Collect unique seed cliques from initial methods
    initial_cliques = [r["clique"] for r in results.values() if r["size"] > 0]
    # Use x_weights from the sample with the best objective
    objectives = [0.5 * (x_vectors[i] @ A @ x_vectors[i]) for i in range(num_samples)]
    best_sample_idx = int(np.argmax(objectives))
    x_weights = x_vectors[best_sample_idx]

    # Apply 1-swap LS to the best initial clique
    if initial_cliques:
        best_initial = max(initial_cliques, key=len)
        c = local_search_1swap(best_initial, A, x_weights=x_weights, max_iters=1000)
        _record("local_search_1swap", c)

        c = local_search_2swap(best_initial, A, x_weights=x_weights, max_iters=500)
        _record("local_search_2swap", c)
    else:
        _record("local_search_1swap", [])
        _record("local_search_2swap", [])

    return results
