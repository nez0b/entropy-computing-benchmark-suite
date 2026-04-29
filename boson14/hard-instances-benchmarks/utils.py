"""I/O helpers for the hard-instances-benchmarks pipeline.

Handles:
- Building hard instances via `_internal.generators`
- Creating 3 permutation variants (front/end/random) per instance
- Writing input CSVs in the format `run_boson14.py` expects
- Post-processing NPZ outputs (add amplitude, num_loops, R fields)
- Renaming results__*.npz/plot__*.png to append `_a{amp}` suffix
- Phase markers for idempotent resume
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

# Reuse boson14_bench utilities (already in boson14/ parent directory)
# Import path: we add boson14/ to sys.path so boson14_bench is importable.
_BOSON14_DIR = Path(__file__).resolve().parents[1]
if str(_BOSON14_DIR) not in sys.path:
    sys.path.insert(0, str(_BOSON14_DIR))

from boson14_bench.core import (  # noqa: E402
    compute_clique_distribution,
    create_targeted_permutation,
    scramble_graph,
    compute_all_max_clique_solutions,
    scaled_objective,
    to_polynomial_json,
    omega_to_scaled_objective,
)
from boson14_bench.problems import motzkin_straus_adjacency  # noqa: E402

from _internal import generators as _gens  # noqa: E402
from _internal.bruteforce import solve_bruteforce  # noqa: E402


# Graphs with more than this many nodes are considered impractical for
# brute-force max-clique enumeration (exponential in n for dense graphs).
BRUTE_FORCE_MAX_N = 90

VARIANTS_PLANTED = ("front", "end", "random")


# ---------------------------------------------------------------------------
# Instance building
# ---------------------------------------------------------------------------

def build_instance(strategy: str, params: dict) -> tuple[np.ndarray, dict]:
    """Dispatch to the correct strategy generator.

    `strategy` is a single letter A/B/C/D/E.
    `params` carries the strategy-specific kwargs plus always-required `n` and `seed`.

    The underlying generators use 0-based node indices, but `boson14_bench`
    utilities expect 1-based. This function converts all node lists in the
    returned metadata to 1-based before returning.
    """
    n = params["n"]
    seed = params["seed"]
    p = params.get("p", 0.5)

    if strategy == "A":
        A, meta = _gens.strategy_a_near_threshold(
            n=n, p=p, seed=seed,
            k_multiplier=params.get("k_multiplier", 2.5),
        )
    elif strategy == "B":
        A, meta = _gens.strategy_b_dense_random(n=n, p=params.get("p", 0.9), seed=seed)
    elif strategy == "C":
        A, meta = _gens.strategy_c_degenerate(
            n=n, k=params["k"],
            num_cliques=params.get("num_cliques", 3),
            p=p, seed=seed,
        )
    elif strategy == "D":
        A, meta = _gens.strategy_d_camouflage(
            n=n, k=params["k"], p=p, seed=seed,
            removal_frac=params.get("removal_frac", 0.4),
        )
    elif strategy == "E":
        A, meta = _gens.strategy_e_overlap(
            n=n, k=params["k"], p=params.get("p", 0.3), seed=seed,
            overlap=params.get("overlap", params["k"] - 2),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Convert 0-based node IDs to 1-based for downstream consistency.
    if meta.get("planted_nodes") is not None:
        meta["planted_nodes"] = [int(v) + 1 for v in meta["planted_nodes"]]
    if meta.get("planted_sets") is not None:
        meta["planted_sets"] = [[int(v) + 1 for v in ps] for ps in meta["planted_sets"]]
    if meta.get("clique_s") is not None:
        meta["clique_s"] = [int(v) + 1 for v in meta["clique_s"]]
    if meta.get("clique_t") is not None:
        meta["clique_t"] = [int(v) + 1 for v in meta["clique_t"]]

    # For strategies where `planted_nodes` is not set directly, derive a reference:
    #   C: use the first planted set
    #   E: use the k-clique S (clique_s) as the reference
    if meta.get("planted_nodes") is None:
        if meta.get("planted_sets"):
            meta["planted_nodes"] = list(meta["planted_sets"][0])
        elif meta.get("clique_s"):
            meta["planted_nodes"] = list(meta["clique_s"])

    return A, meta


def instance_dir_name(meta: dict) -> str:
    """Build a canonical directory name from meta.

    Prefix "hard_" distinguishes from benchmark_* (from full-benchmark.py).
    """
    return f"hard_{meta['name']}"


# ---------------------------------------------------------------------------
# Permutation targets (mirrors full-benchmark.py)
# ---------------------------------------------------------------------------

def compute_variant_targets(
    reference_nodes: list[int],
    n: int,
    k: int,
    seed: int,
) -> dict[str, list[int]]:
    """Compute front/end/random target positions for the k-clique.

    `reference_nodes` are the 1-based positions of the clique in the base graph
    (planted for A/C/D/E, brute-forced for B with n <= 90).
    """
    front_targets = list(range(1, k + 1))
    end_targets = list(range(n - k + 1, n + 1))

    rng = np.random.default_rng(seed + 2000)
    reference_set = set(reference_nodes)
    for _ in range(100):
        random_targets = sorted(int(x) for x in rng.choice(n, size=k, replace=False) + 1)
        if set(random_targets) != reference_set:
            break

    return {"front": front_targets, "end": end_targets, "random": random_targets}


# ---------------------------------------------------------------------------
# Variant building (graph permutation + sanity check)
# ---------------------------------------------------------------------------

def _graph_from_adjacency(A: np.ndarray) -> nx.Graph:
    """Rebuild a 1-based NetworkX graph from a sorted-nodelist adjacency matrix."""
    n = A.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0:
                G.add_edge(i + 1, j + 1)
    return G


def build_planted_variant(
    G: nx.Graph,
    reference_nodes: list[int],
    n: int,
    k: int,
    target_positions: list[int],
    variant_name: str,
) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    """Permute G so the reference clique sits at target_positions.

    Returns (A_variant, forward_perm, inverse_perm).
    Verifies that the submatrix at target_positions is a complete subgraph
    (sanity check; raises RuntimeError on failure).
    """
    forward_perm, inverse_perm = create_targeted_permutation(
        reference_nodes, n, target_positions,
    )
    G_variant = nx.relabel_nodes(G, forward_perm)
    A_variant = motzkin_straus_adjacency(G_variant)

    # Sanity check: target positions (0-based) should be an all-ones minus-diagonal submatrix
    idx = sorted(t - 1 for t in target_positions)
    sub = A_variant[np.ix_(idx, idx)]
    expected = np.ones((k, k)) - np.eye(k)
    if not np.allclose(sub, expected):
        raise RuntimeError(
            f"Variant '{variant_name}' sanity check FAILED: submatrix at "
            f"target positions {target_positions} is not a complete subgraph. "
            f"Permutation or reference clique is wrong."
        )
    return A_variant, forward_perm, inverse_perm


def build_random_variant(
    G: nx.Graph,
    seed: int,
) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    """Random vertex permutation (used for B with n > BRUTE_FORCE_MAX_N)."""
    G_scrambled, forward_perm, inverse_perm = scramble_graph(G, seed=seed)
    A_variant = motzkin_straus_adjacency(G_scrambled)
    return A_variant, forward_perm, inverse_perm


# ---------------------------------------------------------------------------
# Writing variant artifacts (CSV, JSON, meta)
# ---------------------------------------------------------------------------

def write_variant_artifacts(
    variant_dir: Path,
    variant_name: str,
    A_variant: np.ndarray,
    forward_perm: dict[int, int],
    inverse_perm: dict[int, int],
    base_meta: dict,
    target_positions: list[int] | None,
    reference_nodes: list[int],
    k: int,
    R: int,
) -> None:
    """Write {variant}_meta.json, {variant}_boson14.csv, {variant}_boson14.json.

    CSV format matches full-benchmark.py: [C_zeros | A_variant], one row per vertex.
    `target_positions` is None for random-only variants (B with n > 90).
    """
    variant_dir.mkdir(parents=True, exist_ok=True)
    n = A_variant.shape[0]

    # Metadata
    meta = {
        **base_meta,
        "variant": variant_name,
        "target_positions": target_positions,
        "reference_nodes": reference_nodes,
        "forward_perm": {str(k_): int(v) for k_, v in forward_perm.items()},
        "inverse_perm": {str(k_): int(v) for k_, v in inverse_perm.items()},
    }
    meta_path = variant_dir / f"{variant_name}_meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    # CSV: [C_zeros | A] — the input format that run_boson14.py expects
    C_zero = np.zeros(n, dtype=np.float64)
    csv_data = np.column_stack([C_zero, A_variant])
    csv_path = variant_dir / f"{variant_name}_boson14.csv"
    np.savetxt(csv_path, csv_data, delimiter=",", fmt="%g")

    # JSON polynomial (for Dirac-3 compatibility; not used by boson14 hardware)
    poly = to_polynomial_json(C_zero, A_variant)
    payload = {
        "file": poly,
        "job_params": {
            "device_type": "boson14",
            "num_samples": 1000,
            "sum_constraint": R,
        },
        "graph_info": {
            "name": f"{base_meta['instance_name']}_{variant_name}",
            "variant": variant_name,
            "target_positions": target_positions,
            "reference_nodes": reference_nodes,
            "n": n, "k": k, "R": R,
            "known_omega": k,
            "g_star": base_meta.get("g_star"),
        },
    }
    with (variant_dir / f"{variant_name}_boson14.json").open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_theoretical_solutions(
    variant_dir: Path,
    variant_name: str,
    G_variant: nx.Graph,
    R: int,
) -> dict:
    """Compute all max-clique solutions for G_variant and save to NPZ.

    Returns {'omega_verified': int, 'num_max_cliques': int}.
    """
    y_solutions, omega_verified = compute_all_max_clique_solutions(G_variant, R)
    npz_path = variant_dir / f"{variant_name}_solutions.npz"
    np.savez(npz_path, y_solutions=y_solutions)
    return {
        "omega_verified": int(omega_verified),
        "num_max_cliques": int(y_solutions.shape[0]),
    }


# ---------------------------------------------------------------------------
# Hardware output post-processing
# ---------------------------------------------------------------------------

@dataclass
class HardwareFlags:
    """Flags forwarded to run_boson14.py and embedded into the NPZ."""
    amplitude: int
    R: int
    num_samples: int
    num_loops: int
    delay: int
    pulse_width: int
    distance_between_pulses: int
    solps: int

    def as_cli_args(self) -> list[str]:
        return [
            "-a", str(self.amplitude),
            "-R", str(self.R),
            "-s", str(self.num_samples),
            "-l", str(self.num_loops),
            "-d", str(self.delay),
            "-w", str(self.pulse_width),
            "-b", str(self.distance_between_pulses),
            "-solps", str(self.solps),
        ]


def find_latest_result_files(variant_dir: Path, short_filename: str) -> tuple[Path, Path | None]:
    """Find the most recently modified unrenamed results__*.npz + matching plot__.

    Filters to files that have NO `_a{digits}` suffix yet — those are the ones
    just written by run_boson14.py (pre-rename). Files already renamed are
    ignored so we don't mistakenly pick up a previous run's output.

    Returns (npz_path, plot_path_or_None). Raises FileNotFoundError if no NPZ found.
    """
    def _is_unrenamed(p: Path) -> bool:
        # Unrenamed ends with __YYYYMMDD_HHMMSS (no _a<digits> tail)
        tail = p.stem.rsplit("__", 1)[-1]
        return "_a" not in tail or not tail.rsplit("_a", 1)[-1].isdigit()

    npz_candidates = sorted(
        (p for p in variant_dir.glob(f"results__{short_filename}__*.npz") if _is_unrenamed(p)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not npz_candidates:
        raise FileNotFoundError(
            f"No unrenamed results__{short_filename}__*.npz found in {variant_dir} "
            f"after run_boson14.py completed."
        )
    npz_path = npz_candidates[0]

    # Plot file has the same stem pattern as the NPZ (shared timestamp)
    plot_path = npz_path.with_name(
        npz_path.name.replace("results__", "plot__").replace(".npz", ".png")
    )
    return npz_path, (plot_path if plot_path.exists() else None)


def postprocess_npz(
    npz_path: Path,
    flags: HardwareFlags,
    device_time_wrapper_s: float | None = None,
) -> None:
    """Load NPZ, add/overwrite hardware config fields, re-save.

    run_boson14.py saves R, num_samples, num_loops, delay but NOT amplitude.
    We overwrite all hardware config fields for an explicit contract.

    If `device_time_wrapper_s` is provided, it's saved as `device_time_wrapper_s`
    — the wall-clock time this pipeline measured around the subprocess call
    (includes subprocess spawn + hardware compute + result I/O, distinct from
    `computation_time` which `run_boson14.py` measures internally).
    """
    data = dict(np.load(npz_path, allow_pickle=True))
    data["amplitude"] = np.int64(flags.amplitude)
    data["num_loops"] = np.int64(flags.num_loops)
    data["R"] = np.int64(flags.R)
    data["num_samples"] = np.int64(flags.num_samples)
    data["delay"] = np.int64(flags.delay)
    data["pulse_width"] = np.int64(flags.pulse_width)
    data["distance_between_pulses"] = np.int64(flags.distance_between_pulses)
    if device_time_wrapper_s is not None:
        data["device_time_wrapper_s"] = np.float64(device_time_wrapper_s)
    np.savez(npz_path, **data)


def append_timing_entry(
    instance_dir: Path,
    variant_name: str,
    amplitude: int,
    device_time_s: float,
    timestamp: str,
    flags: HardwareFlags,
    result_filename: str,
) -> None:
    """Append one (variant, amplitude, wall-clock-time) entry to timing_summary.json.

    The timing_summary.json lives at the instance root and is the authoritative
    record of hardware runtime for downstream analysis (plots, reports).
    Idempotent: each (variant, amplitude) row is upserted so resumed pipelines
    don't duplicate entries.
    """
    timing_path = instance_dir / "timing_summary.json"
    if timing_path.exists():
        summary = json.loads(timing_path.read_text())
    else:
        summary = {"entries": []}

    entries = [
        e for e in summary["entries"]
        if not (e["variant"] == variant_name and e["amplitude"] == amplitude)
    ]
    entries.append({
        "variant": variant_name,
        "amplitude": amplitude,
        "device_time_wrapper_s": round(device_time_s, 3),
        "timestamp": timestamp,
        "result_filename": result_filename,
        "hardware_flags": {
            "R": flags.R,
            "num_samples": flags.num_samples,
            "num_loops": flags.num_loops,
            "delay": flags.delay,
            "pulse_width": flags.pulse_width,
            "distance_between_pulses": flags.distance_between_pulses,
        },
    })
    # Sort for stable diffs
    entries.sort(key=lambda e: (e["variant"], e["amplitude"]))
    summary["entries"] = entries

    # Roll up totals (helpful for quick inspection)
    summary["totals"] = {
        "num_runs": len(entries),
        "total_device_time_s": round(sum(e["device_time_wrapper_s"] for e in entries), 3),
    }

    with timing_path.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def rename_with_amplitude(
    npz_path: Path,
    plot_path: Path | None,
    amplitude: int,
) -> tuple[Path, Path | None]:
    """Rename results__*.npz and plot__*.png to append `_a{amplitude}`.

    If the file already has an `_a{N}` suffix, return it unchanged (idempotent).
    """
    def _rename(path: Path) -> Path:
        stem = path.stem  # without extension
        # Already has _a<digits> at the end?
        if "_a" in stem:
            tail = stem.rsplit("_a", 1)[1]
            if tail.isdigit():
                return path  # already renamed
        new_stem = f"{stem}_a{amplitude}"
        new_path = path.with_name(new_stem + path.suffix)
        path.rename(new_path)
        return new_path

    new_npz = _rename(npz_path)
    new_plot = _rename(plot_path) if plot_path is not None else None
    return new_npz, new_plot


# ---------------------------------------------------------------------------
# Phase markers (idempotency)
# ---------------------------------------------------------------------------

def phase_marker(instance_dir: Path, phase: str) -> Path:
    return instance_dir / f"phase_{phase}.done"


def mark_phase_done(instance_dir: Path, phase: str) -> None:
    phase_marker(instance_dir, phase).write_text(f"done at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def is_phase_done(instance_dir: Path, phase: str) -> bool:
    return phase_marker(instance_dir, phase).exists()


# ---------------------------------------------------------------------------
# Hardware sample analysis (used at end of hardware phase for plotting)
# ---------------------------------------------------------------------------

def _threshold_support(y: np.ndarray, R: int, k_ref: int, factor: float = 2.0) -> list[int]:
    """Support = {i : y_i > R/(factor*k_ref)}. Returns 0-based indices."""
    thr = R / (factor * max(k_ref, 1))
    return [int(i) for i in np.where(y > thr)[0]]


def _is_clique(support_0based: list[int], A: np.ndarray) -> bool:
    """True iff every pair (i,j) in support has A[i,j] != 0."""
    for a in range(len(support_0based)):
        for b in range(a + 1, len(support_0based)):
            if A[support_0based[a], support_0based[b]] == 0:
                return False
    return True


def analyze_hardware_samples(
    sigma_array: np.ndarray,
    A: np.ndarray,
    R: int,
    k_ref: int,
) -> dict:
    """For each hardware solution vector, compute raw + repaired stats.

    Returns dict with:
        raw_objectives:        list[float]  — g(y) for each sample
        raw_omegas:            list[int]    — omega implied by g(y) via MS inversion
        support_sizes:         list[int]    — |{i: y_i > threshold}| per sample
        supp_is_clique:        list[bool]   — whether support is a valid clique
        repaired_sizes:        list[int]    — |S| if supp is clique, else 0
        repaired_objectives:   list[float]  — (R²/2)(1 - 1/|S|) if valid, else NaN
        fraction_valid:        float        — # valid clique supports / # samples
        clique_size_counts:    dict[int,int] — histogram over valid support sizes
    """
    raw_objectives: list[float] = []
    raw_omegas: list[int] = []
    support_sizes: list[int] = []
    supp_is_clique: list[bool] = []
    repaired_sizes: list[int] = []
    repaired_objectives: list[float] = []
    clique_size_counts: dict[int, int] = {}

    from collections import Counter  # local import for isolation

    for y in sigma_array:
        g = scaled_objective(y, A)
        raw_objectives.append(float(g))
        denom = R * R - 2.0 * g
        if abs(denom) < 1e-12:
            raw_omegas.append(1)
        else:
            raw_omegas.append(int(round(R * R / denom)))

        supp = _threshold_support(y, R, k_ref)
        support_sizes.append(len(supp))
        is_cq = _is_clique(supp, A) and len(supp) >= 2
        supp_is_clique.append(bool(is_cq))

        if is_cq:
            s = len(supp)
            repaired_sizes.append(s)
            repaired_objectives.append(float((R * R / 2.0) * (1.0 - 1.0 / s)))
            clique_size_counts[s] = clique_size_counts.get(s, 0) + 1
        else:
            repaired_sizes.append(0)
            repaired_objectives.append(float("nan"))

    n_samples = len(sigma_array)
    fraction_valid = (sum(supp_is_clique) / n_samples) if n_samples else 0.0

    return {
        "raw_objectives": raw_objectives,
        "raw_omegas": raw_omegas,
        "support_sizes": support_sizes,
        "supp_is_clique": supp_is_clique,
        "repaired_sizes": repaired_sizes,
        "repaired_objectives": repaired_objectives,
        "fraction_valid": fraction_valid,
        "clique_size_counts": dict(sorted(clique_size_counts.items())),
    }


def plot_hardware_distribution(
    analysis: dict,
    bruteforce_dist: dict,
    graph_name: str,
    variant_name: str,
    amplitude: int,
    R: int,
    known_omega: int | None,
    save_path: Path,
) -> Path:
    """Two-panel figure comparing hardware samples to the brute-force landscape.

    Top panel:    valid-clique size histogram (hardware) stacked next to the
                  brute-force maximal-clique size histogram for the same graph.
    Bottom panel: objective distributions — raw g(y) (hardware),
                  repaired g_eq (hardware, valid-clique samples only),
                  brute-force maximal-clique objectives — with vertical
                  reference lines at g*(omega) for each observed clique size.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path.mkdir(parents=True, exist_ok=True)
    fname = save_path / f"hardware_dist_{variant_name}_a{amplitude}.png"

    fig, (ax_size, ax_obj) = plt.subplots(2, 1, figsize=(10, 10))

    # ---- top panel: clique size histograms (hardware vs. brute-force) ----
    hw_size_counts = analysis["clique_size_counts"]
    bf_size_counts = bruteforce_dist.get("size_counts", {})

    all_sizes = sorted(set(hw_size_counts) | set(bf_size_counts))
    if not all_sizes:
        all_sizes = [known_omega or 1]
    x = np.arange(len(all_sizes))
    width = 0.4
    hw_counts = [hw_size_counts.get(s, 0) for s in all_sizes]
    bf_counts = [bf_size_counts.get(s, 0) for s in all_sizes]

    ax_size.bar(x - width / 2, hw_counts, width, color="#4C72B0",
                edgecolor="black", alpha=0.85,
                label=f"Hardware (valid cliques, {sum(hw_counts)} samples)")
    ax_size.bar(x + width / 2, bf_counts, width, color="#DD8452",
                edgecolor="black", alpha=0.7,
                label=f"Brute-force maximal cliques ({sum(bf_counts)} total)")
    ax_size.set_xticks(x)
    ax_size.set_xticklabels([str(s) for s in all_sizes])
    ax_size.set_xlabel("Clique size")
    ax_size.set_ylabel("Count")
    title = (f"{graph_name} / {variant_name} / a={amplitude}: Clique Size Distribution")
    subtitle = (f"Valid-clique fraction: {analysis['fraction_valid']:.1%} "
                f"({sum(hw_counts)}/{len(analysis['raw_objectives'])} samples)")
    if known_omega is not None:
        subtitle += f"  |  Known ω = {known_omega}"
    ax_size.set_title(f"{title}\n{subtitle}")
    ax_size.legend(loc="upper right", fontsize=8)
    ax_size.grid(True, alpha=0.3, axis="y")

    # ---- bottom panel: objective distributions (raw + repaired + BF) ----
    raw_obj = np.array(analysis["raw_objectives"], dtype=float)
    rep_obj = np.array([o for o in analysis["repaired_objectives"] if not np.isnan(o)])
    bf_obj = np.array(bruteforce_dist.get("objectives", []), dtype=float)

    all_g = np.concatenate([
        raw_obj if raw_obj.size else np.array([0.0]),
        rep_obj if rep_obj.size else np.array([0.0]),
        bf_obj if bf_obj.size else np.array([0.0]),
    ])
    g_min = float(np.min(all_g)) if all_g.size else 0.0
    g_max = float(np.max(all_g)) if all_g.size else 1.0
    if g_max <= g_min:
        g_max = g_min + 1.0
    bins = np.linspace(g_min, g_max, 40)

    if raw_obj.size:
        ax_obj.hist(raw_obj, bins=bins, color="#4C72B0", alpha=0.55,
                    edgecolor="black", label=f"Hardware raw g(y)  (n={raw_obj.size})")
    if rep_obj.size:
        ax_obj.hist(rep_obj, bins=bins, color="#55A868", alpha=0.55,
                    edgecolor="black", label=f"Hardware repaired g_eq  (n={rep_obj.size})")
    if bf_obj.size:
        ax_obj.hist(bf_obj, bins=bins, color="#DD8452", alpha=0.45,
                    edgecolor="black", label=f"Brute-force maximal  (n={bf_obj.size})")

    # Vertical reference lines at g*(omega) for observed sizes
    ref_sizes = sorted(set(all_sizes))
    y_max = ax_obj.get_ylim()[1] if ax_obj.get_ylim()[1] > 0 else 1.0
    for s in ref_sizes:
        if s < 2:
            continue
        g_w = (R * R / 2.0) * (1.0 - 1.0 / s)
        if s == known_omega:
            color, style = "#2ca02c", "-"
        elif s == max(ref_sizes):
            color, style = "#d62728", "-"
        else:
            color, style = "#7f7f7f", "--"
        ax_obj.axvline(x=g_w, color=color, linestyle=style, alpha=0.8, linewidth=1.5)
        ax_obj.text(g_w, y_max * 0.92, f" ω={s}",
                    rotation=90, va="bottom", color=color,
                    fontsize=9, fontweight="bold")

    formula = r"$g^*(\omega) = \frac{R^2}{2}\left(1 - \frac{1}{\omega}\right)$"
    ax_obj.text(0.05, 0.95, formula, transform=ax_obj.transAxes,
                fontsize=11, va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))
    ax_obj.set_xlabel(r"Scaled objective $g = \frac{1}{2} y^\top A\, y$")
    ax_obj.set_ylabel("Frequency")
    ax_obj.set_title(f"{graph_name} / {variant_name} / a={amplitude}: Objective Distribution (R={R})")
    ax_obj.legend(loc="upper left", fontsize=8)
    ax_obj.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Alternative solvers (verify-phase optional): classical + Dirac-3 cloud
# ---------------------------------------------------------------------------

# Consistent color palette so repeated runs use identical colors.
_SOLVER_COLOURS = {
    "Greedy":  "#4C72B0",  # blue
    "PGD":     "#DD8452",  # orange
    "SLSQP":   "#55A868",  # green
    "Dirac-3": "#8172B2",  # purple
    "BF":      "#CCB974",  # tan (brute-force)
}


def run_classical_solvers(
    A: np.ndarray,
    R: int,
    true_omega: int | None,
    num_restarts: int = 30,
    seed: int = 42,
) -> dict[str, dict]:
    """Run Greedy, PGD, SLSQP on the unit simplex, scale solutions by R.

    Returns a dict keyed by solver name. Each value has the same schema as
    `_internal.classical_solvers.solve_*` plus `all_y_scaled` (solutions × R)
    for direct comparison with Boson14 hardware samples.
    """
    from _internal.classical_solvers import (
        solve_greedy_degree, solve_pgd, solve_slsqp,
    )

    out: dict[str, dict] = {}
    for name, fn in [("Greedy", solve_greedy_degree),
                     ("PGD",    solve_pgd),
                     ("SLSQP",  solve_slsqp)]:
        print(f"    [{name}] running ({num_restarts} restarts)...")
        res = fn(A, num_restarts=num_restarts, seed=seed, true_omega=true_omega)
        # Scale to Boson14's y-space (sum=R) so threshold analysis matches
        res["all_y_scaled"] = res["all_solutions"] * R
        res["best_y_scaled"] = res["best_solution"] * R
        out[name] = res
        print(f"    [{name}] best ω={res['best_omega']}, "
              f"hit={res['hit_rate']:.0%}, time={res['solve_time']:.1f}s")
    return out


def run_dirac_solver(
    A: np.ndarray,
    R: int,
    num_samples: int = 100,
    relaxation_schedule: int = 2,
    env_path: Path | None = None,
) -> dict:
    """Submit to Dirac-3 cloud; return dict compatible with hardware-analysis pipeline."""
    from _internal.dirac_cloud import solve_dirac_cloud
    return solve_dirac_cloud(
        A, R=R,
        num_samples=num_samples,
        relaxation_schedule=relaxation_schedule,
        env_path=env_path,
    )


def save_alternative_results(
    variant_dir: Path,
    classical_results: dict[str, dict] | None,
    dirac_result: dict | None,
    R: int,
    true_omega: int | None,
) -> None:
    """Persist raw arrays + summary metadata for the alternative solvers.

    Files written (each optional — only if the respective result was provided):
      - {variant}_classical_solutions.npz
          greedy_y / pgd_y / slsqp_y  (each shape (num_restarts, n), sum=R)
          greedy_g / pgd_g / slsqp_g  (per-restart objectives, unit simplex scale)
      - {variant}_classical_meta.json (aggregated stats for plotting/analysis)
      - {variant}_dirac_solutions.npz  (y_vectors shape (num_samples, n))
      - {variant}_dirac_meta.json
    """
    variant_name = variant_dir.name

    if classical_results:
        # NPZ
        npz_kwargs = {}
        for name, res in classical_results.items():
            key = name.lower().replace("-", "_")
            npz_kwargs[f"{key}_y"] = res["all_y_scaled"]
            npz_kwargs[f"{key}_g"] = np.array(res["all_objectives"])
        np.savez(variant_dir / f"{variant_name}_classical_solutions.npz", **npz_kwargs)

        # JSON summary
        meta = {
            "variant": variant_name,
            "R": R,
            "true_omega": true_omega,
            "solvers": {
                name: {
                    "best_omega": res["best_omega"],
                    "best_objective_unit": res["best_objective"],
                    "best_objective_scaled": res["best_objective"] * (R * R),
                    "hit_rate": round(res["hit_rate"], 4),
                    "num_restarts": res["num_restarts"],
                    "solve_time_s": round(res["solve_time"], 3),
                }
                for name, res in classical_results.items()
            },
        }
        (variant_dir / f"{variant_name}_classical_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )

    if dirac_result:
        np.savez(
            variant_dir / f"{variant_name}_dirac_solutions.npz",
            y_vectors=dirac_result["all_solutions"],
            objectives=np.array(dirac_result["all_objectives"]),
        )
        meta = {
            "variant": variant_name,
            "R": R,
            "true_omega": true_omega,
            "best_omega": dirac_result["best_omega"],
            "best_objective_scaled": dirac_result["best_objective"],
            "num_samples": dirac_result["num_samples"],
            "num_finite": dirac_result["num_finite"],
            "relaxation_schedule": dirac_result["relaxation_schedule"],
            "solve_time_s": round(dirac_result["solve_time"], 3),
            "formulation": "J = -0.5*A (standard Dirac-3)",
        }
        (variant_dir / f"{variant_name}_dirac_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )


def _clique_size_hist(y_array: np.ndarray, A: np.ndarray, R: int, k_ref: int) -> dict[int, int]:
    """For each row of y_array, extract threshold support and count valid-clique sizes.

    Uses the same `_threshold_support` and `_is_clique` helpers as
    `analyze_hardware_samples` so classical/Dirac output is analysed the same
    way as Boson14 hardware samples.
    """
    counts: dict[int, int] = {}
    for y in y_array:
        supp = _threshold_support(y, R, k_ref)
        if len(supp) >= 2 and _is_clique(supp, A):
            counts[len(supp)] = counts.get(len(supp), 0) + 1
    return counts


def plot_alternative_comparison(
    classical_results: dict[str, dict] | None,
    dirac_result: dict | None,
    A: np.ndarray,
    bruteforce_dist: dict,
    R: int,
    k_ref: int,
    known_omega: int | None,
    graph_name: str,
    variant_name: str,
    save_path: Path,
) -> list[Path]:
    """Two figures (if both groups are provided):
        classical_dist_{variant}.png  — Greedy + PGD + SLSQP stacked with BF
        dirac_dist_{variant}.png      — Dirac-3 stacked with BF

    Top panel:    valid-clique-size histograms (per solver + BF).
    Bottom panel: scaled objective distribution (per solver + BF), with
                  vertical g*(ω) reference lines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    bf_sizes = bruteforce_dist.get("size_counts", {})
    # compute_clique_distribution already returns objectives in SCALED space
    # (g*(ω, R) = (R²/2)(1 - 1/ω)), so no further multiplication needed.
    bf_obj_scaled = np.array(bruteforce_dist.get("objectives", []), dtype=float)

    # ---- Figure 1: classical solvers ---------------------------------------
    if classical_results:
        fname = save_path / f"classical_dist_{variant_name}.png"
        fig, (ax_size, ax_obj) = plt.subplots(2, 1, figsize=(10, 10))

        solver_size_counts = {
            name: _clique_size_hist(res["all_y_scaled"], A, R, k_ref)
            for name, res in classical_results.items()
        }
        all_sizes = sorted(
            (set.union(*(set(d) for d in solver_size_counts.values())) | set(bf_sizes)) - {0}
        )
        if not all_sizes:
            all_sizes = [known_omega or 1]

        solver_names = list(classical_results.keys())
        x = np.arange(len(all_sizes))

        # Grouped bars on LEFT y-axis (solver counts) + TWIN y-axis on RIGHT (BF counts).
        # BF has orders of magnitude more cliques than a few dozen solver restarts,
        # so a single axis hides the solver data.
        solver_width = 0.8 / (len(solver_names) + 1)
        for i, name in enumerate(solver_names):
            counts = [solver_size_counts[name].get(s, 0) for s in all_sizes]
            offset = (i - len(solver_names) / 2) * solver_width
            ax_size.bar(
                x + offset, counts, solver_width,
                color=_SOLVER_COLOURS.get(name, None), edgecolor="black", alpha=0.85,
                label=f"{name} (valid, {sum(counts)})",
            )
        ax_size.set_xticks(x)
        ax_size.set_xticklabels([str(s) for s in all_sizes])
        ax_size.set_xlabel("Clique size")
        ax_size.set_ylabel("Solver valid-clique count (left)")
        ax_size.legend(loc="upper left", fontsize=8)
        ax_size.grid(True, alpha=0.3, axis="y")

        ax_size_bf = ax_size.twinx()
        bf_counts = [bf_sizes.get(s, 0) for s in all_sizes]
        ax_size_bf.bar(
            x + (len(solver_names) / 2) * solver_width, bf_counts, solver_width,
            color=_SOLVER_COLOURS["BF"], edgecolor="black", alpha=0.5,
            label=f"Brute-force maximal ({sum(bf_counts)})",
        )
        ax_size_bf.set_ylabel("Brute-force count (right)", color=_SOLVER_COLOURS["BF"])
        ax_size_bf.tick_params(axis="y", labelcolor=_SOLVER_COLOURS["BF"])
        ax_size_bf.legend(loc="upper right", fontsize=8)

        title = f"{graph_name} / {variant_name}: Classical Solvers Clique-Size Distribution"
        subtitle = " | ".join(
            f"{n}: best ω={r['best_omega']}, hit={r['hit_rate']:.0%}"
            for n, r in classical_results.items()
        )
        if known_omega is not None:
            subtitle += f"  |  Known ω = {known_omega}"
        ax_size.set_title(f"{title}\n{subtitle}")

        # Bottom: objective distributions. No BF bars — vertical reference lines
        # already mark exactly where the maximal cliques sit, without drowning
        # out the much smaller solver sample.
        all_scaled_g: list[float] = []
        for res in classical_results.values():
            scaled = np.array(res["all_objectives"]) * (R * R)  # unit → scaled
            all_scaled_g.extend(scaled.tolist())
        if not all_scaled_g:
            all_scaled_g = [0.0, 1.0]
        g_min, g_max = float(np.min(all_scaled_g)), float(np.max(all_scaled_g))
        if bf_obj_scaled.size:
            g_min = min(g_min, float(np.min(bf_obj_scaled)))
            g_max = max(g_max, float(np.max(bf_obj_scaled)))
        if g_max <= g_min:
            g_max = g_min + 1.0
        bins = np.linspace(g_min, g_max, 40)

        for name, res in classical_results.items():
            scaled = np.array(res["all_objectives"]) * (R * R)
            ax_obj.hist(
                scaled, bins=bins, color=_SOLVER_COLOURS.get(name, None),
                alpha=0.55, edgecolor="black",
                label=f"{name} g ({len(scaled)} restarts)",
            )

        _draw_omega_reference_lines(ax_obj, all_sizes, R, known_omega)
        ax_obj.set_xlabel(r"Scaled objective $g = \frac{1}{2} y^\top A\, y$")
        ax_obj.set_ylabel("Frequency")
        ax_obj.set_title(
            f"{graph_name} / {variant_name}: Classical Objective Distribution (R={R})"
            f"   (ω lines = brute-force maximal-clique locations)"
        )
        ax_obj.legend(loc="upper left", fontsize=8)
        ax_obj.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(fname)

    # ---- Figure 2: Dirac-3 -------------------------------------------------
    if dirac_result:
        fname = save_path / f"dirac_dist_{variant_name}.png"
        fig, (ax_size, ax_obj) = plt.subplots(2, 1, figsize=(10, 10))

        dirac_sizes = _clique_size_hist(dirac_result["all_solutions"], A, R, k_ref)
        dirac_g = np.array(dirac_result["all_objectives"], dtype=float)  # already scaled

        all_sizes = sorted(set(dirac_sizes) | set(bf_sizes)) or [known_omega or 1]
        x = np.arange(len(all_sizes))
        width = 0.35

        # Left axis: Dirac counts (typically 1-100). Right axis: BF counts (thousands).
        dirac_counts = [dirac_sizes.get(s, 0) for s in all_sizes]
        bf_counts = [bf_sizes.get(s, 0) for s in all_sizes]
        ax_size.bar(x - width / 2, dirac_counts, width,
                    color=_SOLVER_COLOURS["Dirac-3"], edgecolor="black", alpha=0.85,
                    label=f"Dirac-3 (valid, {sum(dirac_counts)})")
        ax_size.set_xticks(x)
        ax_size.set_xticklabels([str(s) for s in all_sizes])
        ax_size.set_xlabel("Clique size")
        ax_size.set_ylabel("Dirac-3 valid-clique count (left)")
        ax_size.legend(loc="upper left", fontsize=8)
        ax_size.grid(True, alpha=0.3, axis="y")

        ax_size_bf = ax_size.twinx()
        ax_size_bf.bar(x + width / 2, bf_counts, width,
                       color=_SOLVER_COLOURS["BF"], edgecolor="black", alpha=0.5,
                       label=f"Brute-force maximal ({sum(bf_counts)})")
        ax_size_bf.set_ylabel("Brute-force count (right)", color=_SOLVER_COLOURS["BF"])
        ax_size_bf.tick_params(axis="y", labelcolor=_SOLVER_COLOURS["BF"])
        ax_size_bf.legend(loc="upper right", fontsize=8)

        frac = sum(dirac_counts) / max(1, dirac_result["num_finite"])
        title = f"{graph_name} / {variant_name}: Dirac-3 Clique-Size Distribution"
        subtitle = (f"Valid-clique fraction: {frac:.1%} "
                    f"({sum(dirac_counts)}/{dirac_result['num_finite']}) | "
                    f"best ω={dirac_result['best_omega']}")
        if known_omega is not None:
            subtitle += f"  |  Known ω = {known_omega}"
        ax_size.set_title(f"{title}\n{subtitle}")

        # Bottom: Dirac objective distribution + g*(ω) reference lines (same style
        # as classical panel — no BF bars, rely on the vertical lines to mark the
        # brute-force maximal-clique locations).
        if dirac_g.size:
            g_min, g_max = float(np.min(dirac_g)), float(np.max(dirac_g))
            if bf_obj_scaled.size:
                g_min = min(g_min, float(np.min(bf_obj_scaled)))
                g_max = max(g_max, float(np.max(bf_obj_scaled)))
            if g_max <= g_min:
                g_max = g_min + 1.0
            bins = np.linspace(g_min, g_max, 40)
            ax_obj.hist(dirac_g, bins=bins, color=_SOLVER_COLOURS["Dirac-3"],
                        alpha=0.55, edgecolor="black",
                        label=f"Dirac-3 g(y) ({len(dirac_g)})")

        _draw_omega_reference_lines(ax_obj, all_sizes, R, known_omega)
        ax_obj.set_xlabel(r"Scaled objective $g = \frac{1}{2} y^\top A\, y$")
        ax_obj.set_ylabel("Frequency")
        ax_obj.set_title(
            f"{graph_name} / {variant_name}: Dirac-3 Objective Distribution (R={R})"
            f"   (ω lines = brute-force maximal-clique locations)"
        )
        ax_obj.legend(loc="upper left", fontsize=8)
        ax_obj.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(fname)

    return saved


def _draw_omega_reference_lines(ax, sizes: list[int], R: int, known_omega: int | None) -> None:
    """Overlay vertical g*(ω) reference lines on an objective-distribution axis."""
    y_max = ax.get_ylim()[1] or 1.0
    for s in sizes:
        if s < 2:
            continue
        g_w = (R * R / 2.0) * (1.0 - 1.0 / s)
        if s == known_omega:
            color, style = "#2ca02c", "-"
        elif sizes and s == max(sizes):
            color, style = "#d62728", "-"
        else:
            color, style = "#7f7f7f", "--"
        ax.axvline(x=g_w, color=color, linestyle=style, alpha=0.8, linewidth=1.5)
        ax.text(g_w, y_max * 0.92, f" ω={s}",
                rotation=90, va="bottom", color=color,
                fontsize=9, fontweight="bold")
