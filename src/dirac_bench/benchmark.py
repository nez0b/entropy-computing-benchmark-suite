"""Benchmark orchestrator — runs solvers on DIMACS graphs and collects results."""

from pathlib import Path

from dirac_bench.io import read_dimacs_graph, get_graph_info
from dirac_bench.problems import motzkin_straus_adjacency
from dirac_bench.utils import save_raw_response
from dirac_bench.plotting import plot_dirac_histogram


# Known clique numbers for standard DIMACS benchmark graphs
KNOWN_OMEGA = {
    "brock200_1": 21, "brock200_2": 12, "brock200_3": 15, "brock200_4": 17,
    "brock400_1": 27, "brock400_2": 29, "brock400_3": 31, "brock400_4": 33,
    "brock800_1": 23, "brock800_2": 24, "brock800_3": 25, "brock800_4": 26,
    "C125.9": 34, "C250.9": 44, "C500.9": 57,
    "c-fat200-1": 12, "c-fat200-2": 24, "c-fat200-5": 58,
    "c-fat500-1": 14, "c-fat500-2": 26, "c-fat500-5": 64, "c-fat500-10": 126,
    "DSJC500.5": 13,
    "gen200_p0.9_44": 44, "gen200_p0.9_55": 55,
    "gen400_p0.9_55": 55, "gen400_p0.9_65": 65, "gen400_p0.9_75": 75,
    "hamming6-2": 32, "hamming6-4": 4, "hamming8-2": 128, "hamming8-4": 16,
    "johnson8-2-4": 4, "johnson8-4-4": 14, "johnson16-2-4": 8, "johnson32-2-4": 16,
    "keller4": 11, "keller5": 27,
    "p_hat300-1": 8, "p_hat300-2": 25, "p_hat300-3": 36,
    "p_hat500-1": 9, "p_hat500-2": 36, "p_hat500-3": 50,
    "p_hat700-1": 11, "p_hat700-2": 44, "p_hat700-3": 62,
    "san200_0.7_1": 30, "san200_0.7_2": 18,
    "san200_0.9_1": 70, "san200_0.9_2": 60, "san200_0.9_3": 44,
    "san400_0.5_1": 13, "san400_0.7_1": 40, "san400_0.7_2": 30, "san400_0.7_3": 22,
    "san400_0.9_1": 100,
    "sanr200_0.7": 18, "sanr200_0.9": 42, "sanr400_0.5": 13, "sanr400_0.7": 21,
}


def run_single_graph(graph_name: str, graph, A, config: dict) -> dict:
    """Run all enabled solvers on a single graph.

    Args:
        graph_name: Stem name of the graph.
        graph: NetworkX graph.
        A: Adjacency matrix.
        config: Dict with keys: skip_dirac, backend, ip, port, num_samples,
                relaxation_schedule, slsqp_restarts, lbfgs_restarts, seed,
                results_dir, plots_dir, no_plot.

    Returns:
        Result row dict.
    """
    info = get_graph_info(graph)
    known = KNOWN_OMEGA.get(graph_name)

    print(f"  |V|={info['nodes']}, |E|={info['edges']}, density={info['density']:.3f}")
    if known is not None:
        print(f"  Known omega = {known}")
    else:
        print(f"  Known omega = ? (not in benchmark table)")

    row = {
        "graph": graph_name,
        "nodes": info["nodes"],
        "edges": info["edges"],
        "density": f"{info['density']:.3f}",
        "known_omega": known if known is not None else "",
    }

    baseline_objectives = {}

    # ── SLSQP ────────────────────────────────────────────────────────
    try:
        from dirac_bench.solvers.slsqp import solve_slsqp
        slsqp = solve_slsqp(A, num_restarts=config.get("slsqp_restarts", 10),
                             seed=config.get("seed", 42))
        row["slsqp_omega"] = slsqp["omega"]
        row["slsqp_objective"] = f"{slsqp['best_objective']:.6f}"
        row["slsqp_time"] = f"{slsqp['solve_time']:.1f}"
        baseline_objectives["SLSQP"] = slsqp["best_objective"]
    except Exception as e:
        print(f"  SLSQP error: {e}")
        row["slsqp_omega"] = "ERR"
        row["slsqp_objective"] = ""
        row["slsqp_time"] = ""

    # ── L-BFGS-B ─────────────────────────────────────────────────────
    try:
        from dirac_bench.solvers.lbfgs import solve_lbfgs
        lbfgs = solve_lbfgs(A, num_restarts=config.get("lbfgs_restarts", 10),
                            seed=config.get("seed", 42))
        row["lbfgs_omega"] = lbfgs["omega"]
        row["lbfgs_objective"] = f"{lbfgs['best_objective']:.6f}"
        row["lbfgs_time"] = f"{lbfgs['solve_time']:.1f}"
        baseline_objectives["L-BFGS-B"] = lbfgs["best_objective"]
    except Exception as e:
        print(f"  L-BFGS-B error: {e}")
        row["lbfgs_omega"] = "ERR"
        row["lbfgs_objective"] = ""
        row["lbfgs_time"] = ""

    # ── Dirac cloud ──────────────────────────────────────────────────
    dirac_objectives = None
    dirac_omega = None

    if not config.get("skip_dirac") and config.get("backend") in ("cloud", "both"):
        try:
            from dirac_bench.solvers.dirac import solve_dirac
            dirac = solve_dirac(
                A,
                num_samples=config.get("num_samples", 100),
                relaxation_schedule=config.get("relaxation_schedule", 2),
            )
            row["dirac_cloud_omega"] = dirac["omega"]
            row["dirac_cloud_objective"] = f"{dirac['best_objective']:.6f}"
            row["dirac_cloud_time"] = f"{dirac['solve_time']:.1f}"
            dirac_objectives = dirac["all_objectives"]
            dirac_omega = dirac["omega"]

            save_raw_response(dirac["raw_response"], graph_name, "dirac",
                              save_dir=config.get("results_dir", "results/raw"))
        except Exception as e:
            print(f"  Dirac cloud error: {e}")
            row["dirac_cloud_omega"] = "ERR"
            row["dirac_cloud_objective"] = ""
            row["dirac_cloud_time"] = ""

    # ── Dirac direct ─────────────────────────────────────────────────
    if not config.get("skip_dirac") and config.get("backend") in ("direct", "both"):
        try:
            from dirac_bench.solvers.dirac_direct import solve_dirac_direct
            direct = solve_dirac_direct(
                A,
                ip_address=config.get("ip", "172.18.41.79"),
                port=config.get("port", "50051"),
                num_samples=config.get("num_samples", 100),
                relaxation_schedule=config.get("relaxation_schedule", 2),
            )
            row["dirac_direct_omega"] = direct["omega"]
            row["dirac_direct_objective"] = f"{direct['best_objective']:.6f}"
            row["dirac_direct_time"] = f"{direct['solve_time']:.1f}"

            # Use direct results for plotting if cloud wasn't run
            if dirac_objectives is None:
                dirac_objectives = direct["all_objectives"]
                dirac_omega = direct["omega"]

            save_raw_response(direct["raw_response"], graph_name, "dirac_direct",
                              save_dir=config.get("results_dir", "results/raw"))
        except Exception as e:
            print(f"  Dirac direct error: {e}")
            row["dirac_direct_omega"] = "ERR"
            row["dirac_direct_objective"] = ""
            row["dirac_direct_time"] = ""

    # ── Plot histogram ───────────────────────────────────────────────
    if dirac_objectives and not config.get("no_plot"):
        plot_dirac_histogram(
            objectives=dirac_objectives,
            graph_name=graph_name,
            computed_omega=dirac_omega,
            known_omega=known,
            baseline_objectives=baseline_objectives or None,
            save_path=config.get("plots_dir", "plots"),
        )

    return row


def run_benchmark(config: dict) -> list[dict]:
    """Run the full benchmark across all graphs.

    Args:
        config: Dict with keys: dimacs_dir, graph, max_nodes, and all solver config.

    Returns:
        List of result row dicts.
    """
    dimacs_dir = Path(config.get("dimacs_dir", "problems/dimacs"))
    graph_filter = config.get("graph")
    max_nodes = config.get("max_nodes")

    files = sorted(dimacs_dir.glob("*.clq"))
    if graph_filter:
        files = [f for f in files if f.stem == graph_filter]

    if not files:
        print(f"No .clq files found in {dimacs_dir}")
        return []

    print(f"Found {len(files)} graph(s) in {dimacs_dir}")
    print()

    rows = []
    for path in files:
        stem = path.stem
        print(f"{'=' * 60}")
        print(f"Graph: {stem}")
        print(f"{'=' * 60}")

        graph = read_dimacs_graph(str(path))
        info = get_graph_info(graph)

        if max_nodes is not None and info["nodes"] > max_nodes:
            print(f"  |V|={info['nodes']} > {max_nodes}, skipping")
            print()
            continue

        A = motzkin_straus_adjacency(graph)
        row = run_single_graph(stem, graph, A, config)
        rows.append(row)
        print()

    return rows


def print_results_table(rows: list[dict]) -> None:
    """Print a formatted results table to the console."""
    if not rows:
        return

    # Determine which solver columns exist
    has_slsqp = any("slsqp_omega" in r for r in rows)
    has_lbfgs = any("lbfgs_omega" in r for r in rows)
    has_cloud = any("dirac_cloud_omega" in r for r in rows)
    has_direct = any("dirac_direct_omega" in r for r in rows)

    # Build header
    parts = [f"{'Graph':<22}", f"{'|V|':>5}", f"{'|E|':>7}", f"{'Dens':>6}", f"{'Known':>6}"]
    if has_slsqp:
        parts.append(f"{'SLSQP':>6}")
    if has_lbfgs:
        parts.append(f"{'LBFGS':>6}")
    if has_cloud:
        parts.append(f"{'Cloud':>6}")
    if has_direct:
        parts.append(f"{'Direct':>6}")

    header = " ".join(parts)
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for r in rows:
        parts = [
            f"{r['graph']:<22}",
            f"{r['nodes']:>5}",
            f"{r['edges']:>7}",
            f"{r['density']:>6}",
            f"{str(r.get('known_omega', '')):>6}",
        ]
        if has_slsqp:
            parts.append(f"{str(r.get('slsqp_omega', '')):>6}")
        if has_lbfgs:
            parts.append(f"{str(r.get('lbfgs_omega', '')):>6}")
        if has_cloud:
            parts.append(f"{str(r.get('dirac_cloud_omega', '')):>6}")
        if has_direct:
            parts.append(f"{str(r.get('dirac_direct_omega', '')):>6}")
        print(" ".join(parts))

    print(sep)
