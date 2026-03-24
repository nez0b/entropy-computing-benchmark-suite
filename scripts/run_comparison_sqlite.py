#!/usr/bin/env python3
"""Cloud vs. Direct Dirac-3 comparison benchmark with SQLite storage.

Runs 4 solvers (SLSQP, L-BFGS-B, Dirac Cloud, Dirac Direct) on
DIMACS max-clique graphs and stores results in a SQLite database for
structured querying. Use --max-nodes to limit by graph size.

Usage:
    uv run python scripts/run_comparison_sqlite.py                    # all solvers, all graphs
    uv run python scripts/run_comparison_sqlite.py --max-nodes 200    # only n≤200
    uv run python scripts/run_comparison_sqlite.py --skip-cloud --skip-direct  # classical only
    uv run python scripts/run_comparison_sqlite.py --graph C125.9     # single graph
"""

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Load QCI credentials before importing Dirac solvers
_env_path = Path(__file__).resolve().parent.parent / "qci-eqc-models" / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)

from dirac_bench.benchmark import KNOWN_OMEGA
from dirac_bench.io import get_graph_info, read_dimacs_graph
from dirac_bench.problems import motzkin_straus_adjacency, objective, objective_to_omega
from dirac_bench.utils import _numpy_converter, save_raw_response

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    num_samples         INTEGER,
    relaxation_schedule INTEGER,
    sum_constraint      INTEGER DEFAULT 1,
    slsqp_restarts      INTEGER,
    lbfgs_restarts       INTEGER,
    seed                INTEGER,
    ip_address          TEXT,
    port                TEXT,
    num_graphs          INTEGER,
    notes               TEXT
);

CREATE TABLE IF NOT EXISTS solves (
    solve_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(run_id),
    graph_name      TEXT NOT NULL,
    num_nodes       INTEGER NOT NULL,
    num_edges       INTEGER NOT NULL,
    density         REAL NOT NULL,
    known_omega     INTEGER,
    solver          TEXT NOT NULL,
    omega           INTEGER,
    best_objective  REAL,
    solve_time      REAL,
    all_objectives      TEXT,
    device_energies     TEXT,
    raw_response        TEXT,
    relaxation_schedule INTEGER,
    hash_id             TEXT,
    raw_output_path     TEXT,
    error               TEXT,
    solved_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS solutions (
    solution_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    solve_id        INTEGER NOT NULL REFERENCES solves(solve_id),
    sample_index    INTEGER NOT NULL,
    objective       REAL,
    omega           INTEGER,
    solution_vector TEXT NOT NULL,
    energy          REAL
);

CREATE INDEX IF NOT EXISTS idx_solves_run ON solves(run_id);
CREATE INDEX IF NOT EXISTS idx_solves_graph ON solves(graph_name);
CREATE INDEX IF NOT EXISTS idx_solves_solver ON solves(solver);
CREATE INDEX IF NOT EXISTS idx_solutions_solve ON solutions(solve_id);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """Create tables if needed and return an open connection."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    # Migrate existing DB: add new columns if missing
    for col, typ in [("relaxation_schedule", "INTEGER"),
                     ("hash_id", "TEXT"),
                     ("raw_output_path", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE solves ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass  # column already exists

    # Backfill relaxation_schedule from runs table for existing rows
    conn.execute("""
        UPDATE solves SET relaxation_schedule = (
            SELECT r.relaxation_schedule FROM runs r WHERE r.run_id = solves.run_id
        ) WHERE relaxation_schedule IS NULL
    """)

    conn.commit()
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_run(conn: sqlite3.Connection, args: argparse.Namespace) -> int:
    """Insert a new run row and return the run_id."""
    cur = conn.execute(
        """INSERT INTO runs
           (started_at, num_samples, relaxation_schedule, sum_constraint,
            slsqp_restarts, lbfgs_restarts, seed, ip_address, port, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            _now_iso(),
            args.num_samples,
            args.relaxation_schedule,
            1,
            args.slsqp_restarts,
            args.lbfgs_restarts,
            args.seed,
            args.ip,
            args.port,
            args.notes,
        ),
    )
    conn.commit()
    return cur.lastrowid


def finalize_run(conn: sqlite3.Connection, run_id: int, num_graphs: int) -> None:
    """Update run with finished_at and num_graphs."""
    conn.execute(
        "UPDATE runs SET finished_at = ?, num_graphs = ? WHERE run_id = ?",
        (_now_iso(), num_graphs, run_id),
    )
    conn.commit()


def insert_solve(
    conn: sqlite3.Connection,
    run_id: int,
    graph_name: str,
    info: dict,
    known_omega: int | None,
    solver: str,
    result: dict | None = None,
    error: str | None = None,
    relaxation_schedule: int | None = None,
    hash_id: str | None = None,
    raw_output_path: str | None = None,
) -> int:
    """Insert a solve row (success or error) and return the solve_id."""
    cur = conn.execute(
        """INSERT INTO solves
           (run_id, graph_name, num_nodes, num_edges, density, known_omega,
            solver, omega, best_objective, solve_time,
            all_objectives, device_energies, raw_response,
            relaxation_schedule, hash_id, raw_output_path,
            error, solved_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            graph_name,
            info["nodes"],
            info["edges"],
            info["density"],
            known_omega,
            solver,
            result["omega"] if result else None,
            result["best_objective"] if result else None,
            result["solve_time"] if result else None,
            json.dumps(result["all_objectives"], default=_numpy_converter)
            if result and "all_objectives" in result
            else None,
            json.dumps(result["device_energies"], default=_numpy_converter)
            if result and "device_energies" in result
            else None,
            json.dumps(result.get("raw_response"), default=_numpy_converter)
            if result and "raw_response" in result
            else None,
            relaxation_schedule,
            hash_id,
            raw_output_path,
            error,
            _now_iso(),
        ),
    )
    return cur.lastrowid


def insert_solutions(
    conn: sqlite3.Connection,
    solve_id: int,
    solver: str,
    result: dict,
    A: np.ndarray,
) -> None:
    """Extract individual solution vectors and insert into solutions table."""
    rows = []

    if solver == "dirac_cloud":
        solutions = result.get("raw_response", {}).get("results", {}).get("solutions", [])
        for i, sol in enumerate(solutions):
            x = np.array(sol, dtype=np.float64)
            obj = objective(x, A)
            omega = objective_to_omega(obj) if np.isfinite(obj) else None
            rows.append((solve_id, i, float(obj) if np.isfinite(obj) else None,
                         omega, json.dumps(x.tolist()), None))

    elif solver == "dirac_direct":
        raw_solutions = np.array(result.get("raw_response", {}).get("solution", []))
        if raw_solutions.ndim == 1:
            raw_solutions = raw_solutions.reshape(1, -1)
        energies = result.get("device_energies", [])
        for i, x in enumerate(raw_solutions):
            x = np.array(x, dtype=np.float64)
            obj = objective(x, A)
            omega = objective_to_omega(obj) if np.isfinite(obj) else None
            energy = energies[i] if i < len(energies) else None
            rows.append((solve_id, i, float(obj) if np.isfinite(obj) else None,
                         omega, json.dumps(x.tolist()), energy))

    elif solver in ("slsqp", "lbfgs"):
        x = result.get("solution")
        if x is not None:
            obj = result["best_objective"]
            omega = result["omega"]
            rows.append((solve_id, 0, obj, omega, json.dumps(x.tolist()), None))

    if rows:
        conn.executemany(
            """INSERT INTO solutions
               (solve_id, sample_index, objective, omega, solution_vector, energy)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )


# ---------------------------------------------------------------------------
# Solver dispatch
# ---------------------------------------------------------------------------

def run_solver(solver: str, A: np.ndarray, args: argparse.Namespace) -> dict:
    """Call the appropriate solver and return its result dict."""
    if solver == "slsqp":
        from dirac_bench.solvers.slsqp import solve_slsqp
        return solve_slsqp(A, num_restarts=args.slsqp_restarts, seed=args.seed)

    if solver == "lbfgs":
        from dirac_bench.solvers.lbfgs import solve_lbfgs
        return solve_lbfgs(A, num_restarts=args.lbfgs_restarts, seed=args.seed)

    if solver == "dirac_cloud":
        from dirac_bench.solvers.dirac import solve_dirac
        return solve_dirac(
            A,
            num_samples=args.num_samples,
            relaxation_schedule=args.relaxation_schedule,
        )

    if solver == "dirac_direct":
        from dirac_bench.solvers.dirac_direct import solve_dirac_direct
        return solve_dirac_direct(
            A,
            ip_address=args.ip,
            port=args.port,
            num_samples=args.num_samples,
            relaxation_schedule=args.relaxation_schedule,
        )

    raise ValueError(f"Unknown solver: {solver}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(conn: sqlite3.Connection, run_id: int) -> None:
    """Query the DB and print a formatted results table."""
    rows = conn.execute(
        """SELECT graph_name, num_nodes, num_edges, density, known_omega,
                  solver, omega, best_objective, solve_time, error
           FROM solves WHERE run_id = ? ORDER BY graph_name, solver""",
        (run_id,),
    ).fetchall()

    if not rows:
        print("No results to display.")
        return

    # Pivot: group by graph, collect solver results
    from collections import OrderedDict
    graphs = OrderedDict()
    solvers_seen = []
    for r in rows:
        gname = r[0]
        solver = r[5]
        if solver not in solvers_seen:
            solvers_seen.append(solver)
        if gname not in graphs:
            graphs[gname] = {
                "nodes": r[1], "edges": r[2], "density": r[3], "known": r[4],
            }
        omega_str = str(r[6]) if r[6] is not None else "ERR"
        time_str = f"{r[8]:.1f}s" if r[8] is not None else ""
        graphs[gname][solver] = f"{omega_str:>3} ({time_str})" if r[9] is None else "ERR"

    # Header
    solver_labels = {
        "slsqp": "SLSQP", "lbfgs": "LBFGS", "dirac_cloud": "Cloud", "dirac_direct": "Direct",
    }
    hdr = f"{'Graph':<28} {'|V|':>5} {'Known':>6}"
    for s in solvers_seen:
        hdr += f"  {solver_labels.get(s, s):>16}"
    sep = "-" * len(hdr)

    print()
    print(sep)
    print(hdr)
    print(sep)
    for gname, g in graphs.items():
        line = f"{gname:<28} {g['nodes']:>5} {str(g.get('known') or '?'):>6}"
        for s in solvers_seen:
            line += f"  {g.get(s, ''):>16}"
        print(line)
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cloud vs. Direct Dirac-3 comparison with SQLite results"
    )

    parser.add_argument(
        "--db-path", type=str, default="results/comparison.db",
        help="SQLite output path (default: results/comparison.db)",
    )
    parser.add_argument(
        "--dimacs-dir", type=str, default="problems/dimacs",
        help="Directory containing .clq files (default: problems/dimacs)",
    )
    parser.add_argument(
        "--graph", type=str, default=None,
        help="Run single graph by stem name (e.g. C125.9)",
    )

    # Dirac options
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Dirac samples per graph (1-100, default: 100)")
    parser.add_argument("--relaxation-schedule", type=int, default=2,
                        help="Dirac relaxation schedule (1-4, default: 2)")
    parser.add_argument("--ip", type=str, default="172.18.41.79",
                        help="Dirac-3 direct solver IP (default: 172.18.41.79)")
    parser.add_argument("--port", type=str, default="50051",
                        help="Dirac-3 direct gRPC port (default: 50051)")

    # Classical options
    parser.add_argument("--slsqp-restarts", type=int, default=10,
                        help="SLSQP random restarts (default: 10)")
    parser.add_argument("--lbfgs-restarts", type=int, default=10,
                        help="L-BFGS-B random restarts (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Solver skip flags
    parser.add_argument("--skip-cloud", action="store_true",
                        help="Skip Dirac cloud solver")
    parser.add_argument("--skip-direct", action="store_true",
                        help="Skip Dirac direct solver")
    parser.add_argument("--skip-classical", action="store_true",
                        help="Skip SLSQP and L-BFGS-B solvers")

    parser.add_argument("--max-nodes", type=int, default=None,
                        help="Only run graphs with n ≤ this value (default: no limit)")
    parser.add_argument("--notes", type=str, default=None,
                        help="Free-text annotation for this run")

    args = parser.parse_args()

    # ── Build solver list ─────────────────────────────────────────────
    solvers = []
    if not args.skip_classical:
        solvers += ["slsqp", "lbfgs"]
    if not args.skip_cloud:
        solvers.append("dirac_cloud")
    if not args.skip_direct:
        solvers.append("dirac_direct")

    if not solvers:
        print("All solvers skipped — nothing to do.")
        return 1

    # ── Discover graphs ───────────────────────────────────────────────
    dimacs_dir = Path(args.dimacs_dir)
    files = sorted(dimacs_dir.glob("*.clq"))
    if args.graph:
        files = [f for f in files if f.stem == args.graph]

    if not files:
        print(f"No .clq files found in {dimacs_dir}")
        return 1

    # Filter by node count (if --max-nodes is set)
    graph_data = []
    for path in files:
        graph = read_dimacs_graph(str(path))
        info = get_graph_info(graph)
        if args.max_nodes is None or info["nodes"] <= args.max_nodes:
            A = motzkin_straus_adjacency(graph)
            graph_data.append((path.stem, graph, info, A))

    if not graph_data:
        limit_msg = f" with n≤{args.max_nodes}" if args.max_nodes else ""
        print(f"No graphs{limit_msg} found.")
        return 1

    limit_msg = f" with n≤{args.max_nodes}" if args.max_nodes else ""
    print(f"Found {len(graph_data)} graph(s){limit_msg}")
    print(f"Solvers: {', '.join(solvers)}")
    print(f"Database: {args.db_path}")
    print()

    # ── Init DB and run ───────────────────────────────────────────────
    conn = init_db(args.db_path)
    run_id = insert_run(conn, args)
    num_graphs = 0

    for graph_name, graph, info, A in graph_data:
        num_graphs += 1
        known = KNOWN_OMEGA.get(graph_name)

        print(f"{'=' * 60}")
        print(f"Graph: {graph_name}  |V|={info['nodes']}  |E|={info['edges']}  "
              f"density={info['density']:.3f}  known_omega={known or '?'}")
        print(f"{'=' * 60}")

        for solver in solvers:
            print(f"\n  [{solver}]")
            try:
                result = run_solver(solver, A, args)

                hash_id = hashlib.sha256(
                    f"{graph_name}:{solver}:{args.relaxation_schedule}:{time.time()}".encode()
                ).hexdigest()[:8]

                raw_output_path = None
                if solver.startswith("dirac") and result.get("raw_response"):
                    raw_output_path = str(save_raw_response(
                        result["raw_response"], graph_name, solver, hash_id=hash_id
                    ))

                solve_id = insert_solve(
                    conn, run_id, graph_name, info, known, solver,
                    result=result,
                    relaxation_schedule=args.relaxation_schedule,
                    hash_id=hash_id,
                    raw_output_path=raw_output_path,
                )
                insert_solutions(conn, solve_id, solver, result, A)
                conn.commit()
            except Exception as e:
                print(f"  ERROR: {e}")
                insert_solve(
                    conn, run_id, graph_name, info, known, solver,
                    error=str(e),
                    relaxation_schedule=args.relaxation_schedule,
                )
                conn.commit()

        print()

    finalize_run(conn, run_id, num_graphs)

    # ── Summary ───────────────────────────────────────────────────────
    print_summary(conn, run_id)

    # Stats
    total_solves = conn.execute(
        "SELECT COUNT(*) FROM solves WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    total_solutions = conn.execute(
        "SELECT COUNT(*) FROM solutions WHERE solve_id IN "
        "(SELECT solve_id FROM solves WHERE run_id = ?)", (run_id,)
    ).fetchone()[0]
    errors = conn.execute(
        "SELECT COUNT(*) FROM solves WHERE run_id = ? AND error IS NOT NULL",
        (run_id,),
    ).fetchone()[0]

    print(f"\nRun {run_id}: {total_solves} solves, {total_solutions} solution vectors, "
          f"{errors} errors")
    print(f"Results saved to {args.db_path}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
