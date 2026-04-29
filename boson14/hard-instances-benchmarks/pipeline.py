#!/usr/bin/env python3
"""Hard-instance Boson14 hardware stress-test pipeline.

Generates difficult max-clique instances (strategies A-E), creates permutation
variants (front/end/random), optionally brute-force verifies omega, and feeds
each variant CSV to run_boson14.py for hardware submission.

Phases (can stop early via --phase):
    generate  — build the hard graph + save adjacency + base_meta.json
    variants  — produce 3 permutation variants with CSV + JSON + meta
    verify    — brute-force compute omega + save theoretical solutions
    hardware  — call run_boson14.py for each variant × amplitude combo
    all       — all of the above in sequence (default)

Usage:
    python pipeline.py --strategies B,D --n 90 --seeds 42 \
        --amplitudes 300,400,500,600 \
        -R 100 -s 1000 -l 256 -d 86 -w 1 -b 28 -solps 0

    # Just generate CSVs, no hardware
    python pipeline.py --strategies D --n 90 --k 12 --seeds 42 \
        --phase variants --verify

    # Resume after an interruption
    python pipeline.py --strategies B,D --n 90 --seeds 42 --skip-existing

run_boson14.py is called as a subprocess with cwd=variant_dir, so all outputs
(results__*.npz, plot__*.png) land next to the CSV. After each run, the NPZ is
post-processed to embed (amplitude, num_loops, R, ...) and renamed to append
the `_a{amplitude}` suffix (matching the existing boson14/output/ convention).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np

import utils
from utils import HardwareFlags
from boson14_bench.core import (
    compute_clique_distribution,
    omega_to_scaled_objective,
    plot_clique_distribution,
)

# Locate run_boson14.py (one level up from this script)
SCRIPT_DIR = Path(__file__).resolve().parent
BOSON14_DIR = SCRIPT_DIR.parent
RUN_BOSON14_PATH = BOSON14_DIR / "run_boson14.py"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _csv_strs(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hard-instance Boson14 hardware stress-test pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Strategy / instance selection
    p.add_argument("--strategies", type=_csv_strs, default=["B", "D"],
                   help="Comma-separated strategies from {A,B,C,D,E} (default: B,D)")
    p.add_argument("--n", type=int, default=90,
                   help="Graph size")
    p.add_argument("--p", type=float, default=None,
                   help="Edge probability (strategy-specific default if unset)")
    p.add_argument("--k", type=int, default=None,
                   help="Planted clique size for A/C/D/E (ignored for B)")
    p.add_argument("--seeds", type=_csv_ints, default=[42],
                   help="Comma-separated random seeds")
    p.add_argument("--k-multiplier", type=float, default=2.5,
                   help="Strategy A only: k = ceil(k_multiplier * log2(n))")
    p.add_argument("--num-cliques", type=int, default=3,
                   help="Strategy C only")
    p.add_argument("--removal-frac", type=float, default=0.4,
                   help="Strategy D only")
    p.add_argument("--overlap", type=int, default=None,
                   help="Strategy E only (default k-2)")
    p.add_argument("--num-random-variants", type=int, default=1,
                   help="Strategy B with n > 90: number of random variants to generate")

    # Pipeline control
    p.add_argument("--phase", choices=["all", "generate", "variants", "verify", "hardware"],
                   default="all", help="Which phase to run")
    p.add_argument("--stop-after", choices=["generate", "variants", "verify"],
                   default=None, help="Run phases up to and including this, then stop")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip phases with existing .done marker")
    p.add_argument("--verify", action="store_true",
                   help="Include brute-force verification of omega (auto-enabled for B with n <= 90)")
    p.add_argument("--no-verify-plot", action="store_true",
                   help="Skip the verify-phase clique-distribution plot (on by default)")
    p.add_argument("--no-hardware-plot", action="store_true",
                   help="Skip the hardware-phase sample-distribution plot (on by default)")

    # Alternative solvers (verify phase, opt-in). Run only on the random variant
    # because all 3 variants are isomorphic copies — so one is enough for
    # solver comparison (unlike hardware which is position-dependent).
    p.add_argument("--classical-solve", action="store_true",
                   help="Run classical solvers (Greedy, PGD, SLSQP) on the random variant")
    p.add_argument("--dirac-solve", action="store_true",
                   help="Run Dirac-3 cloud solver on the random variant")
    p.add_argument("--classical-restarts", type=int, default=30,
                   help="Restarts for Greedy/PGD/SLSQP (default: 30)")
    p.add_argument("--dirac-samples", type=int, default=100,
                   help="Dirac-3 samples (default: 100)")
    p.add_argument("--dirac-schedule", type=int, default=2, choices=[1, 2, 3, 4],
                   help="Dirac-3 relaxation schedule (default: 2)")
    p.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output",
                   help="Output root directory")
    p.add_argument("--dry-run", action="store_true",
                   help="Print hardware commands without executing subprocess calls")

    # Hardware pass-through (forwarded to run_boson14.py)
    p.add_argument("--amplitudes", type=_csv_ints, default=[300, 400, 500, 600],
                   help="Comma-separated amplitudes; one hardware run per variant per amplitude")
    p.add_argument("-R", "--r", dest="R", type=int, default=100,
                   help="Sum constraint (forwarded to run_boson14.py -R)")
    p.add_argument("-s", "--num-samples", dest="num_samples", type=int, default=200,
                   help="Forwarded to run_boson14.py -s")
    p.add_argument("-l", "--num-loops", dest="num_loops", type=int, default=200,
                   help="Forwarded to run_boson14.py -l")
    p.add_argument("-d", "--delay", type=int, default=86,
                   help="Forwarded to run_boson14.py -d")
    p.add_argument("-w", "--pulse-width", dest="pulse_width", type=int, default=1,
                   help="Forwarded to run_boson14.py -w")
    p.add_argument("-b", "--distance-between-pulses", dest="distance_between_pulses",
                   type=int, default=28, help="Forwarded to run_boson14.py -b")
    p.add_argument("-solps", "--solution-plot-show", dest="solps", type=int, default=0,
                   help="Forwarded to run_boson14.py -solps (0 recommended in a pipeline)")

    return p.parse_args()


def _phases_to_run(args: argparse.Namespace) -> list[str]:
    """Resolve which phases to run based on --phase and --stop-after."""
    order = ["generate", "variants", "verify", "hardware"]
    if args.phase == "all":
        phases = list(order)
    else:
        phases = [args.phase]

    if args.stop_after:
        idx = order.index(args.stop_after)
        # keep phases[i] only if order.index(phases[i]) <= idx
        phases = [ph for ph in phases if order.index(ph) <= idx]
    return phases


# ---------------------------------------------------------------------------
# Strategy-specific param bundles (for instance building)
# ---------------------------------------------------------------------------

def _build_params(strategy: str, args: argparse.Namespace, seed: int) -> dict:
    params = {"n": args.n, "seed": seed}
    if args.p is not None:
        params["p"] = args.p
    else:
        # strategy-specific defaults
        params["p"] = 0.9 if strategy == "B" else (0.3 if strategy == "E" else 0.5)

    if strategy == "A":
        params["k_multiplier"] = args.k_multiplier
    elif strategy == "B":
        pass
    elif strategy == "C":
        if args.k is None:
            raise SystemExit("Strategy C requires --k")
        params["k"] = args.k
        params["num_cliques"] = args.num_cliques
    elif strategy == "D":
        if args.k is None:
            raise SystemExit("Strategy D requires --k")
        params["k"] = args.k
        params["removal_frac"] = args.removal_frac
    elif strategy == "E":
        if args.k is None:
            raise SystemExit("Strategy E requires --k")
        params["k"] = args.k
        params["overlap"] = args.overlap if args.overlap is not None else (args.k - 2)
    else:
        raise SystemExit(f"Unknown strategy: {strategy}")
    return params


# ---------------------------------------------------------------------------
# Phase 1: generate
# ---------------------------------------------------------------------------

def cmd_generate(strategy: str, params: dict, output_dir: Path, args: argparse.Namespace) -> Path:
    """Build the hard-instance graph, save adjacency NPZ + base_meta.json."""
    A, meta = utils.build_instance(strategy, params)
    instance_dir = output_dir / utils.instance_dir_name(meta)
    instance_dir.mkdir(parents=True, exist_ok=True)

    # For strategy B with n <= 90: brute-force the max clique here so later
    # phases have a reference clique for permutation targeting.
    reference_nodes = meta.get("planted_nodes")
    if strategy == "B" and params["n"] <= utils.BRUTE_FORCE_MAX_N:
        bf = utils.solve_bruteforce(A)
        reference_nodes = bf["max_clique"]
        meta["reference_nodes"] = reference_nodes
        meta["reference_omega"] = bf["omega"]
        meta["reference_source"] = "brute_force"
        meta["bruteforce"] = {
            "omega": bf["omega"],
            "num_max_cliques": bf["num_max_cliques"],
            "num_maximal_cliques": bf["num_maximal_cliques"],
            "degeneracy": bf["degeneracy"],
            "clique_core_gap": bf["clique_core_gap"],
            "solve_time": bf["solve_time"],
            "timed_out": bf["timed_out"],
        }
        print(f"  [{meta['name']}] brute-force ω = {bf['omega']} "
              f"(reference clique: {reference_nodes})")
    else:
        if reference_nodes is not None:
            meta["reference_nodes"] = reference_nodes
            meta["reference_omega"] = meta.get("planted_omega")
            meta["reference_source"] = "planted"

    # Determine k for the reference clique (for permutation logic)
    if reference_nodes is not None:
        meta["reference_k"] = len(reference_nodes)

    # Compute theoretical g* if we know omega
    omega_ref = meta.get("reference_omega")
    if omega_ref and omega_ref > 1:
        meta["g_star"] = omega_to_scaled_objective(omega_ref, args.R)

    # base_meta.json
    base_meta = {
        "instance_name": meta["name"],
        "strategy": meta["strategy"],
        "n": meta["n"],
        "p": meta.get("p"),
        "seed": meta["seed"],
        "R": args.R,
        "timestamp_generated": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in meta.items() if k not in ("name", "strategy", "n", "p", "seed")},
    }

    with (instance_dir / "base_meta.json").open("w") as f:
        json.dump(base_meta, f, indent=2)
        f.write("\n")

    np.savez(instance_dir / "base_adjacency.npz", A=A)
    utils.mark_phase_done(instance_dir, "generate")
    print(f"  [{meta['name']}] generate: saved to {instance_dir}")
    return instance_dir


# ---------------------------------------------------------------------------
# Phase 2: variants
# ---------------------------------------------------------------------------

def cmd_variants(instance_dir: Path, args: argparse.Namespace) -> None:
    """Create 3 permutation variants (or 1+ random-only for B with n > 90)."""
    base_meta = json.loads((instance_dir / "base_meta.json").read_text())
    A = np.load(instance_dir / "base_adjacency.npz")["A"]
    strategy = base_meta["strategy"]
    n = base_meta["n"]
    R = base_meta["R"]

    reference_nodes = base_meta.get("reference_nodes")
    G = utils._graph_from_adjacency(A)

    if strategy.startswith("B") and reference_nodes is None:
        # n > BRUTE_FORCE_MAX_N branch: random-only variants
        num_variants = args.num_random_variants
        print(f"  [{base_meta['instance_name']}] variants: strategy B, no reference clique "
              f"(n={n} > {utils.BRUTE_FORCE_MAX_N}); generating {num_variants} random variant(s)")
        for i in range(num_variants):
            variant_name = "random" if num_variants == 1 else f"random{i+1}"
            A_v, fwd, inv = utils.build_random_variant(G, seed=base_meta["seed"] + 3000 + i)
            utils.write_variant_artifacts(
                variant_dir=instance_dir / variant_name,
                variant_name=variant_name,
                A_variant=A_v,
                forward_perm=fwd,
                inverse_perm=inv,
                base_meta=base_meta,
                target_positions=None,
                reference_nodes=[],
                k=0,
                R=R,
            )
            print(f"    [{variant_name}] adjacency saved (random scramble)")
        utils.mark_phase_done(instance_dir, "variants")
        return

    if reference_nodes is None:
        raise SystemExit(
            f"Instance {base_meta['instance_name']} has no reference clique. "
            f"For strategy B with n > {utils.BRUTE_FORCE_MAX_N}, no variants can be built. "
            f"For A/C/D/E this is a bug — report it."
        )

    k = base_meta["reference_k"]
    targets = utils.compute_variant_targets(reference_nodes, n, k, base_meta["seed"])

    for variant_name in utils.VARIANTS_PLANTED:
        A_v, fwd, inv = utils.build_planted_variant(
            G, reference_nodes, n, k, targets[variant_name], variant_name,
        )
        utils.write_variant_artifacts(
            variant_dir=instance_dir / variant_name,
            variant_name=variant_name,
            A_variant=A_v,
            forward_perm=fwd,
            inverse_perm=inv,
            base_meta=base_meta,
            target_positions=targets[variant_name],
            reference_nodes=reference_nodes,
            k=k,
            R=R,
        )
        print(f"    [{variant_name}] sanity check PASSED, targets={targets[variant_name]}")

    utils.mark_phase_done(instance_dir, "variants")


# ---------------------------------------------------------------------------
# Phase 3: verify
# ---------------------------------------------------------------------------

def cmd_verify(instance_dir: Path, args: argparse.Namespace) -> None:
    """Brute-force verify omega, save theoretical solutions, plot distribution."""
    base_meta = json.loads((instance_dir / "base_meta.json").read_text())
    R = base_meta["R"]

    G_base = None   # networkx.Graph — populated on first variant iteration
    known_omega: int | None = base_meta.get("reference_omega")

    for variant_dir in sorted(p for p in instance_dir.iterdir() if p.is_dir()):
        variant_name = variant_dir.name
        meta_path = variant_dir / f"{variant_name}_meta.json"
        if not meta_path.exists():
            continue

        # Rebuild graph from CSV to verify
        csv_path = variant_dir / f"{variant_name}_boson14.csv"
        raw = np.loadtxt(csv_path, delimiter=",")
        A_v = raw[:, 1:]  # skip C column
        G_v = utils._graph_from_adjacency(A_v)

        # Save theoretical solutions (skip if no reference clique)
        meta = json.loads(meta_path.read_text())
        if meta.get("reference_nodes"):
            stats = utils.write_theoretical_solutions(variant_dir, variant_name, G_v, R)
            meta.update(stats)
        else:
            # Still brute-force for verification
            bf = utils.solve_bruteforce(A_v)
            meta["omega_verified"] = bf["omega"]
            meta["num_max_cliques"] = bf["num_max_cliques"]

        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
        ov = meta.get("omega_verified", "?")
        print(f"    [{variant_name}] brute-force ω = {ov}")

        if known_omega is None:
            known_omega = meta.get("omega_verified")
        if G_base is None:
            G_base = G_v

    # Verify-phase plot: clique-size + objective histograms (one per instance).
    # The 3 variants are isomorphic (permutations of the same graph) so their
    # maximal-clique distributions are identical — plotting once is sufficient.
    if not args.no_verify_plot and G_base is not None:
        dist = compute_clique_distribution(G_base, R)
        plot_clique_distribution(
            dist,
            graph_name=base_meta["instance_name"],
            R=R,
            known_omega=known_omega,
            save_path=str(instance_dir),
        )
        print(f"    [verify-plot] sizes={dict(sorted(dist['size_counts'].items()))}, "
              f"saved clique_dist_{base_meta['instance_name']}.png")

    # Optional: run alternative solvers (classical + Dirac-3 cloud) on the
    # random variant only. All 3 variants are isomorphic copies of the same
    # graph, so for solver comparison one variant is enough — unlike the
    # hardware phase which is position-dependent.
    if args.classical_solve or args.dirac_solve:
        random_dir = instance_dir / "random"
        random_csv = random_dir / "random_boson14.csv"
        if not random_csv.exists():
            print("    [alt-solvers] no random/random_boson14.csv — skipping")
        else:
            print(f"    [alt-solvers] running on random variant (isomorphic to all variants)")
            A_rand = np.loadtxt(random_csv, delimiter=",")[:, 1:]
            true_omega = known_omega  # from bruteforce above, or reference_omega

            classical_results = None
            dirac_result = None

            if args.classical_solve:
                classical_results = utils.run_classical_solvers(
                    A_rand, R,
                    true_omega=true_omega,
                    num_restarts=args.classical_restarts,
                    seed=base_meta.get("seed", 42),
                )

            if args.dirac_solve:
                try:
                    dirac_result = utils.run_dirac_solver(
                        A_rand, R,
                        num_samples=args.dirac_samples,
                        relaxation_schedule=args.dirac_schedule,
                    )
                    print(f"    [Dirac-3] best ω={dirac_result['best_omega']}, "
                          f"best g={dirac_result['best_objective']:.2f}, "
                          f"time={dirac_result['solve_time']:.1f}s")
                except Exception as e:  # noqa: BLE001
                    print(f"    [Dirac-3] ERROR: {e}")
                    dirac_result = None

            utils.save_alternative_results(
                variant_dir=random_dir,
                classical_results=classical_results,
                dirac_result=dirac_result,
                R=R,
                true_omega=true_omega,
            )

            if classical_results or dirac_result:
                # k_ref for threshold support — use known omega if available, else best_omega
                k_ref = (
                    known_omega
                    or base_meta.get("reference_omega")
                    or base_meta.get("k")
                    or max(3, base_meta["n"] // 4)
                )
                # Need brute-force distribution on the random variant specifically
                # (isomorphic to base, but A_rand is what the y-vectors are scored against)
                G_rand = utils._graph_from_adjacency(A_rand)
                bf_dist_rand = compute_clique_distribution(G_rand, R)
                saved = utils.plot_alternative_comparison(
                    classical_results=classical_results,
                    dirac_result=dirac_result,
                    A=A_rand,
                    bruteforce_dist=bf_dist_rand,
                    R=R,
                    k_ref=k_ref,
                    known_omega=known_omega,
                    graph_name=base_meta["instance_name"],
                    variant_name="random",
                    save_path=random_dir,
                )
                for p_saved in saved:
                    print(f"    [alt-solvers] plot -> {p_saved.name}")

    utils.mark_phase_done(instance_dir, "verify")


# ---------------------------------------------------------------------------
# Phase 4: hardware
# ---------------------------------------------------------------------------

def _analyze_and_plot_hardware(
    instance_dir: Path,
    variant_dir: Path,
    variant_name: str,
    npz_path: Path,
    amp: int,
    args: argparse.Namespace,
) -> dict:
    """Analyze one hardware NPZ, update variant_meta with stats, optionally plot.

    Returns the analysis dict (with `fraction_valid`, `clique_size_counts`, ...).
    """
    base_meta = json.loads((instance_dir / "base_meta.json").read_text())
    R = base_meta["R"]

    # Reference k for threshold extraction (tried in order)
    k_ref = (
        base_meta.get("reference_omega")
        or base_meta.get("planted_omega")
        or base_meta.get("k")
        or max(3, base_meta["n"] // 4)
    )

    csv_path = variant_dir / f"{variant_name}_boson14.csv"
    raw = np.loadtxt(csv_path, delimiter=",")
    A = raw[:, 1:]
    data = np.load(npz_path, allow_pickle=True)
    sigma = data["sigma_array"]

    analysis = utils.analyze_hardware_samples(sigma, A, R, k_ref)

    # Persist per-amplitude summary into variant_meta.json
    meta_path = variant_dir / f"{variant_name}_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        hw_samples = meta.setdefault("hardware_samples", {})
        hw_samples[str(amp)] = {
            "fraction_valid_cliques": round(analysis["fraction_valid"], 4),
            "num_samples": len(analysis["raw_objectives"]),
            "clique_size_counts": analysis["clique_size_counts"],
            "raw_g_mean": float(np.mean(analysis["raw_objectives"])),
            "raw_g_max": float(np.max(analysis["raw_objectives"])),
            "raw_omega_max": int(max(analysis["raw_omegas"])),
        }
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")

    if not args.no_hardware_plot:
        # Brute-force distribution is identical across variants (isomorphic graphs).
        # Rebuild it from this variant's adjacency (same result for any variant).
        G = utils._graph_from_adjacency(A)
        bf_dist = compute_clique_distribution(G, R)
        known_omega = base_meta.get("reference_omega")
        fname = utils.plot_hardware_distribution(
            analysis=analysis,
            bruteforce_dist=bf_dist,
            graph_name=base_meta["instance_name"],
            variant_name=variant_name,
            amplitude=amp,
            R=R,
            known_omega=known_omega,
            save_path=variant_dir,
        )
        print(f"    [{variant_name} a={amp}] plot -> {fname.name} "
              f"(valid-clique fraction {analysis['fraction_valid']:.0%}, "
              f"sizes={analysis['clique_size_counts']})")

    return analysis


def cmd_hardware(instance_dir: Path, args: argparse.Namespace) -> None:
    """For each variant × amplitude, call run_boson14.py, post-process NPZ, rename, analyze."""
    if not RUN_BOSON14_PATH.exists():
        raise SystemExit(f"run_boson14.py not found at {RUN_BOSON14_PATH}")

    for variant_dir in sorted(p for p in instance_dir.iterdir() if p.is_dir()):
        variant_name = variant_dir.name
        csv_path = variant_dir / f"{variant_name}_boson14.csv"
        if not csv_path.exists():
            continue

        for amp in args.amplitudes:
            flags = HardwareFlags(
                amplitude=amp,
                R=args.R,
                num_samples=args.num_samples,
                num_loops=args.num_loops,
                delay=args.delay,
                pulse_width=args.pulse_width,
                distance_between_pulses=args.distance_between_pulses,
                solps=args.solps,
            )

            # Check if this amplitude already ran (idempotent)
            existing = list(variant_dir.glob(f"results__{variant_name}_boson14__*_a{amp}.npz"))
            if existing and args.skip_existing:
                print(f"    [{variant_name} a={amp}] skipping (already exists: {existing[0].name})")
                # Still produce the plot from the existing NPZ so resumed pipelines
                # don't lose their figures.
                _analyze_and_plot_hardware(
                    instance_dir, variant_dir, variant_name, existing[0], amp, args,
                )
                continue

            cmd = [
                sys.executable,
                str(RUN_BOSON14_PATH),
                csv_path.name,   # relative to cwd=variant_dir
                *flags.as_cli_args(),
            ]

            if args.dry_run:
                print(f"    [{variant_name} a={amp}] [dry-run] would run: {' '.join(cmd)} "
                      f"(cwd={variant_dir})")
                continue

            print(f"    [{variant_name} a={amp}] running run_boson14.py...")
            t0 = time.time()
            result = subprocess.run(cmd, cwd=variant_dir, capture_output=False)
            device_time_s = time.time() - t0
            run_timestamp = datetime.now(timezone.utc).isoformat()
            if result.returncode != 0:
                raise SystemExit(
                    f"run_boson14.py failed (exit {result.returncode}) for "
                    f"{variant_dir}/{csv_path.name} at amplitude={amp}. Aborting."
                )

            # Find and post-process the newest NPZ
            short_filename = csv_path.stem  # e.g. "front_boson14"
            try:
                npz_path, plot_path = utils.find_latest_result_files(variant_dir, short_filename)
            except FileNotFoundError as e:
                raise SystemExit(f"Hardware run succeeded but no output NPZ found: {e}")

            utils.postprocess_npz(npz_path, flags, device_time_wrapper_s=device_time_s)
            new_npz, new_plot = utils.rename_with_amplitude(npz_path, plot_path, amp)

            # Record timing entry in the instance-level summary
            utils.append_timing_entry(
                instance_dir=instance_dir,
                variant_name=variant_name,
                amplitude=amp,
                device_time_s=device_time_s,
                timestamp=run_timestamp,
                flags=flags,
                result_filename=new_npz.name,
            )

            print(f"    [{variant_name} a={amp}] saved {new_npz.name}"
                  f"{' + ' + new_plot.name if new_plot else ''} "
                  f"(device_time={device_time_s:.1f}s)")

            # Analyze + plot (writes valid-clique fraction to variant meta)
            _analyze_and_plot_hardware(
                instance_dir, variant_dir, variant_name, new_npz, amp, args,
            )

    if not args.dry_run:
        utils.mark_phase_done(instance_dir, "hardware")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    phases = _phases_to_run(args)
    print(f"\nPipeline phases to run: {phases}")
    print(f"Strategies: {args.strategies}  |  Seeds: {args.seeds}  |  n={args.n}")
    print(f"Output dir: {args.output_dir.resolve()}")

    # Auto-enable verify for strategy B with small n (needed for permutation targeting)
    if "B" in args.strategies and args.n <= utils.BRUTE_FORCE_MAX_N and not args.verify:
        print("  Note: strategy B with n <= 90 — brute-force is run in generate phase automatically.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for strategy, seed in product(args.strategies, args.seeds):
        print(f"\n==== Strategy {strategy}, seed {seed} ====")
        params = _build_params(strategy, args, seed)

        # Resolve instance_dir without running generate if we're skipping it
        if "generate" in phases:
            if args.skip_existing:
                # Need to know the instance name to check marker; build dry to get meta
                A_dry, meta_dry = utils.build_instance(strategy, params)
                probe_dir = args.output_dir / utils.instance_dir_name(meta_dry)
                if utils.is_phase_done(probe_dir, "generate"):
                    print(f"  [{meta_dry['name']}] generate: SKIP (marker exists)")
                    instance_dir = probe_dir
                else:
                    instance_dir = cmd_generate(strategy, params, args.output_dir, args)
            else:
                instance_dir = cmd_generate(strategy, params, args.output_dir, args)
        else:
            A_dry, meta_dry = utils.build_instance(strategy, params)
            instance_dir = args.output_dir / utils.instance_dir_name(meta_dry)
            if not (instance_dir / "base_meta.json").exists():
                raise SystemExit(
                    f"Instance {meta_dry['name']} not generated yet. Run --phase generate first."
                )

        if "variants" in phases:
            if args.skip_existing and utils.is_phase_done(instance_dir, "variants"):
                print(f"  [{instance_dir.name}] variants: SKIP (marker exists)")
            else:
                print(f"  [{instance_dir.name}] variants...")
                cmd_variants(instance_dir, args)

        if "verify" in phases and (args.verify or strategy == "B"):
            if args.skip_existing and utils.is_phase_done(instance_dir, "verify"):
                print(f"  [{instance_dir.name}] verify: SKIP (marker exists)")
            else:
                print(f"  [{instance_dir.name}] verify...")
                cmd_verify(instance_dir, args)

        if "hardware" in phases:
            if args.skip_existing and utils.is_phase_done(instance_dir, "hardware"):
                print(f"  [{instance_dir.name}] hardware: SKIP (marker exists)")
            else:
                print(f"  [{instance_dir.name}] hardware...")
                cmd_hardware(instance_dir, args)

    print("\nPipeline complete.\n")


if __name__ == "__main__":
    main()
