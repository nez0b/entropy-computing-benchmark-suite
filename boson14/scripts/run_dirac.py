"""Submit all six benchmark graphs to Dirac-3 at three relaxation schedules.

Shells out to `boson14/test_dirac.py --from-json ...` for each
(instance, schedule) pair. Idempotent: skips any run whose NPZ already
exists so the orchestrator can be resumed after an interrupted batch.

Output tree:

    boson14/output/dirac_runs/{instance}/rs{S}/{instance_name}/
        {instance_name}_dirac_solutions.npz
        {instance_name}_dirac_meta.json
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from _common import BOSON14_DIR, OUTPUT_DIR

SCHEDULES = (2, 3, 4)

# (instance_dir_name, preferred_variant_for_json_source)
# n100_k63 lacks a random/ folder, so fall back to front/.
INSTANCES = [
    ("benchmark_n20_k5_p0.3_s42",    "random"),
    ("benchmark_n20_k6_p0.2_s42",    "random"),
    ("benchmark_n50_k10_p0.3_s42",   "random"),
    ("benchmark_n50_k22_p0.5_s42",   "random"),
    ("benchmark_n100_k22_p0.2_s42",  "random"),
    ("benchmark_n100_k63_p0.7_s42",  "front"),
]


def expected_dirac_npz(instance: str, schedule: int) -> Path:
    """Path where test_dirac.py will write the NPZ for a given run.

    Mirrors test_dirac.py's logic: {output_dir}/{instance_name}/*_dirac_solutions.npz
    where instance_name is derived from (n,k,p,seed) parsed from the JSON.
    We match on glob because n/k/p/seed aren't trivially re-derivable here.
    """
    out_dir = (
        OUTPUT_DIR / "dirac_runs" / instance / f"rs{schedule}"
    )
    if not out_dir.exists():
        return out_dir / "MISSING.npz"
    # Find *_dirac_solutions.npz under any subfolder.
    matches = list(out_dir.rglob("*_dirac_solutions.npz"))
    return matches[0] if matches else (out_dir / "MISSING.npz")


def run_one(instance: str, variant: str, schedule: int) -> tuple[bool, float]:
    """Run one Dirac submission; return (skipped_or_success, elapsed_seconds)."""
    existing = expected_dirac_npz(instance, schedule)
    if existing.exists():
        print(f"  [{instance} rs{schedule}] already exists — skipping")
        return True, 0.0

    json_path = OUTPUT_DIR / instance / variant / f"{variant}_boson14.json"
    if not json_path.exists():
        print(f"  [{instance} rs{schedule}] MISSING json at {json_path}")
        return False, 0.0

    out_dir = OUTPUT_DIR / "dirac_runs" / instance / f"rs{schedule}"

    cmd = [
        "uv", "run", "python", "test_dirac.py",
        "--from-json", str(json_path.relative_to(BOSON14_DIR)),
        "--num-samples", "100",
        "--relaxation-schedule", str(schedule),
        "--output-dir", str(out_dir.relative_to(BOSON14_DIR)),
    ]
    print(f"  [{instance} rs{schedule}] submitting ...")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=BOSON14_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=900,
        )
    except subprocess.TimeoutExpired:
        print(f"  [{instance} rs{schedule}] TIMED OUT after 15 min")
        return False, time.time() - t0
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [{instance} rs{schedule}] FAILED (exit {result.returncode})")
        # Print the last 20 lines of output for diagnosis.
        print("  ---------- last 20 lines ----------")
        for line in result.stdout.splitlines()[-20:]:
            print(f"    {line}")
        print("  -----------------------------------")
        return False, elapsed

    # Extract "Best Dirac g" from the output for a quick summary line.
    best_g = "?"
    for line in result.stdout.splitlines():
        if line.strip().startswith("Best Dirac g:"):
            best_g = line.split(":", 1)[1].strip()
            break
    print(f"  [{instance} rs{schedule}] done in {elapsed:.1f}s  best g={best_g}")
    return True, elapsed


def main() -> int:
    total_runs = len(INSTANCES) * len(SCHEDULES)
    done = 0
    failed: list[tuple[str, int]] = []
    total_time = 0.0

    print(f"Running {total_runs} Dirac submissions ({len(INSTANCES)} graphs × {len(SCHEDULES)} schedules)")

    for instance, variant in INSTANCES:
        for schedule in SCHEDULES:
            ok, elapsed = run_one(instance, variant, schedule)
            total_time += elapsed
            if ok:
                done += 1
            else:
                failed.append((instance, schedule))

    print(f"\n{'='*60}")
    print(f"DONE: {done}/{total_runs} runs succeeded; total elapsed {total_time:.1f}s")
    if failed:
        print("FAILED:")
        for inst, s in failed:
            print(f"  - {inst} rs{s}")
    print(f"{'='*60}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
