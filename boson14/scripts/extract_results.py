"""Extract raw per-sample data from boson14 result .npz files.

For every `results__{variant}_boson14__{ts}_a{amp}.npz` under
`boson14/output/benchmark_*/{variant}/`, this computes

    g       = 0.5 * y^T A y                     (scaled MS objective)
    omega   = round(R^2 / (R^2 - 2 g))          (estimated clique number)

for each of the 100 samples. Results are written to

    scripts/cache/all_samples.csv   (one row per sample)
    scripts/cache/summary.json      (aggregate stats per run)

Both files are regenerated from scratch on each invocation.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np
import pandas as pd

from _common import CACHE_DIR, RunFile, iter_runs, load_meta

# Pull MS helpers from the existing package.
sys.path.insert(0, str((CACHE_DIR / "../../boson14").resolve()))
from boson14_bench.core import (  # noqa: E402
    scaled_objective,
    scaled_objective_to_omega,
)

# Energy <-> objective self-check tolerance.
# energy_list[i] should equal y^T A y = 2 * g(y).
ENERGY_TOL = 1e-4


def process_run(run: RunFile) -> tuple[list[dict], dict]:
    """Return (per-sample rows, per-run summary) for one results file."""
    meta = load_meta(run)
    k = int(meta["k"])
    n = int(meta["n"])
    R = int(meta["R"])
    g_star = float(meta["g_star"])

    data = np.load(run.npz_path, allow_pickle=True)
    sigma = data["sigma_array"]          # (num_samples, n)
    energy_list = data["energy_list"]    # (num_samples,)
    A = data["quadratic_term"].astype(np.float64)  # verified: +A, not -A
    R_npz = int(data["R"])
    if R_npz != R:
        raise RuntimeError(f"R mismatch for {run.npz_path}: meta={R} npz={R_npz}")

    rows: list[dict] = []
    for i in range(sigma.shape[0]):
        y = sigma[i]
        g = float(scaled_objective(y, A))
        # Consistency check against boson14's reported energy (energy == 2g).
        reported = float(energy_list[i])
        if not np.isclose(reported, 2.0 * g, rtol=0, atol=ENERGY_TOL * max(1.0, abs(g))):
            # Soft warning rather than hard failure — stale files exist in the wild.
            print(
                f"  [warn] {run.npz_path.name} sample {i}: "
                f"energy_list={reported:.6f} vs 2g={2*g:.6f}"
            )
        omega_est = int(scaled_objective_to_omega(g, R))
        rows.append({
            "instance": run.instance,
            "n": n,
            "k": k,
            "variant": run.variant,
            "amplitude": run.amplitude,
            "timestamp": run.timestamp,
            "sample_idx": i,
            "g": g,
            "energy": 2.0 * g,         # boson14 convention for reference
            "omega_est": omega_est,
            "g_star": g_star,
            "omega_planted": k,
        })

    objs = np.array([r["g"] for r in rows])
    omegas = np.array([r["omega_est"] for r in rows])
    num_hit = int(np.sum(omegas >= k))
    within_5pct = int(np.sum(objs >= 0.95 * g_star))

    summary = {
        "instance": run.instance,
        "variant": run.variant,
        "amplitude": run.amplitude,
        "timestamp": run.timestamp,
        "n": n,
        "k": k,
        "R": R,
        "g_star": g_star,
        "num_samples": len(rows),
        "best_g": float(objs.max()),
        "mean_g": float(objs.mean()),
        "best_omega": int(omegas.max()),
        "mean_omega": float(omegas.mean()),
        "frac_omega_at_least_k": num_hit / len(rows),
        "frac_within_5pct_of_gstar": within_5pct / len(rows),
        "omega_counts": {str(k_): int(v) for k_, v in sorted(Counter(omegas.tolist()).items())},
    }
    return rows, summary


def main() -> int:
    runs = iter_runs()
    if not runs:
        print("No runs found under boson14/output — nothing to do.")
        return 1

    print(f"Found {len(runs)} result files across "
          f"{len({r.instance for r in runs})} instances")

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for run in runs:
        rows, summary = process_run(run)
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"  [{run.instance}/{run.variant:>6s} a{run.amplitude}] "
            f"best g={summary['best_g']:8.2f}  "
            f"g*={summary['g_star']:8.2f}  "
            f"omega_hit={summary['frac_omega_at_least_k']:.0%}  "
            f"best_omega={summary['best_omega']}"
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CACHE_DIR / "all_samples.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nWrote {len(all_rows)} rows -> {csv_path}")

    summary_path = CACHE_DIR / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summaries, f, indent=2)
        f.write("\n")
    print(f"Wrote summary for {len(summaries)} runs -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
