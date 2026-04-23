"""Per-graph histogram overlaying Dirac-3 (rs 2/3/4) and boson14 (a=600).

For each of the six benchmark instances, loads:
  - Dirac y_vectors from output/dirac_runs/{instance}/rs{S}/*/_dirac_solutions.npz
    (re-computes g = 0.5 y^T A y using the *base* adjacency regenerated
    from (n,k,p,seed), since Dirac runs on the base graph via --from-json)
  - boson14 g-values from scripts/cache/all_samples.csv filtered to
    amplitude = 600 for whichever variant we submitted to Dirac
    (random, or front for n100_k63).

Produces one PNG per instance at report/figures/dirac_overlay/{instance}.png,
each with 4 overlaid histograms (boson14 a=600, Dirac rs2, rs3, rs4) plus
the MS theoretical line at g*.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import BOSON14_DIR, CACHE_DIR, FIGURES_DIR, OUTPUT_DIR

sys.path.insert(0, str(BOSON14_DIR))
from boson14_bench.planted_clique import generate_planted_clique  # noqa: E402
from boson14_bench.problems import motzkin_straus_adjacency       # noqa: E402
from boson14_bench.core import scaled_objective                    # noqa: E402


# (instance, variant_used_for_json_submission) — mirror run_dirac.py choices.
INSTANCES = [
    ("benchmark_n20_k5_p0.3_s42",    "random"),
    ("benchmark_n20_k6_p0.2_s42",    "random"),
    ("benchmark_n50_k10_p0.3_s42",   "random"),
    ("benchmark_n50_k22_p0.5_s42",   "random"),
    ("benchmark_n100_k22_p0.2_s42",  "random"),
    ("benchmark_n100_k63_p0.7_s42",  "front"),
]

SCHEDULES = (2, 3, 4)

# Palette: boson14 fixed (reused from other figures); Dirac schedules use a
# cool-palette trio so Dirac vs boson14 is visually distinct at a glance.
BOSON14_COLOUR = "#DD8452"          # orange — boson14 @ a=600
DIRAC_COLOURS = {
    2: "#4C72B0",   # blue
    3: "#8172B2",   # purple
    4: "#2F8F9D",   # teal
}


def _load_dirac_y(instance: str, schedule: int) -> np.ndarray | None:
    """Return y_vectors (num_samples, n) or None if run missing."""
    d = OUTPUT_DIR / "dirac_runs" / instance / f"rs{schedule}"
    matches = list(d.rglob("*_dirac_solutions.npz"))
    if not matches:
        return None
    data = np.load(matches[0])
    return data["y_vectors"]


def _regen_adjacency(instance: str) -> tuple[np.ndarray, int, float, int]:
    """Return (A, k, g_star, n) for the base graph."""
    meta = json.load((OUTPUT_DIR / instance / "base_meta.json").open())
    n, k, p, seed = int(meta["n"]), int(meta["k"]), float(meta["p"]), int(meta["seed"])
    g_star = float(meta["g_star"])
    G, _ = generate_planted_clique(n, k, p=p, seed=seed)
    A = motzkin_straus_adjacency(G)
    return A, k, g_star, n


def render_instance(instance: str, variant: str, df_b14: pd.DataFrame, out_path: Path) -> bool:
    A, k, g_star, n = _regen_adjacency(instance)

    # Dirac g-distributions per schedule.
    dirac_g: dict[int, np.ndarray] = {}
    for s in SCHEDULES:
        y = _load_dirac_y(instance, s)
        if y is None:
            continue
        g_vals = np.array([scaled_objective(y[i], A) for i in range(y.shape[0])])
        dirac_g[s] = g_vals[np.isfinite(g_vals)]

    if not dirac_g:
        print(f"  [{instance}] no Dirac results found — skipping")
        return False

    # Boson14 a=600 for the matching variant.
    b14_mask = (
        (df_b14["instance"] == instance)
        & (df_b14["variant"] == variant)
        & (df_b14["amplitude"] == 600)
    )
    b14_vals = df_b14.loc[b14_mask, "g"].to_numpy()

    all_vals = [b14_vals] + list(dirac_g.values())
    all_vals = [v for v in all_vals if len(v)]
    g_min = min(v.min() for v in all_vals)
    g_max = max(max(v.max() for v in all_vals), g_star)
    pad = 0.015 * (g_max - g_min + 1)
    bins = np.linspace(g_min - pad, g_max + pad, 45)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    if len(b14_vals):
        ax.hist(b14_vals, bins=bins, color=BOSON14_COLOUR, edgecolor="white",
                linewidth=0.4, alpha=0.65,
                label=f"boson14 a=600 ({variant}, {len(b14_vals)} samples)")

    for s in SCHEDULES:
        if s not in dirac_g:
            continue
        ax.hist(dirac_g[s], bins=bins, color=DIRAC_COLOURS[s], edgecolor="white",
                linewidth=0.4, alpha=0.55,
                label=f"Dirac-3 rs={s} ({len(dirac_g[s])} samples)")

    ax.axvline(g_star, color="red", linestyle="--", linewidth=1.4,
               label=fr"$g^*(\omega\!=\!{k})={g_star:.1f}$")

    ax.set_title(f"{instance}  (n={n}, k={k})", fontsize=11)
    ax.set_xlabel(r"$g(y) = \frac{1}{2}\, y^T A\, y$")
    ax.set_ylabel("sample count")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    csv_path = CACHE_DIR / "all_samples.csv"
    if not csv_path.exists():
        print(f"Missing {csv_path} — run extract_results.py first.")
        return 1

    df = pd.read_csv(csv_path)
    out_dir = FIGURES_DIR / "dirac_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for instance, variant in INSTANCES:
        out_path = out_dir / f"{instance}.png"
        if render_instance(instance, variant, df, out_path):
            rendered += 1
            print(f"  wrote {out_path.name}")
    print(f"\nRendered {rendered}/{len(INSTANCES)} overlay figures -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
