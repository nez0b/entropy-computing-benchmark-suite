"""Per-instance figure: boson14 raw vs equal-weight vs 1-opt repaired g.

Reads scripts/cache/boson14_repair.csv and emits one PNG per instance at
report/figures/repair/{instance}.png. Focus is amplitude = 600 on the same
variant used for the Dirac overlay (random for most, front for n100_k63),
so these figures slot into the same storyline.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import CACHE_DIR, FIGURES_DIR

INSTANCES = [
    ("benchmark_n20_k5_p0.3_s42",    "random"),
    ("benchmark_n20_k6_p0.2_s42",    "random"),
    ("benchmark_n50_k10_p0.3_s42",   "random"),
    ("benchmark_n50_k22_p0.5_s42",   "random"),
    ("benchmark_n100_k22_p0.2_s42",  "random"),
    ("benchmark_n100_k63_p0.7_s42",  "front"),
]
AMPLITUDE = 600

# Colour scheme: raw vs post-processing variants (all orange family so it's
# visually distinct from the Dirac-vs-boson14 overlays which use blue/teal).
COLOUR_RAW      = "#DD8452"   # boson14 signature orange
COLOUR_EQUAL    = "#1F77B4"   # blue — threshold-support equal-weight (high-contrast)
COLOUR_REPAIR   = "#C44E52"   # red-ish (post-process: 1-opt repaired)


def render_instance(df_i: pd.DataFrame, instance: str, variant: str, out_path) -> bool:
    if df_i.empty:
        print(f"  [{instance}] no rows at a={AMPLITUDE}, variant={variant} — skip")
        return False

    k = int(df_i["k"].iloc[0])
    g_star = float(df_i["g_star"].iloc[0])
    R = int(df_i["R"].iloc[0])

    raw = df_i["g_raw"].to_numpy()
    eq  = df_i["g_equal_weight"].to_numpy()
    eq  = eq[~np.isnan(eq)]
    rep = df_i["g_repaired"].to_numpy()

    # Shared bin range across all three series so they're visually comparable.
    vals = [raw] + ([eq] if len(eq) else []) + [rep]
    g_min = min(v.min() for v in vals)
    g_max = max(max(v.max() for v in vals), g_star)
    pad = 0.01 * (g_max - g_min + 1)
    bins = np.linspace(g_min - pad, g_max + pad, 50)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    # Z-order: draw 1-opt first (background, faint), raw on top of it, and
    # equal-weight last so it's visible where it overlaps the 1-opt peak.
    ax.hist(rep, bins=bins, color=COLOUR_REPAIR, edgecolor=COLOUR_REPAIR,
            linewidth=1.2, alpha=0.18,
            label=f"1-opt repaired ({len(rep)} samples)")
    # Thin step outline on top of the faint 1-opt bar so its position stays
    # readable without obscuring what's underneath.
    ax.hist(rep, bins=bins, histtype="step", edgecolor=COLOUR_REPAIR,
            linewidth=1.2, alpha=0.9)

    ax.hist(raw, bins=bins, color=COLOUR_RAW, edgecolor="white",
            linewidth=0.4, alpha=0.6,
            label=f"raw: $g(y)$  ({len(raw)} samples)")

    if len(eq):
        ax.hist(eq, bins=bins, color=COLOUR_EQUAL, edgecolor="white",
                linewidth=0.4, alpha=0.85,
                label=f"equal-weight on threshold-support  ({len(eq)} valid)")

    # MS-theoretical landmarks at ω, ω-1, ω-2. All three are always drawn
    # and included in xlim so the user can see where each suboptimal
    # clique size would land, even when the gap is large for small k.
    ax.axvline(g_star, color="black", linestyle="--", linewidth=1.3,
               label=fr"$g^*(\omega\!=\!{k})={g_star:.1f}$")
    # Landmark visibility pad: proportional to g* so the leftmost vertical
    # line sits comfortably inside the plot, regardless of how narrow the
    # raw-data range is.
    landmark_pad = 0.02 * g_star
    xmin_candidates = [g_min - pad]
    if k >= 2:
        g_m1 = (R ** 2 / 2.0) * (1.0 - 1.0 / (k - 1))
        ax.axvline(g_m1, color="#555555", linestyle="--", linewidth=0.9,
                   label=fr"$g^*(\omega\!-\!1)={g_m1:.1f}$")
        xmin_candidates.append(g_m1 - landmark_pad)
    if k >= 3:
        g_m2 = (R ** 2 / 2.0) * (1.0 - 1.0 / (k - 2))
        ax.axvline(g_m2, color="#999999", linestyle=":", linewidth=0.9,
                   label=fr"$g^*(\omega\!-\!2)={g_m2:.1f}$")
        xmin_candidates.append(g_m2 - landmark_pad)
    ax.set_xlim(min(xmin_candidates), g_max + pad)

    frac_valid = (df_i["supp_is_clique"].sum() / len(df_i)) * 100.0
    frac_supp_k = ((df_i["supp_is_clique"] & (df_i["supp_size"] == k)).sum()
                   / len(df_i)) * 100.0
    rep_size_mode = int(df_i["repaired_size"].mode().iloc[0])
    frac_rep_k = ((df_i["repaired_size"] == k).sum() / len(df_i)) * 100.0

    subtitle = (
        f"variant={variant}, $a$={AMPLITUDE}  "
        f"|  valid-support: {frac_valid:.0f}%  "
        f"|  $|S|{{=}}k$: {frac_supp_k:.0f}%  "
        f"|  1-opt$\\to k$: {frac_rep_k:.0f}%"
    )
    ax.set_title(f"{instance}  (n={df_i['n'].iloc[0]}, k={k})\n{subtitle}",
                 fontsize=10)
    ax.set_xlabel(r"$g(y) = \frac{1}{2}\, y^T A\, y$")
    ax.set_ylabel("sample count")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    csv_path = CACHE_DIR / "boson14_repair.csv"
    if not csv_path.exists():
        print(f"Missing {csv_path} — run boson14_repair.py first.")
        return 1

    df = pd.read_csv(csv_path)
    out_dir = FIGURES_DIR / "repair"
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for instance, variant in INSTANCES:
        sub = df[
            (df["instance"] == instance)
            & (df["variant"] == variant)
            & (df["amplitude"] == AMPLITUDE)
        ]
        out_path = out_dir / f"{instance}.png"
        if render_instance(sub, instance, variant, out_path):
            rendered += 1
            print(f"  wrote {out_path.name}")
    print(f"\nRendered {rendered}/{len(INSTANCES)} repair figures -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
