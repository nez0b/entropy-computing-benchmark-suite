"""Aggregate success-rate summary across all benchmark instances.

Reads scripts/cache/summary.json and produces two figures:

  report/figures/summary_success_rate.png
      For each variant, a heatmap: rows = instance, cols = amplitude,
      cell value = fraction of 100 samples with omega_est >= planted k.

  report/figures/summary_best_ratio.png
      Same layout, cell value = best_g / g*  (how close the best sample got).

These two together separate "never found the clique" from "came close but not
exact" — a distinction that matters once k grows beyond ~15.
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import CACHE_DIR, FIGURES_DIR, VARIANTS


def _pivot(
    summaries: list[dict], field: str,
) -> tuple[list[str], list[int], dict[str, np.ndarray]]:
    """Return (instances, amplitudes, {variant: matrix})."""
    instances = sorted({s["instance"] for s in summaries})
    amplitudes = sorted({s["amplitude"] for s in summaries})
    by_variant: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        mat = np.full((len(instances), len(amplitudes)), np.nan)
        for s in summaries:
            if s["variant"] != variant:
                continue
            i = instances.index(s["instance"])
            j = amplitudes.index(s["amplitude"])
            mat[i, j] = s[field]
        by_variant[variant] = mat
    return instances, amplitudes, by_variant


def _plot_grid(
    instances: list[str], amplitudes: list[int], matrices: dict[str, np.ndarray],
    title: str, cbar_label: str, vmin: float, vmax: float,
    cmap: str, out_path,
) -> None:
    fig, axes = plt.subplots(
        1, len(VARIANTS),
        figsize=(0.7 * len(amplitudes) * len(VARIANTS) + 3.0,
                 0.55 * len(instances) + 2.0),
        sharey=True,
    )

    for ax, variant in zip(axes, VARIANTS):
        m = matrices[variant]
        im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(variant, fontsize=11)
        ax.set_xticks(range(len(amplitudes)))
        ax.set_xticklabels([f"a{a}" for a in amplitudes], rotation=0, fontsize=8)
        ax.set_xlabel("amplitude")
        # Annotate cells. Both viridis and magma have dark low-ends, so text is
        # white for below-midpoint (or clipped below vmin) and black above.
        mid = 0.5 * (vmin + vmax)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m[i, j]
                if np.isnan(v):
                    txt, colour = "—", "gray"
                else:
                    txt = f"{v:.2f}"
                    v_eff = max(v, vmin)   # clipping makes it dark either way
                    colour = "white" if v_eff < mid else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color=colour)

    axes[0].set_yticks(range(len(instances)))
    axes[0].set_yticklabels(instances, fontsize=8)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    summary_path = CACHE_DIR / "summary.json"
    if not summary_path.exists():
        print(f"Missing {summary_path} — run extract_results.py first.")
        return 1

    summaries = json.load(summary_path.open())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    instances, amplitudes, hit_mats = _pivot(summaries, "frac_omega_at_least_k")
    _plot_grid(
        instances, amplitudes, hit_mats,
        title=r"Success rate: fraction of 100 samples with $\omega_{est} \geq k$",
        cbar_label="fraction",
        vmin=0.0, vmax=1.0, cmap="viridis",
        out_path=FIGURES_DIR / "summary_success_rate.png",
    )
    print("Wrote summary_success_rate.png")

    _, _, ratio_mats = _pivot(summaries, "best_g")
    g_star_for_inst = {s["instance"]: s["g_star"] for s in summaries}
    for variant, mat in ratio_mats.items():
        for i, inst in enumerate(instances):
            mat[i, :] = mat[i, :] / g_star_for_inst[inst]

    _plot_grid(
        instances, amplitudes, ratio_mats,
        title=r"Best-sample closeness: $g_{\mathrm{best}} / g^*$",
        cbar_label="ratio",
        vmin=0.90, vmax=1.0, cmap="magma",
        out_path=FIGURES_DIR / "summary_best_ratio.png",
    )
    print("Wrote summary_best_ratio.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
