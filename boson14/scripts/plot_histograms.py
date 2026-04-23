"""Per-instance histograms of clique-size and MS objective from boson14 samples.

Reads scripts/cache/all_samples.csv (produced by extract_results.py) and emits

  report/figures/{instance}/hist_omega_{instance}.png       (Fig A: 3x4 omega grid)
  report/figures/{instance}/hist_objective_{instance}.png   (Fig B: 3x4 g grid, MS line)
  report/figures/{instance}/hist_overlay_{instance}.png     (Fig C: 4-panel variant overlay)

The MS theoretical optimum g*(ω=k) = (R^2/2)(1 - 1/k) is drawn as a vertical
line on every objective panel.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import CACHE_DIR, FIGURES_DIR, VARIANTS, VARIANT_COLOURS


def _panel_empty(ax: plt.Axes, label: str) -> None:
    ax.text(0.5, 0.5, f"no data\n({label})", ha="center", va="center",
            transform=ax.transAxes, color="gray", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _instance_amplitudes(df: pd.DataFrame) -> list[int]:
    return sorted(int(a) for a in df["amplitude"].unique())


def plot_omega_grid(df: pd.DataFrame, instance: str, out_path) -> None:
    """Figure A: 3 variants x M amplitudes of omega_est histograms."""
    n = int(df["n"].iloc[0])
    k = int(df["k"].iloc[0])
    amplitudes = _instance_amplitudes(df)
    bins = np.arange(0.5, n + 1.5, 1.0)

    fig, axes = plt.subplots(
        len(VARIANTS), len(amplitudes),
        figsize=(3.2 * len(amplitudes), 2.4 * len(VARIANTS)),
        sharex=True, sharey=True,
    )

    for i, variant in enumerate(VARIANTS):
        for j, amp in enumerate(amplitudes):
            ax = axes[i, j]
            cell = df[(df["variant"] == variant) & (df["amplitude"] == amp)]
            if cell.empty:
                _panel_empty(ax, f"{variant} a{amp}")
                continue
            ax.hist(cell["omega_est"], bins=bins,
                    color=VARIANT_COLOURS[variant], edgecolor="white", linewidth=0.5)
            ax.axvline(k, color="green", linestyle="--", linewidth=1.2,
                       label=f"planted k={k}")
            if i == 0:
                ax.set_title(f"a{amp}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{variant}\ncount", fontsize=10)
            if i == len(VARIANTS) - 1:
                ax.set_xlabel(r"$\omega_{\mathrm{est}}$", fontsize=10)

    fig.suptitle(f"{instance} — clique-size distribution (planted k={k})",
                 fontsize=12, y=1.00)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99),
                   fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_objective_grid(df: pd.DataFrame, instance: str, out_path) -> None:
    """Figure B: 3 variants x M amplitudes of g(y) histograms with MS line."""
    k = int(df["k"].iloc[0])
    g_star = float(df["g_star"].iloc[0])
    amplitudes = _instance_amplitudes(df)
    # Share x-range across the full instance so panels are visually comparable.
    g_min, g_max = df["g"].min(), df["g"].max()
    pad = 0.02 * (g_max - g_min + 1)
    bins = np.linspace(g_min - pad, max(g_max, g_star) + pad, 40)

    fig, axes = plt.subplots(
        len(VARIANTS), len(amplitudes),
        figsize=(3.2 * len(amplitudes), 2.4 * len(VARIANTS)),
        sharex=True, sharey=True,
    )

    for i, variant in enumerate(VARIANTS):
        for j, amp in enumerate(amplitudes):
            ax = axes[i, j]
            cell = df[(df["variant"] == variant) & (df["amplitude"] == amp)]
            if cell.empty:
                _panel_empty(ax, f"{variant} a{amp}")
                continue
            ax.hist(cell["g"], bins=bins,
                    color=VARIANT_COLOURS[variant], edgecolor="white", linewidth=0.5)
            ax.axvline(g_star, color="red", linestyle="--", linewidth=1.2,
                       label=fr"$g^*(\omega\!=\!{k})={g_star:.1f}$")
            if i == 0:
                ax.set_title(f"a{amp}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{variant}\ncount", fontsize=10)
            if i == len(VARIANTS) - 1:
                ax.set_xlabel(r"$g(y) = \frac{1}{2}\, y^T A\, y$", fontsize=10)

    fig.suptitle(
        f"{instance} — MS objective distribution  "
        fr"(theoretical $g^* = {g_star:.2f}$)",
        fontsize=12, y=1.00,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99),
                   fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(df: pd.DataFrame, instance: str, out_path) -> None:
    """Figure C: M-panel overlay (one panel per amplitude, 3 variants overlaid)."""
    k = int(df["k"].iloc[0])
    g_star = float(df["g_star"].iloc[0])
    amplitudes = _instance_amplitudes(df)

    # Per-panel x-range is shared across variants; choose from available data.
    g_min, g_max = df["g"].min(), df["g"].max()
    pad = 0.02 * (g_max - g_min + 1)
    bins = np.linspace(g_min - pad, max(g_max, g_star) + pad, 40)

    fig, axes = plt.subplots(
        1, len(amplitudes),
        figsize=(3.6 * len(amplitudes), 3.2),
        sharex=True, sharey=True,
    )

    for j, amp in enumerate(amplitudes):
        ax = axes[j]
        present = False
        for variant in VARIANTS:
            cell = df[(df["variant"] == variant) & (df["amplitude"] == amp)]
            if cell.empty:
                continue
            ax.hist(cell["g"], bins=bins,
                    color=VARIANT_COLOURS[variant], edgecolor="white",
                    linewidth=0.4, alpha=0.55, label=variant)
            present = True
        if not present:
            _panel_empty(ax, f"a{amp}")
            continue
        ax.axvline(g_star, color="red", linestyle="--", linewidth=1.3)
        ax.set_title(f"amplitude = {amp}", fontsize=10)
        if j == 0:
            ax.set_ylabel("sample count", fontsize=10)
        ax.set_xlabel(r"$g(y)$", fontsize=10)

    # Put the MS line and variant legend on the outer axes.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=VARIANT_COLOURS[v], alpha=0.55, label=v)
                      for v in VARIANTS]
    legend_handles.append(Line2D([0], [0], color="red", linestyle="--",
                                 label=fr"$g^*(\omega\!=\!{k})={g_star:.1f}$"))
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, -0.02), ncol=len(legend_handles),
               fontsize=9, frameon=False)

    fig.suptitle(
        f"{instance} — variant comparison across amplitudes",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    csv_path = CACHE_DIR / "all_samples.csv"
    if not csv_path.exists():
        print(f"Missing {csv_path} — run extract_results.py first.")
        return 1

    df = pd.read_csv(csv_path)
    instances = sorted(df["instance"].unique())
    print(f"Plotting {len(instances)} instances from {len(df)} samples")

    for instance in instances:
        sub = df[df["instance"] == instance]
        out_dir = FIGURES_DIR / instance
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_omega_grid(sub, instance, out_dir / f"hist_omega_{instance}.png")
        plot_objective_grid(sub, instance, out_dir / f"hist_objective_{instance}.png")
        plot_overlay(sub, instance, out_dir / f"hist_overlay_{instance}.png")
        print(f"  [{instance}] wrote 3 figures -> {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
