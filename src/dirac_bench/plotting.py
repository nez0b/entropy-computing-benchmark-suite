"""Histogram and comparison plots for benchmark results."""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dirac_bench.problems import omega_to_theoretical_objective


def plot_dirac_histogram(
    objectives: list[float],
    graph_name: str,
    computed_omega: int,
    known_omega: Optional[int] = None,
    baseline_objectives: Optional[dict[str, float]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot a histogram of Dirac-3 sample objectives with theoretical omega lines.

    Args:
        objectives: List of objective values (0.5 * x^T A x) from Dirac samples.
        graph_name: Name for the plot title and filename.
        computed_omega: The omega value computed from the best sample.
        known_omega: Known optimal omega (if available).
        baseline_objectives: Dict of solver_name -> best_objective for classical baselines.
        save_path: Directory to save the PNG. None to skip saving.
        show: Whether to call plt.show().

    Returns:
        Path to saved PNG, or None if not saved.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(objectives, bins=25, edgecolor="black", alpha=0.7, color="#4C72B0",
            label=f"Dirac samples (n={len(objectives)})")

    best_obj = max(objectives)
    ax.axvline(x=best_obj, color="red", linestyle="--", linewidth=2,
               label=f"Best Dirac: {best_obj:.6f}")

    # Classical baseline lines
    baseline_colors = {"SLSQP": "#ff7f0e", "L-BFGS-B": "#9467bd"}
    if baseline_objectives:
        for name, obj in baseline_objectives.items():
            color = baseline_colors.get(name, "#17becf")
            ax.axvline(x=obj, color=color, linestyle="-.", linewidth=2,
                       label=f"{name}: {obj:.6f}")

    # Theoretical omega lines
    omegas_to_show = set()
    omegas_to_show.add(computed_omega)
    if known_omega is not None:
        omegas_to_show.add(known_omega)
    for delta in (-2, -1, 1, 2):
        omegas_to_show.add(computed_omega + delta)
    omegas_to_show = sorted(o for o in omegas_to_show if o >= 2)

    y_max = ax.get_ylim()[1]

    for omega in omegas_to_show:
        f_theory = omega_to_theoretical_objective(omega)
        style = "-" if omega == computed_omega else "--"
        color = "#2ca02c" if omega == known_omega else "#7f7f7f"
        if omega == computed_omega and omega == known_omega:
            color = "#2ca02c"
        elif omega == computed_omega:
            color = "#d62728"

        ax.axvline(x=f_theory, color=color, linestyle=style,
                   alpha=0.8, linewidth=1.5)
        ax.text(f_theory, y_max * 0.92, f" w={omega}",
                rotation=90, va="bottom", color=color,
                fontsize=9, fontweight="bold")

    # Formula annotation
    formula = r"$f^*(\omega) = \frac{1}{2}\left(1-\frac{1}{\omega}\right)$"
    ax.text(0.05, 0.95, formula, transform=ax.transAxes,
            fontsize=11, va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))

    title = f"{graph_name}: Dirac-3 Objective Distribution"
    subtitle = f"Samples: {len(objectives)}, Computed w={computed_omega}"
    if known_omega is not None:
        subtitle += f", Known w={known_omega}"
        match_str = "MATCH" if computed_omega == known_omega else "MISMATCH"
        subtitle += f" [{match_str}]"
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel(r"Objective $\frac{1}{2} x^T A x$")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    saved_path = None
    if save_path is not None:
        out_dir = Path(save_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"histogram_{graph_name}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved histogram -> {fname}")
        saved_path = fname

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path
