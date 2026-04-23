"""Render each of the six test graphs with the planted clique highlighted.

Reproduces each base graph deterministically from base_meta.json via
`boson14_bench.planted_clique.generate_planted_clique(n, k, p, seed)`, then
draws it using a two-shell layout: clique nodes on the inner ring (red),
other nodes on the outer ring (gray). Clique-to-clique edges are drawn on
top in red so the dense non-clique edges don't obscure the planted structure.

Writes report/figures/graphs/{instance}.png for all six instances.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from _common import FIGURES_DIR, OUTPUT_DIR

# Make boson14_bench importable whether or not the editable install picked up.
sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "boson14").resolve()))
from boson14_bench.planted_clique import generate_planted_clique  # noqa: E402


CLIQUE_COLOUR = "#E74C3C"        # red
CLIQUE_EDGE_COLOUR = "#C0392B"   # darker red for outline
OTHER_NODE_COLOUR = "#D0D3D4"    # very light gray
OTHER_EDGE_COLOUR = "#7F8C8D"    # medium gray


def _node_size(n: int) -> tuple[int, int]:
    """Return (clique_size, other_size) scaled to graph order."""
    if n <= 20:
        return 380, 220
    if n <= 50:
        return 160, 90
    return 80, 40


def render_instance(instance_dir: Path, out_path: Path) -> None:
    meta = json.load((instance_dir / "base_meta.json").open())
    n, k, p, seed = int(meta["n"]), int(meta["k"]), float(meta["p"]), int(meta["seed"])
    planted = set(meta["planted_nodes"])

    G, planted_check = generate_planted_clique(n, k, p=p, seed=seed)
    if set(planted_check) != planted:
        raise RuntimeError(
            f"Regenerated clique mismatch for {instance_dir.name}: "
            f"meta={sorted(planted)} regen={sorted(planted_check)}"
        )

    clique_nodes = sorted(planted)
    other_nodes = sorted(set(G.nodes) - planted)

    # Two-shell layout puts the clique on an inner ring so it's spatially
    # distinct regardless of edge density. Rotation offset keeps outer nodes
    # from overlapping inner ones.
    pos = nx.shell_layout(G, nlist=[clique_nodes, other_nodes], rotate=0.0)
    # Pull clique in slightly so it's clearly "inside" the outer ring.
    for v in clique_nodes:
        pos[v] = pos[v] * 0.55

    clique_edges = [(u, v) for u, v in G.edges
                    if u in planted and v in planted]
    other_edges = [(u, v) for u, v in G.edges
                   if not (u in planted and v in planted)]

    fig, ax = plt.subplots(figsize=(5.8, 5.8))

    # Non-clique edges first (under): thin, translucent.
    other_alpha = 0.35 if n <= 20 else (0.18 if n <= 50 else 0.08)
    nx.draw_networkx_edges(
        G, pos, edgelist=other_edges,
        edge_color=OTHER_EDGE_COLOUR, width=0.6, alpha=other_alpha, ax=ax,
    )
    # Clique edges: drawn last so they sit on top.
    nx.draw_networkx_edges(
        G, pos, edgelist=clique_edges,
        edge_color=CLIQUE_COLOUR, width=1.4, alpha=0.85, ax=ax,
    )

    clique_sz, other_sz = _node_size(n)
    nx.draw_networkx_nodes(
        G, pos, nodelist=other_nodes,
        node_color=OTHER_NODE_COLOUR, node_size=other_sz,
        edgecolors="#95A5A6", linewidths=0.5, ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=clique_nodes,
        node_color=CLIQUE_COLOUR, node_size=clique_sz,
        edgecolors=CLIQUE_EDGE_COLOUR, linewidths=0.9, ax=ax,
    )

    # Labels: all nodes for small n, clique-only for medium/large.
    if n <= 20:
        nx.draw_networkx_labels(G, pos, font_size=8, font_color="black", ax=ax)
    else:
        font_sz = 6 if n <= 50 else 5
        nx.draw_networkx_labels(
            G, pos,
            labels={v: str(v) for v in clique_nodes},
            font_size=font_sz, font_color="white", font_weight="bold", ax=ax,
        )

    ax.set_title(
        rf"$n={n}$, $\omega=k={k}$, $p={p}$, edges$=${G.number_of_edges()}",
        fontsize=11,
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    out_dir = FIGURES_DIR / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = sorted(p for p in OUTPUT_DIR.glob("benchmark_*") if p.is_dir())
    if not instances:
        print("No benchmark instances found.")
        return 1

    for instance_dir in instances:
        out_path = out_dir / f"{instance_dir.name}.png"
        render_instance(instance_dir, out_path)
        print(f"  rendered {instance_dir.name} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
