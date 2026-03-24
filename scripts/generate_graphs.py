#!/usr/bin/env python3
"""Generate Erdos-Renyi graphs in DIMACS format and copy C125.9 from sibling repo."""

import shutil
from pathlib import Path

import networkx as nx

from dirac_bench.io import write_dimacs_graph


DIMACS_DIR = Path(__file__).resolve().parent.parent / "problems" / "dimacs"

# Erdos-Renyi configurations: (n, p, seed)
ER_GRAPHS = [
    (50, 0.9, 42),
    (75, 0.5, 42),
    (100, 0.5, 42),
    (100, 0.9, 42),
    (150, 0.9, 42),
]


def generate_er_graphs() -> None:
    """Generate Erdos-Renyi graphs and write to DIMACS format."""
    DIMACS_DIR.mkdir(parents=True, exist_ok=True)

    for n, p, seed in ER_GRAPHS:
        p_str = str(p).replace(".", "")
        name = f"erdos_renyi_{n}_p{p_str}"
        path = DIMACS_DIR / f"{name}.clq"

        G = nx.erdos_renyi_graph(n, p, seed=seed)
        # Relabel to 1-based indexing for DIMACS
        G = nx.relabel_nodes(G, {i: i + 1 for i in range(n)})

        comment = f"Erdos-Renyi G({n}, {p}), seed={seed}"
        write_dimacs_graph(G, path, comment=comment)

        m = G.number_of_edges()
        density = 2 * m / (n * (n - 1)) if n > 1 else 0
        print(f"  Generated {name}: |V|={n}, |E|={m}, density={density:.3f}")


def copy_c125() -> None:
    """Copy C125.9.clq from the sibling mis-spectral-graph-solver repo."""
    src = Path(__file__).resolve().parent.parent.parent / "mis-spectral-graph-solver" / "DIMACS" / "C125.9.clq"
    dst = DIMACS_DIR / "C125.9.clq"

    if src.exists():
        DIMACS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  Copied C125.9.clq from {src}")
    else:
        print(f"  Warning: {src} not found, skipping C125.9 copy")


def main() -> None:
    print("Generating DIMACS graphs...")
    print()

    print("Erdos-Renyi graphs:")
    generate_er_graphs()
    print()

    print("Copying benchmark graphs:")
    copy_c125()
    print()

    print(f"All graphs written to {DIMACS_DIR}")


if __name__ == "__main__":
    main()
