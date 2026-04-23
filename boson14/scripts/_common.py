"""Shared helpers for boson14 benchmark analysis scripts.

Walks the output tree produced by `boson14/full-benchmark.py` and parses the
boson14-hardware result `.npz` files. All paths are computed relative to the
repo root so the scripts can be run from any directory.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BOSON14_DIR = REPO_ROOT / "boson14"
OUTPUT_DIR = BOSON14_DIR / "output"
CACHE_DIR = REPO_ROOT / "scripts" / "cache"
FIGURES_DIR = REPO_ROOT / "report" / "figures"

VARIANTS = ("random", "front", "end")

# Fixed colour palette so every figure uses the same variant colours.
VARIANT_COLOURS = {
    "random": "#4C72B0",  # blue
    "front":  "#DD8452",  # orange
    "end":    "#55A868",  # green
}

# Matches results__<variant>_boson14__<ts>_a<amp>.npz
_RESULTS_RE = re.compile(
    r"^results__(?P<variant>\w+)_boson14__(?P<ts>\d{8}_\d{6})_a(?P<amp>\d+)\.npz$"
)


@dataclass(frozen=True)
class RunFile:
    """One boson14 submission (one .npz file of 100 samples)."""
    instance: str          # benchmark_n50_k10_p0.3_s42
    variant: str           # random | front | end
    amplitude: int         # 300, 400, 500, 600
    timestamp: str         # 20260417_150616
    npz_path: Path         # absolute path to the results file
    variant_dir: Path      # absolute path to the {variant}/ dir (for meta/solutions)

    @property
    def meta_path(self) -> Path:
        return self.variant_dir / f"{self.variant}_meta.json"

    @property
    def solutions_path(self) -> Path:
        return self.variant_dir / f"{self.variant}_solutions.npz"


def iter_runs(output_dir: Path = OUTPUT_DIR) -> list[RunFile]:
    """Discover every results_*.npz under output_dir/benchmark_*/{variant}/."""
    runs: list[RunFile] = []
    for instance_dir in sorted(output_dir.glob("benchmark_*")):
        if not instance_dir.is_dir():
            continue
        for variant in VARIANTS:
            vdir = instance_dir / variant
            if not vdir.is_dir():
                continue
            for npz in sorted(vdir.glob("results__*.npz")):
                m = _RESULTS_RE.match(npz.name)
                if not m or m.group("variant") != variant:
                    continue
                runs.append(RunFile(
                    instance=instance_dir.name,
                    variant=variant,
                    amplitude=int(m.group("amp")),
                    timestamp=m.group("ts"),
                    npz_path=npz,
                    variant_dir=vdir,
                ))
    return runs


def load_meta(run: RunFile) -> dict:
    with run.meta_path.open() as f:
        return json.load(f)


def iter_instances(output_dir: Path = OUTPUT_DIR) -> list[str]:
    return sorted(p.name for p in output_dir.glob("benchmark_*") if p.is_dir())
