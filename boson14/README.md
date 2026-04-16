# Boson14 Planted-Clique Benchmark Suite

A self-contained suite for generating planted-clique instances, testing them on Dirac-3 hardware, and verifying that the recovered clique number matches the planted ground truth.

## Key Features

- **Planted cliques with known omega** — controllable difficulty with verifiable ground truth
- **Scaled simplex formulation** — `sum(y) = R` with integer coupling `J = -A`
- **Degenerate (multi-clique) instances** — plant multiple same-size cliques for landscape analysis
- **Index scrambling** — random vertex permutation for blind testing
- **Multi-R sweep** — verify omega consistency across `R = 1, 10, 100`
- **Dirac-3 cloud integration** — end-to-end submission and result evaluation
- **Visualization** — objective histograms and clique-size distribution plots

## Directory Layout

```
boson14/                     # Self-contained project root
  pyproject.toml             # Dependencies and package config
  uv.lock                    # Pinned versions for reproducibility
  .python-version            # Python 3.12
  .env                       # QCI_API_URL and QCI_TOKEN (gitignored)
  boson14_bench/             # Internal package (copied from dirac_bench)
    core.py                  # Scaled Motzkin-Strauss formulation
    planted_clique.py        # Planted-clique graph generator
    problems.py              # Standard MS helpers
    io.py                    # DIMACS reader/writer
  generate.py                # Instance generation (planted cliques, degenerate, scramble)
  test_dirac.py              # Dirac-3 cloud submission and verification
  run_R_sweep.py             # R-scaling experiment on DIMACS graphs
  inputs/                    # DIMACS .clq files and CSV instances
  output/                    # Generated instances, solutions, and plots
  scaled-motzkin-straus.md   # Mathematical formulation reference
  USAGE.md                   # Full usage guide and CLI reference
```

## Quick Start (self-contained)

From inside the `boson14/` directory:

```bash
# 1. Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create .env with QCI credentials
echo "QCI_API_URL=https://api.qci-prod.com" > .env
echo "QCI_TOKEN=<your-token>" >> .env

# 3. Sync the environment (creates .venv from uv.lock)
uv sync

# 4. Run scripts
uv run python generate.py --n 30 --k 12 --verify
uv run python test_dirac.py --n 14 --k 7 --verify --plot
uv run python run_R_sweep.py --R-values 1,10,100
```

The folder has no dependency on the parent repository — you can copy `boson14/`
to another machine and run `uv sync` to recreate the exact environment.

## Documentation

- **[USAGE.md](USAGE.md)** — Full CLI reference, workflows, output formats, and troubleshooting
- **[scaled-motzkin-straus.md](scaled-motzkin-straus.md)** — Mathematical derivation of the scaled formulation
- **manual.pdf** — Beamer slide deck (compile `manual.tex` with `pdflatex`)
