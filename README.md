# Dirac Test Suite

Benchmark suite for evaluating QCI's Dirac-3 quantum solver on quadratic optimization problems. Uses the Motzkin-Straus max-clique formulation on DIMACS graphs, comparing against SLSQP and L-BFGS-B classical baselines.

## Architecture

```
DIMACS graph → Adjacency matrix A → Motzkin-Straus QP → Solvers → ω (clique number)
                                     max 0.5 x^T A x
                                     s.t. Σx = 1, x ≥ 0
```

The **Motzkin-Straus theorem** states that the optimal objective `f*(ω) = 0.5 * (1 - 1/ω)` exactly encodes the clique number ω of the graph. We invert this via `ω = round(1 / (1 - 2f*))`.

## Solvers

| Solver | Method | Backend |
|--------|--------|---------|
| **SLSQP** | Sequential Least Squares Programming | scipy, local |
| **L-BFGS-B** | Limited-memory BFGS with softmax reparameterization | scipy, local |
| **Dirac-3 Cloud** | QCI quantum solver via cloud API | eqc-models |
| **Dirac-3 Direct** | QCI quantum solver via gRPC to hardware | eqc-direct |

### Bomze Regularization

The standard Motzkin-Straus formulation suffers from **spurious solutions** — local optima whose support is not a clique. [Bomze's regularization](docs/bomze_regularization.md) (`A_bar = A + 0.5*I`) eliminates all spurious local maxima, so every solution corresponds to a maximal clique.

```bash
# Compare standard vs Bomze on Dirac-3 hardware
uv run python scripts/run_bomze_dirac.py --backend cloud
uv run python scripts/run_bomze_dirac.py --backend both --inline-graph 'ER(20,0.7)'
```

## Quick Start

```bash
# Setup
uv venv && uv pip install -e ".[dev]"

# Generate test graphs
uv run python scripts/generate_graphs.py

# Run classical baselines only
uv run python scripts/run_benchmark.py --skip-dirac

# Run with Dirac cloud (requires QCI credentials in qci-eqc-models/.env)
uv run python scripts/run_benchmark.py

# Run tests
uv run pytest tests/ -v
```

## Directory Layout

```
├── src/dirac_bench/        # Core library
│   ├── io.py               # DIMACS reader/writer
│   ├── problems.py         # Motzkin-Straus formulation
│   ├── plotting.py         # Histogram generation
│   ├── utils.py            # JSON/CSV serialization
│   ├── benchmark.py        # Orchestrator
│   └── solvers/            # Solver backends
│       ├── dirac.py        # Dirac-3 cloud
│       ├── dirac_direct.py # Dirac-3 direct hardware
│       ├── slsqp.py        # SLSQP baseline
│       └── lbfgs.py        # L-BFGS-B baseline
├── scripts/
│   ├── run_benchmark.py    # CLI entry point
│   ├── run_bomze_dirac.py  # Bomze vs standard comparison on Dirac-3
│   ├── test_bomze_spurious.py # Spurious elimination demo (SLSQP)
│   └── generate_graphs.py  # Graph generation
├── problems/dimacs/        # DIMACS .clq graph files
├── results/                # Benchmark output (CSV + raw JSON)
├── plots/                  # Generated histogram PNGs
└── tests/                  # Unit tests
```

See [Usage.md](Usage.md) for detailed usage instructions.
