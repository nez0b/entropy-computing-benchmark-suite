# Usage Guide

## Setup

### 1. Python environment

```bash
uv venv
uv pip install -e ".[dev]"
```

### 2. QCI Credentials (for Dirac solvers)

The Dirac cloud solver reads `QCI_API_URL` and `QCI_TOKEN` from `qci-eqc-models/.env`. This file should already exist if you cloned the credentials submodule.

### 3. eqc-direct (for direct hardware)

Direct hardware access requires `eqc-direct >= 1.0.7` (included in dependencies) and network access to the Dirac-3 device.

### 4. Generate test graphs

```bash
uv run python scripts/generate_graphs.py
```

This generates Erdős-Rényi graphs and copies C125.9.clq from the sibling repo.

## Running Benchmarks

### All solvers, all graphs

```bash
uv run python scripts/run_benchmark.py
```

### Classical baselines only

```bash
uv run python scripts/run_benchmark.py --skip-dirac
```

### Single graph

```bash
uv run python scripts/run_benchmark.py --graph C125.9
uv run python scripts/run_benchmark.py --graph erdos_renyi_50_p09 --skip-dirac
```

### Dirac backend selection

```bash
# Cloud API (default)
uv run python scripts/run_benchmark.py --backend cloud

# Direct hardware via gRPC
uv run python scripts/run_benchmark.py --backend direct --ip 172.18.41.79 --port 50051

# Both cloud and direct
uv run python scripts/run_benchmark.py --backend both
```

### Tuning parameters

```bash
# Fewer Dirac samples (faster)
uv run python scripts/run_benchmark.py --num-samples 5

# More classical restarts (more thorough)
uv run python scripts/run_benchmark.py --slsqp-restarts 50 --lbfgs-restarts 50

# Skip large graphs
uv run python scripts/run_benchmark.py --max-nodes 100
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dimacs-dir` | `problems/dimacs` | Directory containing .clq files |
| `--graph` | all | Run a single graph by stem name |
| `--max-nodes` | none | Skip graphs exceeding this node count |
| `--skip-dirac` | false | Skip Dirac solvers |
| `--backend` | `cloud` | Dirac backend: cloud, direct, or both |
| `--ip` | `172.18.41.79` | Dirac hardware IP |
| `--port` | `50051` | Dirac gRPC port |
| `--num-samples` | 100 | Dirac samples per graph |
| `--relaxation-schedule` | 2 | Dirac schedule parameter (1-4) |
| `--slsqp-restarts` | 10 | SLSQP random restarts |
| `--lbfgs-restarts` | 10 | L-BFGS-B random restarts |
| `--seed` | 42 | Random seed |
| `--results-dir` | `results/raw` | Raw JSON output directory |
| `--plots-dir` | `plots` | Histogram PNG output directory |
| `--no-plot` | false | Skip histogram generation |

## Bomze vs Standard Comparison

`scripts/run_bomze_dirac.py` runs the same graph through both standard Motzkin-Straus and Bomze-regularized formulations on Dirac-3 hardware, extracting per-sample x-vectors to count spurious solutions.

### Basic usage

```bash
# Run all inline graphs on cloud backend
uv run python scripts/run_bomze_dirac.py --backend cloud

# Run a specific inline graph on both backends
uv run python scripts/run_bomze_dirac.py --backend both --inline-graph 'ER(20,0.7)'

# Run a DIMACS graph
uv run python scripts/run_bomze_dirac.py --backend cloud --graph C125.9

# Customize solver parameters
uv run python scripts/run_bomze_dirac.py --backend cloud --num-samples 50 --relaxation-schedule 3
```

### Bomze CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `cloud` | Dirac backend: cloud, direct, or both |
| `--graph` | none | Run a DIMACS graph by stem name |
| `--inline-graph` | none | Run a specific inline graph (e.g. `ER(20,0.7)`) |
| `--dimacs-dir` | `problems/dimacs` | Directory containing .clq files |
| `--max-nodes` | none | Skip DIMACS graphs exceeding this node count |
| `--num-samples` | 100 | Dirac samples per run (1-100) |
| `--relaxation-schedule` | 2 | Dirac schedule parameter (1-4) |
| `--ip` | `172.18.41.79` | Dirac hardware IP |
| `--port` | `50051` | Dirac gRPC port |
| `--threshold` | `1e-4` | Support extraction threshold |
| `--results-dir` | `results/bomze_dirac` | Output JSON directory |
| `--no-save` | false | Skip writing JSON files |

Output JSON files are saved to `results/bomze_dirac/` with per-sample x-vectors, support analysis, and summary statistics.

## Interpreting Results

### Summary table

The console output shows a table with omega values from each solver. For DIMACS benchmark graphs with known omega, you can verify correctness.

### Histogram plots

Saved to `plots/histogram_<graph_name>.png`. Shows:
- Distribution of Dirac-3 sample objectives
- Vertical lines at theoretical `f*(ω)` for nearby omega values
- Classical baseline best objectives (if available)

### Raw JSON

Saved to `results/raw/`. Contains the full Dirac API response for debugging and analysis.

### CSV

`results/benchmark_results.csv` has one row per graph with all solver results.

## Adding New Graphs

1. Place `.clq` files in `problems/dimacs/`
2. DIMACS format: `p edge N M` header, then `e u v` edges (1-based)
3. Or add to `scripts/generate_graphs.py` and re-run

## Adding New Solvers

Implement a function matching this interface:

```python
def solve_my_solver(A: np.ndarray, **kwargs) -> dict:
    """
    Args:
        A: Adjacency matrix (n x n, float64).

    Returns:
        Dict with keys:
        - omega: int — computed clique number
        - best_objective: float — best 0.5 * x^T A x value
        - all_objectives: list[float] — objectives from all samples/restarts
        - solve_time: float — wall-clock seconds
    """
```

Then add it to `benchmark.py:run_single_graph()`.

## Troubleshooting

- **"Dirac-3 returned no solutions"**: Check QCI credentials in `qci-eqc-models/.env`
- **gRPC connection errors**: Verify IP/port and network access to Dirac hardware
- **FutureWarning from networkx**: Harmless — networkx 2.x sparse matrix API deprecation
