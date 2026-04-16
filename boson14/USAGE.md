# Boson14 Planted-Clique Benchmark Suite — Usage Guide

A self-contained suite for generating planted-clique benchmark instances, submitting them to Dirac-3 hardware, and verifying results against known ground truth.

## Prerequisites

- **Python 3.10+**
- **uv** (recommended) or pip

## Quick Start

```bash
uv sync
uv run python boson14/generate.py --n 30 --k 12 --verify
uv run python boson14/test_dirac.py --n 14 --k 7 --verify --plot
```

## Installation

From the `dirac-test-suite` repository root:

```bash
uv sync
```

This installs all dependencies including `numpy`, `networkx`, `matplotlib`, and the QCI packages (`eqc-models`, `qci-client`).

## Environment Setup

Dirac-3 cloud submission requires QCI API credentials. The test script loads credentials from `qci-eqc-models/.env` relative to the repository root:

```
QCI_API_URL=https://api.qci-prod.com
QCI_TOKEN=your-api-token-here
```

These are only needed for `test_dirac.py`, not for local instance generation.

---

## Workflow 1: Generate Instances

Generate planted-clique graphs with known clique number and save metadata, solutions, and optional plots.

### Basic generation

```bash
uv run python boson14/generate.py --n 30 --k 12 --verify
```

### Custom planted indices

```bash
uv run python boson14/generate.py --n 50 --k 12 --planted-indices 1,2,3,4,5,6,7,8,9,10,11,12
```

### Degenerate (multi-clique) instance

```bash
uv run python boson14/generate.py --n 50 --k 10 --num-cliques 3 --verify
```

### Scrambled instance

```bash
uv run python boson14/generate.py --n 50 --k 15 --num-cliques 3 --scramble
```

### With distribution plots

```bash
uv run python boson14/generate.py --n 40 --k 15 --plot-distribution --verify
```

### Sample output

```
Generating: n=30, k=12, p=0.5, seed=42, R=100
  Planted nodes: [2, 3, 6, 10, 14, 16, 19, 21, 23, 28, 29, 30]

Theoretical values (R=100, w=12):
  g* = 4583.33
  E* = -9166.67
  y_i = R/w = 100/12 = 8.3333 for planted vertices

Computed from optimal y:
  g(y) = 4583.33
  E(y) = -9166.67
  w(g) = 12
  w(E) = 12

Integer QP: J has entries in {-1, 0}

Brute-force verification:
  Clique number (brute-force): w = 12
  Number of maximum cliques: 1
  CONFIRMED: w = k = 12

Saved solutions -> boson14/output/n30_k12_p0.5_s42/n30_k12_p0.5_s42_solutions.npz  (shape (1, 30))
Saved metadata -> boson14/output/n30_k12_p0.5_s42/n30_k12_p0.5_s42_meta.json
Saved Boson14 JSON -> boson14/output/n30_k12_p0.5_s42/n30_k12_p0.5_s42_boson14.json  (240 terms, maximization)
Saved Boson14 CSV -> boson14/output/n30_k12_p0.5_s42/n30_k12_p0.5_s42_boson14.csv  (30x31, maximization)
```

The Boson14 JSON stores the maximization objective (`+A`, polynomial values `+2.0` per edge) with `sum_constraint = R`. For Dirac-3 submission, negate the polynomial to get `J = -A`.

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n` | int | *required* | Graph size (number of vertices) |
| `--k` | int | *required* | Planted clique size |
| `--p` | float | `0.5` | Edge probability for random edges |
| `--density` | `dense`\|`sparse` | — | Shortcut: `dense` sets p=0.9, `sparse` sets p=0.3 (overrides `--p`) |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--R` | int | `100` | Sum constraint for scaled formulation |
| `--planted-indices` | str | — | Comma-separated 1-based vertices for planted clique |
| `--num-cliques` | int | `1` | Number of planted cliques (degenerate case) |
| `--scramble` | flag | — | Apply random vertex permutation to graph |
| `--plot-distribution` | flag | — | Generate clique objective distribution plots |
| `--output-dir` | path | `boson14/output` | Output directory for results |
| `--verify` | flag | — | Brute-force verify clique number using `nx.find_cliques()` |
| `--output-format` | `json`\|`csv`\|`both` | `both` | Output format for StQP problem file |

---

## Workflow 2: Submit to Dirac-3

Submit planted-clique instances to the Dirac-3 cloud solver and verify that the recovered clique number matches.

### Basic submission

```bash
uv run python boson14/test_dirac.py --n 14 --k 7 --verify --plot
```

### With scrambling

```bash
uv run python boson14/test_dirac.py --n 14 --k 7 --scramble --verify --plot
```

### Degenerate instance

```bash
uv run python boson14/test_dirac.py --n 30 --k 10 --num-cliques 2 --verify --plot
```

### Multi-R sweep

```bash
uv run python boson14/test_dirac.py --n 14 --k 7 --multi-R --verify
```

### From generated JSON

Submit using a Boson14 JSON file produced by `generate.py`. The polynomial is verified against a regenerated graph before submission:

```bash
uv run python boson14/test_dirac.py --from-json boson14/output/n14_k7_p0.5_s42/n14_k7_p0.5_s42_boson14.json --verify --plot
```

### From CSV

Submit using a CSV file (from `generate.py --output-format csv` or external source like DIMACS). Use `--k` to enable theoretical comparison; omit it for unknown instances:

```bash
# With known clique size
uv run python boson14/test_dirac.py --csv boson14/output/n14_k7_p0.5_s42/n14_k7_p0.5_s42_boson14.csv --k 7 --verify

# External file without known k
uv run python boson14/test_dirac.py --csv boson14/inputs/C125.9.csv --verify
```

The CSV format is: no header, n rows x (n+1) columns. Column 0 contains linear coefficients C\_i (all zeros for Motzkin-Straus), columns 1..n contain the full symmetric adjacency matrix A\[i,j\]. Both JSON and CSV use the maximization convention (+A).

External CSV files (e.g. from DIMACS graphs converted with other tools) can be placed in `boson14/inputs/` for convenience.

### Sample output (from CSV with known k)

```
Loaded CSV: boson14/output/n14_k7_p0.5_s42/n14_k7_p0.5_s42_boson14.csv (n=14)
  CSV->JSON round-trip: PASS

============================================================
Boson14 Dirac-3 Test: n=14, k=7, R=100  (from CSV)
============================================================

Theoretical values (R=100, w=7):
  g* = 4285.71
  E* = -8571.43
  y_i = R/w = 100/7 = 14.2857

Brute-force verification:
  Clique number (brute-force): w = 7
  Number of maximum cliques: 2
  CONFIRMED: w = k = 7

--- Dirac-3 Submission (R=100) ---
  Submitting 14-variable scaled QP to Dirac-3 cloud (R=100, samples=100, schedule=2)
  Best g = 4285.65  =>  omega = 7  (161.4s)

Conversion consistency:
  omega_from_g == omega_from_E: 100/100 samples agree

============================================================
SUMMARY
============================================================
  Graph:          n=14, k=7
  Sum constraint:  R=100
  Theoretical:    g*=4285.71, E*=-8571.43
  Best Dirac g:   4285.65
  Best omega:     7
  Known omega:    7
  Result:         MATCH
  Solve time:     161.4s
  Samples:        100 finite / 100 requested
============================================================
```

When `--k` is omitted, the summary reports `k=?` and skips theoretical comparison / MATCH/MISMATCH. Brute-force `--verify` still reports the true omega independently.

### Sample output (planted-clique generation)

```
============================================================
Boson14 Dirac-3 Test: n=14, k=7, p=0.5, seed=42, R=100
============================================================
Planted nodes: [3, 5, 6, 7, 8, 12, 13]

Theoretical values (R=100, w=7):
  g* = 4285.71
  E* = -8571.43
  y_i = R/w = 100/7 = 14.2857

Brute-force verification:
  Clique number (brute-force): w = 7
  Number of maximum cliques: 2
  CONFIRMED: w = k = 7

--- Dirac-3 Submission (R=100) ---
  Submitting 14-variable scaled QP to Dirac-3 cloud (R=100, samples=100, schedule=2)
  Best g = 4285.65  =>  omega = 7  (119.5s)

Conversion consistency:
  omega_from_g == omega_from_E: 100/100 samples agree

============================================================
SUMMARY
============================================================
  Graph:          n=14, k=7, p=0.5
  Sum constraint:  R=100
  Theoretical:    g*=4285.71, E*=-8571.43
  Best Dirac g:   4285.65
  Best omega:     7
  Known omega:    7
  Result:         MATCH
  Solve time:     119.5s
  Samples:        100 finite / 100 requested
============================================================
```

### Multi-R sweep sample output

```
--- Multi-R Sweep ---

  R=1: g*=0.4286, E*=-0.8571
  Submitting 14-variable scaled QP to Dirac-3 cloud (R=1, samples=100, schedule=2)
  Best g = 0.43  =>  omega = 7  (118.2s)

  R=10: g*=42.8571, E*=-85.7143
  Submitting 14-variable scaled QP to Dirac-3 cloud (R=10, samples=100, schedule=2)
  Best g = 42.86  =>  omega = 7  (119.1s)

  R=100: g*=4285.7143, E*=-8571.4286
  Submitting 14-variable scaled QP to Dirac-3 cloud (R=100, samples=100, schedule=2)
  Best g = 4285.65  =>  omega = 7  (119.5s)

      R |         g* |         E* |     best_g | best_omega
  -----+-----------+-----------+-----------+-----------
      1 |     0.4286 |    -0.8571 |     0.4286 |          7
     10 |    42.8571 |   -85.7143 |    42.8571 |          7
    100 |  4285.7143 | -8571.4286 |  4285.6501 |          7

  PASS: All R values recover omega = 7
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n` | int | *required*\* | Graph size (number of vertices) |
| `--k` | int | *required*\* | Planted clique size |
| `--p` | float | `0.5` | Edge probability for random edges |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--R` | int | `100` | Sum constraint for Dirac-3 solver |
| `--num-cliques` | int | `1` | Number of planted cliques (degenerate case) |
| `--num-samples` | int | `100` | Number of samples from Dirac-3 |
| `--relaxation-schedule` | int | `2` | Relaxation schedule 1–4 for Dirac-3 |
| `--scramble` | flag | — | Test vertex permutation roundtrip |
| `--verify` | flag | — | Brute-force verify clique number with networkx |
| `--output-dir` | path | `boson14/output` | Output directory |
| `--plot` | flag | — | Generate visualization plots |
| `--verbose` | flag | — | Print per-sample objective and energy details |
| `--multi-R` | flag | — | Run same graph with R=1,10,100; verify omega consistency |
| `--from-json` | path | — | Load instance from `generate.py` Boson14 JSON (overrides `--n/--k/--p/--seed/--R`) |
| `--csv` | path | — | Load StQP instance from CSV file (overrides `--n`; `--k` optional for verification) |

\* Not required when `--from-json` or `--csv` is given.

---

## Workflow 3: Analyze Results

### Load Dirac-3 solutions

```python
import numpy as np

# Load Dirac-3 y-vectors
data = np.load("boson14/output/n14_k7_p0.5_s42/n14_k7_p0.5_s42_dirac_solutions.npz")
y_vectors = data["y_vectors"]  # shape (num_samples, n)
print(f"Samples: {y_vectors.shape[0]}, Variables: {y_vectors.shape[1]}")
```

### Compute objectives and recover omega

```python
import json

# Load adjacency and compute objectives
from dirac_bench.boson14 import scaled_objective, scaled_objective_to_omega
from dirac_bench.problems import motzkin_straus_adjacency
from dirac_bench.planted_clique import generate_planted_clique

G, _ = generate_planted_clique(14, 7, p=0.5, seed=42)
A = motzkin_straus_adjacency(G)

R = 100
for i, y in enumerate(y_vectors[:5]):
    g = scaled_objective(y, A)
    omega = scaled_objective_to_omega(g, R)
    print(f"Sample {i}: g={g:.2f}, omega={omega}")
```

### Load metadata

```python
with open("boson14/output/n14_k7_p0.5_s42/n14_k7_p0.5_s42_dirac_meta.json") as f:
    meta = json.load(f)
print(f"Best omega: {meta['best_omega']}, Solve time: {meta['solve_time']:.1f}s")
```

---

## Output File Reference

Output files are organized into per-instance subfolders:

```
boson14/output/{instance_name}/
  {instance_name}_meta.json
  {instance_name}_boson14.json
  {instance_name}_boson14.csv
  {instance_name}_solutions.npz
  {instance_name}_dirac_meta.json
  {instance_name}_dirac_solutions.npz
  clique_dist_{instance_name}.png
  scaled_histogram_{instance_name}.png
```

Instance names follow the pattern `n{n}_k{k}_p{p}_s{seed}[_nc{nc}][_scrambled]`. For CSV-loaded instances, the subfolder uses the CSV filename stem (e.g., `C125.9`).

| Suffix | Example | Source | Description |
|--------|---------|--------|-------------|
| `_meta.json` | `n30_k12_p0.5_s42_meta.json` | `generate.py` | Instance metadata |
| `_boson14.json` | `n14_k7_p0.5_s42_boson14.json` | `generate.py` | Boson14 maximization JSON (polynomial + job\_params) |
| `_boson14.csv` | `n14_k7_p0.5_s42_boson14.csv` | `generate.py` | Boson14 CSV (n rows x n+1 cols: C\_i + adjacency A) |
| `_dirac_meta.json` | `n14_k7_p0.5_s42_dirac_meta.json` | `test_dirac.py` | Dirac-3 solver results |
| `_solutions.npz` | `n14_k7_p0.5_s42_solutions.npz` | both | Theoretical max-clique solutions |
| `_dirac_solutions.npz` | `n14_k7_p0.5_s42_dirac_solutions.npz` | `test_dirac.py` | Dirac-3 sample vectors |
| `clique_dist_*.png` | `clique_dist_n14_k7_p0.5_s42.png` | both | Clique size distribution |
| `scaled_histogram_*.png` | `scaled_histogram_n14_k7_p0.5_s42.png` | `test_dirac.py` | Scaled objective histogram |

### Instance metadata (`_meta.json`)

```json
{
  "n": 30,
  "k": 12,
  "p": 0.5,
  "seed": 42,
  "R": 100,
  "num_cliques": 1,
  "planted_sets": [[2, 3, 6, 10, 14, 16, 19, 21, 23, 28, 29, 30]],
  "g_star": 4583.333333333333,
  "E_star": -9166.666666666666,
  "y_planted": 8.333333333333334,
  "omega_verified": 12,
  "num_max_cliques": 1
}
```

Scrambled instances additionally contain `forward_perm` and `inverse_perm` mappings.

### Boson14 maximization JSON (`_boson14.json`)

```json
{
  "file": {
    "file_name": "QuadraticModel",
    "file_config": {
      "polynomial": {
        "num_variables": 14,
        "max_degree": 2,
        "min_degree": 2,
        "data": [
          {"idx": [1, 2], "val": 2.0},
          {"idx": [1, 3], "val": 2.0},
          "..."
        ]
      }
    }
  },
  "job_params": {
    "device_type": "dirac-3",
    "num_samples": 100,
    "relaxation_schedule": 2,
    "sum_constraint": 100
  },
  "graph_info": {
    "name": "n14_k7_p0.5_s42",
    "n": 14, "k": 7, "p": 0.5, "seed": 42, "R": 100,
    "num_cliques": 1, "scrambled": false,
    "edges": 57, "density": 0.6264,
    "known_omega": 7,
    "g_star": 4285.714285714286,
    "E_star": -8571.428571428572
  }
}
```

**Polynomial format notes:**
- Indices are **1-based** (matching eqc\_models convention)
- Quadratic terms `[i+1, j+1]` with value **`+2.0`** per edge (upper-triangular, symmetrized `A`)
- This is the **maximization** convention: polynomial represents `g(y) = 0.5 * y^T A y`
- For Dirac-3 submission, negate to `J = -A` (val `-2.0`); `test_dirac.py --from-json` does this automatically
- Contrast with `hw-benchmark-toolkit` which stores `-1.0` per edge (minimization, `J = -0.5*A`)
- No linear terms (C = 0) for Motzkin-Straus
- `sum_constraint = R` (not 1 as in standard formulation)

### Dirac-3 metadata (`_dirac_meta.json`)

```json
{
  "n": 14,
  "k": 7,
  "p": 0.5,
  "seed": 42,
  "R": 100,
  "num_cliques": 1,
  "num_samples": 100,
  "relaxation_schedule": 2,
  "scrambled": false,
  "omega_verified": 7,
  "num_max_cliques": 2,
  "best_omega": 7,
  "best_objective": 4285.650139734005,
  "solve_time": 119.50580716133118,
  "num_finite_samples": 100
}
```

### Solution NPZ files

| Key | Shape | Description |
|-----|-------|-------------|
| `y_solutions` | `(num_cliques, n)` | Theoretical optimal y-vectors (`_solutions.npz`) |
| `y_vectors` | `(num_samples, n)` | Dirac-3 sample y-vectors (`_dirac_solutions.npz`) |

### Plot descriptions

- **`clique_dist_*.png`** — Bar chart of clique sizes found by brute-force enumeration, with a vertical line at the planted clique size.
- **`scaled_histogram_*.png`** — Histogram of scaled objectives `g(y)` across all Dirac-3 samples, with vertical lines marking theoretical `g*(omega)` for integer omega values. Shows how tightly hardware samples cluster around the known optimum.

---

## Mathematical Reference

The scaled Motzkin-Straus formulation uses four key equations:

**Scaled objective:**
$$g(y) = \tfrac{1}{2}\, y^\top A\, y = R^2 \cdot f(x) \qquad \text{where } y = Rx,\; \sum y_i = R$$

**Optimal value:**
$$g^*(\omega) = \frac{R^2}{2}\Bigl(1 - \frac{1}{\omega}\Bigr)$$

**Hardware energy:**
$$E(y) = y^\top J\, y = -y^\top A\, y = -2\,g(y) \qquad \text{where } J = -A$$

**Omega recovery:**
$$\omega = \operatorname{round}\!\Bigl(\frac{R^2}{R^2 - 2\,g^*}\Bigr) = \operatorname{round}\!\Bigl(\frac{R^2}{R^2 + E^*}\Bigr)$$

For the full derivation, see [scaled-motzkin-straus.md](scaled-motzkin-straus.md).

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'networkx'`
Run `uv sync` from the `dirac-test-suite` root to install dependencies.

### `ModuleNotFoundError: No module named 'eqc_models'`
Run `uv sync` to install QCI packages. These are only needed for `test_dirac.py`, not for `generate.py`.

### `Dirac-3 returned no solutions`
Check that `QCI_API_URL` and `QCI_TOKEN` are set correctly in `qci-eqc-models/.env`. Verify network connectivity to the QCI API.

### `All Dirac-3 solutions produced non-finite objectives`
All returned y-vectors contain NaN or Inf values. Try increasing `--num-samples` or adjusting `--relaxation-schedule`.

### `WARNING: w != k`
The brute-force clique number does not match the planted clique size. This can happen when random edges accidentally form a larger clique. Try a different `--seed` or reduce `--p`.

### Scramble roundtrip mismatch
If unscrambled objectives differ from scrambled objectives, verify that the `forward_perm` and `inverse_perm` in the metadata JSON are consistent. The permutation is deterministic for a given `--seed`.

### Plots not saved
Plots are saved to per-instance subfolders under `--output-dir` (default: `boson14/output/{instance_name}/`). The directory is created automatically. Use `--plot-distribution` for `generate.py` or `--plot` for `test_dirac.py`.
