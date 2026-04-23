# hard-instances-benchmarks/

Hardware stress-test pipeline for the Boson14 solver: generates difficult
max-clique instances (strategies A–E from Walteros–Buchanan / Brockington–
Culberson families), creates permutation variants, and feeds each variant to
`run_boson14.py` (unchanged — this wrapper only handles I/O).

## Design

`run_boson14.py` writes its raw hardware output (`results__*.npz`,
`plot__*.png`) to the current working directory. This pipeline calls it with
`cwd=variant_dir` so outputs land next to the input CSV, then post-processes
each NPZ to embed the hardware config (amplitude, R, num_loops, ...) and
renames files to append an `_a{amplitude}` suffix matching the existing
`boson14/output/benchmark_*/variant/results__*_a{amp}.npz` convention.

## Phases

| Phase       | What it does | Can run without hardware? |
|-------------|--------------|----------------------------|
| `generate`  | Build G(n,p) + hard-instance modification, save adjacency + base_meta | ✅ |
| `variants`  | Produce front/end/random permutation variants, save CSV + polynomial JSON | ✅ |
| `verify`    | Brute-force clique number, save theoretical solution NPZ | ✅ |
| `hardware`  | Call `run_boson14.py` per variant × amplitude, post-process NPZ | ❌ |

Each phase writes a `phase_{name}.done` marker, so `--skip-existing` resumes cleanly.

## Quick usage

```bash
# Full pipeline (default: strategies B and D, all phases)
python pipeline.py --n 90 --seeds 42 \
    --amplitudes 300,400,500,600 \
    -R 100 -s 1000 -l 256 -d 86 -w 1 -b 28 -solps 0

# Just generate CSVs (no hardware)
python pipeline.py --strategies D --n 50 --k 12 --seeds 42 \
    --phase variants --verify

# One strategy, multiple seeds
python pipeline.py --strategies A --n 90 --seeds 42,123,7

# Resume after interruption
python pipeline.py --strategies B,D --n 90 --seeds 42 --skip-existing
```

## Strategy reference

| Code | Name | Planted clique? | Extra flags |
|------|------|-----------------|-------------|
| A    | Near-threshold (k ≈ k_multiplier × log₂ n) | yes (derived k) | `--k-multiplier` |
| B    | Dense random (high p) | no (brute-forced for n ≤ 90) | — |
| C    | Degenerate (multiple planted cliques) | yes | `--k`, `--num-cliques` |
| D    | Camouflaged (degree-suppressed) | yes | `--k`, `--removal-frac` |
| E    | Overlapping near-cliques | yes | `--k`, `--overlap` |

## Strategy B handling

- **n ≤ 90:** Brute-force the max clique during `generate`; use it as the
  reference for front/end/random targeting. Same downstream flow as A/C/D/E.
- **n > 90:** Brute-force is impractical. Skip front/end; emit only `random/`
  variant(s) (controlled by `--num-random-variants`, default 1).

## Output layout

```
output/hard_D_camouflage_n90_k12_rf0.4_s42/
├── base_meta.json              # strategy, n, k, p, seed, reference_nodes, g_star
├── base_adjacency.npz          # raw A (key: 'A')
├── phase_generate.done         # idempotency markers
├── phase_variants.done
├── phase_verify.done
├── phase_hardware.done
├── front/
│   ├── front_meta.json         # variant metadata + forward/inverse perm
│   ├── front_boson14.csv       # [C_zeros | A_variant] — input for run_boson14.py
│   ├── front_boson14.json      # polynomial format (for Dirac-3 if you want it)
│   ├── front_solutions.npz     # theoretical optimal y (from --verify)
│   ├── results__front_boson14__{ts}_a300.npz  # hardware raw (renamed + post-processed)
│   ├── plot__front_boson14__{ts}_a300.png
│   ├── results__front_boson14__{ts}_a600.npz
│   └── plot__front_boson14__{ts}_a600.png
├── end/ …
└── random/ …
```

## Hardware pass-through flags

These match `run_boson14.py` exactly and are forwarded verbatim:

| Flag | Default | Description |
|------|---------|-------------|
| `-a` / `--amplitudes` | `300,400,500,600` | Pulse amplitude(s). List expands to one hardware run per variant per amplitude. |
| `-R` / `--r` | `100` | Sum constraint |
| `-s` / `--num-samples` | `1000` | Hardware samples |
| `-l` / `--num-loops` | `256` | Computation loops |
| `-d` / `--delay` | `86` | Response delay |
| `-w` / `--pulse-width` | `1` | |
| `-b` / `--distance-between-pulses` | `28` | |
| `-solps` | `0` | 0 = don't block on matplotlib in pipeline mode |

The NPZ post-processor records all these fields into each `results__*.npz` for
full hardware provenance, regardless of whether `run_boson14.py`'s default
savez() includes them.

## Self-containment

- `_internal/generators.py` — hard-instance strategy definitions (copied from
  `../../hard-instances/generators.py`)
- `_internal/bruteforce.py` — exact max-clique via Bron-Kerbosch
- `utils.py` — imports `boson14_bench` (the parent package) at runtime; no
  imports from sibling `hard-instances/`. The folder can be moved elsewhere
  as long as a compatible `boson14_bench` is on `sys.path`.
