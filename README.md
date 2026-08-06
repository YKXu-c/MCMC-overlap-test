# A Scaled Susceptibility Interpretation of Algorithmic Overlap in Spin-Model Monte Carlo

Reproducibility data and code for the paper by Y. Xu, Z. Wang and C. Cao (Zhejiang University).

## Overview

We study the configuration and cluster overlaps in Markov-chain Monte Carlo (MCMC) simulations of the two-dimensional ferromagnetic Ising and $q$-state Potts models, for the Metropolis, Swendsen--Wang, and Wolff dynamics. To leading order, the overlap observables are governed by the scaled susceptibility $\chi/N$; their means and variances inherit the critical scaling of $\chi$ and therefore signal the phase transition. The derivation uses the Fortuin--Kasteleyn (FK) random-cluster representation.

This repository contains everything needed to reproduce the figures of the paper: the figure-generation script, the driver scripts that produce the underlying simulation data, the C++ Monte Carlo sources, and the aggregate data.

## Repository layout

```
.
├── generatePic.py          # Master figure-generation script (produces all paper figures)
├── run_phase_b.py          # Produces per-(L, seed) multi-Nt time-series data
├── regen_phase_a.py        # Regenerates the 7-seed FSS family
├── regen_allup.py          # Regenerates the all-up-initialized data set
├── examples/
│   ├── CMakeLists.txt      # CMake build for the C++ sources
│   ├── include/
│   │   ├── mc_base.hpp     # CRTP simulation framework + ObservableRegistry
│   │   ├── potts_model.hpp # q-state Potts model
│   │   └── union_find.hpp  # Union-find with path compression + rank
│   └── src/
│       ├── potts_metropolis.cpp
│       ├── potts_swendsen_wang.cpp
│       └── potts_wolff.cpp
├── overlap_series/        # Aggregate per-(algo,q,L,seed) overlap data (784 .npz)
├── .fss_cache/            # Finite-size-scaling aggregate results (249 .pkl)
└── figures/                # The 30 figures used in the paper (PNG)
```

## Requirements

- **Python 3.13+** with `numpy` and `matplotlib`
- **C++17** compiler (g++ / clang++) and `cmake` (optional, only to re-run simulations)
- The plotting scripts call the C++ binaries from `examples/build/` via `run_mc_binary()`.

## Reproducing the figures

### 1. Build the C++ Monte Carlo binaries

```bash
cd examples
mkdir -p build && cd build
cmake .. && make
```

This produces `potts_metropolis`, `potts_swendsen_wang`, and `potts_wolff` in `examples/build/`. The Python scripts locate them via `MC_BINARY_DIR` (default `examples/build/`).

### 2. (Optional) Regenerate the raw data

The aggregate data in `overlap_series/` and `.fss_cache/` already reproduce all paper figures without re-running simulations. To re-run the simulations from scratch:

```bash
# per-(L, seed) multi-Nt time-series data
python3 run_phase_b.py --L 64 --seed 42 --algo wolff

# FSS families and all-up data sets
python3 regen_phase_a.py
python3 regen_allup.py
```

### 3. Generate the figures

```bash
python3 generatePic.py
```

This regenerates all paper figures into `figures/` (overlap figures for $q \in \{2,3,4\}$, the FSS collapse tests, the $N_t$-convergence study, and the schematics).

## Data notes

- `overlap_series/*.npz` contain the aggregate overlap means/variance per (algorithm, $q$, lattice size $L$, seed), plus the mean FK-cluster size. The raw per-$N_t$ time series are large (≈2.8 GB in total) and are not stored here; they can be regenerated with `run_phase_b.py`.
- `.fss_cache/*.pkl` are the finite-size-scaling collapse-test results keyed by MD5 of the parameter tuple.
- `figures/*.png` are the exact figures embedded in the paper (main text + supplementary material).

## Paper

The accompanying manuscript (Elsevier CAS template) builds with the standard 4-pass `pdflatex -> bibtex -> pdflatex -> pdflatex`. For questions or to request the full raw time series, please contact the corresponding authors.

## License

MIT — see `LICENSE`.
