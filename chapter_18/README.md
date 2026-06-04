# Chapter 18 — Electrostatic Potential Map

Code samples for **Chapter 18: Electrostatic potential map** from *Programming Massively Parallel Processors* (4th ed.).

The chapter implements **Direct Coulomb Summation (DCS)** for computing
electrostatic potential maps, using the VMD application as a case study.
Three progressively optimised kernels are presented.

| File | Section | Strategy | Key idea |
|------|---------|----------|----------|
| `01_dcs_gather.cu` | §18.2 | Gather + constant memory (Fig 18.6) | One thread per grid point; atoms[] in constant cache; broadcast access |
| `02_dcs_coarsen.cu` | §18.3 | Thread coarsening (Fig 18.8) | Each thread computes 4 adjacent grid points; dy²+dz² amortised |
| `03_dcs_coalesce.cu` | §18.4 | Coarsening + coalescing (Fig 18.10) | 4 points spaced blockDim.x apart; adjacent threads write consecutive memory |

---

## The computation (§18.1–18.2)

Direct Coulomb Summation (DCS) at grid point j:

```
energy[j] = Σ_i  charge[i] / √(dx² + dy² + dz²)
```

where dx, dy, dz are distances from atom i to grid point j.

With N_atoms atoms and N_grid grid points, the cost is O(N_atoms × N_grid).

---

## Scatter vs Gather (§18.2)

- **Scatter** (Fig 18.5): one thread per atom, scatters contribution to all
  grid points via `atomicAdd` — heavy contention.
- **Gather** (Fig 18.6): one thread per grid point, accumulates from all atoms
  — no atomics.  All threads access atoms in the *same* order (same index per
  warp iteration) → ideal for constant-memory broadcast.

---

## Thread coarsening (§18.3, Fig 18.8)

Grid points in the same row (same j) share the same y and z distance to every
atom.  Rather than recomputing dy and dz for each grid point independently,
each thread handles COARSEN_FACTOR=4 consecutive grid points and computes dy,
dz, and `dy²+dz²` only once per atom.  This eliminates 3×(COARSEN_FACTOR−1)
redundant constant-memory reads per atom per row.

The writes for the 4 grid points per thread have stride COARSEN_FACTOR between
adjacent threads in a warp — uncoalesced.

---

## Memory coalescing (§18.4, Fig 18.10)

Fix: assign to each thread the 4 grid points at positions
`i, i+blockDim.x, i+2·blockDim.x, i+3·blockDim.x`.

Adjacent threads now write to adjacent memory locations for each of the four
write statements → all four writes are coalesced.

---

## Cutoff binning (§18.5)

The book describes (but does not give a complete kernel for) a cutoff-summation
approach that reduces complexity from O(N_atoms × N_grid) to O(N_grid) by only
considering atoms within a fixed cutoff radius.  Atoms are sorted into spatial
bins; each thread block iterates over the neighbourhood bins, loading atom data
from global memory into shared memory.  The cutoff binning samples in this repo
stop at Fig 18.10.

---

## Building

```bash
make SM_ARCH=sm_89
make SM_ARCH=sm_89 DEBUG=1
```
