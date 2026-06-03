# Chapter 6 — Performance Considerations

Chapter 6 wraps up Part 1 of the book by presenting the *off-chip memory*
(DRAM) architecture and the performance techniques it motivates, then
consolidates every optimisation from Chapters 2–6 into a single actionable
checklist (Table 6.1).

The chapter's three main contributions are:

1. **Memory coalescing** — how the DRAM burst/channel/bank architecture
   rewards warp-level stride-1 accesses and punishes strided accesses.
2. **Thread coarsening** — assigning multiple output elements to one thread
   to amortise the cost of redundant data loading.
3. **Optimisation checklist** (Table 6.1) — a concise guide to when and how
   to apply each technique.

---

## DRAM background (Sections 6.1–6.2)

| Concept | What it means |
|---------|--------------|
| DRAM burst | One address access fetches many consecutive bytes in parallel |
| Coalescing | Hardware combines warp's 32 loads into one burst when they are consecutive |
| Channel | Independent memory controller; multiple channels → parallel bursts |
| Bank | Multiple banks per channel; overlap cell-array latency across banks |
| Interleaved distribution | OS/hardware distributes array elements across banks so concurrent threads hit different banks |

---

## Programs

### `01_coalescing_demo.cu` — Coalesced vs uncoalesced access (Section 6.1, Figures 6.2–6.3)

Two kernels perform the same matrix copy but with different access patterns:

| Kernel | Access pattern | Why | Result |
|--------|---------------|-----|--------|
| `rowReadKernel` | Thread tx reads `M[row*W + tx]` — stride 1 | Consecutive threads → consecutive addresses → one DRAM burst | **COALESCED** |
| `colReadKernel` | Thread tx reads `M[tx*W + col]` — stride W | Consecutive threads → addresses W floats apart → up to 32 bursts | **UNCOALESCED** |

Also explains the matmul access analysis from Figures 6.2–6.3:
- `N[k*Width + col]` where col increments with `threadIdx.x` → stride 1 → COALESCED ✓
- Column-major N: `N_col[col*Width + k]` → stride Width → UNCOALESCED ✗

```bash
nvcc -O2 -arch=sm_89 -o coalescing_demo 01_coalescing_demo.cu
./coalescing_demo
```

Expected: row read is significantly faster (typically 5–20×) than column read.

---

### `02_corner_turning.cu` — Corner turning (Section 6.1, Figure 6.4)

Demonstrates the corner-turning technique for computing C = A × B when B is
stored in column-major layout (e.g., the transpose of a row-major matrix).

**Without corner turning (Figure 6.4A):**  
Loading the B tile assigns thread (ty, tx) to `B_cm[Col * W + ph*TILE + ty]`.  
Consecutive `threadIdx.x` → consecutive `Col` → stride-W accesses in column-major → **UNCOALESCED** ✗

**With corner turning (Figure 6.4B):**  
Swap tx and ty roles so thread (ty, tx) loads `B_cm[(bx*TILE+ty)*W + ph*TILE+tx]`.  
Consecutive `threadIdx.x` → consecutive `ph*TILE+tx` → stride-1 in column-major → **COALESCED** ✓  
The tile is stored transposed in shared memory; the dot product reads `Bds[k][ty]` instead of `Bds[ty][k]`.

```bash
nvcc -O2 -arch=sm_89 -o corner_turning 02_corner_turning.cu -lm
./corner_turning
```

---

### `03_thread_coarsening_matmul.cu` — Thread coarsening (Section 6.3, Figure 6.13)

Reproduces Figure 6.13 exactly.  Each thread block is responsible for
`COARSE_FACTOR` adjacent output tiles (columns of P).  Each thread owns
`COARSE_FACTOR` output elements and accumulates them in `float Pvalue[COARSE_FACTOR]`.

**Key changes from the Chapter 5 tiled kernel:**

| | Tiled (Ch5) | Coarsened (Ch6) |
|---|---|---|
| Column start | `Col = bx*TILE + tx` | `colStart = bx*TILE*CF + tx` |
| Accumulators | `float Pvalue` | `float Pvalue[CF]` |
| M tile loads | 1 per phase | 1 per phase (shared by all CF iterations) |
| N tile loads | 1 per phase | CF per phase (one per output tile) |
| Grid x-dim | `W / TILE` | `W / (TILE * CF)` |

**Result:** M tiles are loaded `CF×` less often, reducing global memory traffic.

```bash
# Default COARSE_FACTOR=4
nvcc -O2 -arch=sm_89 -o thread_coarsening 03_thread_coarsening_matmul.cu -lm
./thread_coarsening

# Experiment with different coarsening factors
nvcc -O2 -arch=sm_89 -DCF=2 -o tc_cf2 03_thread_coarsening_matmul.cu -lm
nvcc -O2 -arch=sm_89 -DCF=8 -o tc_cf8 03_thread_coarsening_matmul.cu -lm
```

**Pitfalls (Section 6.3):**
- Don't coarsen when there's no redundancy cost (e.g. vecAdd, grayscale).
- Too-large `COARSE_FACTOR` → too few thread blocks → hardware under-utilised.
- Extra `Pvalue[]` registers per thread may reduce occupancy.

---

### `04_optimization_progression.cu` — Cumulative optimisations (Table 6.1)

Benchmarks four kernels in sequence, showing each optimisation's incremental
contribution to performance on a 1024×1024 matrix:

| Kernel | Optimisations | Expected speedup |
|--------|--------------|-----------------|
| 1. Naïve (Ch3) | None | 1× baseline |
| 2. Tiled 16×16 (Ch5) | Data reuse + coalescing | ~8–16× |
| 3. Tiled 32×32 (Ch5) | Larger tile → more reuse | ~15–25× |
| 4. Coarsened 32×32×CF4 (Ch6) | + Thread coarsening | ~20–35× |

```bash
nvcc -O2 -arch=sm_89 -o opt_progression 04_optimization_progression.cu -lm
./opt_progression
```

---

## Building all programs

```bash
make SM_ARCH=sm_89                     # release build
make SM_ARCH=sm_89 DEBUG=1             # debug build (-g -G)
make SM_ARCH=sm_89 CF=2 thread_coarsening  # custom coarsening factor
make clean
```

---

## Debugging in VS Code

| File | Good breakpoint | What to observe |
|------|----------------|----------------|
| `01_coalescing_demo.cu` | Inside `colReadKernel`, the `M[row*W+col]` load | Set focus to thread 0 vs thread 1 — their `row` values differ by 1, showing that consecutive threads access addresses `W` floats apart |
| `02_corner_turning.cu` | B load in `matmulColMajorCornerTurned` | Check `(bx*TILE+ty)*W + ph*TILE+tx` for consecutive `tx` — addresses increment by 1 |
| `03_thread_coarsening_matmul.cu` | Inner `c`-loop, `Nds` load | Switch focus to `c=0` vs `c=1` — same `Mds` is reused, different `Nds` loaded |

---

## Optimisation checklist — Table 6.1

| Optimisation | Benefit to compute | Benefit to memory | Key strategy |
|-------------|-------------------|------------------|-------------|
| **Maximise occupancy** | More work hides pipeline latency | More concurrent accesses hide DRAM latency | Tune threads/block, registers, shared memory |
| **Coalesced global access** | Fewer stalls on global memory | Less traffic, better burst utilisation | Stride-1 warp accesses; corner turning for irregular layouts |
| **Minimise control divergence** | High SIMD efficiency | — | Rearrange thread-to-data mapping; rearrange data layout |
| **Tiling of reused data** | Fewer global memory stalls | Less global traffic | Load into shared memory; reuse within block |
| **Privatisation** | Fewer stalls on atomic updates | Less contention | Private partial result per thread; merge at end |
| **Thread coarsening** | Less redundant work/sync | Less redundant global traffic | Assign multiple units of work per thread |

---

## Key formulas

```
Coalesced condition: warp accesses addresses X, X+1, X+2, … (aligned)
Arithmetic intensity with tiling: 0.25 × TILE_WIDTH FLOP/B
Coarsening benefit: M tile loaded CF× fewer times → CF× less M traffic
Coarsening grid: dimGrid.x = Width / (TILE_WIDTH × COARSE_FACTOR)
```
