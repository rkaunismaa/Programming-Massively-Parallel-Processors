# Chapter 7 — Convolution

**Book sections:** §7.1 Background · §7.2 Parallel convolution: a basic algorithm · §7.3 Constant memory and caching · §7.4 Tiled convolution with halo cells · §7.5 Tiled convolution using caches for halo cells

---

## Programs

### `01_conv1d_basic.cu` — 1D convolution: ghost cells and constant memory (§7.1–7.3)

Introduces the two fundamental concepts of Chapter 7 in the simpler 1D case:

- **Ghost cells** (§7.1): output elements near the boundary of the array need input elements that do not exist. The book assumes default value 0 — implemented by an `if` guard that skips out-of-bounds neighbours.
- **Constant memory** (§7.3): the filter `F` is small, read-only, and accessed in the same order by every thread. `__constant__ float F_const[FILTER_DIM]` stores it in a 64 KB constant cache; `cudaMemcpyToSymbol` transfers it from host. The kernel accesses `F_const` as a global variable — no pointer argument needed.

Prints the arithmetic intensity for both variants:

| Kernel | AI |
|--------|-----|
| F in global memory | 0.25 OP/B |
| F in constant cache | 0.50 OP/B |

---

### `02_conv2d_basic.cu` — Basic 2D convolution (§7.2 / Figure 7.7)

Direct implementation of **Figure 7.7**.

- 2D grid of 2D blocks; one thread computes one output element (Figure 7.6).
- `outCol = blockIdx.x*blockDim.x + threadIdx.x` · `outRow = blockIdx.y*blockDim.y + threadIdx.y`
- The double nested loop iterates over the `(2r+1)×(2r+1)` filter patch.
- `if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)` is the ghost-cell guard.
- Filter passed as a linearised pointer `F[fRow*(2r+1)+fCol]`; book notation `F[fRow][fCol]` is shorthand.

Performance is limited by DRAM bandwidth: AI ≈ 0.25 OP/B.

---

### `03_conv2d_constant_mem.cu` — Constant memory for filter (§7.3 / Figure 7.9)

Implements **Figure 7.9** and benchmarks it against the basic kernel.

Key changes from Figure 7.7:
1. `__constant__ float F_c[FILTER_DIM][FILTER_DIM]` declared at file scope.
2. `cudaMemcpyToSymbol(F_c, F_h, size)` on the host replaces `cudaMemcpy`.
3. Kernel drops the `float *F` parameter; accesses `F_c[fRow][fCol]` directly.

The hardware routes all warp accesses to F into a single broadcast from the constant cache (Figure 7.8), eliminating DRAM bandwidth for the filter: AI doubles to 0.50 OP/B.

---

### `04_conv2d_tiled_halo.cu` — Tiled convolution with explicit halo loading (§7.4 / Figure 7.12)

Implements **Figure 7.12** with compile-time tile dimensions.

```
IN_TILE_DIM  = 32   (block size — same as input tile stored in shared memory)
OUT_TILE_DIM = IN_TILE_DIM − 2×FILTER_RADIUS   (e.g. 28 for r=2)
```

Thread → input element mapping:
```
col = blockIdx.x * OUT_TILE_DIM + threadIdx.x − FILTER_RADIUS
row = blockIdx.y * OUT_TILE_DIM + threadIdx.y − FILTER_RADIUS
```
Threads at the edges of the block load halo elements; ghost cells are written as 0.  
After `__syncthreads()` only the inner `OUT_TILE_DIM × OUT_TILE_DIM` threads compute output elements.

Arithmetic intensity formula (§7.4):

```
AI = OUT_TILE_DIM² × (2r+1)² × 2  /  (IN_TILE_DIM² × 4)
   = 28² × 25 × 2 / (32² × 4) ≈ 9.57 OP/B   (matches Figure 7.14)
```

The program also prints the Figure 7.14 table for 5×5 filters at IN_TILE_DIM ∈ {8, 16, 32}.

---

### `05_conv2d_cached_halo.cu` — Cached halo cells (§7.5 / Figure 7.15)

Implements **Figure 7.15** and runs a final four-way benchmark.

The simplification: block size equals the output tile size (TILE_DIM × TILE_DIM).  
Each thread loads exactly one element (its own output position) into `N_s` — no halo threads, no offset arithmetic in the load phase.

The filter loop distinguishes two cases per neighbour:
- Neighbour index is **inside** `N_s` → use shared memory (fast).
- Neighbour index is **outside** `N_s` (halo) → fall through to global memory; hardware L2 cache serves these accesses because neighbouring blocks already loaded them.

Ghost cells (outside the image) contribute 0 — skipped with an `if` guard on the global indices.

Benchmark output (all four kernels, 2048×2048 image, 5×5 filter):

| Kernel | AI | Notes |
|--------|-----|-------|
| Basic (Fig 7.7) | 0.25 OP/B | F in DRAM |
| Const mem (Fig 7.9) | 0.50 OP/B | F in constant cache |
| Tiled halo (Fig 7.12) | ~9.57 OP/B | N tile in shared memory |
| Cached halo (Fig 7.15) | ~9.57 OP/B | simpler code, same AI |

---

## Building

```bash
cd chapter_07

# Default: 5×5 filter (FILTER_RADIUS=2), sm_89
make SM_ARCH=sm_89

# 9×9 filter (FILTER_RADIUS=4)
make SM_ARCH=sm_89 FILTER_RADIUS=4

# Debug build (adds -g -G for cuda-gdb)
make SM_ARCH=sm_89 DEBUG=1

# Single target
make SM_ARCH=sm_89 conv2d_cached
```

```bash
./conv1d
./conv2d_basic
./conv2d_const
./conv2d_tiled
./conv2d_cached
```

---

## Key concepts

| Concept | Where demonstrated |
|---------|-------------------|
| Ghost cells / zero-padding | All files — `if (inRow >= 0 && ...)` guard |
| `__constant__` declaration and `cudaMemcpyToSymbol` | `03`, `04`, `05` |
| Constant cache broadcast (warp-uniform load) | `03` discussion |
| Input tile vs output tile size mismatch | `04` — IN=32, OUT=28 for r=2 |
| Halo threads disabled during output phase | `04` — tileCol/tileRow guard |
| Arithmetic intensity formula (§7.4) | `04` — prints Fig 7.14 table |
| L2 cache exploitation for halo cells | `05` — cached-halo kernel |
| Four-way optimisation progression | `05` — full benchmark table |

---

## Debugging tips

- Breakpoint inside `convolution_cached_tiled_2D_const_mem_kernel` at the `if (shRow >= 0 …)` branch. Switch focus to a corner thread (e.g. block (0,0) thread (0,0)) to watch it take the global-memory (halo) path instead of the shared-memory path.
- Use `-exec print threadIdx` and `-exec print F_c[0][0]` to inspect constant memory from the device debugger.
- To test boundary handling, run with a small image (e.g. 5×5) and print the P array: the first and last `FILTER_RADIUS` rows/columns should have smaller values due to ghost-cell contributions of 0.
