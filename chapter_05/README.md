# Chapter 5 — Memory Architecture and Data Locality

This chapter is the performance turning point of the book.  Chapters 2–4
showed how to write *correct* parallel programs; Chapter 5 shows how to make
them *fast* by understanding and exploiting the GPU memory hierarchy.

The central insight is the **compute to global memory access ratio**
(arithmetic intensity): the naïve matrix multiply performs only **0.25 FLOP/B**,
making it severely memory-bound.  **Tiling** — loading data into shared memory
so it can be reused — raises this ratio by a factor of `TILE_WIDTH`, enabling
a 16× or more improvement in achieved throughput.

---

## Memory hierarchy recap (Section 5.2, Table 5.1)

| Memory | Location | Scope | Lifetime | Declared as | Latency |
|--------|----------|-------|----------|-------------|---------|
| Register | On-chip | Thread | Grid | automatic scalar | 1 cycle |
| Local | Off-chip | Thread | Grid | automatic array | ~600 cycles |
| Shared | On-chip | Block | Grid | `__shared__` | ~5 cycles |
| Global | Off-chip | Grid | Application | `cudaMalloc` / `__device__` | ~600 cycles |
| Constant | Off-chip (cached) | Grid | Application | `__constant__` | 1–5 cycles (broadcast) |

---

## Programs

### `01_memory_types_demo.cu` — CUDA memory types (Section 5.2, Table 5.1)

Demonstrates all five memory types in a single kernel:

- **Registers**: automatic scalar variables (`int i`, `float val`) — fastest, private per-thread
- **Local memory**: automatic array (`float scratch[4]`) — physically in global memory, thread-private
- **Shared memory**: `__shared__ float s_tile[256]` — on-chip, block-shared, requires `__syncthreads()`
- **Global memory**: `__device__ int d_global_counter` — modified via `atomicAdd`, read back with `cudaMemcpyFromSymbol`
- **Constant memory**: `__constant__ float d_scale_factor` — written from host with `cudaMemcpyToSymbol`, read-only broadcast on device

```bash
nvcc -O2 -arch=sm_89 -o memory_types 01_memory_types_demo.cu
./memory_types
```

---

### `02_tiled_matmul.cu` — Tiled matrix multiplication (Sections 5.3–5.4, Figure 5.9)

The central kernel of Chapter 5.  Implements the exact tiling algorithm from
Figure 5.9 and times it against the naïve kernel from Chapter 3.

**Tiling algorithm (Figures 5.7–5.8):**
1. Divide M and N into `TILE_WIDTH×TILE_WIDTH` tiles.
2. For each **phase** `ph`, all `TILE_WIDTH²` threads in a block collaboratively
   load one tile of M (`Mds`) and one tile of N (`Nds`) into shared memory — one element per thread.
3. Two `__syncthreads()` per phase:
   - **Line 21** (read-after-write): all threads must finish loading before any thread reads the tile.
   - **Line 26** (write-after-read): all threads must finish computing before the next phase overwrites the tile.
4. Each thread accumulates `TILE_WIDTH` products per phase into its private `Pvalue` register.

**Global memory traffic reduction:**  
With `TILE_WIDTH=16`, each element of M and N is loaded from global memory
once per block instead of once per thread — a **16× reduction** in global
memory traffic, raising arithmetic intensity from 0.25 to 4.0 FLOP/B.

**Assumption:** `Width` must be a multiple of `TILE_WIDTH`.
See `03_tiled_matmul_boundary.cu` for the general case.

```bash
nvcc -O2 -arch=sm_89 -o tiled_matmul 02_tiled_matmul.cu -lm
./tiled_matmul

# Try a larger tile (must fit in shared memory and block size limit)
nvcc -O2 -arch=sm_89 -DTILE_WIDTH=32 -o tiled_matmul_32 02_tiled_matmul.cu -lm
./tiled_matmul_32
```

Sample output (RTX 4090):
```
Naïve (Ch3)           xx.xx ms    xxx.x GFLOPS  AI = 0.25 FLOP/B
Tiled (Ch5)            x.xx ms   xxxx.x GFLOPS  AI = 4.00 FLOP/B
Speedup: ~16x
```

---

### `03_tiled_matmul_boundary.cu` — Boundary checks (Section 5.5, Figures 5.11–5.13)

Extends the tiled kernel to handle matrices of **any size** — not just
multiples of `TILE_WIDTH`.

**Two boundary problems (Figures 5.11–5.12):**

| Problem | When it occurs | Fix |
|---------|---------------|-----|
| OOB M load | `ph*TILE_WIDTH + tx ≥ Width` | Load `0.0f` instead |
| OOB N load | `ph*TILE_WIDTH + ty ≥ Width` | Load `0.0f` instead |
| OOB P store | `Row ≥ Width` or `Col ≥ Width` | Skip the write |

The `0.0f` padding is safe because `0 × anything = 0` contributes nothing
to the dot product accumulation.

The phase loop uses ceiling division: `(Width + TILE_WIDTH - 1) / TILE_WIDTH`.

Tests verify correctness for:
- The book's 3×3 example with TILE_WIDTH=2 (Figures 5.11–5.12)
- `Width = TILE_WIDTH ± 1` (edge cases)
- Large non-multiples (e.g. 1000 with TILE_WIDTH=16)

```bash
nvcc -O2 -arch=sm_89 -o tiled_matmul_bc 03_tiled_matmul_boundary.cu -lm
./tiled_matmul_bc
```

---

### `04_dynamic_shared_memory.cu` — Dynamic shared memory (Section 5.6, Figure 5.14)

Shows how to replace static shared memory declarations with runtime-configurable
shared memory so the tile size can be selected based on the device's actual
`sharedMemPerBlock` property.

**Static (Figure 5.9):**
```cuda
__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];   // size fixed at compile time
__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
Mds[ty][tx]  // 2D indexing
```

**Dynamic (Figure 5.14):**
```cuda
extern __shared__ float Mds_Nds[];              // size supplied at launch
float* Mds = (float*) Mds_Nds;
float* Nds = (float*) Mds_Nds + Mds_sz;
Mds[ty * tile_w + tx]  // linearised 1D indexing
```

**Launch:**
```c
size_t shmem_bytes = (Mds_sz + Nds_sz) * sizeof(float);
matrixMulDynamic<<<dimGrid, dimBlock, shmem_bytes>>>(...);
//                                    ^^^^^^^^^^^^ 3rd parameter
```

```bash
nvcc -O2 -arch=sm_89 -o dynamic_shmem 04_dynamic_shared_memory.cu -lm
./dynamic_shmem
```

---

### `05_roofline_analysis.cu` — Roofline model (Section 5.1 and sidebar)

Measures and visualises where the naïve and tiled kernels sit on the roofline.

**Arithmetic intensity:**
```
Naïve matmul:  2 FLOPs / 8 bytes = 0.25 FLOP/B   (memory-bound)
16×16 tiling:  0.25 × 16 = 4.00 FLOP/B            (still memory-bound, but much better)
32×32 tiling:  0.25 × 32 = 8.00 FLOP/B
```

**Roofline formula:**
```
Achievable GFLOPS = min(peak_GFLOPS, bandwidth_GB/s × arithmetic_intensity)
```

The program:
1. Measures achieved memory bandwidth with a copy kernel.
2. Times naïve vs tiled matmul and computes achieved GFLOPS.
3. Prints a roofline table showing which kernels are memory-bound vs compute-bound.

```bash
nvcc -O2 -arch=sm_89 -o roofline 05_roofline_analysis.cu -lm
./roofline
```

---

## Building all programs

```bash
make SM_ARCH=sm_89              # release build (all targets)
make SM_ARCH=sm_89 DEBUG=1      # debug build (-g -G)
make SM_ARCH=sm_89 TILE=32      # override tile width for tiled kernels
make clean
```

---

## Debugging in VS Code

| File | Best breakpoint | What to observe |
|------|----------------|----------------|
| `02_tiled_matmul.cu` | Line 21 (`__syncthreads()` after load) | `Mds[ty][tx]` just written; neighbour slots not yet filled — shows *why* the barrier is necessary |
| `02_tiled_matmul.cu` | Line 24 (inner k-loop body) | `Mds[ty][k]` and `Nds[k][tx]` being read; `Pvalue` accumulating |
| `03_tiled_matmul_boundary.cu` | OOB load path (`else Mds[ty][tx] = 0.0f`) | Switch focus to a boundary thread (e.g. thread (1,0) of block (1,0) for a 3×3 matrix) to see the 0.0f substitution |
| `04_dynamic_shared_memory.cu` | After `float* Mds = Mds_Nds` | Inspect the raw `Mds_Nds` pointer and the `Mds`/`Nds` offsets |

---

## Key formulas

```
Arithmetic intensity (AI)  = FLOPs / bytes_from_global_memory
Naïve matmul AI            = 2 / (2 × 4) = 0.25 FLOP/B
Tiled matmul AI            = 0.25 × TILE_WIDTH FLOP/B
Global traffic reduction   = TILE_WIDTH× (one load per block, not per thread)
Shared memory per tile     = 2 × TILE_WIDTH² × 4 bytes
Max tile width (smem only) = sqrt(sharedMemPerBlock / (2 × 4))
```
