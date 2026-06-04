# Chapter 8 — Stencil

**Book sections:** §8.1 Background · §8.2 Parallel stencil: a basic algorithm · §8.3 Shared memory tiling · §8.4 Thread coarsening · §8.5 Register tiling

---

## Programs

### `01_stencil1d_basic.cu` — 1D finite-difference stencil (§8.1)

The conceptual building block.  Stencils arise from discretising differential equations on structured grids using finite differences.  Two examples:

- **First derivative** (Fig 8.2A — three-point, order 1):  
  `FD[i] = (F[i+1] - F[i-1]) / (2h)`  
  Stencil weights: `c_left = -1/(2h)`, `c_center = 0`, `c_right = +1/(2h)`.

- **Laplacian** (second derivative, three-point):  
  `FD2[i] = (F[i-1] - 2*F[i] + F[i+1]) / h²`

Boundary cells (first and last) hold boundary conditions and are **not updated** during a sweep (Fig 8.5). The guard `if (i >= 1 && i < N-1)` enforces this.

Verification uses `f(x) = sin(2πx)` and compares GPU output against the known analytical derivative.

---

### `02_stencil3d_basic.cu` — 3D seven-point stencil: basic kernel (§8.2 / Figure 8.6)

Direct implementation of **Figure 8.6**.

The 3D seven-point stencil (Fig 8.3C, order 1) is the discrete Laplacian:
```
out[i][j][k] = c0·in[i][j][k]
             + c1·in[i][j][k-1] + c2·in[i][j][k+1]    (x face neighbours)
             + c3·in[i][j-1][k] + c4·in[i][j+1][k]    (y face neighbours)
             + c5·in[i-1][j][k] + c6·in[i+1][j][k]    (z face neighbours)
```

Thread assignment (Fig 8.6 lines 02–04):
- `k ← blockIdx.x * blockDim.x + threadIdx.x`  (stride-1 for coalescing)
- `j ← blockIdx.y * blockDim.y + threadIdx.y`
- `i ← blockIdx.z * blockDim.z + threadIdx.z`

Only interior points `[1, N-2]` in each dimension are computed.

Arithmetic intensity: **0.46 OP/B** (13 FLOPs / 28 bytes = 13 / (7×4)).

---

### `03_stencil3d_shared_mem.cu` — Shared memory tiling (§8.3 / Figure 8.8)

Implements **Figure 8.8** using the same tiling strategy as convolution (Chapter 7), adapted for stencil sweep.

Key difference from convolution tiling (§8.3):
- Convolution uses **all** elements in the tile (including corners).
- Stencil uses only **face-adjacent** neighbours — no corner values needed.
- This reduces the arithmetic intensity achievable by tiling.

```
IN_TILE_DIM  = 8   (block size — 8×8×8 = 512 threads)
OUT_TILE_DIM = 6   (= IN_TILE_DIM - 2)
```

Arithmetic intensity with T=8: **1.37 OP/B**. Upper bound as T→∞: **3.25 OP/B** (= 13/4).

The program prints a table showing AI vs tile size, analogous to Figure 7.14.

---

### `04_stencil3d_coarsened.cu` — Thread coarsening in z (§8.4 / Figure 8.10)

Implements **Figure 8.10**: a 2D thread block slides through the z dimension, processing one x-y plane per iteration.

Motivation (§8.4): the 3D shared memory kernel is limited to T=8 blocks (512 threads max). This gives poor reuse (1.37 OP/B) and poor coalescing. Thread coarsening decouples the tile size from the block size.

Sliding window state:
- `inPrev_s[IN_TILE_DIM][IN_TILE_DIM]` — z-1 plane (shared memory)
- `inCurr_s[IN_TILE_DIM][IN_TILE_DIM]` — z   plane (shared memory)
- `inNext_s[IN_TILE_DIM][IN_TILE_DIM]` — z+1 plane (shared memory)

After each z-plane is computed: `inPrev_s ← inCurr_s`, `inCurr_s ← inNext_s`.

```
IN_TILE_DIM  = 32   (2D block: 32×32 = 1024 threads)
OUT_TILE_DIM = 30   (= IN_TILE_DIM - 2)
Shared memory = 3 × 32² × 4 = 12 KB per block
```

Arithmetic intensity with T=32: **2.68 OP/B** (= 13/4 × (30/32)³).

---

### `05_stencil3d_register_tiling.cu` — Register tiling (§8.5 / Figure 8.12)

Implements **Figure 8.12** and runs a four-way final benchmark.

Observation (§8.5): in Fig 8.10, each element of `inPrev_s` and `inNext_s` is accessed by exactly **one thread**.  Private data belongs in **registers**, not shared memory.

Changes from Figure 8.10:
- `inPrev_s` → `float inPrev` (register)
- `inNext_s` → `float inNext` (register)
- `inCurr_s` stays in shared memory (needed by x-y neighbours)
- `inCurr` register copy holds the center value

After each z-plane:
```c
inPrev = inCurr;
inCurr = inNext;
inCurr_s[ty][tx] = inNext;    // keep shared mem in sync for x-y neighbours
```

Shared memory per block is reduced from **12 KB** to **4 KB**. AI is unchanged.

Benchmark output (all four kernels, 256³ grid):

| Kernel | AI | Notes |
|--------|-----|-------|
| Basic (Fig 8.6) | 0.46 OP/B | no data reuse |
| Shared mem 8³ (Fig 8.8) | 1.37 OP/B | limited by T=8 max |
| Coarsened z (Fig 8.10) | 2.68 OP/B | 32×32 tile, 12 KB shmem |
| Register tiling (Fig 8.12) | 2.68 OP/B | 32×32 tile, 4 KB shmem |

---

## Building

```bash
cd chapter_08
make SM_ARCH=sm_89
make SM_ARCH=sm_89 DEBUG=1   # adds -g -G for cuda-gdb
```

```bash
./stencil1d
./stencil3d_basic
./stencil3d_shared
./stencil3d_coarsened
./stencil3d_regtile          # prints four-way benchmark table
```

---

## Key concepts

| Concept | File |
|---------|------|
| Stencil sweep vs convolution: face neighbours only, no corners | All |
| Boundary cells NOT updated (hold initial conditions) | `02` — `if (i>=1 && i<N-1 ...)` |
| AI ceiling for 7-point stencil: 13/4 = 3.25 OP/B | `03` — printed table |
| 3D block size constraint (max 1024 threads → T≤10) | `03` |
| Thread coarsening overcomes the T limit | `04` |
| Sliding window: inPrev_s / inCurr_s / inNext_s | `04` — barrier order |
| Register tiling: private z-planes → registers | `05` — `inPrev`, `inNext` |
| Shared memory reduction 12KB → 4KB with register tiling | `05` |

---

## Debugging tips

- In `04` and `05`, set a breakpoint inside the z-iteration loop and trace how `inPrev_s` / `inCurr_s` / `inNext_s` change as `i` increments.
- To observe the register tiling benefit: use `-exec info registers` in cuda-gdb after breaking inside `stencil3d_regtile_kernel`. You should see `inPrev`, `inCurr`, and `inNext` as distinct register variables.
- For the 3D basic kernel, set a breakpoint at the output write on a boundary block (i=0 or i=N-1) and verify it is never reached — the `if` guard prevents it.
