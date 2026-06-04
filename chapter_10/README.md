# Chapter 10 — Reduction and Minimizing Divergence

**Book sections:** §10.1 Background · §10.2 Reduction trees · §10.3 Simple kernel · §10.4 Minimizing control divergence · §10.5 Minimizing memory divergence · §10.6 Minimizing global memory accesses · §10.7 Hierarchical multiblock reduction · §10.8 Thread coarsening

---

## Programs

### `01_reduction_simple.cu` — Simple kernel with divergence analysis (§10.3 / Figure 10.6)

The baseline parallel sum reduction. Each thread owns the even-indexed location `input[2*threadIdx.x]`. Stride **doubles** each iteration:

```c
for (stride = 1; stride <= blockDim.x; stride *= 2)
    if (threadIdx.x % stride == 0)
        input[2*threadIdx.x] += input[2*threadIdx.x + stride];
```

**Problems diagnosed:**

| Problem | Cause |
|---------|-------|
| Control divergence (§10.4) | Active threads are 0, stride, 2*stride, … — spread across warps |
| Memory divergence (§10.5) | Thread i reads `input[2i]` and `input[2i+stride]` — stride-2 access (non-coalesced) |

The program prints a divergence analysis table showing execution resource utilisation ≈ 35% for N=256.

---

### `02_reduction_convergent.cu` — Convergent kernel (§10.4/10.5 / Figure 10.9)

One change from Fig 10.6: `i = threadIdx.x` (not `2*threadIdx.x`) and stride **halves**:

```c
for (stride = blockDim.x; stride >= 1; stride /= 2)
    if (threadIdx.x < stride)
        input[i] += input[i + stride];
```

Active threads are always a contiguous prefix `[0, stride)`. For stride ≥ 32, all warps are either fully active or fully idle — **no warp-level divergence**. Divergence only occurs in the final 5 iterations (stride < 32).

Access pattern: thread i reads `input[i]` and `input[i+stride]` — adjacent threads access adjacent locations → **coalesced**.

Execution resource utilisation improves from ~35% to ~66%.

---

### `03_reduction_shared_mem.cu` — Shared memory kernel (§10.6 / Figure 10.11)

Keeps all intermediate partial sums in shared memory to eliminate repeated global memory writes:

```c
input_s[t] = input[t] + input[t + BLOCK_DIM];   // single coalesced global load
for (stride = blockDim.x/2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) input_s[t] += input_s[t + stride];   // shared memory only
}
if (t == 0) *output = input_s[0];   // single global write
```

Total global memory accesses: **N + 1** (N reads at start + 1 write at end).  
Compare with Fig 10.9: ~36 global requests for N=256 → ~4× fewer.  
The original input array is **not modified** (non-destructive).

---

### `04_reduction_multiblock.cu` — Segmented multiblock reduction (§10.7 / Figure 10.13)

Extends the shared memory kernel to arbitrary input length using multiple blocks. Each block independently reduces a segment of `2 * blockDim.x` elements, then contributes its partial sum to the output with `atomicAdd`:

```c
segment = 2 * blockDim.x * blockIdx.x
i = segment + threadIdx.x
input_s[t] = input[i] + input[i + BLOCK_DIM]
// ... reduction tree ...
if (t == 0) atomicAdd(output, input_s[0]);
```

No cross-block synchronisation is needed: blocks are independent and can run in any order. The host must initialise `output` to 0 before the launch.

Tested for N ∈ {2048, 64K, 1M, 4M}.

---

### `05_reduction_coarsened.cu` — Thread coarsening (§10.8 / Figure 10.15)

When N is large, the segmented kernel launches more blocks than the hardware can run simultaneously. Surplus blocks are serialised, each paying the full reduction-tree overhead (synchronisation, shared memory accesses) — wasteful.

Thread coarsening assigns `COARSE_FACTOR × 2` elements per thread. Each thread first accumulates its elements serially (no `__syncthreads()`, no shared memory), then participates in the shared-memory reduction tree:

```c
float sum = input[i];
for (tile = 1; tile < COARSE_FACTOR*2; tile++)
    sum += input[i + tile*BLOCK_DIM];   // serial accumulation (all threads active)
input_s[t] = sum;
// ... reduction tree in shared memory ...
```

Benefits (Fig 10.16):
- `COARSE_FACTOR × fewer` blocks → `COARSE_FACTOR × fewer` reduction trees
- All threads active during the serial phase → hardware fully utilised
- Fewer synchronisation points overall

All five kernels benchmarked side-by-side.

---

## Building

```bash
cd chapter_10
make SM_ARCH=sm_89
make SM_ARCH=sm_89 COARSE_FACTOR=8   # vary coarsening factor
make SM_ARCH=sm_89 DEBUG=1
```

```bash
./reduction_simple        # shows divergence analysis table
./reduction_convergent    # compares simple vs convergent
./reduction_shared        # shared memory vs global memory
./reduction_multiblock    # tests N from 2K to 4M
./reduction_coarsened     # all five kernels, full benchmark
```

---

## Optimisation progression

| Kernel | Figure | Key change | Problem fixed |
|--------|--------|-----------|---------------|
| Simple | 10.6 | stride doubles, `i = 2*tx` | baseline |
| Convergent | 10.9 | stride halves, `i = tx` | control divergence, memory coalescing |
| Shared memory | 10.11 | intermediate sums in `__shared__` | global memory traffic |
| Multiblock | 10.13 | segment per block + `atomicAdd` | single-block limitation |
| Coarsened | 10.15 | `COARSE_FACTOR` elements per thread | parallelisation overhead |

---

## Key CUDA concepts

| Concept | Where |
|---------|-------|
| Reduction tree: log₂N time steps | §10.2 / Fig 10.3/10.5 |
| Control divergence: `if (tx % stride == 0)` vs `if (tx < stride)` | `01` vs `02` |
| Memory coalescing: `input[2*tx]` vs `input[tx]` | `01` vs `02` |
| Shared memory for partial sums (N+1 global accesses) | `03` |
| `atomicAdd` for cross-block accumulation | `04` |
| Coarsening: serial phase + tree phase | `05` |

---

## Debugging tips

- To observe divergence in Fig 10.6: set a breakpoint at line 05 (`input[i] += ...`). Use `info cuda threads` and notice that during iteration 3 (stride=4), only threads 0, 4, 8, … are at the breakpoint; threads 1, 2, 3, … skip it. The warp still advances to the `__syncthreads()` with those threads idle.
- In the shared memory kernel (Fig 10.11), break at the `__syncthreads()` inside the loop. Run `print input_s[0]` before and after the first iteration to watch the partial sum form.
- For the multiblock kernel, break inside the `atomicAdd` line. Run `print input_s[0]` to see each block's contribution before it is added.
