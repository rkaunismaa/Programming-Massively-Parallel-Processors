# Chapter 11 — Prefix Sum (Scan)

**Book sections:** §11.1 Background · §11.2 Kogge-Stone algorithm · §11.3 Work efficiency · §11.4 Brent-Kung algorithm · §11.5 Coarsening · §11.6 Segmented hierarchical scan · §11.7 Single-pass scan

---

## Programs

### `01_scan_kogge_stone.cu` — Kogge-Stone algorithm (§11.2 / Figure 11.3)

Inclusive prefix sum using doubling stride (Kogge-Stone adder design):

```c
for (stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    float temp = XY[tx];
    if (tx >= stride) temp += XY[tx - stride];   // read into temp
    __syncthreads();                               // barrier before write
    XY[tx] = temp;
}
```

Two `__syncthreads()` per iteration are required to avoid the write-after-read race condition (§11.2). Thread tx reads `XY[tx - stride]` which may be written by thread `tx - stride` in the same iteration.

Exclusive variant (Fig 11.4): load `X[i-1]` into `XY[tx]`, identity (0) at position 0 — equivalent to shifting the inclusive result right by one.

**Work complexity (§11.3):** O(N·log₂N) — NOT work efficient. For N=1024: ~10× more work than sequential. Fast (log₂N time steps) when hardware has plenty of parallel resources.

---

### `02_scan_brent_kung.cu` — Brent-Kung algorithm (§11.4 / Figure 11.7)

Two-phase work-efficient scan (Brent-Kung adder design):

**Phase 1 — Reduction tree (upsweep):**  
Stride doubles. Thread index maps to `(tx+1)*2*stride - 1` so a contiguous prefix of threads stays active — less warp divergence than Kogge-Stone.

**Phase 2 — Distribution tree (downsweep):**  
Stride halves. `XY[index + stride] += XY[index]` pushes partial sums toward the leaves.

**Work complexity:** 2N − 2 − log₂N ≈ 2N = **O(N)** — work-efficient. At most 2× the sequential work regardless of N. Requires only N/2 threads (each thread loads 2 elements).

Trade-off vs Kogge-Stone:
- Brent-Kung: 2·log₂N time steps, O(N) work
- Kogge-Stone: log₂N time steps, O(N·log₂N) work
- Kogge-Stone wins when hardware has surplus parallel resources; Brent-Kung wins when resources are limited or energy matters.

---

### `03_scan_coarsened.cu` — Coarsened scan (§11.5 / Figure 11.8)

Three-phase coarsened inclusive scan that further reduces parallelisation overhead:

```
Phase 1: Sequential scan per thread subsection (no barriers — fully parallel)
Phase 2: Brent-Kung on BLOCK_DIM last elements (block-wide collaboration)
Phase 3: Add predecessor sum to remaining elements
```

Input is loaded coalesced into shared memory; each thread then processes its CFACTOR-element subsection independently. The Phase 2 Brent-Kung operates on only BLOCK_DIM elements (much smaller than the full section).

Work analysis for N=1024, T=256, CF=4:
- Phase 1: N−T = 768 additions
- Phase 2: ~2T = 512 additions (BK on T=256 elements)
- Phase 3: N−T = 768 additions
- Total: ~2048 vs Kogge-Stone on N: ~10240

---

### `04_scan_hierarchical.cu` — Hierarchical segmented scan (§11.6 / Figures 11.9–11.10)

Extends single-block scan to arbitrary N using three kernels:

| Kernel | Purpose |
|--------|---------|
| `scan_local_kernel` | Brent-Kung per block; writes `S[blockIdx.x]` = block's total |
| `scan_S_kernel` | Brent-Kung scan on S[] (single block) |
| `add_block_sums_kernel` | Each thread adds `S[blockIdx.x - 1]` to its Y element |

Example (Fig 11.10, 16 elements, 4 blocks of 4):
```
X: [2 1 3 1 | 0 4 1 2 | 0 3 1 2 | 5 3 1 2]
K1 Y: [2 3 6 7 | 0 4 5 7 | 0 3 4 6 | 5 8 9 11]   S=[7,7,6,11]
K2 S: [7 14 20 31]
K3 Y: [2 3 6 7 | 7 11 12 14 | 14 17 18 20 | 25 28 29 31]  ✓
```

Maximum N: `SECTION_SIZE²` (two-level hierarchy). For SECTION_SIZE=2048: ~4M elements.

Limitation: S[] is written to global memory between K1 and K2, then reloaded by K2 — this round-trip adds latency (§11.7 motivation).

---

### `05_scan_single_pass.cu` — Single-pass domino-style scan (§11.7)

Eliminates the global memory round-trip by processing all input in one kernel. Blocks communicate sequentially via `scan_value[]` and `flags[]` arrays:

```c
// Dynamic block index (prevents deadlock)
if (tx == 0) bid_s = atomicAdd(blockCounter, 1);
__syncthreads();
int bid = bid_s;

// ... local Brent-Kung scan ...

if (tx == 0) {
    if (bid == 0) {
        scan_value[0] = local_sum;
        __threadfence();
        atomicAdd(&flags[0], 1);          // signal ready
    } else {
        while (atomicAdd(&flags[bid-1], 0) == 0) { }  // spin-wait
        float prev = scan_value[bid-1];
        scan_value[bid] = prev + local_sum;
        __threadfence();
        atomicAdd(&flags[bid], 1);
    }
}
```

`__threadfence()` ensures `scan_value[bid]` is globally visible before the flag is set. `atomicAdd(&flags[bid], 0)` provides a cached spin-wait with acquire semantics. Dynamic bid guarantees block `bid-1` executes before block `bid`, preventing deadlock.

---

## Building

```bash
cd chapter_11
make SM_ARCH=sm_89
make SM_ARCH=sm_89 SECTION_SIZE=512 CFACTOR=4   # smaller sections
make SM_ARCH=sm_89 DEBUG=1
```

```bash
./scan_kogge_stone     # includes work efficiency analysis
./scan_brent_kung      # includes KS vs BK comparison table
./scan_coarsened       # three-phase breakdown
./scan_hierarchical    # tests N from 2K to 4M
./scan_single_pass     # domino vs hierarchical benchmark
```

---

## Algorithm comparison

| Algorithm | Work | Time steps | Notes |
|-----------|------|-----------|-------|
| Sequential | O(N) | N | baseline |
| Kogge-Stone (§11.2) | O(N·log₂N) | log₂N | fast but work-inefficient |
| Brent-Kung (§11.4) | O(N) | 2·log₂N | work-efficient, half the threads |
| Coarsened (§11.5) | O(N) | ~N/T + log₂T steps | reduces overhead for large N |
| Hierarchical (§11.6) | O(N) | 3 kernel launches | handles arbitrary N |
| Single-pass (§11.7) | O(N) | 1 kernel launch | avoids S[] round-trip |

---

## Debugging tips

- Break inside the Brent-Kung reduction loop and inspect `XY[]`. At stride=1, XY[1] should be X[0]+X[1], XY[3] should be X[2]+X[3], etc. Only threads mapping to odd positions updated the array.
- For the single-pass kernel, break on the `while(atomicAdd(...) == 0)` spin. Use `info cuda threads` — only thread 0 of each block should be spinning. Other threads wait at the barrier after the spin.
- Set a watchpoint on `flags[1]` to see exactly when block 0 signals block 1.
