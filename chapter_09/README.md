# Chapter 9 — Parallel Histogram

**Book sections:** §9.1 Background · §9.2 Atomic operations · §9.3 Latency and throughput · §9.4 Privatization · §9.5 Coarsening · §9.6 Aggregation

---

## Programs

### `01_histogram_atomic_basic.cu` — Atomic operations (§9.2 / Figure 9.6)

Demonstrates the three stages of the histogram story:

1. **Sequential CPU** (Fig 9.2) — correct reference: `histo[pos/4]++`
2. **Naïve GPU without atomics** — produces wrong results due to race conditions (Figs 9.4–9.5): concurrent read-modify-write sequences overlap, causing lost updates.
3. **Atomic GPU** (Fig 9.6) — correct: `atomicAdd(&histo[pos/4], 1)` serialises concurrent updates to the same location.

Key insight (§9.3 / Fig 9.7): atomic operations at the **same** memory location are serialised. Throughput ≈ 1 operation per (2 × memory latency). For a 200-cycle DRAM system: max ~2.5 M atomics/second per location. Spreading updates across 7 bins improves this ~7×.

---

### `02_histogram_privatized_global.cu` — Privatization in global memory (§9.4 / Figure 9.9)

Each block gets its own private copy of the histogram, offset by `blockIdx.x * NUM_BINS`. Within-block atomic operations contend only within the block (not across all blocks).

After the data pass, each non-zero block merges its copy into block 0's copy:
```c
atomicAdd(&histo[blockIdx.x*NUM_BINS + bin], 1)   // data pass
atomicAdd(&histo[bin], myBlockValue)               // merge to block 0
```

Host must allocate `gridDim.x × NUM_BINS` elements. The final histogram is in the first `NUM_BINS` positions.

---

### `03_histogram_privatized_shared.cu` — Privatization in shared memory (§9.4 / Figure 9.10)

Improvement over global-memory privatization: the private copy lives in `__shared__` memory (a few cycles of latency vs hundreds for DRAM → much higher atomic throughput).

Three-phase structure:
```c
// Phase 1: initialise private bins
for (bin = tx; bin < NUM_BINS; bin += blockDim.x) histo_s[bin] = 0;
__syncthreads();
// Phase 2: atomic update into shared memory
atomicAdd(&histo_s[pos/4], 1);
__syncthreads();
// Phase 3: commit to global memory (at most NUM_BINS atomics per block)
if (histo_s[bin] > 0) atomicAdd(&histo[bin], histo_s[bin]);
```

For NUM_BINS=7, the shared allocation is only 28 bytes per block — essentially free. The commit phase performs at most 7 global atomics per block regardless of how many input elements the block processes.

---

### `04_histogram_coarsened.cu` — Thread coarsening (§9.5 / Figures 9.12 and 9.14)

Reduces the number of thread blocks (and thus commit-phase overhead) by having each thread process CFACTOR input elements.

Two partitioning strategies:

| Strategy | Access pattern | Coalesced? |
|----------|---------------|------------|
| **Contiguous** (Fig 9.12) | `data[tid*CF .. (tid+1)*CF-1]` | No — adjacent threads access distant locations |
| **Interleaved** (Fig 9.14) | `data[tid], data[tid+stride], …` | **Yes** — adjacent threads access adjacent locations |

GPU prefers **interleaved** because the stride-1 access pattern enables memory coalescing.

---

### `05_histogram_aggregated.cu` — Aggregation (§9.6 / Figure 9.15)

For datasets with long runs of the same value (e.g., sky images with many identical pixels), aggregation batches consecutive identical-bin updates into a single `atomicAdd`:

```c
if (bin == prevBinIdx) {
    ++accumulator;          // extend the streak — no atomic
} else {
    atomicAdd(&histo_s[prevBinIdx], accumulator);  // flush previous streak
    accumulator = 1;
    prevBinIdx  = bin;
}
// After loop: flush remaining streak
if (accumulator > 0) atomicAdd(&histo_s[prevBinIdx], accumulator);
```

Benchmark uses two datasets to show when aggregation matters:
- **Uniform random** — no benefit (bin changes every element)
- **Biased 90% 'm'** — large benefit (long streaks of bin 3)

---

## Building

```bash
cd chapter_09
make SM_ARCH=sm_89
make SM_ARCH=sm_89 CFACTOR=8   # larger coarsening factor
make SM_ARCH=sm_89 DEBUG=1
```

```bash
./histogram_basic          # shows race condition vs atomicAdd
./histogram_priv_global
./histogram_priv_shared
./histogram_coarsened      # compares contiguous vs interleaved
./histogram_aggregated     # full benchmark on uniform and biased data
```

---

## Key concepts

| Concept | File |
|---------|------|
| Race condition on `histo[bin]++` | `01` — naive kernel gives wrong result |
| `atomicAdd(&addr, val)` — Fig 9.6 | `01` |
| Serialisation bottleneck (§9.3) | `01` — contention analysis |
| Per-block private copies in global memory | `02` — Fig 9.9 |
| Per-block private copies in shared memory | `03` — Fig 9.10 |
| Commit phase: ≤ NUM_BINS atomics per block | `03` |
| Contiguous vs interleaved partitioning | `04` — Figs 9.12/9.14 |
| Interleaved preferred for coalescing on GPU | `04` |
| `accumulator` / `prevBinIdx` aggregation | `05` — Fig 9.15 |
| Aggregation benefit on biased input | `05` — biased dataset benchmark |

---

## Debugging tips

- To observe the race condition: add a breakpoint in `histo_naive_kernel` on the `histo[bin]++` line. Use `info cuda threads` to list all active threads. Multiple threads in the same warp or different warps may be at the same breakpoint simultaneously with the same `bin` value.
- In `histo_private_shared_kernel`, break at the Phase 3 commit. Run `print histo_s[3]` — you should see the within-block count for bin "m-p". Compare with the global `histo[3]` before and after the atomic add.
- For aggregation debugging: break inside the `else` branch of `histo_aggregated_kernel` on the `atomicAdd` line and inspect `prevBinIdx` and `accumulator` to see streak lengths.
