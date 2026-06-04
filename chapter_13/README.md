# Chapter 13 — Sorting

Code samples for **Chapter 13: Sorting** from *Programming Massively Parallel Processors* (4th ed.).

| File | Section | Algorithm |
|------|---------|-----------|
| `01_radix_sort_sequential.c` | §13.2 | Sequential LSD radix sort (baseline) |
| `02_radix_sort_parallel.cu` | §13.3 | Parallel radix sort — one thread per key (Fig 13.4) |
| `03_radix_sort_coalesced.cu` | §13.4 | Coalesced radix sort — shared-memory local buckets (Figs 13.5–13.6) |
| `04_parallel_merge_sort.cu` | §13.7 | Parallel merge sort — bitonic segment sort + co-rank iterative merge (Fig 13.11) |

---

## Key concepts

### §13.2 Radix sort
LSD (least-significant-digit) radix sort processes one bit at a time, performing a **stable** two-way partition per iteration.  Stability is essential: keys sorted by earlier iterations remain in order within each bucket of later iterations, so after 32 passes all 32-bit keys are fully sorted.

### §13.3 Parallel radix sort (Fig 13.4)
Each GPU thread handles one key.  Per bit-iteration:
1. Each thread extracts its key's current bit.
2. An **exclusive prefix sum (scan)** on the bits array gives every thread the count of 1-bits before its position (`ones_before`).
3. Each thread computes its output index and scatters its key:
   - `dest(0-bit) = index − ones_before`
   - `dest(1-bit) = N − ones_total + ones_before`

The scan is implemented as a two-level Kogge-Stone (same pattern as Chapter 11).

**Limitation**: the scatter writes are not memory-coalesced because consecutive threads may write to non-adjacent addresses.

### §13.4 Coalesced radix sort (Figs 13.5–13.6)
Each block performs a **local 1-bit radix sort in shared memory**, producing a locally sorted chunk with its 0-bucket first and 1-bucket second.  Per-block bucket sizes are stored to a global table; an exclusive scan on that table yields the global start position of each local bucket.  The scatter then writes each local bucket sequentially to global memory, so **consecutive threads write to consecutive addresses → coalesced stores**.

### §13.7 Parallel merge sort (Fig 13.11)
Two phases:
1. **Bitonic sort** — each block sorts one `SEG_SIZE`-element segment in shared memory.
2. **Iterative merge** — each round doubles the sorted segment size by merging adjacent pairs.  Multiple merge operations run in parallel across blocks, and within each merge the tiled co-rank kernel from Chapter 12 parallelises across thread blocks.

---

## Building

```bash
make SM_ARCH=sm_89          # build all four targets
make SM_ARCH=sm_89 DEBUG=1  # add -g -G for cuda-gdb
```

| GPU family | `SM_ARCH` |
|------------|-----------|
| RTX 40xx   | `sm_89`   |
| RTX 30xx   | `sm_86`   |
| A100       | `sm_80`   |
| V100       | `sm_70`   |

Optional build variables:

| Variable | Default | Effect |
|----------|---------|--------|
| `SEG_SIZE` | `256` | Initial sorted segment size for merge sort (must be power of 2, ≤ 1024) |
| `TILE_SIZE` | `256` | Tile size used in the co-rank merge kernel |
