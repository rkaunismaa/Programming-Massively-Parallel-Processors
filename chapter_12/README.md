# Chapter 12 — Merge

**Book sections:** §12.1 Background · §12.2 Sequential merge · §12.3 Parallelization approach · §12.4 Co-rank function · §12.5 Basic parallel merge kernel · §12.6 Tiled merge kernel · §12.7 Circular buffer merge kernel · §12.8 Thread coarsening for merge

---

## Programs

### `01_merge_sequential.cu` — Sequential merge (§12.2 / Figure 12.2)

Baseline ordered merge of two sorted arrays A (m elements) and B (n elements) into sorted C (m+n elements). Stability: when A[i] == B[j], A goes first — preserves prior orderings.

```c
while (i < m && j < n) {
    if (A[i] <= B[j]) C[k++] = A[i++];   // A takes precedence on tie (stability)
    else               C[k++] = B[j++];
}
```

Complexity: O(m+n). Used by §12.3–12.7 as the per-thread sequential merge building block.

---

### `02_merge_basic.cu` — Co-rank function + basic parallel merge kernel (§12.4–12.5 / Figures 12.5, 12.9)

**Co-rank function (Fig 12.5):** Binary search to find i such that C[0..k-1] = merge(A[0..i-1], B[0..j-1]) where j = k - i. O(log(max(m,n))).

```
Invariant: i + j = k throughout.
If A[i-1] > B[j]: i too high → decrease i, increase j.
If B[j-1] >= A[i]: j too high → decrease j, increase i.
Otherwise: found.
```

**Basic kernel (Fig 12.9):** Each thread owns `ceil((m+n)/(gridDim*blockDim))` output elements. Two `co_rank` calls to find input subarrays, then one `merge_sequential`.

```c
int k_curr = tid * elementsPerThread;
int i_curr = co_rank(k_curr, A, m, B, n);
int i_next = co_rank(k_next, A, m, B, n);
merge_sequential(&A[i_curr], i_next-i_curr, &B[j_curr], j_next-j_curr, &C[k_curr]);
```

Limitation: co_rank accesses A and B directly from global memory with irregular (binary search) patterns → not coalesced (§12.6 motivation).

---

### `03_merge_tiled.cu` — Tiled merge kernel (§12.6 / Figures 12.11–12.13)

Uses shared memory to improve coalescing:

| Phase | What happens |
|-------|-------------|
| **Part 1 (Fig 12.11)** | One thread computes block-level co-ranks; block's A and B subarrays identified |
| **Part 2 (Fig 12.12)** | All threads cooperatively load `tile_size` A elements + `tile_size` B elements into A_S / B_S (coalesced: thread i loads element i) |
| **Part 3 (Fig 12.13)** | Each thread runs `co_rank` on shared memory and calls `merge_sequential` |

Global memory accesses in the tile loading loops are coalesced. The co_rank binary search operates on shared memory — no uncoalesced global accesses.

**Deficiency:** Only ~half the loaded 2×tile_size data is actually used per iteration. The unused remainder (elements that weren't consumed) is overwritten in the next iteration → 50% bandwidth waste. §12.7 fixes this.

---

### `04_merge_circular_buffer.cu` — Circular buffer merge kernel (§12.7 / Figures 12.16, 12.18–12.20)

Eliminates the 50% bandwidth waste by reusing unconsumed tile elements:

- `A_S_start`, `B_S_start`: pointers into circular buffers A_S and B_S
- Each iteration: only refill the `A_S_consumed` and `B_S_consumed` slots, writing them at `(A_S_start + remaining) % tile_size`
- After iteration: advance starts with `(A_S_start + A_S_consumed) % tile_size`

**Simplified model (Fig 12.17):** `co_rank_circular` and `merge_sequential_circular` accept virtual 0-based offsets; internally apply `(start + offset) % tile_size` to form actual indices. The binary search logic is unchanged.

```c
// co_rank_circular: only difference from co_rank
int i_cir     = (A_S_start + i)   % tile_size;
int i_m_1_cir = (A_S_start + i-1) % tile_size;
// use A_S[i_m_1_cir] and B_S[j_cir] in the comparison
```

**Thread coarsening (§12.8):** Each thread handles `tile_size / blockDim.x` output elements. This amortizes the O(log N) binary search cost across multiple outputs. Without coarsening, every single output element would require a separate binary search — prohibitively expensive.

---

## Building

```bash
cd chapter_12
make SM_ARCH=sm_89
make SM_ARCH=sm_89 TILE_SIZE=1024
make SM_ARCH=sm_89 DEBUG=1
```

```bash
./merge_sequential       # Fig 12.1/12.2 example + large random test
./merge_basic            # co-rank unit tests + Fig 12.9 kernel
./merge_tiled            # tiled kernel with coalesced loads
./merge_circular_buffer  # circular buffer: full tile utilization
```

---

## Algorithm progression

| Kernel | Global mem coalescing | Tile utilization | Notes |
|--------|----------------------|-----------------|-------|
| Basic (§12.5) | Poor (irregular binary search + merge) | N/A | Simple, uncoalesced |
| Tiled (§12.6) | Good (cooperative tile loading) | ~50% | Reloads consumed elements |
| Circular buffer (§12.7) | Good | ~100% | Higher register usage |

## Key concept: dynamic input identification

Unlike all previous parallel patterns (reduction, scan, histogram, stencil, convolution), the merge pattern cannot determine its input data range from the thread/block index alone — it depends on the actual data values. The co-rank binary search is the mechanism that resolves this data-dependent input identification in O(log N) time.
