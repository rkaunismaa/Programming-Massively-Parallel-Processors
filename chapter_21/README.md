# Chapter 21 — CUDA Dynamic Parallelism

Code samples for **Chapter 21: CUDA dynamic parallelism** from *Programming Massively Parallel Processors* (4th ed.).

CUDA Dynamic Parallelism (CDP) lets kernels launch other kernels from the device, without returning control to the host. This enables algorithms whose work is discovered at runtime — recursion, adaptive refinement, irregular parallelism — to be expressed naturally and without the CPU-GPU round-trips that would otherwise be required (§21.1–21.2).

| File | Section | Key idea |
|------|---------|----------|
| `01_cdp_overview.cu` | §21.2, Figs 21.4/21.5 | Parent/child kernel split; variable inner-loop becomes child grid |
| `02_bezier_no_cdp.cu` | §21.3, Fig 21.6 | Bezier tessellation: one block per curve, fixed vertex array |
| `03_bezier_cdp.cu` | §21.3, Figs 21.7/21.12 | Bezier with CDP: parent allocates memory, launches sized child; per-thread streams |
| `04_quadtree_cdp.cu` | §21.4, Figs 21.10/21.11, A21.1 | Recursive quadtree: each block subdivides its quadrant and launches 4 children |

---

## §21.2 — Dynamic parallelism overview (Fig 21.4 / 21.5)

**Without CDP** (Fig 21.4): each parent thread owns a range `[start[i], end[i])` and serialises the inner loop. Threads with long ranges stall the warp.

**With CDP** (Fig 21.5): the parent thread replaces the loop with a child launch sized to the actual range. Control divergence in the parent is eliminated; inner work is parallelised.

```
kernel_parent <<< grid, block >>> (...)
  └─ kernel_child <<< ceil(nwork/256), 256 >>> (...)   per thread
```

---

## §21.3 — Bezier curve tessellation

### Without CDP (Fig 21.6)

One block per curve. All threads in the block collaboratively compute the tessellation points using a strided loop. Blocks with high-curvature curves (many vertices) do more iterations; blocks with low-curvature curves idle early — **workload imbalance across SMs**.

### With CDP (Figs 21.7 / 21.12)

Two improvements:

1. **Parent kernel** (one thread per curve): determines the vertex count, calls `cudaMalloc` to allocate *exactly* the needed device memory, then launches a child grid sized to that count.

2. **Child kernel**: tessellates the points for a single curve. Grid size matches work exactly — no idle threads.

**Stream optimisation (Fig 21.12)**: by default all children launched by threads in the same parent block share the block's NULL stream and are serialised. Creating a per-thread non-blocking stream places each child in its own stream, enabling concurrent execution.

**Memory management**: device-side `cudaMalloc` allocations must be freed by a device kernel (`freeVertexMem`, Fig 21.7 lines 42-47).

**Pending launch pool (§21.5)**: the runtime fixes the pool at 2048 launches by default. For `N_LINES > 2048`, increase it:
```c
cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, N_LINES);
```

---

## §21.4 — Recursive quadtree (Figs 21.8–21.11, A21.1)

A quadtree partitions a 2-D plane by recursively dividing each node into four equal quadrants (Fig 21.8). Each block owns one node (quadrant). The block:

1. Checks termination: ≤ `MIN_POINTS_PER_NODE` points **or** depth ≥ `MAX_DEPTH`.
2. Computes the bounding-box centre.
3. **Counts** points in each child quadrant (shared memory + atomics, Fig 21.11).
4. **Scans** the four counts into placement offsets.
5. **Reorders** points into the output buffer (ping-pong, Fig 21.9).
6. Thread 0 atomically allocates 4 child nodes and launches `build_quadtree_kernel<<<4, BLOCK_DIM>>>` recursively (Fig 21.10).

Support types (`Points`, `Bounding_box`, `Quadtree_node`, `Parameters`) are defined per Appendix A21.1.

> **Implementation note**: the book's Fig 21.10 uses a level-based node-offset scheme (`&nodes[params.num_nodes_at_level]`) that assumes a *full* tree. This implementation uses a global `__device__ int g_node_count` atomic counter so that partial trees (where only some nodes subdivide) are allocated correctly.

---

## §21.5 — Important considerations

| Topic | Key point |
|-------|-----------|
| Memory visibility | Writes by the parent before a child launch are visible to the child; child writes are visible to the parent only after synchronisation |
| Pending launch pool | Default 2048 slots; virtualised pool beyond that incurs slowdown — raise with `cudaDeviceSetLimit` |
| Streams | Default NULL stream serialises children in the same block; use per-thread non-blocking streams for concurrency |
| Nesting depth | Hardware limit is 24 levels; synchronisation depth imposes additional constraints |

---

## Building

```bash
# All targets (requires compute capability ≥ 3.5 for CDP)
make SM_ARCH=sm_89

# Individual targets
make SM_ARCH=sm_89 bezier_cdp
make SM_ARCH=sm_89 quadtree_cdp

# Debug build
make SM_ARCH=sm_89 DEBUG=1 quadtree_cdp
```

> **Requires**: CUDA 5.0+, GPU with compute capability ≥ 3.5 (Kepler or newer).  
> CDP kernels are compiled with `-rdc=true` (relocatable device code).
