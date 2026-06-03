# Chapter 4 — Compute Architecture and Scheduling

This chapter explains *why* CUDA programs behave the way they do by looking
inside the GPU hardware.  Unlike Chapters 2–3, which focused on writing
correct parallel code, Chapter 4 focuses on **writing fast** parallel code by
understanding the execution model.

The central theme: a GPU SM can hold many more threads than it can run
simultaneously.  This deliberate oversubscription lets the hardware hide
long-latency operations (global memory reads, arithmetic pipelines) by
switching to a ready warp when a stalled warp is waiting.

---

## Hardware concepts covered

| Concept | Section | One-line summary |
|---------|---------|-----------------|
| SM architecture | 4.1 | GPU = array of SMs; each SM has cores, shared memory, register file |
| Block scheduling | 4.2 | Blocks are assigned to SMs in any order — transparent scalability |
| `__syncthreads()` | 4.3 | Barrier: all threads in a block must arrive before any may pass |
| Transparent scalability | 4.3 | No cross-block sync → same code runs on 2 or 2000 SMs |
| Warps and SIMD | 4.4 | 32 threads share one instruction fetch/dispatch unit |
| 2-D thread linearisation | 4.4 | Row-major linear order before warp partitioning (Figure 4.7) |
| Control divergence | 4.5 | If threads in a warp split, the hardware makes multiple passes |
| Latency tolerance | 4.6 | SM switches to a ready warp when current warp stalls — zero overhead |
| Occupancy | 4.7 | Warps-assigned / max-warps; limited by threads, blocks, registers, smem |
| Performance cliff | 4.7 | Adding 1 register per thread can halve occupancy |
| `cudaGetDeviceProperties` | 4.8 | Query SM count, warp size, register file size, etc. at runtime |

---

## Programs

### `01_barrier_synchronization.cu` — `__syncthreads()` (Section 4.3, Figures 4.3–4.4)

Demonstrates correct and incorrect barrier usage.

**Part A — Block-wise array reversal:**  
Two phases — all threads load into shared memory, then a `__syncthreads()`
ensures every load completes before any thread reads a neighbour's slot,
then all threads write in reversed order.  Removing the barrier produces
incorrect results because some threads read slots before other threads have
filled them.

**Part B — Parallel prefix sum:**  
A multi-pass scan within a block requires a barrier at both the *read* and
the *write* boundary of each pass — demonstrates that a single kernel can
need many `__syncthreads()` calls in sequence.

**Part C — The INCORRECT pattern from Figure 4.4** (inside `#if 0`):  
Even threads go to `__syncthreads()_A`; odd threads go to `__syncthreads()_B`.
Neither set ever reaches the other's barrier → deadlock.

```bash
nvcc -O2 -arch=sm_89 -o barrier_sync 01_barrier_synchronization.cu
./barrier_sync
```

---

### `02_warp_partitioning.cu` — Warps and SIMD (Section 4.4, Figures 4.6–4.7)

Shows how blocks are divided into warps and how 2-D blocks linearise before
that partitioning occurs.

**1-D partitioning:** threads 0–31 → warp 0, 32–63 → warp 1, etc.  
Blocks whose size is not a multiple of 32 have padding inactive threads in
their last warp (e.g. a 48-thread block → warp 0 full, warp 1 has 16 active
+ 16 inactive).

**2-D linearisation (Figure 4.7):**  
For a 4×16 block, the linear order is:
```
row y=0: T(0,0)–T(0,15) → linear 0–15  ┐
row y=1: T(1,0)–T(1,15) → linear 16–31 ┘ warp 0
row y=2: T(2,0)–T(2,15) → linear 32–47 ┐
row y=3: T(3,0)–T(3,15) → linear 48–63 ┘ warp 1
```

Includes a GPU kernel that records `warpId = linear / 32` for every thread
and verifies it matches the expected partition.

```bash
nvcc -O2 -arch=sm_89 -o warp_partitioning 02_warp_partitioning.cu
./warp_partitioning
```

---

### `03_control_divergence.cu` — Control Divergence (Section 4.5, Figures 4.9–4.10)

Measures the performance cost of warp divergence through timing experiments.

**Part A — If-divergence (Figure 4.9):**  
```c
if (threadIdx.x < 24) { /* path A */ } else { /* path B */ }
```
Every warp splits: threads 0–23 take path A, threads 24–31 take path B.
The hardware executes two passes per warp.  Expected overhead: ~2×.

**Part B — Loop-divergence (Figure 4.10):**  
Trip count = `a[threadIdx.x]`, varying 4–8 within each warp.  The warp
continues iterating until the *last* thread finishes, with shorter threads
masked out during their extra iterations.

**Part C — Boundary divergence analysis:**  
For `if (i < n)` in vecAdd: only the very last warp diverges.  The table
shows how the fraction of divergent warps shrinks as n grows — less than
1% for n ≥ 10,000.

```bash
nvcc -O2 -arch=sm_89 -o control_divergence 03_control_divergence.cu
./control_divergence
```

---

### `04_query_device_properties.cu` — Device Properties (Section 4.8)

Comprehensive device capability dump using `cudaGetDeviceProperties`.

**Fields printed:**

| Field | Purpose |
|-------|---------|
| `multiProcessorCount` | Number of SMs |
| `warpSize` | Always 32 on current hardware |
| `maxThreadsPerBlock` | Upper bound for block size |
| `maxThreadsPerMultiProcessor` | Threads an SM can hold simultaneously |
| `maxBlocksPerMultiProcessor` | Blocks an SM can hold simultaneously |
| `regsPerMultiprocessor` | Register file size; limits register-heavy kernels |
| `sharedMemPerBlock` | Max shared mem per block |
| `clockRate` | SM clock in kHz |
| `major`, `minor` | Compute capability (e.g. 8, 9 = sm_89) |

Also prints the **block-size occupancy table** and **register-cliff example**
from Section 4.7 using the queried values for your specific GPU.

```bash
nvcc -O2 -arch=sm_89 -o query_device 04_query_device_properties.cu
./query_device
```

Sample output (RTX 4090):
```
Device: NVIDIA GeForce RTX 4090  (sm_89)
  Streaming Multiprocessors: 128
  Max threads/SM:            1536
  Registers/SM:              65536
  ...
```

---

### `05_occupancy.cu` — Occupancy (Section 4.7)

Demonstrates the CUDA occupancy API and the performance cliff.

**Part A — Manual calculation:**  
Shows how block size determines blocks/SM and thread-slot occupancy using
the hardware limits queried from `cudaDeviceProp`.

**Part B — `cudaOccupancyMaxActiveBlocksPerMultiprocessor`:**  
The proper way to compute occupancy — accounts for register file, shared
memory, block slots, and thread slots simultaneously.

```c
int blocksPerSM;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocksPerSM, myKernel, blockSize, dynamicSharedMem);
float occupancy = (blocksPerSM * blockSize) / (float)maxThreadsPerSM;
```

**Part C — Performance cliff:**  
The Volta/Ampere example from Section 4.7: at 512 threads/block with
32 registers/thread, 4 blocks fit → 100% occupancy.  Adding one register
(33/thread) forces the SM down to 3 blocks → 1536/2048 threads → 75%.

**Part D — `cudaOccupancyMaxPotentialBlockSize`:**  
Finds the block size that maximises occupancy for a given kernel.

```bash
nvcc -O2 -arch=sm_89 -o occupancy 05_occupancy.cu
./occupancy
```

---

## Building all programs

```bash
make SM_ARCH=sm_89           # release build
make SM_ARCH=sm_89 DEBUG=1   # debug build (-g -G)
make clean
```

---

## Debugging in VS Code

Chapter 4 programs are good for observing the *scheduling* behaviour of the GPU.

| File | Good breakpoint | What you observe |
|------|----------------|-----------------|
| `01_barrier_synchronization.cu` | Inside `reverseKernel`, after `s[t] = in[i]` | Shared memory `s[]` filling up; some slots still 0 before barrier |
| `01_barrier_synchronization.cu` | After `__syncthreads()` in reverseKernel | All slots in `s[]` are now filled — barrier took effect |
| `02_warp_partitioning.cu` | Inside `recordWarpIds2D` | `threadIdx.x`, `threadIdx.y`, `linear` warp ID for the focused thread |
| `03_control_divergence.cu` | Inside `ifDivergentKernel`, the `if` body | Only threads 0–23 reach this; switch focus to thread 24 to see the else path |

Add `"breakOnLaunch": true` in `launch.json` to stop at the very first kernel
launch — useful for inspecting warp 0, thread 0 before any work begins.

See the [root README](../README.md) for full VS Code debugger setup.

---

## Key formulas

```
warp ID within block = linear_thread_id / 32
linear_thread_id (2D) = threadIdx.y * blockDim.x + threadIdx.x
warps per SM = maxThreadsPerMultiProcessor / warpSize
occupancy (%) = (active_threads / maxThreadsPerMultiProcessor) × 100
max threads by registers = regsPerMultiprocessor / regsPerThread
                         (rounded down to multiple of warpSize)
```
