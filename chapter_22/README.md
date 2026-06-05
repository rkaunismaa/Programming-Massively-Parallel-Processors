# Chapter 22 — Advanced Practices and Future Evolution

Code samples for **Chapter 22: Advanced practices and future evolution** from *Programming Massively Parallel Processors* (4th ed.).

Chapter 22 surveys advanced CUDA features that support high-performance, production-quality applications. The only code figure in the chapter is Fig. 22.1 (unified memory porting example). The four programs here cover every runnable concept from §22.1–22.4.

| File | Section | Key idea |
|------|---------|----------|
| `01_unified_memory.cu` | §22.1, Fig 22.1 | `cudaMallocManaged`: single pointer shared by CPU and GPU; eliminates explicit `cudaMemcpy` |
| `02_zero_copy.cu` | §22.1 | `cudaHostAlloc(cudaHostAllocMapped)`: kernel reads host memory over PCIe; suitable for sparse table access |
| `03_concurrent_streams.cu` | §22.2 | Serial vs concurrent kernel execution; H→D + compute + D→H pipeline across segments |
| `04_cooperative_kernel.cu` | §22.2 | `cudaLaunchCooperativeKernel` + `grid.sync()`: device-wide barrier enables single-launch multi-pass algorithm |

---

## §22.1 — Model of host/device interaction

Three generations of host/device memory interaction:

### Traditional model (Chapter 2)
Separate host and device allocations require explicit `cudaMemcpy` transfers in both directions. I/O devices can only DMA to host memory, so data read from disk passes through a host buffer and two PCIe transfers before reaching the GPU.

### Zero-copy memory (CUDA 2.2)
`cudaHostAlloc(cudaHostAllocMapped)` pins host memory and maps it into the GPU virtual address space. A kernel can dereference the pointer returned by `cudaHostGetDevicePointer()` directly, accessing host memory over PCIe. Bandwidth is < 10% of device DRAM; suitable for data accessed **occasionally or sparsely**.

### Unified Virtual Addressing — UVA (CUDA 4)
A single virtual address space shared by host and device: every pointer is unambiguously host or device based on its VA. Removes the need to specify the direction in `cudaMemcpy` and enables direct GPU peer access to other GPUs on the same PCIe fabric.

### Unified memory — `cudaMallocManaged` (CUDA 6)
A single pointer backed by a managed pool that migrates pages between CPU and GPU on demand. Fig 22.1 shows that porting a CPU function to CUDA requires only three API changes:

```
malloc(N)          →  cudaMallocManaged(&ptr, N)
free(ptr)          →  cudaFree(ptr)
cpu_fn(ptr)        →  kernel<<<...>>>(ptr) + cudaDeviceSynchronize()
```

Pascal and later GPUs add hardware page-fault support, removing the requirement to flush all managed data to the device before each kernel launch.

---

## §22.2 — Kernel execution control

### Function calls within kernels
Early CUDA required the compiler to inline all device function bodies. Kepler (CUDA 5+) added hardware call-frame stacks, enabling true function calls, recursion, and library-linkable device code.

### Simultaneous grid execution
Fermi and later GPUs can execute multiple grids from the same application concurrently. CUDA streams are the programmer-facing mechanism: kernels in the same stream serialise; kernels in different streams may overlap.

```
Default stream:       K0 → K1 → K2   (serial)
Separate streams:     K0 ─┐
                      K1 ─┤ (concurrent, subject to resources)
                      K2 ─┘
```

**Pipelined H→D + compute + D→H**: issuing all three operations for each segment into its own stream overlaps the stages across segments — the same technique used for MPI+GPU communication hiding in Chapter 20.

### Cooperative kernels (CUDA 11)
`cudaLaunchCooperativeKernel()` guarantees that **all** thread blocks are resident simultaneously. This enables `cg::this_grid().sync()` — a device-wide barrier that is safe from deadlock. Without cooperative kernels, multi-pass algorithms require separate launches with a host-side `cudaDeviceSynchronize()` between passes.

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void two_pass(...) {
    cg::grid_group grid = cg::this_grid();

    // Pass 1: all blocks contribute
    for (int i = ...; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(&hist[data[i] % bins], 1);

    grid.sync();   // device-wide barrier

    // Pass 2: all blocks read the completed histogram
    for (int b = ...; b < bins; b += gridDim.x * blockDim.x)
        pdf[b] = (float)hist[b] / n;
}
```

The grid must not exceed the number of blocks that can all be resident simultaneously. Query this limit with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`; use a grid-stride loop to cover inputs larger than the residency limit.

---

## §22.3 — Memory bandwidth and compute throughput (survey)

| Feature | Introduced | Benefit |
|---------|-----------|---------|
| Double-precision speed | Fermi | ~½ of single-precision throughput (vs 8× slower on early GPUs) |
| Half-precision (FP16) | Pascal | Tensor cores: 156 TFLOPS on A100 vs 19.5 TFLOPS FP32 |
| Configurable L1/shared | Fermi | Configurable split between L1 cache and shared memory scratchpad |
| Enhanced atomics | Kepler/Maxwell | Faster atomics over shared memory; reduces need for scan/sort pre-processing |
| HBM2 / NVLink | Pascal | 3× memory bandwidth and 5× GPU-GPU bandwidth vs Maxwell |

---

## §22.4 — Programming environment (survey)

| Feature | Notes |
|---------|-------|
| Unified device address space | Single load/store ISA covers global, local, and shared memory (Fermi+) |
| OpenACC / Thrust / CUDA Fortran | Higher-level programming models that generate CUDA code |
| Critical path analysis | CUDA 8 Visual Profiler grays out off-critical-path activities (Fig 22.3) |

---

## Building

```bash
make SM_ARCH=sm_89          # all targets
make SM_ARCH=sm_89 cooperative_kernel   # individual target

make SM_ARCH=sm_89 DEBUG=1 unified_memory  # debug build
```

> **`cooperative_kernel`** is compiled with `-rdc=true` (required for `cg::this_grid()`).  
> Cooperative kernel launch requires compute capability ≥ 6.0 (Pascal or newer).  
> Zero-copy requires `cudaDeviceMapHost` device flag and `prop.canMapHostMemory == 1`.
