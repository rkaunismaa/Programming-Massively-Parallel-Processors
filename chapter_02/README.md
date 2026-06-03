# Chapter 2 — Heterogeneous Data Parallel Computing

This chapter introduces the CUDA C programming model through a single running
example: **vector addition** (`C = A + B`).  Starting from a sequential C
baseline, it incrementally builds the complete CUDA program — memory allocation
on the device, host-to-device data transfer, a parallel kernel, and result
retrieval — and explains *why* each step exists.

A second motivating example, **color-to-grayscale conversion**, is used to
introduce the concept of data parallelism before any CUDA code is written.

---

## Programs

### `01_vec_add_sequential.c` — Sequential baseline (Figure 2.4)

The traditional C vector addition loop that the chapter will parallelise.
Variable names are suffixed `_h` (host) to distinguish them from device
variables introduced later.

**Key concept:** the for-loop iterates over every element sequentially.
The CUDA kernel replaces this loop with a grid of threads.

```bash
gcc -O2 -o vec_add_sequential 01_vec_add_sequential.c -lm
./vec_add_sequential
```

---

### `02_vec_add_cuda.cu` — Complete CUDA implementation (Figures 2.5, 2.8, 2.10, 2.12, 2.13)

The full CUDA program, bringing together every concept introduced in the chapter.

| Code region | Book figure | What it does |
|-------------|-------------|-------------|
| `cudaMalloc` / `cudaFree` | Figure 2.6 | Allocate / free device global memory |
| `cudaMemcpy(...HostToDevice)` | Figure 2.7 | Copy inputs from CPU RAM to GPU global memory |
| `vecAddKernel` (`__global__`) | Figure 2.10 | Kernel: each thread computes one element `C[i] = A[i] + B[i]` |
| `<<<ceil(n/256.0), 256>>>` | Figure 2.12 | Launch configuration — ceiling division ensures enough blocks |
| `cudaMemcpy(...DeviceToHost)` | Figure 2.13 | Copy result back to host |

**Key concepts:**
- **Loop parallelism**: the grid of threads *is* the for-loop; each thread handles one iteration.
- **Thread index formula**: `i = blockIdx.x * blockDim.x + threadIdx.x`
- **Guard clause**: `if (i < n)` handles vector lengths that are not multiples of the block size.
- **Naming convention**: `_h` = host pointer, `_d` = device pointer.

```bash
nvcc -O2 -arch=sm_89 -o vec_add_cuda 02_vec_add_cuda.cu -lm
./vec_add_cuda
```

---

### `03_error_checking.cu` — CUDA error handling (sidebar, page 35)

Every CUDA API function returns a `cudaError_t`.  This program demonstrates
the `CUDA_CHECK` macro pattern described in the book's sidebar and applies it
to every API call in the vector addition program.

| API call | Purpose |
|----------|---------|
| `cudaError_t err = cudaMalloc(...)` | Capture return value |
| `cudaSuccess` | The value returned on success |
| `cudaGetErrorString(err)` | Human-readable error description |
| `cudaGetLastError()` | Detect invalid kernel launch configuration |
| `cudaDeviceSynchronize()` | Wait for kernel; surface async device errors |

**Key concept:** kernel launches are asynchronous — the host moves on
immediately.  Errors that occur during kernel execution are only visible
after a synchronisation point.

```bash
nvcc -O2 -arch=sm_89 -o error_checking 03_error_checking.cu
./error_checking
```

---

### `04_data_parallelism_grayscale.cu` — Data parallelism (Section 2.1, Figures 2.1–2.2)

The motivating example from Section 2.1.  Converting a colour image to
grayscale requires computing `L = 0.21r + 0.72g + 0.07b` for every pixel,
and each pixel is completely independent of every other pixel — this is
**data parallelism**.

The kernel uses a 1-D thread organisation (one thread per pixel) consistent
with the Chapter 2 material.  The 2-D thread organisation for images is
introduced in Chapter 3.

**Key concept:** Figure 2.2 shows that `O[0], O[1], … O[N-1]` can all be
computed simultaneously because none depends on another.

```bash
nvcc -O2 -arch=sm_89 -o grayscale 04_data_parallelism_grayscale.cu
./grayscale
```

---

### `05_function_qualifiers.cu` — CUDA C function qualifiers (Figure 2.11)

Demonstrates all three CUDA C qualifier keywords and their rules.

| Qualifier | Callable from | Runs on | Launches new grid? |
|-----------|--------------|---------|-------------------|
| `__host__` (default) | Host | Host | No |
| `__global__` | Host (or device†) | Device | **Yes** |
| `__device__` | Device | Device | No |
| `__host__ __device__` | Both | Both | No |

†Dynamic Parallelism (Chapter 21) allows `__global__` from the device.

**Key concept:** using both `__host__` and `__device__` tells NVCC to compile
two versions of a function — one for the CPU and one for the GPU.  This is
useful for math utility functions shared between host and device code.

```bash
nvcc -O2 -arch=sm_89 -o function_qualifiers 05_function_qualifiers.cu
./function_qualifiers
```

---

## Building all programs

```bash
# Release build
make SM_ARCH=sm_89

# Debug build (adds -g -G for cuda-gdb)
make SM_ARCH=sm_89 DEBUG=1

# Clean
make clean
```

---

## Debugging in VS Code

1. Build with `DEBUG=1` (adds `-g -G` to enable device-side breakpoints).
2. Open `02_vec_add_cuda.cu` and click in the gutter next to line 44 (`if (i < n)`) to set a kernel breakpoint.
3. Select **"Ch02: CUDA vecAdd"** in the Run and Debug panel and press **F5**.
4. When the breakpoint is hit, the Variables panel shows `threadIdx`, `blockIdx`, `i`, and local variables for the focused GPU thread.

See the [root README](../README.md) for full `.vscode/launch.json` setup.

---

## Concepts covered

| Concept | Where |
|---------|-------|
| Data parallelism | Section 2.1 |
| CUDA C program structure (host + device) | Section 2.2 |
| `__global__` kernel, `threadIdx`, `blockIdx`, `blockDim` | Section 2.5 |
| `cudaMalloc`, `cudaFree`, `cudaMemcpy` | Section 2.4 |
| Execution configuration `<<<gridDim, blockDim>>>` | Section 2.6 |
| NVCC compilation pipeline (host C++ + device PTX) | Section 2.7 |
| Error checking with `cudaError_t` | Sidebar, page 35 |
