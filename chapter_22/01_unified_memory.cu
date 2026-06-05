// §22.1  Unified memory — model of host/device interaction
//        Fig 22.1
//
// The simple host/device model (separate cudaMalloc + cudaMemcpy) has two
// practical limitations (§22.1):
//   1. I/O devices see the host memory; copying data from disk to device
//      memory requires an extra host buffer and two PCIe transfers.
//   2. Large host data structures must be manually partitioned to fit in
//      the smaller device memory.
//
// CUDA 6 introduced unified memory (cudaMallocManaged) to address both.
// A single pointer is shared by CPU and GPU; the CUDA runtime and hardware
// migrate pages on demand.  Fig 22.1 shows the two key API changes needed
// to port a CPU sort() function to CUDA:
//
//   CPU                          CUDA 6 with Unified Memory
//   malloc(N)           →        cudaMallocManaged(&ptr, N)
//   free(ptr)           →        cudaFree(ptr)
//   cpu_function(ptr)   →        kernel<<<...>>>(ptr) + cudaDeviceSynchronize()
//
// This file demonstrates the same concept with a vector-scale kernel:
// both the traditional and the unified-memory variants produce identical
// results while the unified variant eliminates the explicit H→D and D→H
// cudaMemcpy calls.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N      (1 << 22)   // 4 M floats
#define BLOCK  256
#define FACTOR 2.5f

__global__ void scale(float *data, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= factor;
}

// ── Traditional: separate host / device allocations ──────────────────────────
static void run_traditional(const float *h_src, float *h_result) {
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // Explicit H→D transfer
    cudaMemcpy(d_data, h_src, N * sizeof(float), cudaMemcpyHostToDevice);

    scale<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_data, FACTOR, N);
    cudaDeviceSynchronize();

    // Explicit D→H transfer
    cudaMemcpy(h_result, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

// ── Unified memory (Fig 22.1): single pointer shared by CPU and GPU ──────────
static void run_unified(const float *h_src, float *h_result) {
    float *data;
    cudaMallocManaged(&data, N * sizeof(float));  // Fig 22.1: replace malloc

    // CPU writes directly through the managed pointer — no H→D copy needed
    for (int i = 0; i < N; i++) data[i] = h_src[i];

    scale<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(data, FACTOR, N);
    cudaDeviceSynchronize();   // Fig 22.1: replace cpu_function() call

    // CPU reads directly through the same managed pointer — no D→H copy needed
    for (int i = 0; i < N; i++) h_result[i] = data[i];

    cudaFree(data);            // Fig 22.1: replace free()
}

int main(void) {
    printf("=== Unified Memory (§22.1, Fig 22.1) ===\n\n");

    float *h_src    = (float *)malloc(N * sizeof(float));
    float *h_trad   = (float *)malloc(N * sizeof(float));
    float *h_unified = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) h_src[i] = (float)(i % 1024);

    run_traditional(h_src, h_trad);
    run_unified    (h_src, h_unified);

    // Verify both produce the same result
    int fail = 0;
    for (int i = 0; i < N; i++)
        if (fabsf(h_trad[i] - h_unified[i]) > 1e-5f) { fail++; break; }

    printf("Traditional vs unified memory: %s\n\n", fail == 0 ? "PASS" : "FAIL");

    printf("API differences (Fig 22.1):\n");
    printf("  Traditional:    malloc + cudaMalloc + cudaMemcpy(H→D) +\n");
    printf("                  kernel + cudaMemcpy(D→H) + free + cudaFree\n");
    printf("  Unified memory: cudaMallocManaged + kernel +\n");
    printf("                  cudaDeviceSynchronize + cudaFree\n\n");
    printf("  • CPU and GPU share a single virtual pointer — no manual transfers\n");
    printf("  • CUDA runtime migrates pages between host and device on demand\n");
    printf("  • Pascal and later GPUs add hardware page-fault support (§22.1):\n");
    printf("    the CPU no longer needs to flush all managed data before a launch\n");
    printf("  • Simplifies porting CPU code: change malloc→cudaMallocManaged,\n");
    printf("    wrap compute in a kernel launch, change free→cudaFree\n");

    free(h_src); free(h_trad); free(h_unified);
    return 0;
}
