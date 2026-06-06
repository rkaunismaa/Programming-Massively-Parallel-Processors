// §22.2  Cooperative kernels
//
// CUDA 11 introduced cooperative kernels (§22.2) launched via
// cudaLaunchCooperativeKernel().  The CUDA runtime guarantees that ALL
// thread blocks are resident simultaneously, enabling a device-wide
// barrier — grid.sync() from <cooperative_groups.h>.
//
// Without cooperative kernels, a multi-pass algorithm requires separate
// kernel launches with a host-side cudaDeviceSynchronize() between passes.
// Each host sync adds launch overhead and forces the device to flush the
// pipeline.  With a cooperative kernel, the barrier is device-side:
//
//   Pass 1: all blocks build a histogram  (atomicAdd into d_hist[])
//   grid.sync()                            device-wide barrier
//   Pass 2: all blocks normalise the bins  (d_pdf[b] = d_hist[b] / N)
//
// Without the guarantee that all blocks run concurrently, grid.sync()
// would deadlock (a block waiting for another that hasn't been scheduled).
//
// This file runs the same histogram-normalisation computation two ways:
//
//   Two-kernel:   build_hist + cudaDeviceSynchronize + normalise_hist
//   Cooperative:  hist_and_normalise_coop via cudaLaunchCooperativeKernel
//
// Both produce identical results; the cooperative version eliminates the
// host round-trip and issues both passes as a single grid launch.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define N     (1 << 22)   // 4 M data points
#define BINS  256
#define BLOCK 256

// ── Two-kernel approach ───────────────────────────────────────────────────────
__global__ void build_hist(const unsigned int *data, int n,
                             int *hist, int bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(&hist[data[i] % (unsigned)bins], 1);
}

__global__ void normalise_hist(const int *hist, float *pdf, int n, int bins) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < bins)
        pdf[b] = (float)hist[b] / (float)n;
}

// ── Single cooperative kernel (§22.2) ────────────────────────────────────────
// Grid-stride loops cover N > coop_blocks * BLOCK.
// grid.sync() provides the device-wide barrier between the two passes.
__global__ void hist_and_normalise_coop(const unsigned int *data, int n,
                                         int *hist, float *pdf, int bins) {
    cg::grid_group grid = cg::this_grid();

    // Pass 1: build histogram
    for (int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); i < n;
             i += (int)(gridDim.x * blockDim.x))
        atomicAdd(&hist[data[i] % (unsigned)bins], 1);

    grid.sync();   // device-wide barrier: all threads finish Pass 1 before Pass 2

    // Pass 2: normalise
    for (int b = (int)(blockIdx.x * blockDim.x + threadIdx.x); b < bins;
             b += (int)(gridDim.x * blockDim.x))
        pdf[b] = (float)hist[b] / (float)n;
}

int main(void) {
    printf("=== Cooperative Kernels (§22.2) ===\n\n");

    // Check hardware support
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!prop.cooperativeLaunch) {
        printf("Device does not support cooperative kernel launch (need cc ≥ 6.0).\n");
        return 0;
    }
    /* Consumer Pascal (GTX 10xx, cc 6.1) reports cooperativeLaunch=true but
       grid.sync() deadlocks in practice — only Volta+ and enterprise Pascal
       (P100/cc 6.0) reliably support device-wide barriers. */
    if (prop.major == 6 && prop.minor != 0) {
        printf("Device: %s  (%d SMs, cc %d.%d)\n",
               prop.name, prop.multiProcessorCount, prop.major, prop.minor);
        printf("Skipping cooperative launch: consumer Pascal does not reliably\n"
               "support grid.sync() despite reporting cooperativeLaunch=true.\n"
               "Use Volta (cc 7.0) or later for reliable cooperative kernels.\n");
        return 0;
    }
    printf("Device: %s  (%d SMs, cc %d.%d)\n\n",
           prop.name, prop.multiProcessorCount,
           prop.major, prop.minor);

    // Generate random data
    unsigned int *h_data = (unsigned int *)malloc(N * sizeof(unsigned int));
    srand(42);
    for (int i = 0; i < N; i++) h_data[i] = (unsigned int)rand();

    unsigned int *d_data;
    int   *d_hist,  *d_hist2;
    float *d_pdf,   *d_pdf2;
    cudaMalloc(&d_data,  N    * sizeof(unsigned int));
    cudaMalloc(&d_hist,  BINS * sizeof(int));
    cudaMalloc(&d_hist2, BINS * sizeof(int));
    cudaMalloc(&d_pdf,   BINS * sizeof(float));
    cudaMalloc(&d_pdf2,  BINS * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // ── Two-kernel approach ───────────────────────────────────────────────────
    cudaMemset(d_hist, 0, BINS * sizeof(int));
    int blocks = (N + BLOCK - 1) / BLOCK;
    build_hist    <<<blocks, BLOCK>>>(d_data, N, d_hist, BINS);
    cudaDeviceSynchronize();                           // host-side barrier
    normalise_hist<<<(BINS + BLOCK-1)/BLOCK, BLOCK>>>(d_hist, d_pdf, N, BINS);
    cudaDeviceSynchronize();

    // ── Cooperative kernel ────────────────────────────────────────────────────
    cudaMemset(d_hist2, 0, BINS * sizeof(int));

    // Determine max blocks that can all be resident simultaneously
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, hist_and_normalise_coop, BLOCK, 0);
    int coop_blocks = max_blocks_per_sm * prop.multiProcessorCount;

    int   n_val    = N;
    int   bins_val = BINS;
    void *args[]   = { (void *)&d_data,
                       (void *)&n_val,
                       (void *)&d_hist2,
                       (void *)&d_pdf2,
                       (void *)&bins_val };

    cudaLaunchCooperativeKernel((void *)hist_and_normalise_coop,
                                 dim3(coop_blocks), dim3(BLOCK),
                                 args, 0, NULL);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Cooperative launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // ── Verify ────────────────────────────────────────────────────────────────
    float *h_pdf  = (float *)malloc(BINS * sizeof(float));
    float *h_pdf2 = (float *)malloc(BINS * sizeof(float));
    cudaMemcpy(h_pdf,  d_pdf,  BINS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pdf2, d_pdf2, BINS * sizeof(float), cudaMemcpyDeviceToHost);

    int fail_match = 0;
    for (int b = 0; b < BINS; b++)
        if (fabsf(h_pdf[b] - h_pdf2[b]) > 1e-7f) fail_match++;

    float sum = 0.f;
    for (int b = 0; b < BINS; b++) sum += h_pdf2[b];

    printf("Two-kernel vs cooperative kernel: %s",
           fail_match == 0 ? "PASS" : "FAIL");
    if (fail_match) printf("  (%d bins differ)", fail_match);
    printf("\n");
    printf("PDF sums to 1 (cooperative):      %s  (sum = %.6f)\n\n",
           fabsf(sum - 1.0f) < 1e-4f ? "PASS" : "FAIL", sum);

    printf("Cooperative kernel details (§22.2):\n");
    printf("  • coop_blocks = %d  (%d SMs × %d blocks/SM)\n",
           coop_blocks, prop.multiProcessorCount, max_blocks_per_sm);
    printf("  • Runtime guarantees all %d blocks are resident simultaneously\n",
           coop_blocks);
    printf("  • grid.sync() is a device-wide barrier — safe from deadlock\n");
    printf("    because ALL blocks are guaranteed to reach it eventually\n");
    printf("  • Grid-stride loops handle N=%d > coop_blocks×BLOCK=%d\n",
           N, coop_blocks * BLOCK);
    printf("  • Eliminates host-side cudaDeviceSynchronize between passes\n");
    printf("  • Requires compute capability ≥ 6.0 and -rdc=true\n");

    free(h_data); free(h_pdf); free(h_pdf2);
    cudaFree(d_data); cudaFree(d_hist); cudaFree(d_hist2);
    cudaFree(d_pdf);  cudaFree(d_pdf2);
    return 0;
}
