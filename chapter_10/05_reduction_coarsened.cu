// §10.8 Thread coarsening for reduced overhead — Figure 10.15
//
// Problem with the multiblock kernel (Fig 10.13):
//   To maximise parallelism, we launch N/2 threads (N/2/BLOCK_DIM blocks).
//   When N is very large, many more blocks are launched than the hardware
//   can run simultaneously.  The hardware serialises the surplus blocks.
//   Serialised blocks still pay the full startup + synchronisation overhead
//   of the reduction tree.  This is wasted work (§10.8 / Fig 10.16A).
//
// Solution: thread coarsening (§10.8 / Figs 10.14–10.15):
//   Each thread accumulates COARSE_FACTOR*2 elements independently
//   (no __syncthreads() or shared memory needed for this phase).
//   Then threads collaborate for the final reduction tree.
//   One coarsened block does the work of COARSE_FACTOR original blocks,
//   but the hardware-underutilised reduction tree is executed only once
//   instead of COARSE_FACTOR times.
//
// Figure 10.15 structure:
//   Line 03: segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x
//   Lines 06–09: serial coarsening loop: sum += input[i + tile*BLOCK_DIM]
//   Line 10: store partial sum to shared memory
//   Lines 11–16: convergent tree reduction in shared memory
//   Lines 17–19: atomicAdd for final output
//
// All five kernels are benchmarked here for a final comparison table.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_DIM    1024
#ifndef COARSE_FACTOR
#define COARSE_FACTOR 4
#endif

// ── Figure 10.15: coarsened multiblock reduction ──────────────────────────────
__global__ void CoarsenedSumReductionKernel(const float *input, float *output) {
    __shared__ float input_s[BLOCK_DIM];

    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i       = segment + threadIdx.x;
    unsigned int t       = threadIdx.x;

    // ── Serial coarsening: each thread adds COARSE_FACTOR*2 elements ──────────
    // No __syncthreads() needed here — threads are completely independent.
    float sum = input[i];
    for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; tile++)
        sum += input[i + tile * BLOCK_DIM];

    // Store partial sum into shared memory for the tree reduction
    input_s[t] = sum;

    // ── Convergent tree reduction in shared memory ─────────────────────────────
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride)
            input_s[t] += input_s[t + stride];
    }
    if (t == 0)
        atomicAdd(output, input_s[0]);
}

// ── Figure 10.6: simple (baseline) ───────────────────────────────────────────
__global__ void SimpleSumReductionKernel(float *input, float *output) {
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) input[i] += input[i + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) *output = input[0];
}

// ── Figure 10.9: convergent (reduced divergence) ──────────────────────────────
__global__ void ConvergentSumReductionKernel(float *input, float *output) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) input[i] += input[i + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) *output = input[0];
}

// ── Figure 10.11: shared memory (one block) ───────────────────────────────────
__global__ void SharedMemorySumReductionKernel(const float *input, float *output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t + BLOCK_DIM];
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) input_s[t] += input_s[t + stride];
    }
    if (threadIdx.x == 0) *output = input_s[0];
}

// ── Figure 10.13: segmented multiblock ────────────────────────────────────────
__global__ void SegmentedSumReductionKernel(const float *input, float *output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int i = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int t = threadIdx.x;
    input_s[t] = input[i] + input[i + BLOCK_DIM];
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) input_s[t] += input_s[t + stride];
    }
    if (t == 0) atomicAdd(output, input_s[0]);
}

static double cpu_sum(const float *arr, unsigned int n) {
    double s = 0.0;
    for (unsigned int i = 0; i < n; i++) s += arr[i];
    return s;
}

int main(void) {
    const unsigned int N_SMALL = 2 * BLOCK_DIM;           // for single-block kernels
    const unsigned int N_LARGE = COARSE_FACTOR * 2 * BLOCK_DIM * 1024; // for multiblock

    printf("BLOCK_DIM=%d  COARSE_FACTOR=%d\n\n", BLOCK_DIM, COARSE_FACTOR);

    // ── Part 1: single-block kernels (N_SMALL) ─────────────────────────────────
    {
        float *h = (float *)malloc(N_SMALL * sizeof(float));
        srand(42);
        for (unsigned int i = 0; i < N_SMALL; i++) h[i] = (float)rand() / RAND_MAX;
        double expected = cpu_sum(h, N_SMALL);

        float *d_in, *d_buf, *d_out;
        cudaMalloc(&d_in,  N_SMALL * sizeof(float));
        cudaMalloc(&d_buf, N_SMALL * sizeof(float));
        cudaMalloc(&d_out, sizeof(float));
        cudaMemcpy(d_in, h, N_SMALL * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        float ms, result_h;

        printf("Single-block kernels, N=%u:\n", N_SMALL);

        // Simple
        cudaMemcpy(d_buf, d_in, N_SMALL*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemset(d_out, 0, sizeof(float));
        cudaEventRecord(t0);
        SimpleSumReductionKernel<<<1, BLOCK_DIM>>>(d_buf, d_out);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        cudaMemcpy(&result_h, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        printf("  Simple     (Fig 10.6):  %s  %.4f  %.4f ms  (divergent, non-coalesced)\n",
               fabs(result_h-expected)<1.0?"PASS":"FAIL", result_h, ms);

        // Convergent
        cudaMemcpy(d_buf, d_in, N_SMALL*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemset(d_out, 0, sizeof(float));
        cudaEventRecord(t0);
        ConvergentSumReductionKernel<<<1, BLOCK_DIM>>>(d_buf, d_out);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        cudaMemcpy(&result_h, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        printf("  Convergent (Fig 10.9):  %s  %.4f  %.4f ms  (less divergence, coalesced)\n",
               fabs(result_h-expected)<1.0?"PASS":"FAIL", result_h, ms);

        // Shared memory
        cudaMemset(d_out, 0, sizeof(float));
        cudaEventRecord(t0);
        SharedMemorySumReductionKernel<<<1, BLOCK_DIM>>>(d_in, d_out);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        cudaMemcpy(&result_h, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        printf("  Shared mem (Fig 10.11): %s  %.4f  %.4f ms  (N+1 global accesses)\n",
               fabs(result_h-expected)<1.0?"PASS":"FAIL", result_h, ms);

        free(h); cudaFree(d_in); cudaFree(d_buf); cudaFree(d_out);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    // ── Part 2: multi-block kernels (N_LARGE) ──────────────────────────────────
    {
        float *h = (float *)malloc(N_LARGE * sizeof(float));
        for (unsigned int i = 0; i < N_LARGE; i++) h[i] = 1.0f;
        double expected = N_LARGE;

        float *d_in, *d_out;
        cudaMalloc(&d_in,  N_LARGE * sizeof(float));
        cudaMalloc(&d_out, sizeof(float));
        cudaMemcpy(d_in, h, N_LARGE * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        float ms, result_h;

        dim3 blockD(BLOCK_DIM);
        dim3 gridSeg(N_LARGE / (2 * BLOCK_DIM));
        dim3 gridCoarse(N_LARGE / (COARSE_FACTOR * 2 * BLOCK_DIM));

        printf("\nMulti-block kernels, N=%u:\n", N_LARGE);

        // Warm-up
        cudaMemset(d_out, 0, sizeof(float));
        SegmentedSumReductionKernel<<<gridSeg, blockD>>>(d_in, d_out);
        cudaMemset(d_out, 0, sizeof(float));
        CoarsenedSumReductionKernel<<<gridCoarse, blockD>>>(d_in, d_out);
        cudaDeviceSynchronize();

        // Segmented
        cudaMemset(d_out, 0, sizeof(float));
        cudaEventRecord(t0);
        SegmentedSumReductionKernel<<<gridSeg, blockD>>>(d_in, d_out);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        cudaMemcpy(&result_h, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        printf("  Segmented  (Fig 10.13): %s  %.0f  %.3f ms  (%u blocks)\n",
               fabs(result_h-expected)/expected<1e-4?"PASS":"FAIL",
               result_h, ms, gridSeg.x);

        // Coarsened
        cudaMemset(d_out, 0, sizeof(float));
        cudaEventRecord(t0);
        CoarsenedSumReductionKernel<<<gridCoarse, blockD>>>(d_in, d_out);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        cudaMemcpy(&result_h, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        printf("  Coarsened  (Fig 10.15): %s  %.0f  %.3f ms  (%u blocks, CF=%d)\n",
               fabs(result_h-expected)/expected<1e-4?"PASS":"FAIL",
               result_h, ms, gridCoarse.x, COARSE_FACTOR);

        printf("\nCoarsening effect:\n");
        printf("  Blocks: segmented=%u → coarsened=%u  (%dx fewer blocks)\n",
               gridSeg.x, gridCoarse.x, gridSeg.x/gridCoarse.x);
        printf("  Each coarsened block does %d original blocks' work\n", COARSE_FACTOR);
        printf("  %d reduction trees instead of %d → reduced synchronisation overhead\n",
               gridCoarse.x, gridSeg.x);

        free(h); cudaFree(d_in); cudaFree(d_out);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    return 0;
}
