// §10.3 A simple reduction kernel — Figure 10.6
//
// The simplest parallel sum reduction kernel.  One block, N/2 threads.
// Each thread owns the even-indexed location input[2*threadIdx.x].
//
// Algorithm (Fig 10.7):
//   Iteration 1: stride=1.  Thread i adds input[2i+1] into input[2i].
//   Iteration 2: stride=2.  Thread i adds input[2i+2] into input[2i] if i%2==0.
//   Iteration k: stride=2^(k-1).  Active threads: those where threadIdx.x % stride == 0.
//   After log₂(blockDim.x) iterations, input[0] holds the total sum.
//
// Control divergence problem (§10.4):
//   Active threads are: those with threadIdx.x % stride == 0.
//   After 5 iterations (stride=32), only 1/32 threads per warp are active.
//   Execution resource utilisation for N=256: only ~35%.
//   This is because active threads are spread across many warps — every warp
//   must still execute even when only one of its 32 threads is doing work.
//
// Memory divergence problem (§10.5):
//   In each iteration, thread i accesses input[2i] and input[2i+stride].
//   Adjacent threads access locations 2 apart (stride-2 in global memory).
//   This stride-2 access → two DRAM bursts per warp → 50% bandwidth waste.
//
// IMPORTANT: This kernel modifies the input array in place (destructive).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 1024   // threads per block; input must have 2*BLOCK_DIM elements

// ── Figure 10.6: simple divergent reduction ───────────────────────────────────
__global__ void SimpleSumReductionKernel(float *input, float *output) {
    unsigned int i = 2 * threadIdx.x;    // each thread owns an even index

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0)   // active threads: indices 0, stride, 2*stride, ...
            input[i] += input[i + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        *output = input[0];
}

int main(void) {
    const unsigned int N = 2 * BLOCK_DIM;   // exactly 2 * blockDim.x elements

    float *input_h  = (float *)malloc(N * sizeof(float));
    float *input_d;
    float *output_d;
    float output_h = 0.0f;

    // Fill with 1.0 → expected sum = N
    for (unsigned int i = 0; i < N; i++) input_h[i] = 1.0f;
    float expected = N;

    cudaMalloc(&input_d,  N * sizeof(float));
    cudaMalloc(&output_d, sizeof(float));

    // ── Run and verify ─────────────────────────────────────────────────────────
    cudaMemcpy(input_d, input_h, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0, sizeof(float));

    SimpleSumReductionKernel<<<1, BLOCK_DIM>>>(input_d, output_d);
    cudaDeviceSynchronize();
    cudaMemcpy(&output_h, output_d, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Simple reduction (Fig 10.6):\n");
    printf("  N=%u  expected=%.0f  got=%.0f  %s\n",
           N, expected, output_h,
           fabsf(output_h - expected) < 0.5f ? "PASS" : "FAIL");

    // ── Control divergence analysis (§10.4) ────────────────────────────────────
    printf("\nControl divergence analysis for N=%u (blockDim=%d):\n", N, BLOCK_DIM);
    printf("  Iteration  stride  active_threads  warps_used  active_warps\n");

    unsigned int active  = BLOCK_DIM;
    double total_consumed = 0.0;   // warp-slots consumed
    double total_useful   = 0.0;   // work done (additions)

    for (unsigned int stride = 1, iter = 1; stride <= BLOCK_DIM; stride *= 2, iter++) {
        unsigned int active_threads = BLOCK_DIM / stride;
        unsigned int warps_launched = (active_threads + 31) / 32;
        // But ALL warps from the block still execute (they just skip the if-body)
        unsigned int all_warps = (BLOCK_DIM + 31) / 32;
        (void)warps_launched;
        total_consumed += all_warps;
        total_useful   += active_threads;
        printf("  %9d  %6d  %14d  %10d  %12d\n",
               iter, stride, active_threads, all_warps, warps_launched);
    }
    printf("\nExecution resource utilisation: %.0f/%.0f = %.1f%%\n",
           total_useful, total_consumed * 32.0,
           100.0 * total_useful / (total_consumed * 32.0));
    printf("(§10.4 says ~35%% for N=256; we print for N=%u)\n", N);

    free(input_h);
    cudaFree(input_d); cudaFree(output_d);
    return 0;
}
