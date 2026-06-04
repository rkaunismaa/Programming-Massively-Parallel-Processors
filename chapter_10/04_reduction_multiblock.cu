// §10.7 Hierarchical reduction for arbitrary input length — Figure 10.13
//
// Problem with single-block kernels (Figs 10.6, 10.9, 10.11):
//   __syncthreads() only synchronises threads within the same block.
//   A single block can have at most 1024 threads → handles at most 2048 elements.
//   For large inputs (millions of elements), we need multiple blocks.
//
// Solution (§10.7 / Fig 10.12): segmented multiblock reduction
//   Partition the input into segments of size 2 * blockDim.x.
//   Each block independently reduces its segment (using the shared memory
//   approach from Fig 10.11) and commits its partial sum to the output
//   with atomicAdd.
//
// Figure 10.13 structure:
//   Line 03: segment = 2 * blockDim.x * blockIdx.x   (segment start)
//   Line 04: i = segment + threadIdx.x               (global index)
//   Line 05: t = threadIdx.x                         (shared memory index)
//   Line 06: initial load: input_s[t] = input[i] + input[i + BLOCK_DIM]
//   Lines 07–12: convergent reduction in shared memory
//   Lines 13–15: thread 0 atomically adds its segment's sum to output
//
// The final sum is in output[0] after all blocks complete.
// Host must initialise output to 0 before launching.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 1024    // threads per block; each block reduces 2*BLOCK_DIM elements

// ── Figure 10.13: segmented multiblock reduction ──────────────────────────────
__global__ void SegmentedSumReductionKernel(const float *input, float *output) {
    __shared__ float input_s[BLOCK_DIM];

    unsigned int segment = 2 * blockDim.x * blockIdx.x;   // start of this block's segment
    unsigned int i       = segment + threadIdx.x;          // global index (first half)
    unsigned int t       = threadIdx.x;                    // shared memory index

    // Load both halves of this segment and add them into shared memory
    input_s[t] = input[i] + input[i + BLOCK_DIM];

    // Convergent reduction tree in shared memory
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride)
            input_s[t] += input_s[t + stride];
    }

    // Thread 0 commits this block's partial sum atomically (Fig 10.12 bottom)
    if (t == 0)
        atomicAdd(output, input_s[0]);
}

static double cpu_sum(const float *arr, unsigned int n) {
    double s = 0.0;
    for (unsigned int i = 0; i < n; i++) s += arr[i];
    return s;
}

int main(void) {
    // Test with various sizes including non-power-of-two segment counts
    const unsigned int SIZES[] = {2048, 1 << 16, 1 << 20, 1 << 22};
    const int NUM_SIZES = 4;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    for (int si = 0; si < NUM_SIZES; si++) {
        unsigned int N = SIZES[si];
        // N must be a multiple of 2*BLOCK_DIM for this kernel
        N = (N / (2 * BLOCK_DIM)) * (2 * BLOCK_DIM);

        float *input_h = (float *)malloc(N * sizeof(float));
        srand(42 + si);
        for (unsigned int i = 0; i < N; i++)
            input_h[i] = 1.0f;   // sum = N (easy to verify)
        double expected = cpu_sum(input_h, N);

        float *input_d, *output_d;
        cudaMalloc(&input_d,  N * sizeof(float));
        cudaMalloc(&output_d, sizeof(float));
        cudaMemcpy(input_d, input_h, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(output_d, 0, sizeof(float));

        dim3 block(BLOCK_DIM);
        dim3 grid(N / (2 * BLOCK_DIM));

        // Warm-up
        SegmentedSumReductionKernel<<<grid, block>>>(input_d, output_d);
        cudaDeviceSynchronize();

        // Timed run
        cudaMemset(output_d, 0, sizeof(float));
        cudaEventRecord(t0);
        SegmentedSumReductionKernel<<<grid, block>>>(input_d, output_d);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);

        float result_h;
        cudaMemcpy(&result_h, output_d, sizeof(float), cudaMemcpyDeviceToHost);

        double rel_err = fabs((double)result_h - expected) / (expected + 1e-10);
        printf("N=%-10u  blocks=%-6u  result=%.0f  expected=%.0f  %s  %.3f ms\n",
               N, grid.x, result_h, expected,
               rel_err < 1e-4 ? "PASS" : "FAIL", ms);

        free(input_h);
        cudaFree(input_d); cudaFree(output_d);
    }

    printf("\nKey points (§10.7):\n");
    printf("  - Each block handles 2*%d = %d elements independently\n",
           BLOCK_DIM, 2*BLOCK_DIM);
    printf("  - No cross-block __syncthreads() needed — blocks are independent\n");
    printf("  - Final accumulation: atomicAdd(output, block_partial_sum)\n");
    printf("  - Host must initialise output to 0 before launch\n");
    printf("  - Works for any N that is a multiple of %d\n", 2 * BLOCK_DIM);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
