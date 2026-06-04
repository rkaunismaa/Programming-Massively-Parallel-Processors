// §10.6 Minimizing global memory accesses — Figure 10.11
//
// Problem with Fig 10.9 (convergent kernel):
//   Threads write partial sums back to global memory every iteration.
//   Total global memory requests for N=256: 36 (much better than 141 for Fig 10.6
//   but still more than necessary).  Partial sums are written to global memory
//   only to be read back by the same or adjacent threads in the next iteration.
//
// Solution: shared memory (§10.6 / Fig 10.10):
//   Load the 2*blockDim.x input elements from global memory ONCE.
//   Perform all reduction iterations entirely in shared memory.
//   Write the final sum to global memory exactly ONCE (via the atomic add).
//
// Figure 10.11 structure:
//   Line 04: each thread loads two adjacent global elements and sums them into
//             shared memory: input_s[t] = input[t] + input[t + BLOCK_DIM]
//   Lines 05–10: convergent reduction entirely within input_s[]
//   Lines 11–13: thread 0 writes input_s[0] as the output
//
// Global memory accesses for N=256 (Fig 10.11 vs Fig 10.9):
//   Fig 10.9: 36 global memory requests
//   Fig 10.11: (N/32 + 1)*3 = (8+1)*3 = 27 → actually (N/32)*1 + 1 = 9
//   The book says: N+1 total (N loads for the input + 1 write to output).
//   This is a ~4× improvement over Fig 10.9.
//
// Input is NOT modified: the original array is preserved after the reduction.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 1024

// ── Figure 10.11: shared memory reduction ────────────────────────────────────
// Input size must be exactly 2 * BLOCK_DIM.
__global__ void SharedMemorySumReductionKernel(const float *input, float *output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;

    // Load two adjacent global elements and merge immediately into shared mem.
    // This counts as the first reduction step at zero extra cost.
    input_s[t] = input[t] + input[t + BLOCK_DIM];

    // Convergent reduction entirely in shared memory
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();                  // barrier before each read
        if (t < stride)
            input_s[t] += input_s[t + stride];
    }
    if (threadIdx.x == 0)
        *output = input_s[0];            // single global write
}

// ── Fig 10.9: convergent (global memory) — for comparison ────────────────────
__global__ void ConvergentSumReductionKernel(float *input, float *output) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) input[i] += input[i + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) *output = input[0];
}

static float cpu_sum(const float *arr, unsigned int n) {
    double s = 0.0;
    for (unsigned int i = 0; i < n; i++) s += arr[i];
    return (float)s;
}

int main(void) {
    const unsigned int N = 2 * BLOCK_DIM;

    float *input_h = (float *)malloc(N * sizeof(float));
    float *input_d, *buf_d, *output_d;
    float output_h;

    srand(42);
    for (unsigned int i = 0; i < N; i++) input_h[i] = (float)rand() / RAND_MAX;
    float expected = cpu_sum(input_h, N);

    cudaMalloc(&input_d,  N * sizeof(float));
    cudaMalloc(&buf_d,    N * sizeof(float));
    cudaMalloc(&output_d, sizeof(float));
    cudaMemcpy(input_d, input_h, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms_conv, ms_shmem;

    // Warm-up
    cudaMemcpy(buf_d, input_d, N * sizeof(float), cudaMemcpyDeviceToDevice);
    ConvergentSumReductionKernel<<<1, BLOCK_DIM>>>(buf_d, output_d);
    SharedMemorySumReductionKernel<<<1, BLOCK_DIM>>>(input_d, output_d);
    cudaDeviceSynchronize();

    // ── Convergent kernel (global memory) ─────────────────────────────────────
    cudaMemcpy(buf_d, input_d, N * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemset(output_d, 0, sizeof(float));
    cudaEventRecord(t0);
    ConvergentSumReductionKernel<<<1, BLOCK_DIM>>>(buf_d, output_d);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_conv, t0, t1);
    cudaMemcpy(&output_h, output_d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Convergent global mem (Fig 10.9):  %s  %.4f  time=%.3f ms\n",
           fabsf(output_h - expected) < 1.0f ? "PASS" : "FAIL", output_h, ms_conv);

    // ── Shared memory kernel ───────────────────────────────────────────────────
    cudaMemset(output_d, 0, sizeof(float));
    cudaEventRecord(t0);
    SharedMemorySumReductionKernel<<<1, BLOCK_DIM>>>(input_d, output_d);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_shmem, t0, t1);
    cudaMemcpy(&output_h, output_d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Shared memory       (Fig 10.11): %s  %.4f  time=%.3f ms\n",
           fabsf(output_h - expected) < 1.0f ? "PASS" : "FAIL", output_h, ms_shmem);

    printf("\nN=%u  expected=%.4f\n", N, expected);
    printf("Global memory accesses:\n");
    printf("  Fig 10.9  (convergent): each iteration writes partial sums to global → ~36 requests (N=256)\n");
    printf("  Fig 10.11 (shared mem): N loads + 1 write = N+1 total → ~4x fewer\n");
    printf("  Input array PRESERVED by Fig 10.11 (non-destructive)\n");

    free(input_h);
    cudaFree(input_d); cudaFree(buf_d); cudaFree(output_d);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
