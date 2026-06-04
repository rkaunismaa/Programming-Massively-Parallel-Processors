// §10.4 Minimizing control divergence — Figure 10.9
// §10.5 Minimizing memory divergence
//
// The convergent kernel fixes both problems in Fig 10.6 with one change:
//   - Thread owns position i = threadIdx.x  (not 2*threadIdx.x)
//   - Stride DECREASES from blockDim.x down to 1 (not increases)
//   - Active condition: threadIdx.x < stride  (contiguous low threads active)
//
// Control divergence improvement (§10.4):
//   Active threads are: 0..stride-1  (a contiguous prefix).
//   For stride ≥ 32, all warps are either fully active or fully inactive.
//   Only the final 5 iterations (stride < 32) have divergence within a warp.
//   For N=256: 384 warp-slots consumed vs 736 before → ~66% utilisation vs ~35%.
//
// Memory divergence improvement (§10.5):
//   Thread i accesses input[i] and input[i + stride].
//   Adjacent threads access adjacent locations → COALESCED.
//   Total global memory requests: ((N/64 + N/64*½ + …) + 1 + 5)*3 = 36 for N=256
//   vs 141 for the simple kernel — a 3.9× improvement.
//
// NOTE: like Fig 10.6 this still modifies the input array in place.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 1024

// ── Figure 10.6: simple (divergent, non-coalesced) — for comparison ───────────
__global__ void SimpleSumReductionKernel(float *input, float *output) {
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0)
            input[i] += input[i + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) *output = input[0];
}

// ── Figure 10.9: convergent (less divergence, coalesced) ─────────────────────
__global__ void ConvergentSumReductionKernel(float *input, float *output) {
    unsigned int i = threadIdx.x;      // own position = thread index (adjacent!)

    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride)      // active = contiguous low threads
            input[i] += input[i + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) *output = input[0];
}

static float cpu_sum(const float *arr, unsigned int n) {
    float s = 0.0f;
    for (unsigned int i = 0; i < n; i++) s += arr[i];
    return s;
}

int main(void) {
    const unsigned int N = 2 * BLOCK_DIM;

    float *input_h = (float *)malloc(N * sizeof(float));
    float *buf_d;
    float *output_d;
    float output_h;

    srand(42);
    for (unsigned int i = 0; i < N; i++) input_h[i] = (float)rand() / RAND_MAX;
    float expected = cpu_sum(input_h, N);

    cudaMalloc(&buf_d,    N * sizeof(float));
    cudaMalloc(&output_d, sizeof(float));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms_simple, ms_conv;

    // ── Simple kernel ──────────────────────────────────────────────────────────
    cudaMemcpy(buf_d, input_h, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0, sizeof(float));
    cudaEventRecord(t0);
    SimpleSumReductionKernel<<<1, BLOCK_DIM>>>(buf_d, output_d);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_simple, t0, t1);
    cudaMemcpy(&output_h, output_d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Simple   (Fig 10.6): %s  result=%.2f  expected=%.2f  time=%.3f ms\n",
           fabsf(output_h - expected) < 0.5f ? "PASS" : "FAIL",
           output_h, expected, ms_simple);

    // ── Convergent kernel ──────────────────────────────────────────────────────
    cudaMemcpy(buf_d, input_h, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0, sizeof(float));
    cudaEventRecord(t0);
    ConvergentSumReductionKernel<<<1, BLOCK_DIM>>>(buf_d, output_d);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_conv, t0, t1);
    cudaMemcpy(&output_h, output_d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Convergent (Fig 10.9): %s  result=%.2f  expected=%.2f  time=%.3f ms\n",
           fabsf(output_h - expected) < 0.5f ? "PASS" : "FAIL",
           output_h, expected, ms_conv);

    printf("\nControl divergence comparison (N=%u, warp=32):\n", N);
    printf("  Simple:    stride increases 1→%d, threads spread across ALL warps each iteration\n",
           BLOCK_DIM);
    printf("  Convergent: stride decreases %d→1, first k warps fully active, rest fully idle\n",
           BLOCK_DIM);
    printf("  Divergence only in final 5 iterations (stride < 32)\n");

    printf("\nMemory coalescing comparison:\n");
    printf("  Simple:    thread i accesses input[2i] and input[2i+stride] → STRIDED (non-coalesced)\n");
    printf("  Convergent: thread i accesses input[i] and input[i+stride] → COALESCED\n");

    free(input_h);
    cudaFree(buf_d); cudaFree(output_d);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
