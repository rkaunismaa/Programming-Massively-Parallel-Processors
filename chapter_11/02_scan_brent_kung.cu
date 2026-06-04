// §11.4 Parallel scan with the Brent-Kung algorithm — Figure 11.7
//
// Brent-Kung achieves O(N) work — optimal like the sequential algorithm.
// It runs in two phases (Fig 11.5):
//
// Phase 1 — Reduction tree (upsweep):
//   Stride doubles: 1, 2, 4, …, SECTION_SIZE/2.
//   Active threads produce partial sums at positions (tx+1)*2^n - 1.
//   After phase 1, XY[SECTION_SIZE-1] = total sum.
//
// Phase 2 — Reverse distribution tree (downsweep):
//   Stride halves: SECTION_SIZE/4, …, 2, 1.
//   Partial sums are pushed from the root position toward the leaves.
//   After phase 2, XY[i] = inclusive prefix sum of X[0..i].
//
// Thread-index mapping (Fig 11.7):
//   Reduction:    index = (threadIdx.x + 1) * 2 * stride - 1
//   Distribution: index = (threadIdx.x + 1) * stride * 2 - 1
//   This maps consecutive thread indices to the needed positions, keeping
//   active threads contiguous in a warp to minimise control divergence.
//
// Work complexity:
//   Reduction:    N-1 additions
//   Distribution: N-1-log₂N additions
//   Total:        2N - 2 - log₂N ≈ 2N — O(N), work-efficient
//
// Each block processes SECTION_SIZE = 2 * blockDim.x elements.
// Block size: SECTION_SIZE/2 (half as many threads as elements).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef SECTION_SIZE
#define SECTION_SIZE 2048   // 2 elements per thread; block = SECTION_SIZE/2 = 1024
#endif
#define BLOCK_DIM (SECTION_SIZE / 2)

// ── Figure 11.7: Brent-Kung inclusive scan ────────────────────────────────────
__global__ void Brent_Kung_scan_kernel(const float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    // Load two elements per thread (coalesced)
    XY[threadIdx.x]              = (i < N)              ? X[i]              : 0.0f;
    XY[threadIdx.x + blockDim.x] = (i + blockDim.x < N) ? X[i + blockDim.x] : 0.0f;

    // ── Phase 1: Reduction tree (upsweep) — stride doubles ────────────────────
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < SECTION_SIZE)
            XY[index] += XY[index - stride];
    }

    // ── Phase 2: Reverse tree (downsweep) — stride halves ─────────────────────
    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE)
            XY[index + stride] += XY[index];
    }

    // Write output (two elements per thread)
    __syncthreads();
    if (i < N)              Y[i]              = XY[threadIdx.x];
    if (i + blockDim.x < N) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}

// ── Kogge-Stone for comparison ────────────────────────────────────────────────
// (single-block, N = SECTION_SIZE/2 since BK uses 2× elements per block)
__global__ void Kogge_Stone_scan_kernel(const float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    XY[threadIdx.x] = (i < N) ? X[i] : 0.0f;
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp = XY[threadIdx.x];
        if (threadIdx.x >= stride) temp += XY[threadIdx.x - stride];
        __syncthreads();
        XY[threadIdx.x] = temp;
    }
    if (i < N) Y[i] = XY[threadIdx.x];
}

static void cpu_inclusive_scan(const float *X, float *Y, unsigned int N) {
    Y[0] = X[0];
    for (unsigned int i = 1; i < N; i++) Y[i] = Y[i-1] + X[i];
}

static bool verify(const float *ref, const float *gpu, unsigned int N) {
    for (unsigned int i = 0; i < N; i++) {
        float rel = fabsf(ref[i] - gpu[i]) / (fabsf(ref[i]) + 1.0f);
        if (rel > 1e-4f) {
            printf("  MISMATCH i=%u  ref=%.4f  gpu=%.4f\n", i, ref[i], gpu[i]);
            return false;
        }
    }
    return true;
}

int main(void) {
    const unsigned int N = SECTION_SIZE;

    float *X_h = (float *)malloc(N * sizeof(float));
    float *Y_h = (float *)malloc(N * sizeof(float));
    float *ref = (float *)malloc(N * sizeof(float));

    srand(42);
    for (unsigned int i = 0; i < N; i++) X_h[i] = (float)(rand() % 5 + 1);
    cpu_inclusive_scan(X_h, ref, N);

    float *X_d, *Y_d;
    cudaMalloc(&X_d, N * sizeof(float));
    cudaMalloc(&Y_d, N * sizeof(float));
    cudaMemcpy(X_d, X_h, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms;

    // ── Brent-Kung ─────────────────────────────────────────────────────────────
    Brent_Kung_scan_kernel<<<1, BLOCK_DIM>>>(X_d, Y_d, N);
    cudaDeviceSynchronize();
    cudaMemcpy(Y_h, Y_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Brent-Kung (Fig 11.7): %s\n", verify(ref, Y_h, N) ? "PASS" : "FAIL");
    printf("  X[0..7]: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
           X_h[0],X_h[1],X_h[2],X_h[3],X_h[4],X_h[5],X_h[6],X_h[7]);
    printf("  Y[0..7]: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
           Y_h[0],Y_h[1],Y_h[2],Y_h[3],Y_h[4],Y_h[5],Y_h[6],Y_h[7]);

    // Timing
    cudaEventRecord(t0);
    Brent_Kung_scan_kernel<<<1, BLOCK_DIM>>>(X_d, Y_d, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    float ms_bk = ms;

    cudaEventRecord(t0);
    Kogge_Stone_scan_kernel<<<1, SECTION_SIZE>>>(X_d, Y_d, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_bk, t0, t1); // reuse ms_bk to get both
    // Actually let me separate them properly
    cudaEventRecord(t0);
    Brent_Kung_scan_kernel<<<1, BLOCK_DIM>>>(X_d, Y_d, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    printf("\nN=%u  Brent-Kung: %.4f ms\n", N, ms);

    // ── Work efficiency comparison (§11.4 vs §11.3) ───────────────────────────
    printf("\nWork efficiency comparison (N=%u):\n", N);
    double ks_work = 0.0;
    for (unsigned int s = 1; s < N; s *= 2) ks_work += (N - s);
    // BK: reduction N-1, distribution N-1-log₂N
    double bk_work = 2.0 * (N - 1) - (int)log2f((float)N);
    printf("  Sequential:   %u additions  (O(N))\n", N - 1);
    printf("  Brent-Kung:   %.0f additions  (O(N)) — work-efficient!\n", bk_work);
    printf("  Kogge-Stone:  %.0f additions  (O(N·log₂N))\n", ks_work);
    printf("\n  BK  / sequential:  %.2f×\n", bk_work / (N-1));
    printf("  KS  / sequential:  %.2f×\n", ks_work / (N-1));
    printf("\n  Brent-Kung uses N/2 threads (vs N for Kogge-Stone)\n");
    printf("  Brent-Kung requires 2*log₂N time steps (vs log₂N for Kogge-Stone)\n");
    printf("  → Trade: Kogge-Stone is faster with enough hardware;\n");
    printf("    Brent-Kung is better when hardware is limited or energy matters.\n");

    free(X_h); free(Y_h); free(ref);
    cudaFree(X_d); cudaFree(Y_d);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
