// §11.6 Segmented parallel scan for arbitrary-length inputs — Figures 11.9/11.10
//
// Single-block scan kernels are limited to SECTION_SIZE elements (≤2048 for BK).
// For large inputs (millions of elements), three kernels implement a hierarchical
// two-level scan (Figs 11.9–11.10):
//
// Kernel 1 — Local scan + collect block sums:
//   Run Brent-Kung scan on each block's section independently.
//   The last thread of each block writes its section sum to S[blockIdx.x].
//   After Kernel 1, Y[i] has the correct scan within its block but misses
//   the contributions from all preceding blocks.
//
// Kernel 2 — Scan S:
//   Run Brent-Kung on the S array (single block, gridDim.x elements ≤ SECTION_SIZE).
//   After Kernel 2, S[b] = sum of all X elements in blocks 0..b.
//
// Kernel 3 — Propagate block sums:
//   Each thread in block b > 0 adds S[b-1] to its Y element.
//   Block 0 needs no update (its elements are already correct).
//
// Hierarchical example (Fig 11.10):
//   X: [2,1,3,1, 0,4,1,2, 0,3,1,2, 5,3,1,2]  (4 blocks of 4 elements each)
//   After K1: Y = [2,3,6,7, 0,4,5,7, 0,3,4,6, 5,8,9,11]  S = [7,7,6,11]
//   After K2: S = [7,14,20,31]
//   After K3: Y = [2,3,6,7, 7,11,12,14, 14,17,18,20, 25,28,29,31]  ✓
//
// Maximum N: SECTION_SIZE^2 (since Kernel 2 scans S in one block).
// For SECTION_SIZE=2048: max N = 4M elements.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef SECTION_SIZE
#define SECTION_SIZE 2048
#endif
#define BLOCK_DIM (SECTION_SIZE / 2)

// ── Kernel 1: Brent-Kung scan per block, collect block sums ──────────────────
__global__ void scan_local_kernel(const float *X, float *Y, float *S,
                                   unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    XY[threadIdx.x]              = (i < N)              ? X[i]              : 0.0f;
    XY[threadIdx.x + blockDim.x] = (i + blockDim.x < N) ? X[i + blockDim.x] : 0.0f;

    // Brent-Kung reduction tree
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int idx = (threadIdx.x + 1) * 2 * stride - 1;
        if (idx < SECTION_SIZE) XY[idx] += XY[idx - stride];
    }
    // Brent-Kung distribution tree
    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int idx = (threadIdx.x + 1) * stride * 2 - 1;
        if (idx + stride < SECTION_SIZE) XY[idx + stride] += XY[idx];
    }
    __syncthreads();

    if (i < N)              Y[i]              = XY[threadIdx.x];
    if (i + blockDim.x < N) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];

    // Last thread writes this block's total to S (§11.6)
    if (threadIdx.x == blockDim.x - 1)
        S[blockIdx.x] = XY[SECTION_SIZE - 1];
}

// ── Kernel 2: scan S (single block; gridDim.x ≤ SECTION_SIZE) ────────────────
__global__ void scan_S_kernel(float *S, unsigned int num_blocks) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = threadIdx.x;

    XY[i]              = (i < num_blocks)              ? S[i]              : 0.0f;
    XY[i + blockDim.x] = (i + blockDim.x < num_blocks) ? S[i + blockDim.x] : 0.0f;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int idx = (threadIdx.x + 1) * 2 * stride - 1;
        if (idx < SECTION_SIZE) XY[idx] += XY[idx - stride];
    }
    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int idx = (threadIdx.x + 1) * stride * 2 - 1;
        if (idx + stride < SECTION_SIZE) XY[idx + stride] += XY[idx];
    }
    __syncthreads();

    if (i < num_blocks)              S[i]              = XY[i];
    if (i + blockDim.x < num_blocks) S[i + blockDim.x] = XY[i + blockDim.x];
}

// ── Kernel 3: add S[blockIdx.x - 1] to each element in block blockIdx.x ──────
__global__ void add_block_sums_kernel(float *Y, const float *S, unsigned int N) {
    if (blockIdx.x == 0) return;   // block 0 already correct

    unsigned int i = blockIdx.x * SECTION_SIZE + threadIdx.x;
    float offset = S[blockIdx.x - 1];

    if (i < N)              Y[i]              += offset;
    if (i + blockDim.x < N) Y[i + blockDim.x] += offset;
}

static void cpu_scan(const float *X, float *Y, unsigned int N) {
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
    const unsigned int SIZES[] = {
        SECTION_SIZE,        // single block
        SECTION_SIZE * 4,    // 4 blocks
        1 << 20,             // 1M
        1 << 22              // 4M (near max for 2-level hierarchical with SS=2048)
    };
    const int NUM = 4;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    for (int si = 0; si < NUM; si++) {
        unsigned int N = SIZES[si];
        // Pad N to multiple of SECTION_SIZE
        unsigned int N_padded = ((N + SECTION_SIZE - 1) / SECTION_SIZE) * SECTION_SIZE;
        unsigned int num_blocks = N_padded / SECTION_SIZE;

        float *X_h = (float *)malloc(N * sizeof(float));
        float *Y_h = (float *)malloc(N * sizeof(float));
        float *ref = (float *)malloc(N * sizeof(float));
        srand(42 + si);
        for (unsigned int i = 0; i < N; i++) X_h[i] = 1.0f;  // easy to verify: Y[i]=i+1
        cpu_scan(X_h, ref, N);

        float *X_d, *Y_d, *S_d;
        cudaMalloc(&X_d, N * sizeof(float));
        cudaMalloc(&Y_d, N * sizeof(float));
        cudaMalloc(&S_d, num_blocks * sizeof(float));
        cudaMemcpy(X_d, X_h, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(Y_d, 0, N * sizeof(float));

        dim3 block(BLOCK_DIM);
        dim3 grid_scan(num_blocks);
        dim3 grid_add(num_blocks);

        // Warm-up
        scan_local_kernel<<<grid_scan, block>>>(X_d, Y_d, S_d, N);
        if (num_blocks > 1) scan_S_kernel<<<1, BLOCK_DIM>>>(S_d, num_blocks);
        if (num_blocks > 1) add_block_sums_kernel<<<grid_add, block>>>(Y_d, S_d, N);
        cudaDeviceSynchronize();

        // Timed run
        cudaMemset(Y_d, 0, N * sizeof(float));
        cudaEventRecord(t0);
        scan_local_kernel<<<grid_scan, block>>>(X_d, Y_d, S_d, N);
        if (num_blocks > 1) scan_S_kernel<<<1, BLOCK_DIM>>>(S_d, num_blocks);
        if (num_blocks > 1) add_block_sums_kernel<<<grid_add, block>>>(Y_d, S_d, N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);

        cudaMemcpy(Y_h, Y_d, N * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = verify(ref, Y_h, N);
        printf("N=%-8u  blocks=%-4u  %s  %.3f ms  (Y[N-1]=%.0f, expected %.0f)\n",
               N, num_blocks, ok ? "PASS" : "FAIL", ms, Y_h[N-1], ref[N-1]);

        free(X_h); free(Y_h); free(ref);
        cudaFree(X_d); cudaFree(Y_d); cudaFree(S_d);
    }

    printf("\nHierarchical approach (§11.6 / Fig 11.9):\n");
    printf("  Kernel 1: Brent-Kung per block + write last element to S[]\n");
    printf("  Kernel 2: Brent-Kung scan on S[] (single block)\n");
    printf("  Kernel 3: Y[i] += S[blockIdx.x - 1] for block > 0\n");
    printf("  Max N: SECTION_SIZE^2 = %llu (two-level hierarchy)\n",
           (unsigned long long)SECTION_SIZE * SECTION_SIZE);
    printf("  Limitation: extra global memory round-trips between K1→K2→K3\n");
    printf("  → see 05_scan_single_pass.cu for the domino approach that avoids this.\n");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
