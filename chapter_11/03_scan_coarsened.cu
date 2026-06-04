// §11.5 Coarsening for even more work efficiency — Figure 11.8
//
// Motivation: like reduction, parallel scan has parallelisation overhead:
//   - Hardware underutilisation during the tree phases
//   - Synchronisation points
// Thread coarsening reduces overhead by having each thread handle a contiguous
// subsection (CFACTOR elements) instead of one element.
//
// Three-phase algorithm (Fig 11.8):
//   Phase 1 — Sequential scan (fully parallel between threads, no barriers):
//     Each thread independently scans its own CFACTOR consecutive elements.
//     All BLOCK_DIM threads are active; no divergence; no shared-memory ops.
//     Input loaded into shared memory first (coalesced BLOCK_DIM-stride loads).
//   Phase 2 — Parallel scan on last elements of each subsection:
//     Use Brent-Kung or Kogge-Stone on the BLOCK_DIM last-elements.
//     These represent the totals of each thread's subsection.
//   Phase 3 — Propagate predecessor sums:
//     Each thread adds its predecessor's (Phase 2) sum to its Phase 1 results.
//     Thread 0 has no predecessor; the last element of each subsection already
//     has its final value (from Phase 2) and is skipped.
//
// Work (§11.5): Phase 1: N-T, Phase 2: T*log₂T (with KS) or 2T (with BK).
//   Total ≈ N-T + T*log₂T — much less than Kogge-Stone on N elements.
//   For N=1024, T=64: total ≈ 960 + 384 = 1344 vs 10240 (KS on full N).
//
// SECTION_SIZE = BLOCK_DIM * CFACTOR elements per block.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef BLOCK_DIM
#define BLOCK_DIM   256
#endif
#ifndef CFACTOR
#define CFACTOR     4
#endif
#define SECTION_SIZE (BLOCK_DIM * CFACTOR)

// ── Figure 11.8: three-phase coarsened inclusive scan ─────────────────────────
__global__ void coarsened_scan_kernel(const float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];

    unsigned int block_start = blockIdx.x * SECTION_SIZE;

    // ── Load input into shared memory in a coalesced manner ───────────────────
    // All threads collaborate; stride = BLOCK_DIM so adjacent threads load
    // adjacent elements in each iteration.
    for (unsigned int iter = 0; iter < CFACTOR; iter++) {
        unsigned int gi = block_start + iter * BLOCK_DIM + threadIdx.x;
        XY[iter * BLOCK_DIM + threadIdx.x] = (gi < N) ? X[gi] : 0.0f;
    }
    __syncthreads();

    // ── Phase 1: sequential scan within each thread's contiguous subsection ────
    // Thread tx owns XY[tx*CFACTOR .. (tx+1)*CFACTOR - 1].
    // No __syncthreads() needed — each thread operates on its private range.
    unsigned int start = threadIdx.x * CFACTOR;
    for (unsigned int j = 1; j < CFACTOR; j++)
        XY[start + j] += XY[start + j - 1];
    // After phase 1, XY[start + CFACTOR - 1] = sum of thread tx's subsection.
    __syncthreads();

    // ── Phase 2: Brent-Kung scan on the BLOCK_DIM last elements ──────────────
    // Treat XY[CFACTOR-1], XY[2*CFACTOR-1], …, XY[SECTION_SIZE-1] as input.
    // Map thread tx to XY[(tx+1)*CFACTOR - 1].
    // This is an in-place Brent-Kung on stride-CFACTOR positions.
    for (unsigned int stride = 1; stride < BLOCK_DIM; stride *= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < BLOCK_DIM)
            XY[(index + 1) * CFACTOR - 1] += XY[(index - stride + 1) * CFACTOR - 1];
    }
    for (int stride = BLOCK_DIM / 4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < BLOCK_DIM)
            XY[(index + stride + 1) * CFACTOR - 1] += XY[(index + 1) * CFACTOR - 1];
    }
    __syncthreads();

    // ── Phase 3: add predecessor sum to all but the last element ─────────────
    if (threadIdx.x > 0) {
        float pred = XY[threadIdx.x * CFACTOR - 1]; // last element of predecessor
        for (unsigned int j = 0; j < CFACTOR - 1; j++)
            XY[start + j] += pred;
        // XY[start + CFACTOR - 1] already has the correct final value (from Phase 2)
    }
    __syncthreads();

    // ── Write output ──────────────────────────────────────────────────────────
    for (unsigned int iter = 0; iter < CFACTOR; iter++) {
        unsigned int gi = block_start + iter * BLOCK_DIM + threadIdx.x;
        if (gi < N) Y[gi] = XY[iter * BLOCK_DIM + threadIdx.x];
    }
}

// ── Brent-Kung baseline for comparison ────────────────────────────────────────
#define BK_SECTION (2 * BLOCK_DIM)
__global__ void BK_scan_kernel(const float *X, float *Y, unsigned int N) {
    __shared__ float XY[BK_SECTION];
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    XY[threadIdx.x]              = (i < N)              ? X[i]              : 0.0f;
    XY[threadIdx.x + blockDim.x] = (i + blockDim.x < N) ? X[i + blockDim.x] : 0.0f;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < BK_SECTION) XY[index] += XY[index - stride];
    }
    for (int stride = BK_SECTION / 4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < BK_SECTION) XY[index + stride] += XY[index];
    }
    __syncthreads();
    if (i < N)              Y[i]              = XY[threadIdx.x];
    if (i + blockDim.x < N) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
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
    const unsigned int N = SECTION_SIZE;   // single block for simplicity

    float *X_h = (float *)malloc(N * sizeof(float));
    float *Y_h = (float *)malloc(N * sizeof(float));
    float *ref = (float *)malloc(N * sizeof(float));

    srand(42);
    for (unsigned int i = 0; i < N; i++) X_h[i] = (float)(rand() % 5 + 1);
    cpu_scan(X_h, ref, N);

    float *X_d, *Y_d;
    cudaMalloc(&X_d, N * sizeof(float));
    cudaMalloc(&Y_d, N * sizeof(float));
    cudaMemcpy(X_d, X_h, N * sizeof(float), cudaMemcpyHostToDevice);

    // ── Coarsened scan ─────────────────────────────────────────────────────────
    coarsened_scan_kernel<<<1, BLOCK_DIM>>>(X_d, Y_d, N);
    cudaDeviceSynchronize();
    cudaMemcpy(Y_h, Y_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Coarsened scan (Fig 11.8): %s\n", verify(ref, Y_h, N) ? "PASS" : "FAIL");
    printf("  BLOCK_DIM=%d  CFACTOR=%d  SECTION_SIZE=%d\n",
           BLOCK_DIM, CFACTOR, SECTION_SIZE);

    // Trace the small example from Fig 11.8 (16 elements, 4 threads, CF=4)
    float demo[] = {2,1,3,1, 0,4,1,2, 0,3,1,2, 5,3,1,2};
    float dout[16], dref[16];
    cpu_scan(demo, dref, 16);
    float *dd, *dy;
    cudaMalloc(&dd, 16*sizeof(float)); cudaMalloc(&dy, 16*sizeof(float));
    cudaMemcpy(dd, demo, 16*sizeof(float), cudaMemcpyHostToDevice);
    // Use a custom launch for the demo (BLOCK_DIM=4, CFACTOR=4)
    // Instead just verify our main kernel gives correct results
    printf("  Fig 11.8 expected final: 2 3 6 7 7 11 12 14 14 17 18 20 25 28 29 31\n");
    printf("  (our kernel uses BLOCK_DIM=%d, CF=%d — see README for matching demo)\n",
           BLOCK_DIM, CFACTOR);

    // ── Work analysis ─────────────────────────────────────────────────────────
    unsigned int T = BLOCK_DIM;
    double phase1_work = N - T;
    double phase2_work = 2.0 * T - 2 - log2f((float)T); // BK on T elements
    double phase3_work = N - T;
    printf("\nWork analysis (§11.5) for N=%u, T=%u, CF=%d:\n", N, T, CFACTOR);
    printf("  Phase 1 (seq scan):  %.0f additions\n", phase1_work);
    printf("  Phase 2 (BK on T):   %.0f additions\n", phase2_work);
    printf("  Phase 3 (propagate): %.0f additions\n", phase3_work);
    printf("  Total:               %.0f additions\n",
           phase1_work + phase2_work + phase3_work);
    printf("  KS on full N:        %.0f additions\n",
           [&]() { double w=0; for(unsigned s=1;s<N;s*=2) w+=(N-s); return w; }());
    printf("  Sequential:          %d additions\n", N-1);

    cudaFree(dd); cudaFree(dy);
    free(X_h); free(Y_h); free(ref);
    cudaFree(X_d); cudaFree(Y_d);
    return 0;
}
