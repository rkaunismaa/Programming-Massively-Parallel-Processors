/*
 * Chapter 4 — Section 4.5: Control Divergence
 *             Figures 4.9, 4.10
 *
 * SIMD hardware executes all 32 threads in a warp with a single instruction
 * fetch/dispatch.  When threads in a warp follow different control flow paths
 * (if-else, loops with data-dependent trip counts), the hardware must make
 * MULTIPLE PASSES — one per divergent path.  Threads not on the active path
 * are masked out (inactive) during each pass.  This is called control divergence.
 *
 * Cost: if a warp has k divergent paths, execution takes up to k times
 * longer than a fully uniform warp.  The SIMD efficiency of a warp is
 *   (active threads in that pass) / (warp size)
 *
 * Key observations from Section 4.5:
 *   • Divergence decreases as dataset size grows (fewer divergent warps
 *     as a fraction of the total).
 *   • The boundary guard `if (i < n)` in vecAdd/matMul causes divergence
 *     only in the LAST warp of the grid — negligible for large n.
 *   • Conditionals based on threadIdx are the most common source of
 *     full-warp divergence (every warp may split).
 *
 * Programs:
 *   A) If-divergence: `if (threadIdx.x < 24)` — Figure 4.9 pattern
 *   B) Loop-divergence: trip count = a[threadIdx.x] — Figure 4.10 pattern
 *   C) Boundary divergence analysis for vecAdd
 *   D) Timing comparison: uniform vs divergent kernels
 *
 * Build:
 *   nvcc -O2 -arch=sm_89 -o control_divergence 03_control_divergence.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N       (1 << 22)   /* 4M elements for timing */
#define BLOCK   128

/* -----------------------------------------------------------------------
 * A) If-divergence — Figure 4.9 pattern
 *
 * In a warp of 32 threads, threads 0-23 take the if-path and threads
 * 24-31 take the else-path.  The hardware makes two passes:
 *   pass 1: threads 0-23 execute A; threads 24-31 are inactive
 *   pass 2: threads 24-31 execute B; threads 0-23 are inactive
 *   (then all reconverge at C)
 *
 * The condition `threadIdx.x < 24` affects EVERY warp the same way, so
 * ALL warps diverge — maximum divergence impact.
 * ----------------------------------------------------------------------- */
__global__
void ifDivergentKernel(float* a, float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (threadIdx.x < 24) {
            /* path A */
            out[i] = a[i] * 2.0f + b[i];
        } else {
            /* path B */
            out[i] = a[i] + b[i] * 2.0f;
        }
        /* reconvergence point C */
    }
}

/* Same work, no divergence — baseline for timing */
__global__
void uniformKernel(float* a, float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] * 2.0f + b[i];
}

/* -----------------------------------------------------------------------
 * B) Loop-divergence — Figure 4.10 pattern
 *
 * Each thread runs a different number of loop iterations based on its
 * data value.  Within a warp, the hardware must continue executing the
 * loop body until the LAST active thread has finished, masking out
 * threads that have already completed their iterations.
 * ----------------------------------------------------------------------- */
__global__
void loopDivergentKernel(float* a, int* trip_counts, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = a[i];
        int iters = trip_counts[i];   /* varies per thread → divergence */
        for (int j = 0; j < iters; j++) {
            sum += 1.0f;
        }
        out[i] = sum;
    }
}

/* Same loop, uniform trip count — baseline */
__global__
void loopUniformKernel(float* a, float* out, int n, int uniform_trips) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = a[i];
        for (int j = 0; j < uniform_trips; j++) {
            sum += 1.0f;
        }
        out[i] = sum;
    }
}

/* -----------------------------------------------------------------------
 * Timing helper
 * ----------------------------------------------------------------------- */
static float time_kernel(void (*launch)(float*, float*, float*, int,
                                        int*, float*),
                          float* d_a, float* d_b, float* d_out, int n,
                          int* d_trips, float* d_out2)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (n + BLOCK - 1) / BLOCK;

    /* warm up */
    ifDivergentKernel<<<blocks, BLOCK>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    launch(d_a, d_b, d_out, n, d_trips, d_out2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    /* ── C) Boundary divergence analysis ────────────────────────── */
    printf("=== C) Boundary divergence in vecAdd (Section 4.5) ===\n");
    {
        int test_sizes[] = {100, 1000, 10000, 1000000};
        int block = 64;
        for (int s = 0; s < 4; s++) {
            int n = test_sizes[s];
            int num_warps = ((n + block - 1) / block) * (block / 32);
            int total_warps = num_warps;
            /* Only the last warp of the last block may diverge */
            int divergent_warps = (n % 32 != 0) ? 1 : 0;
            printf("  n=%-8d  total warps=%-6d  divergent warps=%d  (%.1f%%)\n",
                   n, total_warps, divergent_warps,
                   100.0f * divergent_warps / total_warps);
        }
    }

    /* ── A) If-divergence timing ─────────────────────────────────── */
    printf("\n=== A) If-divergence (threadIdx.x < 24) timing ===\n");
    {
        size_t bytes = N * sizeof(float);
        float *d_a, *d_b, *d_out;
        cudaMalloc((void**)&d_a,  bytes);
        cudaMalloc((void**)&d_b,  bytes);
        cudaMalloc((void**)&d_out, bytes);

        /* Fill with 1s */
        cudaMemset(d_a, 0, bytes);  /* zero then add bias below is cleaner */
        float one = 1.0f;
        for (int i = 0; ; ) {
            /* init on device cheaply via a small kernel */
            break;
        }
        /* Use cudaMemset to approximate — just need non-zero data for timing */

        int blocks = (N + BLOCK - 1) / BLOCK;

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);

        /* Warm-up */
        uniformKernel<<<blocks, BLOCK>>>(d_a, d_b, d_out, N);
        cudaDeviceSynchronize();

        cudaEventRecord(t0);
        for (int r = 0; r < 20; r++)
            uniformKernel<<<blocks, BLOCK>>>(d_a, d_b, d_out, N);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms_uniform;
        cudaEventElapsedTime(&ms_uniform, t0, t1);

        cudaEventRecord(t0);
        for (int r = 0; r < 20; r++)
            ifDivergentKernel<<<blocks, BLOCK>>>(d_a, d_b, d_out, N);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms_divergent;
        cudaEventElapsedTime(&ms_divergent, t0, t1);

        printf("  Uniform kernel:    %.3f ms (avg over 20 runs)\n",
               ms_uniform / 20);
        printf("  If-divergent:      %.3f ms (avg over 20 runs)\n",
               ms_divergent / 20);
        printf("  Overhead factor:   %.2fx\n", ms_divergent / ms_uniform);
        printf("  (Expected ~2x for 2-path divergence where every warp splits)\n");

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    }

    /* ── B) Loop-divergence timing ───────────────────────────────── */
    printf("\n=== B) Loop-divergence (data-dependent trip count) timing ===\n");
    {
        size_t bytes = N * sizeof(float);
        float *d_a, *d_out;
        int *d_trips;
        cudaMalloc((void**)&d_a,    bytes);
        cudaMalloc((void**)&d_out,  bytes);
        cudaMalloc((void**)&d_trips, N * sizeof(int));

        /* Build trip-count array: values cycle 4..8 (as in Figure 4.10) */
        int* h_trips = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) h_trips[i] = 4 + (i % 5);  /* 4,5,6,7,8 */
        cudaMemcpy(d_trips, h_trips, N * sizeof(int), cudaMemcpyHostToDevice);
        int uniform_trips = 6;  /* mean of 4..8 */

        int blocks = (N + BLOCK - 1) / BLOCK;
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);

        /* Warm-up */
        loopUniformKernel<<<blocks, BLOCK>>>(d_a, d_out, N, uniform_trips);
        cudaDeviceSynchronize();

        cudaEventRecord(t0);
        for (int r = 0; r < 20; r++)
            loopUniformKernel<<<blocks, BLOCK>>>(d_a, d_out, N, uniform_trips);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms_uniform;
        cudaEventElapsedTime(&ms_uniform, t0, t1);

        cudaEventRecord(t0);
        for (int r = 0; r < 20; r++)
            loopDivergentKernel<<<blocks, BLOCK>>>(d_a, d_trips, d_out, N);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms_divergent;
        cudaEventElapsedTime(&ms_divergent, t0, t1);

        printf("  Uniform loop (%d iters):       %.3f ms\n",
               uniform_trips, ms_uniform / 20);
        printf("  Divergent loop (4-8 iters):   %.3f ms\n",
               ms_divergent / 20);
        printf("  Overhead factor:              %.2fx\n",
               ms_divergent / ms_uniform);
        printf("  (Warp executes max(trip_counts) iters; shorter threads masked)\n");

        free(h_trips);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(d_a); cudaFree(d_out); cudaFree(d_trips);
    }

    printf("\nKey takeaway (Section 4.5):\n");
    printf("  Control divergence cost DECREASES as dataset size grows —\n");
    printf("  only boundary warps diverge, which is a tiny fraction of all warps.\n");
    printf("  Conditionals based on threadIdx affect EVERY warp equally,\n");
    printf("  so they have the largest relative impact.\n");

    return 0;
}
