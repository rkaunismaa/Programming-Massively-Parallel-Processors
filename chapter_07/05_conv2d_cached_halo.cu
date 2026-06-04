// §7.5 Tiled convolution using caches for halo cells
// Figure 7.15: simplified tiled kernel where input and output tiles have the
// SAME dimension, and halo accesses fall through to the L2 cache.
//
// Insight (§7.5):
//   Halo cells of a block's input tile are internal elements of neighbouring
//   blocks' input tiles.  Blocks execute concurrently, so by the time a block
//   needs its halo cells, they are very likely already in L2 cache from
//   neighbouring blocks' shared-memory loads.  We can therefore leave halo
//   cells in the N array and rely on hardware caching rather than loading them
//   explicitly into shared memory.
//
// Design simplification vs Fig 7.12:
//   - Block size = TILE_DIM × TILE_DIM  (same as output tile)
//   - Each thread loads exactly ONE element into N_s (no halo threads to disable)
//   - Loading condition: just the image boundary check, no ghost-cell offsets
//   - Output calculation: condition split into shared-mem (interior) vs global (halo)
//
// The body of the filter loop:
//   if neighbour index is within N_s  → use N_s  (fast shared memory)
//   else if neighbour is inside image → use N    (global, likely L2 cache)
//   else                              → skip     (ghost cell, treat as 0)
//
// All four kernels are included here for a final side-by-side benchmark.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef FILTER_RADIUS
#define FILTER_RADIUS 2
#endif
#define FILTER_DIM   (2 * FILTER_RADIUS + 1)
#define TILE_DIM     32
#define IN_TILE_DIM  TILE_DIM
#define OUT_TILE_DIM (TILE_DIM - 2 * FILTER_RADIUS)

#if OUT_TILE_DIM <= 0
#error "FILTER_RADIUS too large for TILE_DIM=32"
#endif

// Filter in constant memory
__constant__ float F_c[FILTER_DIM][FILTER_DIM];

// ── Fig 7.15: cached-halo tiled kernel ───────────────────────────────────────
__global__ void convolution_cached_tiled_2D_const_mem_kernel(const float *N,
                                                              float *P,
                                                              int width,
                                                              int height) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load only interior tile elements (same size as output tile)
    __shared__ float N_s[TILE_DIM][TILE_DIM];
    if (row < height && col < width)
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    else
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();

    if (col >= width || row >= height) return;

    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < FILTER_DIM; fRow++) {
        for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
            int shRow = threadIdx.y - FILTER_RADIUS + fRow;
            int shCol = threadIdx.x - FILTER_RADIUS + fCol;

            if (shRow >= 0 && shRow < TILE_DIM &&
                shCol >= 0 && shCol < TILE_DIM) {
                // Neighbour is inside the shared-memory tile
                Pvalue += F_c[fRow][fCol] * N_s[shRow][shCol];
            } else {
                // Neighbour is in the halo — fall through to global/L2 cache
                int globalRow = row - FILTER_RADIUS + fRow;
                int globalCol = col - FILTER_RADIUS + fCol;
                if (globalRow >= 0 && globalRow < height &&
                    globalCol >= 0 && globalCol < width)
                    Pvalue += F_c[fRow][fCol] *
                              N[globalRow * width + globalCol];
                // else: ghost cell — contributes 0, skip
            }
        }
    }
    P[row * width + col] = Pvalue;
}

// ── Fig 7.12: tiled with explicit halo loading ───────────────────────────────
__global__ void convolution_tiled_2D_kernel(const float *N, float *P,
                                             int width, int height) {
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < height && col >= 0 && col < width)
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    else
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();

    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM &&
            tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < FILTER_DIM; fRow++)
                for (int fCol = 0; fCol < FILTER_DIM; fCol++)
                    Pvalue += F_c[fRow][fCol] *
                              N_s[tileRow + fRow][tileCol + fCol];
            P[row * width + col] = Pvalue;
        }
    }
}

// ── Fig 7.9: constant-memory kernel (no tiling) ───────────────────────────────
__global__ void convolution_2D_const_mem_kernel(const float *N, float *P,
                                                 int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outCol >= width || outRow >= height) return;

    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < FILTER_DIM; fRow++)
        for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
            int inRow = outRow - FILTER_RADIUS + fRow;
            int inCol = outCol - FILTER_RADIUS + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                Pvalue += F_c[fRow][fCol] * N[inRow * width + inCol];
        }
    P[outRow * width + outCol] = Pvalue;
}

// ── Fig 7.7: basic kernel (F in global memory) ───────────────────────────────
__global__ void convolution_2D_basic_kernel(const float *N, const float *F,
                                             float *P, int r,
                                             int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outCol >= width || outRow >= height) return;

    float Pvalue = 0.0f;
    int fd = 2*r+1;
    for (int fRow = 0; fRow < fd; fRow++)
        for (int fCol = 0; fCol < fd; fCol++) {
            int inRow = outRow - r + fRow, inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                Pvalue += F[fRow*fd + fCol] * N[inRow*width + inCol];
        }
    P[outRow * width + outCol] = Pvalue;
}

// ── CPU reference ─────────────────────────────────────────────────────────────
static void conv2d_cpu(const float *N, const float *F, float *P,
                       int r, int width, int height) {
    int fd = 2*r+1;
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++) {
            float v = 0.0f;
            for (int fRow = 0; fRow < fd; fRow++)
                for (int fCol = 0; fCol < fd; fCol++) {
                    int ir = row-r+fRow, ic = col-r+fCol;
                    if (ir >= 0 && ir < height && ic >= 0 && ic < width)
                        v += F[fRow*fd+fCol] * N[ir*width+ic];
                }
            P[row*width+col] = v;
        }
}

static int verify(const float *ref, const float *gpu, int n) {
    for (int i = 0; i < n; i++) {
        float err = fabsf(ref[i] - gpu[i]);
        if (err > 1e-4f * (fabsf(ref[i]) + 1.0f)) {
            printf("  MISMATCH i=%d  ref=%.6f  gpu=%.6f\n", i, ref[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

static float time_kernel(cudaEvent_t t0, cudaEvent_t t1) {
    float ms = 0.0f;
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    return ms;
}

int main(void) {
    const int W  = 2048;
    const int H  = 2048;
    const int R  = FILTER_RADIUS;
    const int FD = FILTER_DIM;
    const long NE = (long)W * H;
    const long FE = (long)FD * FD;

    float *N_h   = (float *)malloc(NE * sizeof(float));
    float *F_h   = (float *)malloc(FE * sizeof(float));
    float *P_ref = (float *)malloc(NE * sizeof(float));
    float *P_gpu = (float *)malloc(NE * sizeof(float));

    srand(42);
    for (long i = 0; i < NE; i++) N_h[i] = (float)rand() / RAND_MAX;
    for (long f = 0; f < FE; f++) F_h[f] = 1.0f / FE;

    float *N_d, *F_d, *P_d;
    cudaMalloc(&N_d, NE * sizeof(float));
    cudaMalloc(&F_d, FE * sizeof(float));   // basic kernel only
    cudaMalloc(&P_d, NE * sizeof(float));
    cudaMemcpy(N_d, N_h, NE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F_h, FE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F_c, F_h, FE * sizeof(float));

    dim3 block16(16, 16);
    dim3 gridBase((W + 15) / 16, (H + 15) / 16);

    dim3 blockTiled(IN_TILE_DIM, IN_TILE_DIM);
    dim3 gridTiled((W + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                   (H + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    dim3 blockCached(TILE_DIM, TILE_DIM);
    dim3 gridCached((W + TILE_DIM - 1) / TILE_DIM,
                    (H + TILE_DIM - 1) / TILE_DIM);

    // Warm-up passes
    convolution_2D_basic_kernel<<<gridBase, block16>>>(N_d, F_d, P_d, R, W, H);
    convolution_2D_const_mem_kernel<<<gridBase, block16>>>(N_d, P_d, W, H);
    convolution_tiled_2D_kernel<<<gridTiled, blockTiled>>>(N_d, P_d, W, H);
    convolution_cached_tiled_2D_const_mem_kernel<<<gridCached, blockCached>>>(N_d, P_d, W, H);
    cudaDeviceSynchronize();

    // Correctness: verify cached-halo kernel vs CPU
    conv2d_cpu(N_h, F_h, P_ref, R, W, H);
    convolution_cached_tiled_2D_const_mem_kernel<<<gridCached, blockCached>>>(N_d, P_d, W, H);
    cudaDeviceSynchronize();
    cudaMemcpy(P_gpu, P_d, NE * sizeof(float), cudaMemcpyDeviceToHost);
    int ok = verify(P_ref, P_gpu, (int)NE);
    printf("Cached-halo kernel: %s\n\n", ok ? "PASS" : "FAIL");

    // Benchmark all four kernels
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms_basic, ms_const, ms_tiled, ms_cached;
    double flops = 2.0 * (double)NE * FE;

    cudaEventRecord(t0);
    convolution_2D_basic_kernel<<<gridBase, block16>>>(N_d, F_d, P_d, R, W, H);
    cudaEventRecord(t1);
    ms_basic = time_kernel(t0, t1);

    cudaEventRecord(t0);
    convolution_2D_const_mem_kernel<<<gridBase, block16>>>(N_d, P_d, W, H);
    cudaEventRecord(t1);
    ms_const = time_kernel(t0, t1);

    cudaEventRecord(t0);
    convolution_tiled_2D_kernel<<<gridTiled, blockTiled>>>(N_d, P_d, W, H);
    cudaEventRecord(t1);
    ms_tiled = time_kernel(t0, t1);

    cudaEventRecord(t0);
    convolution_cached_tiled_2D_const_mem_kernel<<<gridCached, blockCached>>>(N_d, P_d, W, H);
    cudaEventRecord(t1);
    ms_cached = time_kernel(t0, t1);

    // AI values
    float ai_basic  = (float)(2*FE) / (float)(2*FE*4);            // 0.25 OP/B
    float ai_const  = (float)(2*FE) / (float)(FE*4);              // 0.50 OP/B
    float ai_tiled  = (float)(OUT_TILE_DIM*OUT_TILE_DIM) * FE * 2.0f /
                      ((float)(IN_TILE_DIM*IN_TILE_DIM) * 4.0f);  // §7.4 formula
    float ai_cached = ai_tiled; // same effective ratio (halo in L2)

    printf("Image %dx%d   Filter %dx%d (r=%d)\n\n", W, H, FD, FD, R);
    printf("%-34s  %6.3f ms  %5.1f GFLOPS  AI=%.2f OP/B  (Fig 7.7)\n",
           "1. Basic (F global):", ms_basic, flops/(ms_basic*1e6), ai_basic);
    printf("%-34s  %6.3f ms  %5.1f GFLOPS  AI=%.2f OP/B  (Fig 7.9)\n",
           "2. Const mem:", ms_const, flops/(ms_const*1e6), ai_const);
    printf("%-34s  %6.3f ms  %5.1f GFLOPS  AI=%.2f OP/B  (Fig 7.12)\n",
           "3. Tiled explicit halo:", ms_tiled, flops/(ms_tiled*1e6), ai_tiled);
    printf("%-34s  %6.3f ms  %5.1f GFLOPS  AI=%.2f OP/B  (Fig 7.15)\n",
           "4. Tiled cached halo:", ms_cached, flops/(ms_cached*1e6), ai_cached);
    printf("\nSpeedup over basic:  const=%.2fx  tiled=%.2fx  cached=%.2fx\n",
           ms_basic/ms_const, ms_basic/ms_tiled, ms_basic/ms_cached);

    free(N_h); free(F_h); free(P_ref); free(P_gpu);
    cudaFree(N_d); cudaFree(F_d); cudaFree(P_d);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ok ? 0 : 1;
}
