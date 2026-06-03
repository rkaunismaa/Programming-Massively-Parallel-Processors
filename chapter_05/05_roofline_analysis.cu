/*
 * Chapter 5 — Section 5.1: Importance of Memory Access Efficiency
 *             The Roofline Model sidebar
 *
 * ARITHMETIC INTENSITY (Section 5.1):
 *   Arithmetic intensity = FLOPs performed / bytes accessed from global memory
 *   Also called "compute to global memory access ratio" or "computational intensity".
 *
 *   For the naïve matrix multiply inner loop (Figure 5.1):
 *     2 FLOPs (multiply + add) per 8 bytes (2 × 4-byte floats) = 0.25 FLOP/B
 *   This makes it severely memory-bound on any modern GPU.
 *
 *   With TILE_WIDTH tiling:
 *     Same FLOPs, TILE_WIDTH× fewer global memory bytes → 0.25 × TILE_WIDTH FLOP/B
 *
 * THE ROOFLINE MODEL:
 *   Achievable GFLOPS = min(peak_GFLOPS, bandwidth_GB/s × arithmetic_intensity)
 *   Below the "ridge point", performance is memory-bound.
 *   Above it, performance is compute-bound.
 *
 * This program measures:
 *   A) Peak global memory bandwidth  (copy kernel)
 *   B) Achieved GFLOPS for naïve and tiled matmul
 *   C) Arithmetic intensity of each, and where it sits on the roofline
 *
 * Build:
 *   nvcc -O2 -arch=sm_89 -o roofline 05_roofline_analysis.cu -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define N_RUNS     20

/* -----------------------------------------------------------------------
 * Memory bandwidth benchmark — copies N floats from src to dst.
 * Achieved BW = 2 * N * sizeof(float) / elapsed_seconds
 * (factor 2: one read + one write per element)
 * ----------------------------------------------------------------------- */
__global__
void copyKernel(float* __restrict__ dst, const float* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

static double measure_bandwidth_GBs(int n) {
    size_t bytes = n * sizeof(float);
    float *d_src, *d_dst;
    cudaMalloc((void**)&d_src, bytes);
    cudaMalloc((void**)&d_dst, bytes);
    cudaMemset(d_src, 1, bytes);

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    /* warm-up */
    copyKernel<<<blocks, threads>>>(d_dst, d_src, n);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < N_RUNS; r++)
        copyKernel<<<blocks, threads>>>(d_dst, d_src, n);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    double bw = (2.0 * bytes * N_RUNS) / (ms * 1e-3) / 1e9;
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_src); cudaFree(d_dst);
    return bw;
}

/* -----------------------------------------------------------------------
 * Naïve kernel (Chapter 3)
 * ----------------------------------------------------------------------- */
__global__
void naiveKernel(const float* M, const float* N, float* P, int W) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < W && col < W) {
        float v = 0.0f;
        for (int k = 0; k < W; ++k) v += M[row*W+k] * N[k*W+col];
        P[row*W+col] = v;
    }
}

/* -----------------------------------------------------------------------
 * Tiled kernel (Chapter 5, Figure 5.9)
 * ----------------------------------------------------------------------- */
__global__
void tiledKernel(const float* M, const float* N, float* P, int W) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx=blockIdx.x, by=blockIdx.y, tx=threadIdx.x, ty=threadIdx.y;
    int Row=by*TILE_WIDTH+ty, Col=bx*TILE_WIDTH+tx;
    float Pv=0.0f;
    for (int ph=0; ph<W/TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[Row*W + ph*TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph*TILE_WIDTH+ty)*W + Col];
        __syncthreads();
        for (int k=0; k<TILE_WIDTH; ++k) Pv += Mds[ty][k]*Nds[k][tx];
        __syncthreads();
    }
    P[Row*W+Col] = Pv;
}

/* -----------------------------------------------------------------------
 * Time a kernel, return ms (average over N_RUNS launches)
 * ----------------------------------------------------------------------- */
static float time_ms(dim3 grid, dim3 block, bool tiled,
                     const float* M, const float* N, float* P, int W) {
    /* warm-up */
    if (tiled) tiledKernel<<<grid,block>>>(M,N,P,W);
    else       naiveKernel<<<grid,block>>>(M,N,P,W);
    cudaDeviceSynchronize();

    cudaEvent_t t0,t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r=0; r<N_RUNS; r++) {
        if (tiled) tiledKernel<<<grid,block>>>(M,N,P,W);
        else       naiveKernel<<<grid,block>>>(M,N,P,W);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms,t0,t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / N_RUNS;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("=== Roofline Analysis — Section 5.1 ===\n");
    printf("Device: %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    /* ── A) Measure global memory bandwidth ────────────────────── */
    printf("--- A) Global Memory Bandwidth ---\n");
    {
        /* Theoretical peak from device props */
        double theory_bw = 2.0 * prop.memoryClockRate * 1000.0
                           * (prop.memoryBusWidth / 8.0) / 1e9;
        printf("  Theoretical peak BW: %.1f GB/s\n", theory_bw);

        double achieved_bw = measure_bandwidth_GBs(1 << 25);  /* 128M floats */
        printf("  Achieved copy BW:    %.1f GB/s  (%.0f%% of peak)\n\n",
               achieved_bw, 100.0 * achieved_bw / theory_bw);
    }

    /* ── B) Matmul timing and GFLOPS ────────────────────────────── */
    printf("--- B) Matrix Multiply Performance (W=%d, TILE=%d) ---\n", 1024, TILE_WIDTH);
    {
        const int W = 1024;
        size_t bytes = W * W * sizeof(float);
        float *d_M, *d_N, *d_P;
        cudaMalloc((void**)&d_M, bytes);
        cudaMalloc((void**)&d_N, bytes);
        cudaMalloc((void**)&d_P, bytes);

        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid(W/TILE_WIDTH, W/TILE_WIDTH);

        double flops = 2.0 * W * W * W;

        float ms_n = time_ms(dimGrid, dimBlock, false, d_M, d_N, d_P, W);
        float ms_t = time_ms(dimGrid, dimBlock, true,  d_M, d_N, d_P, W);

        double gf_naive = (flops / ms_n) / 1e6;
        double gf_tiled = (flops / ms_t) / 1e6;

        /* Arithmetic intensity:
         *   Naïve: each of W² output elements reads 2W floats = 8W bytes.
         *          Total bytes = W² * 2W * 4 = 8W³.
         *          AI = 2W³ FLOPs / 8W³ bytes = 0.25 FLOP/B
         *
         *   Tiled: TILE_WIDTH× fewer global bytes.
         *          AI = 0.25 * TILE_WIDTH FLOP/B
         */
        double ai_naive = 0.25;
        double ai_tiled = 0.25 * TILE_WIDTH;

        printf("  %-20s  %7.2f ms  %8.1f GFLOPS  AI = %.2f FLOP/B\n",
               "Naïve (Ch3)",   ms_n, gf_naive, ai_naive);
        printf("  %-20s  %7.2f ms  %8.1f GFLOPS  AI = %.2f FLOP/B\n",
               "Tiled (Ch5)",   ms_t, gf_tiled, ai_tiled);
        printf("  Speedup: %.1fx\n\n", ms_n / ms_t);

        cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
    }

    /* ── C) Roofline model ──────────────────────────────────────── */
    printf("--- C) Roofline Model (Section 5.1 sidebar) ---\n");
    {
        double bw   = 2.0 * prop.memoryClockRate * 1000.0
                      * (prop.memoryBusWidth / 8.0) / 1e9;
        /* Use SP FLOPS/s from clock + core count as a rough peak */
        double peak_gflops = 2.0 * prop.multiProcessorCount * 128
                             * (prop.clockRate / 1e6);   /* rough SM FLOPS */

        printf("  Peak BW:     %.0f GB/s\n", bw);
        printf("  Peak FLOPS:  %.0f GFLOPS (rough estimate)\n", peak_gflops);

        double ridge = peak_gflops / bw;   /* FLOP/B at ridge point */
        printf("  Ridge point: %.1f FLOP/B\n\n", ridge);

        double ai_naive = 0.25;
        double ai_tiled = 0.25 * TILE_WIDTH;

        printf("  Kernel AI placement on roofline:\n");
        for (int tw = 1; tw <= 32; tw *= 2) {
            double ai = 0.25 * tw;
            double roof = fmin(peak_gflops, bw * ai);
            const char* bound = (ai < ridge) ? "memory-bound" : "compute-bound";
            printf("    TILE=%2d  AI=%5.2f FLOP/B  roofline limit=%.0f GFLOPS  (%s)\n",
                   tw, ai, roof, bound);
        }

        printf("\n  Book quote (Section 5.1):\n");
        printf("  'The naïve kernel performs 0.25 FLOP/B.'\n");
        printf("  'With 16×16 tiles: 4 FLOP/B — a 16× improvement.'\n");
        printf("  'This raises achievable throughput from ~389 to ~6220 GFLOPS on A100.'\n");
        printf("  (A100 peak BW 1555 GB/s × 4 FLOP/B = 6220 GFLOPS)\n");
    }

    return 0;
}
