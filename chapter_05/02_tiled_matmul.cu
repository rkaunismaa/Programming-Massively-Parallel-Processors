/*
 * Chapter 5 — Sections 5.3–5.4: Tiling for Reduced Memory Traffic
 *             Figure 5.9: tiled matrix multiplication kernel
 *             Figures 5.7–5.8: tiling concept and execution phases
 *
 * PROBLEM WITH THE NAÏVE KERNEL (Chapter 3):
 *   For an N×N matrix multiply, each thread reads an entire row of M and an
 *   entire column of N — Width global memory accesses per output element.
 *   2 FLOPs (multiply + add) per 8 bytes → 0.25 FLOP/B.
 *   An A100 at 1555 GB/s delivers only 389 GFLOPS this way, 2% of its peak.
 *
 * THE TILING SOLUTION (Section 5.3):
 *   Partition M and N into TILE_WIDTH×TILE_WIDTH tiles.
 *   Each block loads one tile of M and one tile of N collaboratively into
 *   shared memory (one element per thread).  The tile fits in shared memory
 *   and every element is reused TILE_WIDTH times within the block before the
 *   next tile is loaded.  This reduces global memory traffic by TILE_WIDTH×.
 *
 *   With TILE_WIDTH=16: 0.25 × 16 = 4 FLOP/B  (16× improvement)
 *   With TILE_WIDTH=32: 0.25 × 32 = 8 FLOP/B  (32× improvement)
 *
 * TWO __syncthreads() PER PHASE (Section 5.4, Figure 5.9 lines 21 & 26):
 *   Line 21: read-after-write dependence — all threads must finish loading
 *             the tile before anyone starts consuming it.
 *   Line 26: write-after-read dependence — all threads must finish using
 *             the tile before anyone overwrites it with the next tile.
 *
 * ASSUMPTION: Width is a multiple of TILE_WIDTH.
 *   → See 03_tiled_matmul_boundary.cu for the general case.
 *
 * Build:
 *   nvcc -O2 -arch=sm_89 -o tiled_matmul 02_tiled_matmul.cu -lm
 *   To change tile size: nvcc -DTILE_WIDTH=32 ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

/* -----------------------------------------------------------------------
 * Tiled matrix multiplication kernel — Figure 5.9
 *
 * Square matrices: M[Width×Width], N[Width×Width] → P[Width×Width]
 * Assumption: Width is a multiple of TILE_WIDTH.
 *
 * The outer loop (ph) iterates over Width/TILE_WIDTH phases.
 * In each phase:
 *   1. All TILE_WIDTH² threads collaboratively load one tile of M (Mds)
 *      and one tile of N (Nds) from global into shared memory.
 *   2. __syncthreads() ensures the load completes before use.
 *   3. Each thread accumulates TILE_WIDTH products into Pvalue.
 *   4. __syncthreads() ensures all threads finish before the next phase
 *      overwrites Mds and Nds.
 * ----------------------------------------------------------------------- */
__global__
void matrixMulKernel(float* M, float* N, float* P, int Width) {
    /* Shared memory tiles — one version created per block */
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    /* Save threadIdx / blockIdx into registers for repeated use */
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    /* Identify the P element this thread is responsible for (Figure 5.10) */
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    /* Accumulate the dot product across all phases */
    float Pvalue = 0.0f;

    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        /* Collaborative load: each thread loads one M and one N element
         *
         * M tile: row is fixed (Row), column advances by TILE_WIDTH each phase
         *   column index = ph*TILE_WIDTH + tx
         *
         * N tile: row advances by TILE_WIDTH each phase, column is fixed (Col)
         *   row index    = ph*TILE_WIDTH + ty
         */
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];

        /* Barrier 1 (line 21): read-after-write
         * All threads must finish writing Mds/Nds before any thread reads */
        __syncthreads();

        /* Compute partial dot product using the loaded tiles */
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        /* Barrier 2 (line 26): write-after-read
         * All threads must finish reading Mds/Nds before the next phase
         * overwrites them with fresh tile data */
        __syncthreads();
    }

    P[Row * Width + Col] = Pvalue;
}

/* -----------------------------------------------------------------------
 * Naïve kernel from Chapter 3 (Figure 3.11) — for timing comparison
 * ----------------------------------------------------------------------- */
__global__
void naiveMatMulKernel(float* M, float* N, float* P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < Width && col < Width) {
        float Pvalue = 0.0f;
        for (int k = 0; k < Width; ++k)
            Pvalue += M[row * Width + k] * N[k * Width + col];
        P[row * Width + col] = Pvalue;
    }
}

/* -----------------------------------------------------------------------
 * CPU reference
 * ----------------------------------------------------------------------- */
static void cpu_matmul(float* M, float* N, float* P, int W) {
    for (int r = 0; r < W; r++)
        for (int c = 0; c < W; c++) {
            float v = 0.0f;
            for (int k = 0; k < W; k++) v += M[r*W+k] * N[k*W+c];
            P[r*W+c] = v;
        }
}

/* -----------------------------------------------------------------------
 * Timing helper: run a kernel N_RUNS times and return average ms
 * ----------------------------------------------------------------------- */
#define N_RUNS 10
static float time_kernel(dim3 grid, dim3 block, size_t shmem,
                          float* M, float* N, float* P, int W, bool tiled) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    /* warm-up */
    if (tiled) matrixMulKernel<<<grid, block>>>(M, N, P, W);
    else       naiveMatMulKernel<<<grid, block>>>(M, N, P, W);
    cudaDeviceSynchronize();

    cudaEventRecord(t0);
    for (int r = 0; r < N_RUNS; r++) {
        if (tiled) matrixMulKernel<<<grid, block>>>(M, N, P, W);
        else       naiveMatMulKernel<<<grid, block>>>(M, N, P, W);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / N_RUNS;
}

int main() {
    /* Width must be a multiple of TILE_WIDTH for the tiled kernel */
    const int W = 1024;   /* 1024 × 1024 */
    int n = W * W;
    size_t bytes = n * sizeof(float);

    float* h_M   = (float*)malloc(bytes);
    float* h_N   = (float*)malloc(bytes);
    float* h_P   = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    float *d_M, *d_N, *d_P;

    srand(42);
    for (int i = 0; i < n; i++) {
        h_M[i] = (float)rand() / RAND_MAX;
        h_N[i] = (float)rand() / RAND_MAX;
    }

    cudaMalloc((void**)&d_M, bytes);
    cudaMalloc((void**)&d_N, bytes);
    cudaMalloc((void**)&d_P, bytes);
    cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(W / TILE_WIDTH, W / TILE_WIDTH, 1);

    /* ── Correctness check against naïve ──────────────────────────── */
    naiveMatMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, W);
    cudaMemcpy(h_ref, d_P, bytes, cudaMemcpyDeviceToHost);

    matrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, W);
    cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(h_P[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("Tiled vs naïve result: max error = %e  [%s]\n",
           max_err, max_err < 1e-2f ? "PASSED" : "FAILED");

    /* ── Timing comparison ─────────────────────────────────────────── */
    float ms_naive = time_kernel(dimGrid, dimBlock, 0, d_M, d_N, d_P, W, false);
    float ms_tiled = time_kernel(dimGrid, dimBlock, 0, d_M, d_N, d_P, W, true);

    /* FLOPs: 2 * W^3 (one multiply + one add per inner loop iteration) */
    double flops = 2.0 * W * W * W;
    double gflops_naive = (flops / ms_naive) / 1e6;
    double gflops_tiled = (flops / ms_tiled) / 1e6;

    /* Global memory bytes accessed (naïve: each element accessed W times) */
    double bytes_naive = 2.0 * n * sizeof(float) * W;  /* W reads per element */
    double arith_naive = flops / bytes_naive;
    double arith_tiled = arith_naive * TILE_WIDTH;       /* TILE_WIDTH× reduction */

    printf("\nTILE_WIDTH = %d,  Matrix = %d×%d\n", TILE_WIDTH, W, W);
    printf("%-20s  %8s  %10s  %12s\n", "Kernel", "ms", "GFLOPS", "FLOP/B (arith. intensity)");
    printf("%-20s  %8.2f  %10.1f  %.3f\n",
           "Naïve (Ch3)",   ms_naive, gflops_naive, arith_naive);
    printf("%-20s  %8.2f  %10.1f  %.3f\n",
           "Tiled (Ch5)",   ms_tiled, gflops_tiled, arith_tiled);
    printf("Speedup: %.2fx\n", ms_naive / ms_tiled);
    printf("\nTheoretical: tiling reduces global memory traffic by %dx\n", TILE_WIDTH);
    printf("(Section 5.4: 'global memory accesses reduced by factor of TILE_WIDTH')\n");

    free(h_M); free(h_N); free(h_P); free(h_ref);
    cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
    return 0;
}
