/*
 * Chapter 6 — Section 6.3: Thread Coarsening
 *             Figure 6.13: coarsened tiled matrix multiplication kernel
 *             Figure 6.12: memory access pattern for two adjacent output tiles
 *
 * THE REDUNDANCY PROBLEM (Section 6.3):
 *   In the tiled kernel from Chapter 5, each output tile is processed by a
 *   separate thread block.  When two horizontally adjacent output tiles (same
 *   rows of P, adjacent column ranges) are computed:
 *   - They need DIFFERENT N tiles (different column ranges).
 *   - They need THE SAME M tiles (same row range).
 *   If these blocks run on different SMs they each load their own copy of
 *   the M tiles — redundant global memory traffic.
 *
 * THREAD COARSENING FIX (Figure 6.13):
 *   Assign ONE thread block to process COARSE_FACTOR adjacent output tiles.
 *   Each thread is responsible for COARSE_FACTOR output elements (one per tile).
 *   The M tile is loaded ONCE and reused for all COARSE_FACTOR N tiles.
 *
 *   Changes from the Figure 5.9 kernel:
 *     • colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx
 *       (block covers COARSE_FACTOR times the normal column range)
 *     • float Pvalue[COARSE_FACTOR]  (one accumulator per output element)
 *     • M tile loaded ONCE per (ph, outer loop) — no change to M load
 *     • Inner loop (c) over COARSE_FACTOR: loads one N tile and accumulates
 *     • Final loop stores COARSE_FACTOR results back to P
 *
 * PITFALLS (Section 6.3):
 *   1. Don't coarsen when there's no price for parallelism (e.g. vecAdd).
 *   2. Too large a COARSE_FACTOR → not enough parallelism exposed to hardware.
 *   3. Extra Pvalue[] registers may reduce occupancy.
 *
 * Build:
 *   nvcc -O2 -arch=sm_89 -o thread_coarsening 03_thread_coarsening_matmul.cu -lm
 *   Override factor: nvcc -DCOARSE_FACTOR=2 ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef TILE_WIDTH
#define TILE_WIDTH   32
#endif
#ifndef COARSE_FACTOR
#define COARSE_FACTOR 4
#endif

#define N_RUNS 10

/* -----------------------------------------------------------------------
 * Reference: plain tiled kernel from Chapter 5 (Figure 5.9)
 * Width must be a multiple of TILE_WIDTH.
 * ----------------------------------------------------------------------- */
__global__
void tiledKernel(const float* M, const float* N, float* P, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx=blockIdx.x, by=blockIdx.y, tx=threadIdx.x, ty=threadIdx.y;
    int Row = by*TILE_WIDTH + ty;
    int Col = bx*TILE_WIDTH + tx;
    float Pvalue = 0.0f;

    for (int ph=0; ph<width/TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[Row*width + ph*TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*width + Col];
        __syncthreads();
        for (int k=0; k<TILE_WIDTH; ++k) Pvalue += Mds[ty][k]*Nds[k][tx];
        __syncthreads();
    }
    P[Row*width + Col] = Pvalue;
}

/* -----------------------------------------------------------------------
 * Thread-coarsened tiled kernel — Figure 6.13
 *
 * Each thread block covers COARSE_FACTOR×TILE_WIDTH columns of P.
 * Each thread accumulates COARSE_FACTOR dot products in Pvalue[].
 *
 * Grid dimensions change: gridDim.x = width / (TILE_WIDTH * COARSE_FACTOR)
 * Block dimensions:       TILE_WIDTH × TILE_WIDTH (same as before)
 * ----------------------------------------------------------------------- */
__global__
void coarsenedKernel(const float* M, const float* N, float* P, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    /* Row index of P element this thread works on */
    int row = by * TILE_WIDTH + ty;

    /* Starting column for the COARSE_FACTOR output elements this thread owns.
     * Each block covers TILE_WIDTH*COARSE_FACTOR consecutive columns.       */
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    /* One accumulator per output element (Figure 6.13 lines 15-19) */
    float Pvalue[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; ++c) Pvalue[c] = 0.0f;

    /* Loop over phases — one M tile + COARSE_FACTOR N tiles per phase */
    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {

        /* Load M tile ONCE — shared by all COARSE_FACTOR coarsening iterations
         * (Figure 6.13 line 24-25) */
        Mds[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];

        /* For each of the COARSE_FACTOR output columns this block is responsible
         * for, load the corresponding N tile and accumulate (Figure 6.13 lines 27-40) */
        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int col = colStart + c * TILE_WIDTH;

            /* Load N tile for this output column section */
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
            __syncthreads();

            /* Accumulate dot product for output element at (row, col) */
            for (int k = 0; k < TILE_WIDTH; ++k)
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];

            __syncthreads();
        }
    }

    /* Write results — one store per coarsened output element
     * (Figure 6.13 lines 44-47) */
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int col = colStart + c * TILE_WIDTH;
        P[row * width + col] = Pvalue[c];
    }
}

/* -----------------------------------------------------------------------
 * Helpers
 * ----------------------------------------------------------------------- */
static void cpu_matmul(const float* M, const float* N, float* P, int W) {
    for (int r=0; r<W; r++)
        for (int c=0; c<W; c++) {
            float v=0.0f;
            for (int k=0; k<W; k++) v += M[r*W+k]*N[k*W+c];
            P[r*W+c]=v;
        }
}

static float time_kernel(dim3 g, dim3 b, bool coarsened,
                          const float* M, const float* N, float* P, int W) {
    if (coarsened) coarsenedKernel<<<g,b>>>(M,N,P,W);
    else           tiledKernel<<<g,b>>>(M,N,P,W);
    cudaDeviceSynchronize();
    cudaEvent_t t0,t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r=0; r<N_RUNS; r++) {
        if (coarsened) coarsenedKernel<<<g,b>>>(M,N,P,W);
        else           tiledKernel<<<g,b>>>(M,N,P,W);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms,t0,t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms/N_RUNS;
}

int main() {
    /* Width must be multiple of TILE_WIDTH * COARSE_FACTOR */
    const int W = TILE_WIDTH * COARSE_FACTOR * 8;  /* e.g. 32*4*8 = 1024 */
    int n = W*W;
    size_t bytes = n*sizeof(float);

    float *h_M=(float*)malloc(bytes), *h_N=(float*)malloc(bytes);
    float *h_P=(float*)malloc(bytes), *h_ref=(float*)malloc(bytes);
    float *d_M, *d_N, *d_P;

    srand(31);
    for (int i=0; i<n; i++) { h_M[i]=(float)rand()/RAND_MAX; h_N[i]=(float)rand()/RAND_MAX; }

    cudaMalloc((void**)&d_M, bytes);
    cudaMalloc((void**)&d_N, bytes);
    cudaMalloc((void**)&d_P, bytes);
    cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);

    /* Grid for tiled kernel: one block per TILE_WIDTH×TILE_WIDTH output tile */
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid_tiled(W/TILE_WIDTH, W/TILE_WIDTH, 1);

    /* Grid for coarsened kernel: each block handles COARSE_FACTOR tiles horizontally */
    dim3 dimGrid_coarsened(W / (TILE_WIDTH * COARSE_FACTOR), W / TILE_WIDTH, 1);

    /* ── Correctness ─────────────────────────────────────────────── */
    tiledKernel<<<dimGrid_tiled, dimBlock>>>(d_M, d_N, d_P, W);
    cudaMemcpy(h_ref, d_P, bytes, cudaMemcpyDeviceToHost);

    coarsenedKernel<<<dimGrid_coarsened, dimBlock>>>(d_M, d_N, d_P, W);
    cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i=0; i<n; i++) { float e=fabsf(h_P[i]-h_ref[i]); if(e>max_err) max_err=e; }
    printf("Coarsened vs tiled: max error = %e [%s]\n\n",
           max_err, max_err<1e-2f?"PASSED":"FAILED");

    /* ── Timing ──────────────────────────────────────────────────── */
    float ms_tiled    = time_kernel(dimGrid_tiled,    dimBlock, false, d_M, d_N, d_P, W);
    float ms_coarsened= time_kernel(dimGrid_coarsened, dimBlock, true,  d_M, d_N, d_P, W);

    double flops = 2.0*W*W*W;
    printf("TILE_WIDTH=%d  COARSE_FACTOR=%d  W=%d\n", TILE_WIDTH, COARSE_FACTOR, W);
    printf("%-25s %7.2f ms  %6.1f GFLOPS\n",
           "Tiled (Ch5)",    ms_tiled,    (flops/ms_tiled)/1e6);
    printf("%-25s %7.2f ms  %6.1f GFLOPS\n",
           "Coarsened (Ch6)", ms_coarsened,(flops/ms_coarsened)/1e6);
    printf("Speedup: %.2fx\n\n", ms_tiled/ms_coarsened);

    printf("Grid dimensions:\n");
    printf("  Tiled:     (%d, %d) blocks of (%d, %d) threads\n",
           dimGrid_tiled.x, dimGrid_tiled.y, TILE_WIDTH, TILE_WIDTH);
    printf("  Coarsened: (%d, %d) blocks of (%d, %d) threads\n",
           dimGrid_coarsened.x, dimGrid_coarsened.y, TILE_WIDTH, TILE_WIDTH);
    printf("  Coarsened uses %d× fewer blocks — M tiles loaded %d× less\n",
           COARSE_FACTOR, COARSE_FACTOR);

    free(h_M); free(h_N); free(h_P); free(h_ref);
    cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
    return 0;
}
