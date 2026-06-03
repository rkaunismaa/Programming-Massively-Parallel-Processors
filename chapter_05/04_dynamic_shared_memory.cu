/*
 * Chapter 5 — Section 5.6: Dynamic Shared Memory (Figure 5.14)
 *
 * MOTIVATION (Section 5.6):
 *   The static tile declaration in Figure 5.9:
 *     __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
 *     __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
 *   hardwires TILE_WIDTH at compile time.  If we want to tune the tile
 *   size at runtime — e.g. to adapt to the hardware's shared memory
 *   capacity or to pick the best size for a given matrix size — we need
 *   dynamic shared memory.
 *
 * DYNAMIC SHARED MEMORY SYNTAX (Section 5.6):
 *   Instead of a fixed-size array, declare one extern unsized array:
 *     extern __shared__ float Mds_Nds[];
 *   Then manually partition it into sections for Mds and Nds:
 *     float* Mds = (float*) Mds_Nds;
 *     float* Nds = (float*) Mds_Nds + Mds_sz;   // Mds_sz floats
 *   And pass the total byte size as the third launch parameter:
 *     <<<dimGrid, dimBlock, total_shared_bytes>>>(...)
 *
 * ACCESSING THE MANUALLY PARTITIONED ARRAYS (Figure 5.14):
 *   Static:  Mds[ty][tx]  → Dynamic: Mds[ty * TILE_WIDTH + tx]
 *   Static:  Nds[k][tx]   → Dynamic: Nds[k  * TILE_WIDTH + tx]
 *
 * This file:
 *   A) Shows the dynamic shared memory kernel (Figure 5.14 style)
 *   B) Demonstrates host-side tile-size selection based on device properties
 *   C) Compares static vs dynamic kernel results for correctness
 *
 * Build:
 *   nvcc -O2 -arch=sm_89 -o dynamic_shmem 04_dynamic_shared_memory.cu -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define STATIC_TILE 16   /* used only for the static reference kernel */

/* -----------------------------------------------------------------------
 * Static tiled kernel (Figure 5.9) — reference for correctness comparison
 * ----------------------------------------------------------------------- */
__global__
void matrixMulStatic(float* M, float* N, float* P, int Width) {
    __shared__ float Mds[STATIC_TILE][STATIC_TILE];
    __shared__ float Nds[STATIC_TILE][STATIC_TILE];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * STATIC_TILE + ty;
    int Col = bx * STATIC_TILE + tx;
    float Pvalue = 0.0f;

    for (int ph = 0; ph < Width / STATIC_TILE; ++ph) {
        Mds[ty][tx] = M[Row * Width + ph * STATIC_TILE + tx];
        Nds[ty][tx] = N[(ph * STATIC_TILE + ty) * Width + Col];
        __syncthreads();
        for (int k = 0; k < STATIC_TILE; ++k)
            Pvalue += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
    }
    P[Row * Width + Col] = Pvalue;
}

/* -----------------------------------------------------------------------
 * Dynamic shared memory kernel — Figure 5.14
 *
 * tile_width: the tile dimension (passed at runtime)
 * Mds_sz:     number of floats in the Mds section  (= tile_width²)
 * Nds_sz:     number of floats in the Nds section  (= tile_width²)
 *
 * Total shared bytes allocated by host: (Mds_sz + Nds_sz) * sizeof(float)
 * ----------------------------------------------------------------------- */
__global__
void matrixMulDynamic(float* M, float* N, float* P, int Width,
                      int tile_width, unsigned Mds_sz, unsigned Nds_sz) {
    /* Single extern unsized shared array — split manually (Figure 5.14) */
    extern __shared__ float Mds_Nds[];
    float* Mds = (float*) Mds_Nds;
    float* Nds = (float*) Mds_Nds + Mds_sz;   /* Nds starts after Mds */

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * tile_width + ty;
    int Col = bx * tile_width + tx;
    float Pvalue = 0.0f;

    /* Boundary-checked phase loop (handles non-multiples too) */
    int num_phases = (Width + tile_width - 1) / tile_width;

    for (int ph = 0; ph < num_phases; ++ph) {
        /* Linearised 2D index: [ty][tx] → ty * tile_width + tx */
        if (Row < Width && (ph * tile_width + tx) < Width)
            Mds[ty * tile_width + tx] = M[Row * Width + ph * tile_width + tx];
        else
            Mds[ty * tile_width + tx] = 0.0f;

        if ((ph * tile_width + ty) < Width && Col < Width)
            Nds[ty * tile_width + tx] = N[(ph * tile_width + ty) * Width + Col];
        else
            Nds[ty * tile_width + tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < tile_width; ++k)
            Pvalue += Mds[ty * tile_width + k] * Nds[k * tile_width + tx];

        __syncthreads();
    }

    if (Row < Width && Col < Width)
        P[Row * Width + Col] = Pvalue;
}

/* -----------------------------------------------------------------------
 * Host: select tile size at runtime based on device shared memory
 * ----------------------------------------------------------------------- */
static int select_tile_width(int max_shared_bytes, int max_threads_per_block) {
    /* Find the largest power-of-2 tile that:
     *   1. Fits in shared memory: 2 * tile² * 4 bytes ≤ max_shared_bytes
     *   2. Fits in a block: tile² ≤ max_threads_per_block                */
    for (int tw = 32; tw >= 2; tw /= 2) {
        int shmem = 2 * tw * tw * (int)sizeof(float);
        if (shmem <= max_shared_bytes && tw * tw <= max_threads_per_block)
            return tw;
    }
    return 2;
}

int main() {
    /* ── Query device to pick optimal tile width at runtime ─────── */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int tile_w = select_tile_width((int)prop.sharedMemPerBlock,
                                   prop.maxThreadsPerBlock);

    printf("Device: %s\n", prop.name);
    printf("  sharedMemPerBlock:  %zu bytes\n", prop.sharedMemPerBlock);
    printf("  maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("  Selected tile_width at runtime: %d\n\n", tile_w);

    /* ── Prepare matrices ──────────────────────────────────────────
     * Use a Width that is an exact multiple of STATIC_TILE=16 so both
     * the static and dynamic kernels can be compared                  */
    const int W = 512;
    int n = W * W;
    size_t bytes = n * sizeof(float);
    float *h_M   = (float*)malloc(bytes);
    float *h_N   = (float*)malloc(bytes);
    float *h_P   = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    float *d_M, *d_N, *d_P;

    srand(99);
    for (int i = 0; i < n; i++) {
        h_M[i] = (float)rand() / RAND_MAX;
        h_N[i] = (float)rand() / RAND_MAX;
    }
    cudaMalloc((void**)&d_M, bytes);
    cudaMalloc((void**)&d_N, bytes);
    cudaMalloc((void**)&d_P, bytes);
    cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);

    /* ── Static kernel (baseline) ──────────────────────────────── */
    dim3 dimBlock_s(STATIC_TILE, STATIC_TILE, 1);
    dim3 dimGrid_s(W / STATIC_TILE, W / STATIC_TILE, 1);
    matrixMulStatic<<<dimGrid_s, dimBlock_s>>>(d_M, d_N, d_P, W);
    cudaMemcpy(h_ref, d_P, bytes, cudaMemcpyDeviceToHost);
    printf("Static  kernel (TILE=%d): complete\n", STATIC_TILE);

    /* ── Dynamic kernel with runtime-selected tile ─────────────── */
    unsigned Mds_sz = tile_w * tile_w;   /* floats in Mds section */
    unsigned Nds_sz = tile_w * tile_w;   /* floats in Nds section */
    size_t shmem_bytes = (Mds_sz + Nds_sz) * sizeof(float);

    dim3 dimBlock_d(tile_w, tile_w, 1);
    dim3 dimGrid_d((W + tile_w - 1) / tile_w,
                   (W + tile_w - 1) / tile_w, 1);

    /* Third launch parameter: dynamic shared memory bytes */
    matrixMulDynamic<<<dimGrid_d, dimBlock_d, shmem_bytes>>>(
        d_M, d_N, d_P, W, tile_w, Mds_sz, Nds_sz);
    cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);
    printf("Dynamic kernel (TILE=%d): complete  (shared=%zu bytes)\n\n",
           tile_w, shmem_bytes);

    /* ── Correctness ────────────────────────────────────────────── */
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(h_P[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("Dynamic vs static result: max error = %e  [%s]\n",
           max_err, max_err < 1e-2f ? "PASSED" : "FAILED");

    /* ── Show key syntax differences ───────────────────────────── */
    printf("\nStatic vs dynamic shared memory syntax:\n");
    printf("  Static:  __shared__ float Mds[%d][%d];\n",
           STATIC_TILE, STATIC_TILE);
    printf("           Mds[ty][tx]  (2D indexing)\n");
    printf("  Dynamic: extern __shared__ float Mds_Nds[];\n");
    printf("           float* Mds = Mds_Nds;\n");
    printf("           Mds[ty * tile_w + tx]  (linearised 1D indexing)\n");
    printf("  Launch:  kernel<<<grid, block, (Mds_sz+Nds_sz)*sizeof(float)>>>(...)\n");

    free(h_M); free(h_N); free(h_P); free(h_ref);
    cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
    return 0;
}
