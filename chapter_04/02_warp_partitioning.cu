/*
 * Chapter 4 — Section 4.4: Warps and SIMD Hardware
 *             Figures 4.6, 4.7
 *
 * A warp is the unit of thread scheduling in CUDA SMs.  On all current
 * hardware the warp size is 32 threads.  When a block is assigned to an SM,
 * it is divided into warps of 32 consecutive threads (Section 4.4).
 *
 * 1-D blocks: warp n = threads [32n … 32n+31]
 * 2-D blocks: threads are first linearised in row-major order:
 *   linear_id = threadIdx.y * blockDim.x + threadIdx.x
 *   then grouped into warps of 32 (Figure 4.7)
 * 3-D blocks: similarly linearised with z varying slowest, then y, then x.
 *
 * If a block size is not a multiple of 32, the last warp is padded with
 * inactive threads (Section 4.4).
 *
 * This file demonstrates:
 *   A) 1-D block → warp assignment
 *   B) 2-D block linearisation into warps (Figure 4.7)
 *   C) Padding: a 48-thread block → 2 warps, second warp 50% inactive
 *
 * Build:
 *   nvcc -O2 -arch=sm_89 -o warp_partitioning 02_warp_partitioning.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

/* -----------------------------------------------------------------------
 * Kernel: each thread records its warp-within-block ID.
 * warp_id = linear_thread_id / WARP_SIZE
 * For 1-D blocks: linear_thread_id = threadIdx.x
 * For 2-D blocks: linear_thread_id = threadIdx.y * blockDim.x + threadIdx.x
 * ----------------------------------------------------------------------- */
__global__
void recordWarpIds(int* warp_ids, int* linear_ids) {
    int linear = threadIdx.z * (blockDim.y * blockDim.x)
               + threadIdx.y * blockDim.x
               + threadIdx.x;
    int warp_in_block = linear / WARP_SIZE;
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;  /* 1D only */

    /* Store for the host to verify */
    warp_ids[blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) + linear]
        = warp_in_block;
    linear_ids[blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) + linear]
        = linear;
    (void)global_thread;
}

/* 2-D specialised version so we can launch a 2-D block cleanly */
__global__
void recordWarpIds2D(int* warp_ids, int* linear_ids,
                     int total_x, int total_y) {
    int linear = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_in_block = linear / WARP_SIZE;

    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_y < total_y && global_x < total_x) {
        int global_linear = global_y * total_x + global_x;
        warp_ids[global_linear]   = warp_in_block;
        linear_ids[global_linear] = linear;
    }
}

/* -----------------------------------------------------------------------
 * Host helper: print a warp assignment table for a small block
 * ----------------------------------------------------------------------- */
static void print_1d_warp_table(int block_size) {
    printf("\n  1-D block of %d threads → %d warp(s):\n",
           block_size, (block_size + WARP_SIZE - 1) / WARP_SIZE);
    printf("  Thread  0.. %d  → warp 0\n", WARP_SIZE - 1 < block_size - 1
                                             ? WARP_SIZE - 1 : block_size - 1);
    for (int w = 1; w < (block_size + WARP_SIZE - 1) / WARP_SIZE; w++) {
        int lo = w * WARP_SIZE;
        int hi = lo + WARP_SIZE - 1;
        if (hi >= block_size) {
            printf("  Thread %2d..%2d  → warp %d  (%d inactive padding threads)\n",
                   lo, block_size - 1, w, hi - (block_size - 1));
        } else {
            printf("  Thread %2d..%2d  → warp %d\n", lo, hi, w);
        }
    }
}

int main() {
    /* ── A) 1-D block warp assignments ──────────────────────────── */
    printf("=== A) 1-D block warp assignments (Section 4.4) ===");
    print_1d_warp_table(256);
    print_1d_warp_table(128);

    /* Padding case: 48 threads → 2 warps, second has 16 inactive */
    print_1d_warp_table(48);

    /* ── B) 2-D block → linearisation → warps (Figure 4.7) ──────── */
    printf("\n=== B) 2-D block linearisation into warps (Figure 4.7) ===\n");
    printf("\nA 4×4 block (16 threads) linearises as:\n");
    printf("  row-major: T(y,x) → linear = y*4 + x\n");
    for (int y = 0; y < 4; y++) {
        printf("  y=%d: ", y);
        for (int x = 0; x < 4; x++) {
            int linear = y * 4 + x;
            printf("T(%d,%d)→%2d  ", y, x, linear);
        }
        printf("\n");
    }
    printf("All 16 threads → warp 0 (padded to 32 with 16 inactive)\n");

    printf("\nA 4×8 block (32 threads) → exactly warp 0\n");
    printf("  T(0,0)–T(0,7)  linear 0–7\n");
    printf("  T(1,0)–T(1,7)  linear 8–15\n");
    printf("  T(2,0)–T(2,7)  linear 16–23\n");
    printf("  T(3,0)–T(3,7)  linear 24–31  → warp 0 (full, no padding)\n");

    printf("\nA 4×16 block (64 threads) → warp 0 (threads 0-31) and warp 1 (32-63):\n");
    printf("  warp 0: rows y=0,1  (linear 0-31)\n");
    printf("  warp 1: rows y=2,3  (linear 32-63)\n");

    /* ── C) GPU verification: 2-D block warp IDs ────────────────── */
    printf("\n=== C) GPU verification of 2-D warp IDs ===\n");
    {
        /* 4×16 block: 2 warps.  All of y=0,1 should be warp 0;
         * y=2,3 should be warp 1. */
        const int bx = 16, by = 4;
        const int n = bx * by;
        int *h_wids = (int*)malloc(n * sizeof(int));
        int *h_lids = (int*)malloc(n * sizeof(int));
        int *d_wids, *d_lids;
        cudaMalloc((void**)&d_wids, n * sizeof(int));
        cudaMalloc((void**)&d_lids, n * sizeof(int));

        dim3 dimBlock(bx, by, 1);
        dim3 dimGrid(1, 1, 1);
        recordWarpIds2D<<<dimGrid, dimBlock>>>(d_wids, d_lids, bx, by);
        cudaDeviceSynchronize();
        cudaMemcpy(h_wids, d_wids, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_lids, d_lids, n * sizeof(int), cudaMemcpyDeviceToHost);

        int ok = 1;
        for (int y = 0; y < by; y++) {
            for (int x = 0; x < bx; x++) {
                int linear = y * bx + x;
                int expected_warp = linear / WARP_SIZE;
                if (h_wids[y * bx + x] != expected_warp) ok = 0;
            }
        }
        printf("4×16 block: warp IDs %s\n", ok ? "CORRECT" : "WRONG");

        /* Print the warp map */
        printf("\nWarp map for 4×16 block (W = warp_id):\n     x:");
        for (int x = 0; x < bx; x++) printf(" %2d", x);
        printf("\n");
        for (int y = 0; y < by; y++) {
            printf("  y=%d: ", y);
            for (int x = 0; x < bx; x++)
                printf(" W%d", h_wids[y * bx + x]);
            printf("\n");
        }

        free(h_wids); free(h_lids);
        cudaFree(d_wids); cudaFree(d_lids);
    }

    /* ── SM capacity example from Section 4.4 ───────────────────── */
    printf("\n=== SM warp capacity (A100 example from Section 4.4) ===\n");
    {
        int block_size = 256;
        int warps_per_block = block_size / WARP_SIZE;  /* 8 */
        int max_blocks_per_sm = 32;   /* A100: 32 blocks/SM */
        int max_warps_per_sm  = 64;   /* A100: 2048 threads / 32 = 64 warps */

        int blocks_limited_by_warps = max_warps_per_sm / warps_per_block;
        int actual_blocks = blocks_limited_by_warps < max_blocks_per_sm
                          ? blocks_limited_by_warps : max_blocks_per_sm;

        printf("  Block size:          %d threads = %d warps\n",
               block_size, warps_per_block);
        printf("  A100 SM limits:      %d blocks, %d warps (2048 threads)\n",
               max_blocks_per_sm, max_warps_per_sm);
        printf("  Blocks limited by warps: %d\n", blocks_limited_by_warps);
        printf("  Actual blocks/SM:    %d\n", actual_blocks);
        printf("  Warps active/SM:     %d / %d\n",
               actual_blocks * warps_per_block, max_warps_per_sm);
    }

    return 0;
}
