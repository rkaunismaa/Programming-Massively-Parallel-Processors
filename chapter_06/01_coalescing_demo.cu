/*
 * Chapter 6 — Section 6.1: Memory Coalescing
 *             Figures 6.1–6.3
 *
 * DRAM BURSTS (Section 6.1):
 *   Each time a DRAM location is accessed, a range of consecutive locations
 *   is fetched together as a burst.  When consecutive threads in a warp
 *   access consecutive addresses, their accesses can be merged ("coalesced")
 *   into a single burst transaction.  When they access addresses that are
 *   far apart, each thread requires its own transaction — a severe bandwidth
 *   penalty.
 *
 * RULE: a warp's global memory accesses coalesce when
 *   thread 0 accesses X, thread 1 accesses X+1, thread 2 accesses X+2, …
 *   (all accesses fall in the same aligned cache line / DRAM burst window)
 *
 * This file demonstrates three access patterns using a Width×Width matrix
 * stored in row-major order:
 *
 *   A) Row read (Figure 6.2): thread i reads M[row][i] — stride 1 → COALESCED
 *   B) Column read (Figure 6.3): thread i reads M[i][col] — stride Width → UNCOALESCED
 *   C) The matmul access pattern: which of M and N are coalesced in the
 *      naïve kernel vs when N is stored in column-major layout.
 *
 * Build:
 *   nvcc -O2 -arch=sm_89 -o coalescing_demo 01_coalescing_demo.cu
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH  4096
#define BLOCK  32
#define N_RUNS 50

/* -----------------------------------------------------------------------
 * A) Coalesced: thread tx reads M[row][tx] — stride-1 access
 *    Each warp reads a contiguous segment of one row.
 *    All 32 accesses land in the same (or adjacent) cache lines.
 * ----------------------------------------------------------------------- */
__global__
void rowReadKernel(const float* __restrict__ M, float* out, int W) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  /* varies in warp */
    int row = blockIdx.y * blockDim.y + threadIdx.y;  /* fixed in warp  */
    if (row < W && col < W)
        out[row * W + col] = M[row * W + col];  /* stride 1 — coalesced */
}

/* -----------------------------------------------------------------------
 * B) Uncoalesced: thread tx reads M[tx][col] — stride-Width access
 *    Each warp reads one element from each of 32 different rows.
 *    Consecutive threads access addresses W floats (4W bytes) apart.
 * ----------------------------------------------------------------------- */
__global__
void colReadKernel(const float* __restrict__ M, float* out, int W) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  /* varies in warp */
    int col = blockIdx.y * blockDim.y + threadIdx.y;  /* fixed in warp  */
    if (row < W && col < W)
        out[col * W + row] = M[row * W + col];  /* stride W — uncoalesced */
}

/* -----------------------------------------------------------------------
 * Timing helper — returns achieved memory bandwidth in GB/s
 * data_bytes: total bytes read + written by the kernel
 * ----------------------------------------------------------------------- */
static double bw_GBs(cudaEvent_t t0, cudaEvent_t t1, double data_bytes) {
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    return (data_bytes * N_RUNS) / (ms * 1e-3) / 1e9;
}

int main() {
    size_t bytes = (size_t)WIDTH * WIDTH * sizeof(float);
    float *d_M, *d_out;
    cudaMalloc((void**)&d_M,  bytes);
    cudaMalloc((void**)&d_out, bytes);
    cudaMemset(d_M, 1, bytes);

    dim3 dimBlock(BLOCK, BLOCK, 1);
    dim3 dimGrid(WIDTH / BLOCK, WIDTH / BLOCK, 1);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    /* ── Warm-up ──────────────────────────────────────────────────── */
    rowReadKernel<<<dimGrid, dimBlock>>>(d_M, d_out, WIDTH);
    cudaDeviceSynchronize();

    /* ── A) Coalesced row read ─────────────────────────────────────── */
    cudaEventRecord(t0);
    for (int r = 0; r < N_RUNS; r++)
        rowReadKernel<<<dimGrid, dimBlock>>>(d_M, d_out, WIDTH);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    double bw_row = bw_GBs(t0, t1, 2.0 * bytes);  /* 1 read + 1 write */

    /* ── B) Uncoalesced column read ───────────────────────────────── */
    cudaEventRecord(t0);
    for (int r = 0; r < N_RUNS; r++)
        colReadKernel<<<dimGrid, dimBlock>>>(d_M, d_out, WIDTH);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    double bw_col = bw_GBs(t0, t1, 2.0 * bytes);

    printf("=== Memory Coalescing Demo (Section 6.1) ===\n");
    printf("Matrix: %d×%d  (%.1f MB)\n", WIDTH, WIDTH, bytes / 1e6);
    printf("\n%-30s  %8.1f GB/s\n", "A) Row read   (COALESCED)",   bw_row);
    printf("%-30s  %8.1f GB/s\n", "B) Column read (UNCOALESCED)", bw_col);
    printf("Coalesced is %.1fx faster\n\n", bw_row / bw_col);

    printf("Explanation (Figures 6.2–6.3):\n");
    printf("  Row read:    thread tx accesses M[row*W + tx]  — stride 1\n");
    printf("               32 consecutive floats → 1 DRAM burst\n");
    printf("  Column read: thread tx accesses M[tx*W + col]  — stride W=%d\n", WIDTH);
    printf("               32 floats, each %d bytes apart → up to 32 bursts\n\n",
           WIDTH * 4);

    printf("Matmul access analysis (Figure 6.2 code):\n");
    printf("  N[k*Width + col], col = blockIdx.x*blockDim.x + threadIdx.x\n");
    printf("  Consecutive threadIdx.x → consecutive col → stride-1 → COALESCED ✓\n");
    printf("  M[row*Width + k], row = blockIdx.y*blockDim.y + threadIdx.y\n");
    printf("  Same threadIdx.y in a warp → same row → broadcast → OK ✓\n");
    printf("\n  Column-major N (Figure 6.3):\n");
    printf("  N_col[col*Width + k] — consecutive col → stride Width → UNCOALESCED ✗\n");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_M); cudaFree(d_out);
    return 0;
}
