/*
 * Chapter 6 — Section 6.1: Corner Turning
 *             Figure 6.4: coalescing accesses to a column-major matrix
 *
 * PROBLEM:
 *   In C = A × B, if B is stored in column-major layout (a common case
 *   when working with the transpose of a row-major matrix), loading the
 *   B input tile in the same way as Chapter 5 produces UNCOALESCED accesses.
 *
 *   Column-major storage: B[row][col] is at B_cm[col * Height + row].
 *   Loading Bds[ty][tx] = B_cm[(ph*TILE + ty) * Width + col]
 *     where col = bx*TILE + tx:
 *     → consecutive threadIdx.x → consecutive col → stride Height → UNCOALESCED ✗
 *
 * THE CORNER TURNING FIX (Figure 6.4B):
 *   Swap tx and ty roles when computing the load index for B:
 *     Bds[tx][ty] = B_cm[col_for_tx * Height + row_element_for_ty]
 *
 *   In column-major layout, elements in the SAME COLUMN are adjacent.
 *   By assigning each thread to load a different ROW of the same column
 *   (instead of a different COLUMN of the same row), consecutive threads
 *   access consecutive addresses → COALESCED ✓.
 *
 *   After loading, Bds holds B transposed in shared memory.  The inner
 *   product loop reads Bds[tx][k] instead of Bds[k][tx] — equivalent
 *   computation, but the shared memory access pattern is now transposed.
 *
 * This file:
 *   A) Uncoalesced matmul — loads B tiles naïvely (column-major B, no fix)
 *   B) Corner-turned matmul — coalesced B tile loads
 *   Verifies both give the same result, then times them.
 *
 * Build:
 *   nvcc -O2 -arch=sm_89 -o corner_turning 02_corner_turning.cu -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE  16
#define N_RUNS 10

/* -----------------------------------------------------------------------
 * Helper: convert a row-major matrix to column-major in-place on host
 * ----------------------------------------------------------------------- */
static void to_col_major(float* dst, const float* src, int W) {
    for (int r = 0; r < W; r++)
        for (int c = 0; c < W; c++)
            dst[c * W + r] = src[r * W + c];
}

/* -----------------------------------------------------------------------
 * A) Naïve tiled matmul where N is column-major — UNCOALESCED B loads
 *
 *    Loading: Nds[ty][tx] = N_cm[(ph*TILE+ty)*Width + Col]
 *    Col = bx*TILE + tx → consecutive tx → stride Width in column-major → ✗
 * ----------------------------------------------------------------------- */
__global__
void matmulColMajorNaive(const float* M, const float* N_cm,
                          float* P, int Width) {
    __shared__ float Mds[TILE][TILE];
    __shared__ float Nds[TILE][TILE];

    int bx = blockIdx.x,  by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = by * TILE + ty;
    int Col = bx * TILE + tx;
    float Pvalue = 0.0f;

    for (int ph = 0; ph < Width / TILE; ++ph) {
        /* M tile (row-major, stride-1) — COALESCED ✓ */
        Mds[ty][tx] = M[Row * Width + ph * TILE + tx];

        /* N tile (column-major, stride Width) — UNCOALESCED ✗
         * N_cm[row][col] = N_cm[col * Width + row]
         * row = ph*TILE + ty, col = Col */
        Nds[ty][tx] = N_cm[Col * Width + ph * TILE + ty];  /* uncoalesced */
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            Pvalue += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
    }
    P[Row * Width + Col] = Pvalue;
}

/* -----------------------------------------------------------------------
 * B) Corner-turned tiled matmul — COALESCED B loads (Figure 6.4B)
 *
 *    Each thread loads a different ROW of the same column segment of N,
 *    which in column-major layout are consecutive addresses.
 *
 *    Load: Nds[tx][ty] = N_cm[(Col_for_tx) * Width + ph*TILE + ty]
 *    where Col_for_tx = bx*TILE + tx → each tx picks a COLUMN
 *    and ty selects which ROW within that column.
 *    In column-major: column c starts at N_cm[c*Width], so
 *    N_cm[(bx*TILE+tx)*Width + ph*TILE+ty] — consecutive tx → stride Width → ✗?
 *
 *    Wait — the key is that consecutive THREADS load consecutive ROWS of
 *    the SAME column → addresses are 1 apart in column-major → COALESCED ✓.
 *    We achieve this by assigning: thread (ty,tx) loads N_cm at row=(ph*TILE+tx),
 *    col=(bx*TILE+ty).
 *
 *    Addresses: N_cm[(bx*TILE+ty)*Width + ph*TILE+tx]
 *    Consecutive tx → ph*TILE+tx increments by 1 → stride 1 → COALESCED ✓
 *
 *    This stores N transposed in Nds: Nds[tx][ty] holds N[ph*TILE+tx][bx*TILE+ty].
 *    The dot product then reads Nds[k][ty] (not Nds[ty][k]).
 * ----------------------------------------------------------------------- */
__global__
void matmulColMajorCornerTurned(const float* M, const float* N_cm,
                                  float* P, int Width) {
    __shared__ float Mds[TILE][TILE];
    __shared__ float Nds[TILE][TILE];

    int bx = blockIdx.x,  by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = by * TILE + ty;
    int Col = bx * TILE + tx;
    float Pvalue = 0.0f;

    for (int ph = 0; ph < Width / TILE; ++ph) {
        /* M tile (row-major) — COALESCED ✓ */
        Mds[ty][tx] = M[Row * Width + ph * TILE + tx];

        /* N tile with corner turning — COALESCED ✓
         * Thread (ty,tx) loads N_cm at row=(ph*TILE+tx), col=(bx*TILE+ty).
         * In column-major: N_cm[col*Width + row] = N_cm[(bx*TILE+ty)*Width + ph*TILE+tx].
         * Consecutive tx → ph*TILE+tx consecutive → stride-1 → COALESCED ✓
         * Store transposed: Nds[tx][ty] = N[ph*TILE+tx][bx*TILE+ty]             */
        Nds[tx][ty] = N_cm[(bx * TILE + ty) * Width + ph * TILE + tx];
        __syncthreads();

        /* Dot product: Nds[k][ty] gives N[ph*TILE+k][Col] — same values as before */
        for (int k = 0; k < TILE; ++k)
            Pvalue += Mds[ty][k] * Nds[k][ty];
        __syncthreads();
    }
    P[Row * Width + Col] = Pvalue;
}

/* -----------------------------------------------------------------------
 * CPU reference
 * ----------------------------------------------------------------------- */
static void cpu_matmul(const float* M, const float* N, float* P, int W) {
    for (int r = 0; r < W; r++)
        for (int c = 0; c < W; c++) {
            float v = 0.0f;
            for (int k = 0; k < W; k++) v += M[r*W+k] * N[k*W+c];
            P[r*W+c] = v;
        }
}

static float time_ms(dim3 grid, dim3 block, bool corner,
                     const float* dM, const float* dN, float* dP, int W) {
    if (corner) matmulColMajorCornerTurned<<<grid, block>>>(dM, dN, dP, W);
    else        matmulColMajorNaive<<<grid, block>>>(dM, dN, dP, W);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < N_RUNS; r++) {
        if (corner) matmulColMajorCornerTurned<<<grid, block>>>(dM, dN, dP, W);
        else        matmulColMajorNaive<<<grid, block>>>(dM, dN, dP, W);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / N_RUNS;
}

int main() {
    const int W = 512;
    int n = W * W;
    size_t bytes = n * sizeof(float);

    float *h_M   = (float*)malloc(bytes);
    float *h_N   = (float*)malloc(bytes);
    float *h_Ncm = (float*)malloc(bytes);
    float *h_P   = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    float *d_M, *d_Ncm, *d_P;

    srand(7);
    for (int i = 0; i < n; i++) { h_M[i] = (float)rand()/RAND_MAX; h_N[i] = (float)rand()/RAND_MAX; }
    to_col_major(h_Ncm, h_N, W);
    cpu_matmul(h_M, h_N, h_ref, W);

    cudaMalloc((void**)&d_M,   bytes);
    cudaMalloc((void**)&d_Ncm, bytes);
    cudaMalloc((void**)&d_P,   bytes);
    cudaMemcpy(d_M,   h_M,   bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ncm, h_Ncm, bytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE, TILE, 1);
    dim3 dimGrid(W/TILE, W/TILE, 1);

    /* ── Correctness ─────────────────────────────────────────────── */
    auto check = [&](const char* name, bool corner) {
        if (corner) matmulColMajorCornerTurned<<<dimGrid, dimBlock>>>(d_M, d_Ncm, d_P, W);
        else        matmulColMajorNaive<<<dimGrid, dimBlock>>>(d_M, d_Ncm, d_P, W);
        cudaDeviceSynchronize();
        cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);
        float max_err = 0.0f;
        for (int i = 0; i < n; i++) { float e = fabsf(h_P[i]-h_ref[i]); if(e>max_err) max_err=e; }
        printf("%-30s max_err=%e [%s]\n", name, max_err, max_err<1e-2f?"PASSED":"FAILED");
    };
    check("Naive (uncoalesced B):", false);
    check("Corner-turned (coalesced B):", true);

    /* ── Timing ──────────────────────────────────────────────────── */
    float ms_naive  = time_ms(dimGrid, dimBlock, false, d_M, d_Ncm, d_P, W);
    float ms_corner = time_ms(dimGrid, dimBlock, true,  d_M, d_Ncm, d_P, W);

    printf("\nTiming (W=%d, TILE=%d, %d runs):\n", W, TILE, N_RUNS);
    printf("  Naive (uncoalesced B):    %.2f ms\n", ms_naive);
    printf("  Corner-turned (coalesced): %.2f ms\n", ms_corner);
    printf("  Speedup: %.2fx\n\n", ms_naive / ms_corner);

    printf("Key insight (Section 6.1, Figure 6.4):\n");
    printf("  Without corner turning: thread tx loads N_cm[Col*W + ph*TILE+ty]\n");
    printf("    → consecutive tx → consecutive Col → stride W → UNCOALESCED\n");
    printf("  With corner turning: thread tx loads N_cm[(bx*TILE+ty)*W + ph*TILE+tx]\n");
    printf("    → consecutive tx → ph*TILE+tx consecutive → stride 1 → COALESCED\n");
    printf("  Nds is stored transposed; dot product reads Nds[k][ty] instead of Nds[ty][k]\n");

    free(h_M); free(h_N); free(h_Ncm); free(h_P); free(h_ref);
    cudaFree(d_M); cudaFree(d_Ncm); cudaFree(d_P);
    return 0;
}
