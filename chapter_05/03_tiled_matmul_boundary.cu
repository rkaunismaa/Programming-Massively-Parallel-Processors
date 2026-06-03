/*
 * Chapter 5 — Section 5.5: Boundary Checks
 *             Figure 5.13: tiled matrix multiplication with boundary conditions
 *             Figures 5.11–5.12: why boundary checks are needed
 *
 * The simple tiled kernel (Figure 5.9) makes two simplifying assumptions:
 *   1. Matrix Width is a multiple of TILE_WIDTH.
 *   2. Matrices are square.
 *
 * Both assumptions can fail in practice.  This file adds boundary checks
 * that allow the kernel to handle arbitrary matrix dimensions.
 *
 * TWO BOUNDARY PROBLEMS (Figures 5.11–5.12):
 *
 *   Problem 1 — Loading M tiles (Figure 5.11):
 *     thread_{0,1} of block_{0,0} may attempt to load M_{0,3} from a 3×3 M
 *     matrix when Width=3, TILE_WIDTH=2.  M_{0,3} does not exist.
 *     Fix: if (ph*TILE_WIDTH + tx) >= Width, load 0.0f instead.
 *
 *   Problem 2 — Loading N tiles (Figure 5.11):
 *     thread_{1,0} of block_{0,0} may attempt to load N_{3,0} from a 3×3 N.
 *     Fix: if (ph*TILE_WIDTH + ty) >= Width, load 0.0f instead.
 *
 *   Problem 3 — Storing P elements (Figure 5.13, line 29):
 *     Threads in boundary blocks may compute P elements outside the matrix.
 *     Fix: only write if Row < Width && Col < Width.
 *
 * NOTE on "phantom" 0.0f elements:
 *   Loading 0.0f for out-of-bounds positions is safe because multiplying by 0
 *   contributes nothing to the dot product.  This is a standard padding trick.
 *
 * This kernel also shows how to extend to RECTANGULAR matrices (Section 5.5
 * end): replace single Width argument with j, k, l for an I×J × J×K → I×K
 * multiplication.  That extension is left as an exercise (as in the book);
 * the kernel here handles square matrices of arbitrary Width.
 *
 * Build:
 *   nvcc -O2 -arch=sm_89 -o tiled_matmul_bc 03_tiled_matmul_boundary.cu -lm
 *   Override tile size: nvcc -DTILE_WIDTH=8 ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

/* -----------------------------------------------------------------------
 * Tiled matmul with boundary checks — Figure 5.13 (extended from 5.9)
 *
 * Works for ANY Width, not just multiples of TILE_WIDTH.
 * ----------------------------------------------------------------------- */
__global__
void matrixMulKernelBC(float* M, float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    /* Phase loop: ceiling division handles Width not divisible by TILE_WIDTH */
    int num_phases = (Width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int ph = 0; ph < num_phases; ++ph) {

        /* Boundary-checked load for M tile (Figure 5.13 lines 18-19):
         * The column index of M being loaded = ph*TILE_WIDTH + tx.
         * Both Row and that column index must be within [0, Width). */
        if (Row < Width && (ph * TILE_WIDTH + tx) < Width)
            Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0f;   /* phantom element — contributes 0 to dot product */

        /* Boundary-checked load for N tile (Figure 5.13 lines 20-21):
         * The row index of N being loaded = ph*TILE_WIDTH + ty.
         * Both that row index and Col must be within [0, Width). */
        if ((ph * TILE_WIDTH + ty) < Width && Col < Width)
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        else
            Nds[ty][tx] = 0.0f;

        __syncthreads();   /* barrier 1: read-after-write */

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();   /* barrier 2: write-after-read */
    }

    /* Only write valid P elements (Figure 5.13 lines 28-29) */
    if (Row < Width && Col < Width)
        P[Row * Width + Col] = Pvalue;
}

/* -----------------------------------------------------------------------
 * CPU reference — works for any Width
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
 * Run a single test for a given Width and print PASS/FAIL
 * ----------------------------------------------------------------------- */
static void test_width(int W) {
    int n = W * W;
    size_t bytes = n * sizeof(float);
    float *h_M   = (float*)malloc(bytes);
    float *h_N   = (float*)malloc(bytes);
    float *h_P   = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    float *d_M, *d_N, *d_P;

    srand(W);   /* deterministic seed per width */
    for (int i = 0; i < n; i++) {
        h_M[i] = (float)(rand() % 5);
        h_N[i] = (float)(rand() % 5);
    }
    cpu_matmul(h_M, h_N, h_ref, W);

    cudaMalloc((void**)&d_M, bytes);
    cudaMalloc((void**)&d_N, bytes);
    cudaMalloc((void**)&d_P, bytes);
    cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_P, 0, bytes);

    /* Grid covers the full output — ceiling division */
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((W + TILE_WIDTH - 1) / TILE_WIDTH,
                 (W + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    matrixMulKernelBC<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, W);
    cudaDeviceSynchronize();
    cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(h_P[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }

    int multiple = (W % TILE_WIDTH == 0);
    printf("  Width=%-5d  TILE_WIDTH=%-3d  (%s multiple)  max_err=%e  [%s]\n",
           W, TILE_WIDTH, multiple ? "exact" : "non  ",
           max_err, max_err < 1e-1f ? "PASSED" : "FAILED");

    free(h_M); free(h_N); free(h_P); free(h_ref);
    cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
}

int main() {
    printf("=== Tiled MatMul with Boundary Checks (Figure 5.13) ===\n");
    printf("TILE_WIDTH = %d\n\n", TILE_WIDTH);

    /* ── Book example: 3×3 with TILE_WIDTH=2 (Figures 5.11–5.12) ── */
    printf("Small cases from Figures 5.11-5.12:\n");
    test_width(3);    /* non-multiple: threads attempt OOB loads */
    test_width(2);    /* exact multiple */
    test_width(4);    /* exact multiple */

    /* ── Systematic: multiples and non-multiples of TILE_WIDTH ────── */
    printf("\nSystematic tests:\n");
    int widths[] = {
        TILE_WIDTH,       /* exact */
        TILE_WIDTH - 1,   /* one short */
        TILE_WIDTH + 1,   /* one over */
        TILE_WIDTH * 4,   /* large exact multiple */
        TILE_WIDTH * 4 - 3,/* large non-multiple */
        TILE_WIDTH * 4 + 7,/* large non-multiple */
        500,
        512,
        1000,
        1024
    };
    for (int i = 0; i < (int)(sizeof(widths)/sizeof(widths[0])); i++)
        if (widths[i] > 0) test_width(widths[i]);

    printf("\nKey insight (Section 5.5):\n");
    printf("  Out-of-bounds loads use 0.0f — safe because 0 * anything = 0.\n");
    printf("  Out-of-bounds stores are suppressed by the Row/Col guard.\n");
    printf("  The phase count uses ceiling division: (Width + TW - 1) / TW.\n");

    return 0;
}
