/*
 * Chapter 3 — Section 3.4: Matrix Multiplication
 *             Figure 3.11: MatrixMulKernel
 *             Figure 3.12: execution example with 4×4 matrices, BLOCK_WIDTH=2
 *
 * Matrix multiplication C = A × B where A is I×J, B is J×K, C is I×K.
 * The book uses square matrices (I=J=K=Width) and calls them M, N, P.
 * We follow the same naming to match Figure 3.11.
 *
 * Thread-to-data mapping (Section 3.4):
 *   Each thread computes exactly ONE element of the output matrix P.
 *   • row = blockIdx.y * blockDim.y + threadIdx.y   → which row of P
 *   • col = blockIdx.x * blockDim.x + threadIdx.x   → which col of P
 *   This is the same mapping used in colorToGrayscaleConversion.
 *
 * Row-major index arithmetic (Figure 3.11, Section 3.4):
 *   M[row][k]  →  M[row*Width + k]       (k-th element of row)
 *   N[k][col]  →  N[k*Width + col]       (k-th element of column col)
 *
 * Limitations of this "naïve" kernel (Section 3.4):
 *   • Grid size limits bound the largest matrix that fits in one launch.
 *   • Each element of M and N is re-read from global memory for every
 *     thread that needs it — highly redundant.  Chapters 5 & 6 address
 *     this with tiling and shared memory (tiled matmul).
 *
 * Build:
 *   nvcc -O2 -arch=sm_70 -o matrix_multiply 04_matrix_multiply.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_WIDTH 16   /* 16×16 = 256 threads per block */

/* -----------------------------------------------------------------------
 * MatrixMulKernel — Figure 3.11
 *
 * Square matrices only: M[Width×Width], N[Width×Width] → P[Width×Width]
 * All stored in row-major order as 1-D arrays of length Width*Width.
 * ----------------------------------------------------------------------- */
__global__
void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    /* Thread coordinates → P element to compute */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < Width) && (col < Width)) {
        float Pvalue = 0.0f;

        /* Inner product of row `row` of M and column `col` of N */
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k]   /* M[row][k] */
                    * N[k   * Width + col]; /* N[k][col] */
        }

        P[row * Width + col] = Pvalue;
    }
}

/* -----------------------------------------------------------------------
 * Host wrapper — allocates device memory, launches kernel, retrieves result
 * ----------------------------------------------------------------------- */
void matMul(float* M_h, float* N_h, float* P_h, int Width) {
    int bytes = Width * Width * sizeof(float);
    float *M_d, *N_d, *P_d;

    cudaMalloc((void**)&M_d, bytes);
    cudaMalloc((void**)&N_d, bytes);
    cudaMalloc((void**)&P_d, bytes);

    cudaMemcpy(M_d, M_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, bytes, cudaMemcpyHostToDevice);

    /* Grid: enough 16×16 blocks to cover the Width×Width output */
    dim3 dimGrid((int)ceil(Width / (float)BLOCK_WIDTH),
                 (int)ceil(Width / (float)BLOCK_WIDTH), 1);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    MatrixMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    cudaDeviceSynchronize();

    cudaMemcpy(P_h, P_d, bytes, cudaMemcpyDeviceToHost);
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

/* -----------------------------------------------------------------------
 * CPU reference — straightforward O(N^3) triple loop
 * ----------------------------------------------------------------------- */
static void cpu_matmul(float* M, float* N, float* P, int Width) {
    for (int r = 0; r < Width; r++)
        for (int c = 0; c < Width; c++) {
            float v = 0.0f;
            for (int k = 0; k < Width; k++)
                v += M[r * Width + k] * N[k * Width + c];
            P[r * Width + c] = v;
        }
}

/* -----------------------------------------------------------------------
 * Print a small matrix (for the 4×4 trace-through from Figure 3.12)
 * ----------------------------------------------------------------------- */
static void print_matrix(const char* name, float* M, int Width) {
    printf("%s (%d×%d):\n", name, Width, Width);
    for (int r = 0; r < Width; r++) {
        printf("  [ ");
        for (int c = 0; c < Width; c++) printf("%6.1f ", M[r * Width + c]);
        printf("]\n");
    }
}

int main() {
    /* ---------------------------------------------------------------
     * Trace-through: the 4×4 example from Figure 3.12 / Section 3.4
     * BLOCK_WIDTH=2 → 4 blocks of 2×2 threads covering the 4×4 P matrix.
     * Thread (0,0) of block (0,0) → P[0][0], thread (0,0) of block (1,0) → P[2][0]
     * --------------------------------------------------------------- */
    {
        const int W = 4;
        float M[16], N[16], P[16], ref[16];

        /* Fill M with 1..16, N with identity so P should equal M */
        for (int i = 0; i < W * W; i++) M[i] = (float)(i + 1);
        for (int r = 0; r < W; r++)
            for (int c = 0; c < W; c++)
                N[r * W + c] = (r == c) ? 1.0f : 0.0f;

        matMul(M, N, P, W);
        cpu_matmul(M, N, ref, W);

        print_matrix("M", M, W);
        print_matrix("N (identity)", N, W);
        print_matrix("P = M×N (GPU)", P, W);

        float max_err = 0.0f;
        for (int i = 0; i < W * W; i++) {
            float e = fabsf(P[i] - ref[i]);
            if (e > max_err) max_err = e;
        }
        printf("4×4 trace-through: max error = %e [%s]\n\n",
               max_err, max_err < 1e-4f ? "PASSED" : "FAILED");
    }

    /* ---------------------------------------------------------------
     * Larger test: random 512×512 matrices
     * --------------------------------------------------------------- */
    {
        const int W = 512;
        int n = W * W;
        float* M   = (float*)malloc(n * sizeof(float));
        float* N   = (float*)malloc(n * sizeof(float));
        float* P   = (float*)malloc(n * sizeof(float));
        float* ref = (float*)malloc(n * sizeof(float));

        srand(42);
        for (int i = 0; i < n; i++) {
            M[i] = (float)rand() / RAND_MAX;
            N[i] = (float)rand() / RAND_MAX;
        }

        matMul(M, N, P, W);
        cpu_matmul(M, N, ref, W);

        float max_err = 0.0f;
        for (int i = 0; i < n; i++) {
            float e = fabsf(P[i] - ref[i]);
            if (e > max_err) max_err = e;
        }
        printf("512×512 matmul: max error = %e [%s]\n",
               max_err, max_err < 1e-2f ? "PASSED" : "FAILED");

        free(M); free(N); free(P); free(ref);
    }

    return 0;
}
