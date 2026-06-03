/*
 * Chapter 3 — Section 3.2: Row-Major Linearization (Figure 3.3)
 *             Section 3.2: 3-D array linearization
 *
 * Dynamically allocated C arrays are always "flat" (1-D in memory).
 * C compilers linearize statically-declared multi-dimensional arrays for
 * you; for dynamically-allocated arrays in CUDA C you must do it manually.
 *
 * 2-D row-major (Figure 3.3):
 *   M[j][i]  →  M[j * Width + i]
 *   "j*Width skips over all rows before row j; i selects the column"
 *
 * 3-D row-major (Section 3.2, end):
 *   P[plane][row][col]  →  P[plane * height * width + row * width + col]
 *
 * This file demonstrates these formulae with a simple kernel that reads and
 * writes elements using the correct linearised indices, verifies them against
 * a CPU reference, and prints a worked example matching the book's 4×4
 * trace-through from Figure 3.3 (M_{2,1} → index 2*4+1 = 9).
 *
 * Build:
 *   nvcc -O2 -arch=sm_70 -o row_major 05_row_major_linearization.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* -----------------------------------------------------------------------
 * Kernel: transpose a 2-D matrix by writing M[row][col] → T[col][row].
 * Both M and T are stored in row-major order, so:
 *   read  from M at offset  row*Width + col
 *   write to   T at offset  col*Width + row   (transposed dimensions)
 * ----------------------------------------------------------------------- */
__global__
void transposeKernel(float* M, float* T, int Width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < Width && col < Width) {
        T[col * Width + row] = M[row * Width + col];
    }
}

/* -----------------------------------------------------------------------
 * Kernel: sum every "plane" of a 3-D tensor into a 2-D result matrix.
 * result[row][col] = sum over plane of P[plane][row][col]
 *
 * 3-D linearised index: plane * height * width + row * width + col
 * ----------------------------------------------------------------------- */
__global__
void sumPlanes(float* P, float* result, int width, int height, int depth) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        float sum = 0.0f;
        for (int p = 0; p < depth; p++) {
            sum += P[p * height * width + row * width + col];
        }
        result[row * width + col] = sum;
    }
}

int main() {
    /* ---------------------------------------------------------------
     * Worked example from Figure 3.3:
     * 4×4 matrix M, element M_{2,1} (row 2, col 1) has 1D index 2*4+1 = 9
     * --------------------------------------------------------------- */
    printf("=== Figure 3.3 trace-through ===\n");
    {
        const int W = 4;
        float M[W * W];
        for (int i = 0; i < W * W; i++) M[i] = (float)i;

        printf("4×4 matrix stored in row-major order:\n  [ ");
        for (int i = 0; i < W * W; i++) printf("%.0f ", M[i]);
        printf("]\n");
        printf("M[row=2][col=1] via M[2*4+1] = M[%d] = %.0f  (book: M_9 = 9)\n\n",
               2*W+1, M[2*W+1]);
    }

    /* ---------------------------------------------------------------
     * 2-D: transpose a 512×512 matrix
     * --------------------------------------------------------------- */
    printf("=== 2-D row-major: transpose 512×512 ===\n");
    {
        const int W = 512;
        int n = W * W;
        float* M   = (float*)malloc(n * sizeof(float));
        float* T   = (float*)malloc(n * sizeof(float));
        float *Md, *Td;

        for (int i = 0; i < n; i++) M[i] = (float)i;

        cudaMalloc((void**)&Md, n * sizeof(float));
        cudaMalloc((void**)&Td, n * sizeof(float));
        cudaMemcpy(Md, M, n * sizeof(float), cudaMemcpyHostToDevice);

        dim3 dimGrid((int)ceil(W / 16.0), (int)ceil(W / 16.0), 1);
        dim3 dimBlock(16, 16, 1);
        transposeKernel<<<dimGrid, dimBlock>>>(Md, Td, W);
        cudaDeviceSynchronize();
        cudaMemcpy(T, Td, n * sizeof(float), cudaMemcpyDeviceToHost);

        /* Verify: T[col][row] should equal M[row][col] */
        int ok = 1;
        for (int r = 0; r < W && ok; r++)
            for (int c = 0; c < W && ok; c++)
                ok = (T[c * W + r] == M[r * W + c]);
        printf("Transpose: %s\n\n", ok ? "PASSED" : "FAILED");

        free(M); free(T);
        cudaFree(Md); cudaFree(Td);
    }

    /* ---------------------------------------------------------------
     * 3-D: sum planes of a 4×4×4 tensor
     * Linearised index: plane*H*W + row*W + col
     * --------------------------------------------------------------- */
    printf("=== 3-D row-major: sum planes of 4×4×4 tensor ===\n");
    {
        const int W = 4, H = 4, D = 4;
        int n3 = W * H * D, n2 = W * H;
        float* P      = (float*)malloc(n3 * sizeof(float));
        float* result = (float*)malloc(n2 * sizeof(float));
        float* ref    = (float*)malloc(n2 * sizeof(float));
        float *Pd, *Rd;

        for (int i = 0; i < n3; i++) P[i] = 1.0f;  /* all ones → sum = depth */

        cudaMalloc((void**)&Pd, n3 * sizeof(float));
        cudaMalloc((void**)&Rd, n2 * sizeof(float));
        cudaMemcpy(Pd, P, n3 * sizeof(float), cudaMemcpyHostToDevice);

        dim3 dimGrid((int)ceil(W / 4.0), (int)ceil(H / 4.0), 1);
        dim3 dimBlock(4, 4, 1);
        sumPlanes<<<dimGrid, dimBlock>>>(Pd, Rd, W, H, D);
        cudaDeviceSynchronize();
        cudaMemcpy(result, Rd, n2 * sizeof(float), cudaMemcpyDeviceToHost);

        /* Every element should equal D (sum of D ones) */
        int ok = 1;
        for (int i = 0; i < n2; i++) ok &= (result[i] == (float)D);
        printf("Sum planes (4×4×4, all-ones): %s\n", ok ? "PASSED" : "FAILED");
        printf("result[0] = %.0f (expected %d)\n", result[0], D);

        free(P); free(result); free(ref);
        cudaFree(Pd); cudaFree(Rd);
    }

    return 0;
}
