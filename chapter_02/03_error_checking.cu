/*
 * Chapter 2 — "Error Checking and Handling in CUDA" sidebar (page 35)
 *
 * CUDA API functions return a cudaError_t flag.  For brevity the book's
 * examples omit the checks, but real code must handle them.  This file
 * shows the pattern described in the sidebar and wraps every API call in
 * a CUDA_CHECK macro — the approach the book recommends.
 *
 * Key points from the sidebar:
 *   • cudaError_t  — return type of every CUDA runtime API function
 *   • cudaSuccess  — the value returned when the call succeeded
 *   • cudaGetErrorString(err) — human-readable description of the error
 *   • After a kernel launch, call cudaGetLastError() to catch launch errors
 *   • Call cudaDeviceSynchronize() to wait for the kernel and catch async errors
 *
 * Build:
 *   nvcc -O2 -arch=sm_70 -o error_checking 03_error_checking.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* -----------------------------------------------------------------------
 * CUDA_CHECK macro — wraps any CUDA API call
 *
 * Usage:  CUDA_CHECK(cudaMalloc((void**)&ptr, size));
 *
 * If the call fails, prints the file, line number, and error string, then
 * exits.  __FILE__ and __LINE__ are standard C preprocessor macros.
 * ----------------------------------------------------------------------- */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s line %d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* -----------------------------------------------------------------------
 * Kernel — identical to the one in 02_vec_add_cuda.cu
 * ----------------------------------------------------------------------- */
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

/* -----------------------------------------------------------------------
 * vecAdd with full error checking on every API call
 * ----------------------------------------------------------------------- */
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc((void**)&A_d, size));
    CUDA_CHECK(cudaMalloc((void**)&B_d, size));
    CUDA_CHECK(cudaMalloc((void**)&C_d, size));

    /* Copy inputs host → device */
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    /* Launch kernel */
    vecAddKernel<<<(int)ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    /*
     * Kernel launches are asynchronous — the host moves on immediately.
     * Two steps to catch errors:
     *   1. cudaGetLastError()     — detects invalid launch configuration
     *   2. cudaDeviceSynchronize()— waits for the kernel, then detects
     *                               any runtime errors that arose during
     *                               execution
     */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result device → host */
    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    /* Free device memory */
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

int main() {
    int n = 1024;
    size_t size = n * sizeof(float);

    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        A[i] = (float)i;
        B[i] = (float)i * 2.0f;   /* B[i] = 2*i, so C[i] = 3*i */
    }

    vecAdd(A, B, C, n);

    printf("C[0]   = %.1f  (expected 0.0)\n",      C[0]);
    printf("C[1]   = %.1f  (expected 3.0)\n",      C[1]);
    printf("C[511] = %.1f  (expected %.1f)\n",     C[511], 511 * 3.0f);
    printf("C[n-1] = %.1f  (expected %.1f)\n",     C[n-1], (n-1) * 3.0f);

    free(A);
    free(B);
    free(C);
    return 0;
}
