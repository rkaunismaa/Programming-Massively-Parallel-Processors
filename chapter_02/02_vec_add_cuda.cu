/*
 * Chapter 2 — Sections 2.3–2.6: A Complete CUDA Vector Addition
 *
 * This file brings together every concept introduced in Chapter 2:
 *
 *  Section 2.2  CUDA C program structure — host/device split, grids of threads
 *  Section 2.3  vecAdd kernel (Figure 2.10)
 *  Section 2.4  Device global memory & data transfer — cudaMalloc, cudaFree,
 *               cudaMemcpy (Figures 2.6, 2.7, 2.8)
 *  Section 2.5  Kernel functions & threading — __global__, threadIdx, blockIdx,
 *               blockDim, the loop-parallelism pattern (Figures 2.9–2.11)
 *  Section 2.6  Calling kernel functions — <<<gridDim, blockDim>>> syntax,
 *               ceiling-division for block count (Figures 2.12, 2.13)
 *  Section 2.7  Compilation — NVCC separates host code (gcc) and device code (PTX)
 *
 * Build:
 *   nvcc -O2 -arch=sm_70 -o vec_add_cuda 02_vec_add_cuda.cu
 *
 * Note: replace sm_70 with your GPU's compute capability (e.g. sm_86 for RTX 30xx).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* -----------------------------------------------------------------------
 * Section 2.5 — Kernel function declaration using __global__
 *
 * The __global__ qualifier (Figure 2.11) marks this as a CUDA kernel:
 *   • Executed on the device (GPU)
 *   • Called from the host (CPU)
 *   • Launches a new grid of threads
 *
 * Each thread computes exactly one element of C = A + B.
 * This replaces the for-loop of the sequential version — the grid of
 * threads IS the loop (Section 2.5, "loop parallelism").
 *
 * Thread indexing (Figure 2.9):
 *   blockIdx.x  — which block this thread is in
 *   blockDim.x  — how many threads per block
 *   threadIdx.x — thread's position within its block
 *   i = blockIdx.x * blockDim.x + threadIdx.x   → unique global index
 * ----------------------------------------------------------------------- */
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /*
     * Guard clause: not all vector lengths are multiples of the block size.
     * If n = 100 and blockDim.x = 32 we launch 4 blocks (128 threads), but
     * only the first 100 should write.  The last 28 threads skip this body.
     */
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

/* -----------------------------------------------------------------------
 * Section 2.4 — Device memory allocation and data transfer
 * Section 2.6 — Kernel launch
 *
 * The revised vecAdd function (Figures 2.5, 2.8, 2.13) acts as an
 * "outsourcing agent": it ships data to the device, runs the kernel,
 * and collects the result.  Three logical parts:
 *
 *   Part 1 — Allocate device memory, copy inputs host→device
 *   Part 2 — Launch the kernel
 *   Part 3 — Copy result device→host, free device memory
 *
 * Variable naming convention (Section 2.3):
 *   _h suffix → pointer lives in host memory
 *   _d suffix → pointer lives in device global memory
 * ----------------------------------------------------------------------- */
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    /* --- Part 1: allocate device global memory (Figure 2.6) ------------ *
     * cudaMalloc(address_of_pointer, size_in_bytes)
     *   First arg : address of the pointer variable (cast to void**)
     *   Second arg: number of bytes to allocate
     * After the call A_d points into device global memory.
     * -------------------------------------------------------------------- */
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    /* cudaMemcpy(dst, src, bytes, direction)  (Figure 2.7)
     * cudaMemcpyHostToDevice copies from CPU RAM to GPU global memory.    */
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    /* --- Part 2: launch kernel (Figure 2.12) ---------------------------- *
     * Execution configuration <<<gridDim, blockDim>>>:
     *   gridDim  = number of blocks  = ⌈n / 256⌉
     *   blockDim = threads per block = 256
     *
     * Using 256.0 (not 256) forces floating-point division so that ceil()
     * rounds up correctly.  E.g. n=1000 → ceil(1000/256.0)=4 blocks,
     * giving 1024 threads total; the kernel guard (i<n) silences the last 24.
     *
     * The book recommends block sizes that are multiples of 32 for hardware
     * efficiency (Section 2.5).
     * -------------------------------------------------------------------- */
    vecAddKernel<<<(int)ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    /* --- Part 3: retrieve result, free device memory -------------------- */
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    /* cudaFree only needs the pointer value, not its address (Section 2.4) */
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

/* -----------------------------------------------------------------------
 * Main: allocates host arrays, calls vecAdd, verifies the result.
 * The main function itself is standard C — no CUDA keywords needed.
 * ----------------------------------------------------------------------- */
int main() {
    int n = 1 << 20;    /* ~1 M elements */
    size_t size = n * sizeof(float);

    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    vecAdd(A, B, C, n);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(C[i] - 3.0f);
        if (err > max_err) max_err = err;
    }
    printf("CUDA vecAdd (n=%d) — max error: %e  [%s]\n",
           n, max_err, max_err < 1e-5f ? "PASSED" : "FAILED");

    free(A);
    free(B);
    free(C);
    return 0;
}
