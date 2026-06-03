/*
 * Chapter 2 — Section 2.5: CUDA C Function Qualifiers (Figure 2.11)
 *
 * CUDA C extends C with three qualifier keywords for function declarations.
 * This file demonstrates all three and the dual __host__ __device__ form.
 *
 * Summary table (Figure 2.11):
 * ┌──────────────┬────────────────┬─────────────┬────────────────────────┐
 * │  Qualifier   │  Callable from │ Executes on │      Launched by       │
 * ├──────────────┼────────────────┼─────────────┼────────────────────────┤
 * │  __host__    │     Host       │    Host     │  Caller host thread    │
 * │  __global__  │  Host(Device*) │   Device    │  New grid of threads   │
 * │  __device__  │    Device      │   Device    │  Caller device thread  │
 * └──────────────┴────────────────┴─────────────┴────────────────────────┘
 *  *Dynamic Parallelism (Chapter 21) allows __global__ from the device.
 *
 * Build:
 *   nvcc -O2 -arch=sm_70 -o function_qualifiers 05_function_qualifiers.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* -----------------------------------------------------------------------
 * __host__ function (Section 2.5)
 * This is the default — any function without a qualifier is __host__.
 * Runs on the CPU; can only be called from other host functions.
 * ----------------------------------------------------------------------- */
__host__
float host_square(float x) {
    return x * x;
}

/* -----------------------------------------------------------------------
 * __device__ function (Section 2.5)
 * Runs on the GPU; called by kernels or other device functions.
 * Does NOT launch a new grid of threads.
 * ----------------------------------------------------------------------- */
__device__
float device_square(float x) {
    return x * x;
}

/* -----------------------------------------------------------------------
 * __host__ __device__ function (Section 2.5)
 * Using both qualifiers tells NVCC to compile two versions:
 *   • a host version  — called from CPU code
 *   • a device version — called from GPU code
 * Useful for math utility functions used in both contexts.
 * ----------------------------------------------------------------------- */
__host__ __device__
float clamp01(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

/* -----------------------------------------------------------------------
 * __global__ kernel (Section 2.5)
 * Called from host, executes on device, launches a grid of threads.
 * Calls the __device__ helper function device_square.
 * ----------------------------------------------------------------------- */
__global__
void squareKernel(float* A, float* B, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        B[i] = device_square(A[i]);   /* call a __device__ function */
    }
}

/* -----------------------------------------------------------------------
 * Demonstrate __host__ __device__ clamp01 called from both sides
 * ----------------------------------------------------------------------- */
__global__
void clampKernel(float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = clamp01(in[i]);   /* device version of clamp01 */
    }
}

int main() {
    /* --- demonstrate __host__ function ---------------------------------- */
    printf("host_square(4.0f) = %.1f\n", host_square(4.0f));
    printf("clamp01(-0.5f)    = %.1f  (host version)\n", clamp01(-0.5f));
    printf("clamp01(1.5f)     = %.1f  (host version)\n", clamp01(1.5f));

    /* --- demonstrate __global__ kernel with __device__ helper ----------- */
    int n = 8;
    size_t size = n * sizeof(float);

    float h_in[]  = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float h_out[8];

    float *d_in, *d_out;
    cudaMalloc((void**)&d_in,  size);
    cudaMalloc((void**)&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    /* Small grid: 1 block of 8 threads is enough for 8 elements */
    squareKernel<<<1, 8>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    printf("\ndevice_square results:\n");
    for (int i = 0; i < n; i++) {
        printf("  square(%.0f) = %.0f\n", h_in[i], h_out[i]);
    }

    /* --- demonstrate __host__ __device__ clamp on the GPU -------------- */
    float h_clamp_in[]  = {-1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
    float h_clamp_out[5];
    int m = 5;

    float *d_ci, *d_co;
    cudaMalloc((void**)&d_ci, m * sizeof(float));
    cudaMalloc((void**)&d_co, m * sizeof(float));
    cudaMemcpy(d_ci, h_clamp_in, m * sizeof(float), cudaMemcpyHostToDevice);

    clampKernel<<<1, m>>>(d_ci, d_co, m);
    cudaDeviceSynchronize();
    cudaMemcpy(h_clamp_out, d_co, m * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nclamp01 results (device version):\n");
    for (int i = 0; i < m; i++) {
        printf("  clamp01(%.1f) = %.1f\n", h_clamp_in[i], h_clamp_out[i]);
    }

    cudaFree(d_in);  cudaFree(d_out);
    cudaFree(d_ci);  cudaFree(d_co);
    return 0;
}
