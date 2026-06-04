// §7.3 Constant memory and caching
// Figure 7.9: 2D convolution kernel using __constant__ memory for the filter.
//
// Why F is a perfect constant-memory candidate (§7.3):
//   1. Small — radius ≤ 7 means ≤ 343 elements (≤ 1372 bytes for float).
//   2. Read-only during kernel execution.
//   3. Uniform access — all threads iterate F in the same order (F[0][0]…).
// The hardware routes warp-uniform loads to a specialised constant cache,
// providing the full bandwidth of an L1 hit with zero DRAM traffic for F.
//
// Host side change (§7.3):
//   cudaMemcpyToSymbol(F_c, F_h, size) — informs the runtime that F_c is
//   constant; ordinary cudaMemcpy to a __constant__ variable is not allowed.
//
// Arithmetic intensity with constant caching:
//   Each output element needs (2r+1)^2 × 4 bytes from N (no caching of N).
//   F is served entirely from the constant cache → 0 DRAM bytes for F.
//   AI ≈ 2*(2r+1)^2 / ((2r+1)^2*4) = 0.50 OP/B  (double the basic kernel).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef FILTER_RADIUS
#define FILTER_RADIUS 2
#endif
#define FILTER_DIM (2 * FILTER_RADIUS + 1)

// §7.3: global-scope __constant__ declaration.
// The kernel accesses F_c as a global variable — no pointer parameter needed.
// Scoping rules: if host and kernel code live in separate files, the kernel
// file must include an extern declaration for F_c.
__constant__ float F_c[FILTER_DIM][FILTER_DIM];

// ── Fig 7.9 ──────────────────────────────────────────────────────────────────
// Kernel is almost identical to Fig 7.7; only difference:
//   - F pointer parameter is gone
//   - F[fRow][fCol] is now F_c[fRow][fCol]  (2D constant array)
__global__ void convolution_2D_const_mem_kernel(const float *N, float *P,
                                                 int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outCol >= width || outRow >= height) return;

    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2*r+1; fRow++) {
        for (int fCol = 0; fCol < 2*r+1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                Pvalue += F_c[fRow][fCol] * N[inRow*width + inCol];
        }
    }
    P[outRow*width + outCol] = Pvalue;
}

// ── Basic kernel for comparison ───────────────────────────────────────────────
__global__ void convolution_2D_basic_kernel(const float *N, const float *F,
                                             float *P, int r,
                                             int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outCol >= width || outRow >= height) return;

    float Pvalue = 0.0f;
    int fd = 2*r+1;
    for (int fRow = 0; fRow < fd; fRow++) {
        for (int fCol = 0; fCol < fd; fCol++) {
            int inRow = outRow - r + fRow, inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                Pvalue += F[fRow*fd + fCol] * N[inRow*width + inCol];
        }
    }
    P[outRow*width + outCol] = Pvalue;
}

static void conv2d_cpu(const float *N, const float *F, float *P,
                       int r, int width, int height) {
    int fd = 2*r+1;
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++) {
            float v = 0.0f;
            for (int fRow = 0; fRow < fd; fRow++)
                for (int fCol = 0; fCol < fd; fCol++) {
                    int ir = row-r+fRow, ic = col-r+fCol;
                    if (ir >= 0 && ir < height && ic >= 0 && ic < width)
                        v += F[fRow*fd+fCol] * N[ir*width+ic];
                }
            P[row*width+col] = v;
        }
}

static int verify(const float *ref, const float *gpu, int n) {
    for (int i = 0; i < n; i++) {
        float err = fabsf(ref[i] - gpu[i]);
        if (err > 1e-4f * (fabsf(ref[i]) + 1.0f)) {
            printf("  MISMATCH i=%d  ref=%.6f  gpu=%.6f\n", i, ref[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

static float timed_kernel(void (*launch)(dim3,dim3), dim3 grid, dim3 block) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    launch(grid, block);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

int main(void) {
    const int W  = 2048;
    const int H  = 2048;
    const int R  = FILTER_RADIUS;
    const int FD = 2*R + 1;
    const long NE = (long)W * H;
    const long FE = (long)FD * FD;

    float *N_h   = (float *)malloc(NE * sizeof(float));
    float *F_h   = (float *)malloc(FE * sizeof(float));
    float *P_ref = (float *)malloc(NE * sizeof(float));
    float *P_gpu = (float *)malloc(NE * sizeof(float));

    srand(42);
    for (long i = 0; i < NE; i++) N_h[i] = (float)rand() / RAND_MAX;
    for (long f = 0; f < FE; f++) F_h[f] = 1.0f / FE;

    float *N_d, *F_d, *P_d;
    cudaMalloc(&N_d, NE * sizeof(float));
    cudaMalloc(&F_d, FE * sizeof(float));   // for basic kernel comparison
    cudaMalloc(&P_d, NE * sizeof(float));
    cudaMemcpy(N_d, N_h, NE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F_h, FE * sizeof(float), cudaMemcpyHostToDevice);

    // §7.3: upload filter to __constant__ memory
    cudaMemcpyToSymbol(F_c, F_h, FE * sizeof(float));

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    // Warm-up both kernels
    convolution_2D_basic_kernel<<<grid, block>>>(N_d, F_d, P_d, R, W, H);
    convolution_2D_const_mem_kernel<<<grid, block>>>(N_d, P_d, R, W, H);
    cudaDeviceSynchronize();

    // Correctness check
    conv2d_cpu(N_h, F_h, P_ref, R, W, H);
    convolution_2D_const_mem_kernel<<<grid, block>>>(N_d, P_d, R, W, H);
    cudaDeviceSynchronize();
    cudaMemcpy(P_gpu, P_d, NE * sizeof(float), cudaMemcpyDeviceToHost);
    int ok = verify(P_ref, P_gpu, (int)NE);
    printf("Constant-memory kernel:   %s\n", ok ? "PASS" : "FAIL");

    // Timing
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms_basic, ms_const;

    cudaEventRecord(t0);
    convolution_2D_basic_kernel<<<grid, block>>>(N_d, F_d, P_d, R, W, H);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_basic, t0, t1);

    cudaEventRecord(t0);
    convolution_2D_const_mem_kernel<<<grid, block>>>(N_d, P_d, R, W, H);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_const, t0, t1);

    double flops = 2.0 * (double)NE * FE;
    printf("\nImage: %dx%d   Filter: %dx%d\n", W, H, FD, FD);
    printf("%-28s  %6.3f ms  %6.1f GFLOPS  AI=0.25 OP/B\n",
           "Basic (F global):", ms_basic, flops / (ms_basic * 1e6));
    printf("%-28s  %6.3f ms  %6.1f GFLOPS  AI=0.50 OP/B\n",
           "Const mem (F cached):", ms_const, flops / (ms_const * 1e6));
    printf("Speedup from constant memory: %.2fx\n", ms_basic / ms_const);

    free(N_h); free(F_h); free(P_ref); free(P_gpu);
    cudaFree(N_d); cudaFree(F_d); cudaFree(P_d);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ok ? 0 : 1;
}
