// §7.1 Background  §7.2 Basic algorithm  §7.3 Constant memory and caching
//
// 1D convolution — the conceptual foundation for Chapter 7.
//   P[i] = Σ_{f=0}^{2r} F[f] * N[i - r + f]   (zero-padding for ghost cells)
//
// Two kernels:
//   conv1d_basic_kernel  — F passed as ordinary global-memory pointer
//   conv1d_const_kernel  — F in __constant__ memory (cudaMemcpyToSymbol)
//
// Arithmetic intensity (per output element, ignoring N inter-thread reuse):
//   Basic:     2*FD FLOP / (2*FD*4 B)  = 0.25 OP/B   (§7.2, same as 2D)
//   Const mem: 2*FD FLOP / (  FD*4 B)  = 0.50 OP/B   (§7.3, F from cache)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef FILTER_RADIUS
#define FILTER_RADIUS 3
#endif
#define FILTER_DIM (2 * FILTER_RADIUS + 1)

// §7.3 / Fig 7.8: filter in __constant__ memory.
// The CUDA runtime caches constant variables aggressively because they are
// small, read-only, and accessed in the same order by all threads.
__constant__ float F_const[FILTER_DIM];

// ── Basic 1D kernel: F in global memory ──────────────────────────────────────
__global__ void conv1d_basic_kernel(const float *N, const float *F,
                                     float *P, int r, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;

    float Pvalue = 0.0f;
    for (int f = 0; f < 2*r+1; f++) {
        int in_i = i - r + f;
        if (in_i >= 0 && in_i < width)   // §7.1: skip ghost cells (treat as 0)
            Pvalue += F[f] * N[in_i];
    }
    P[i] = Pvalue;
}

// ── Constant-memory 1D kernel: F in __constant__ (§7.3) ──────────────────────
// F_const is accessed as a global variable — no pointer parameter needed.
// Hardware routes all warp accesses to a single broadcast from the constant cache.
__global__ void conv1d_const_kernel(const float *N, float *P,
                                     int r, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;

    float Pvalue = 0.0f;
    for (int f = 0; f < 2*r+1; f++) {
        int in_i = i - r + f;
        if (in_i >= 0 && in_i < width)
            Pvalue += F_const[f] * N[in_i];
    }
    P[i] = Pvalue;
}

// ── CPU reference ─────────────────────────────────────────────────────────────
static void conv1d_cpu(const float *N, const float *F, float *P,
                       int r, int width) {
    for (int i = 0; i < width; i++) {
        float v = 0.0f;
        for (int f = 0; f < 2*r+1; f++) {
            int idx = i - r + f;
            if (idx >= 0 && idx < width) v += F[f] * N[idx];
        }
        P[i] = v;
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

int main(void) {
    const int W  = 1 << 20;        // 1 M-element signal
    const int R  = FILTER_RADIUS;
    const int FD = 2*R + 1;

    float *N_h   = (float *)malloc(W  * sizeof(float));
    float *F_h   = (float *)malloc(FD * sizeof(float));
    float *P_ref = (float *)malloc(W  * sizeof(float));
    float *P_gpu = (float *)malloc(W  * sizeof(float));

    srand(42);
    for (int i = 0; i < W;  i++) N_h[i] = (float)rand() / RAND_MAX;
    for (int f = 0; f < FD; f++) F_h[f] = 1.0f / FD;   // uniform (box) filter

    float *N_d, *F_d, *P_d;
    cudaMalloc(&N_d, W  * sizeof(float));
    cudaMalloc(&F_d, FD * sizeof(float));
    cudaMalloc(&P_d, W  * sizeof(float));
    cudaMemcpy(N_d, N_h, W  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F_h, FD * sizeof(float), cudaMemcpyHostToDevice);

    // §7.3: load filter into constant memory
    cudaMemcpyToSymbol(F_const, F_h, FD * sizeof(float));

    dim3 block(256);
    dim3 grid((W + 255) / 256);

    conv1d_cpu(N_h, F_h, P_ref, R, W);

    // Basic kernel
    conv1d_basic_kernel<<<grid, block>>>(N_d, F_d, P_d, R, W);
    cudaDeviceSynchronize();
    cudaMemcpy(P_gpu, P_d, W * sizeof(float), cudaMemcpyDeviceToHost);
    printf("1D basic (F global mem):   %s\n", verify(P_ref, P_gpu, W) ? "PASS" : "FAIL");

    // Constant-memory kernel
    conv1d_const_kernel<<<grid, block>>>(N_d, P_d, R, W);
    cudaDeviceSynchronize();
    cudaMemcpy(P_gpu, P_d, W * sizeof(float), cudaMemcpyDeviceToHost);
    printf("1D const mem (F cached):   %s\n", verify(P_ref, P_gpu, W) ? "PASS" : "FAIL");

    printf("\nFilter radius r=%d  filter size=%d  signal length=%d\n", R, FD, W);
    printf("Arithmetic intensity (per output, no caching of N):\n");
    printf("  Basic:    2*%d / (2*%d*4) = %.2f OP/B\n",
           FD, FD, (float)(2*FD) / (float)(2*FD*4));
    printf("  Const:    2*%d / (  %d*4) = %.2f OP/B  (F served from constant cache)\n",
           FD, FD, (float)(2*FD) / (float)(FD*4));

    free(N_h); free(F_h); free(P_ref); free(P_gpu);
    cudaFree(N_d); cudaFree(F_d); cudaFree(P_d);
    return 0;
}
