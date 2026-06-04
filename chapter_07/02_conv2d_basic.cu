// §7.2 Parallel convolution: a basic algorithm
// Figure 7.7: 2D convolution kernel with ghost cell boundary handling.
//
// Thread organization (Fig 7.6):
//   2D grid of 2D blocks — one thread computes one output element P[row][col].
//   outCol = blockIdx.x * blockDim.x + threadIdx.x
//   outRow = blockIdx.y * blockDim.y + threadIdx.y
//
// Ghost cells: N elements whose row or col index falls outside [0,width) or
// [0,height) are outside the image boundary.  §7.1 assumes the default value
// is 0, so we simply skip those contributions (the if-statement on line 09 of
// Fig 7.7 lets Pvalue stay 0 for missing neighbours).
//
// Performance analysis (§7.3 intro):
//   Operations per output: 2 * (2r+1)^2
//   Bytes loaded (no caching): N and F each (2r+1)^2 elements × 4 bytes
//   Arithmetic intensity ≈ 0.25 OP/B — severely memory-bound.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef FILTER_RADIUS
#define FILTER_RADIUS 2
#endif
#define FILTER_DIM (2 * FILTER_RADIUS + 1)

// ── Fig 7.7 ──────────────────────────────────────────────────────────────────
// F is a 1-D linearised pointer: F[fRow*(2r+1)+fCol] ≡ F[fRow][fCol].
__global__ void convolution_2D_basic_kernel(const float *N, const float *F,
                                             float *P, int r,
                                             int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outCol >= width || outRow >= height) return;

    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2*r+1; fRow++) {
        for (int fCol = 0; fCol < 2*r+1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            // Fig 7.7 line 09: ghost cells treated as 0 — skip the multiply
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                Pvalue += F[fRow*(2*r+1) + fCol] * N[inRow*width + inCol];
        }
    }
    P[outRow*width + outCol] = Pvalue;
}

// ── CPU reference ─────────────────────────────────────────────────────────────
static void conv2d_cpu(const float *N, const float *F, float *P,
                       int r, int width, int height) {
    int fd = 2*r+1;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float v = 0.0f;
            for (int fRow = 0; fRow < fd; fRow++) {
                for (int fCol = 0; fCol < fd; fCol++) {
                    int ir = row - r + fRow, ic = col - r + fCol;
                    if (ir >= 0 && ir < height && ic >= 0 && ic < width)
                        v += F[fRow*fd+fCol] * N[ir*width+ic];
                }
            }
            P[row*width+col] = v;
        }
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
    for (long f = 0; f < FE; f++) F_h[f] = 1.0f / FE;   // box (averaging) filter

    float *N_d, *F_d, *P_d;
    cudaMalloc(&N_d, NE * sizeof(float));
    cudaMalloc(&F_d, FE * sizeof(float));
    cudaMalloc(&P_d, NE * sizeof(float));
    cudaMemcpy(N_d, N_h, NE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F_h, FE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    // Warm-up
    convolution_2D_basic_kernel<<<grid, block>>>(N_d, F_d, P_d, R, W, H);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    convolution_2D_basic_kernel<<<grid, block>>>(N_d, F_d, P_d, R, W, H);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(P_gpu, P_d, NE * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU reference (full image — fast enough for 2048×2048 with small filter)
    conv2d_cpu(N_h, F_h, P_ref, R, W, H);
    int ok = verify(P_ref, P_gpu, (int)NE);

    double flops   = 2.0 * (double)NE * FE;
    double gflops  = flops / (ms * 1e6);
    float  ai      = (float)(2 * FE) / (float)(2 * FE * 4); // 0.25 OP/B

    printf("2D basic convolution:  %s\n", ok ? "PASS" : "FAIL");
    printf("Image: %dx%d   Filter: %dx%d (%dx%d)\n", W, H, FD, FD, 2*R+1, 2*R+1);
    printf("Time:  %.3f ms   GFLOPS: %.1f\n", ms, gflops);
    printf("Arithmetic intensity:  %.2f OP/B  (memory-bound — see §7.3)\n", ai);

    free(N_h); free(F_h); free(P_ref); free(P_gpu);
    cudaFree(N_d); cudaFree(F_d); cudaFree(P_d);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ok ? 0 : 1;
}
