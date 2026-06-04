// §11.2 Parallel scan with the Kogge-Stone algorithm — Figure 11.3
// §11.3 Speed and work efficiency consideration
//
// An inclusive prefix sum (scan) computes:
//   Y[i] = X[0] + X[1] + ... + X[i]   for i = 0..N-1
//
// An exclusive prefix sum computes:
//   Y[0] = 0,  Y[i] = X[0] + ... + X[i-1]   (identity + shift right by 1)
//
// Kogge-Stone algorithm (Fig 11.3):
//   Doubling stride: 1, 2, 4, …, blockDim.x/2.
//   In each iteration: XY[i] += XY[i - stride]  if i >= stride.
//   After log₂N iterations, XY[i] holds the inclusive prefix sum to position i.
//
// Race condition (§11.2):
//   Write-after-read hazard: thread i writes XY[i] while thread i+stride reads XY[i].
//   Fix: use a temp variable (lines 13-16 of Fig 11.3) + two __syncthreads()
//   per iteration (one before reading, one before writing).
//
// Work complexity (§11.3):
//   Total additions = Σ (N - stride) for stride = 1, 2, 4, …, N/2
//                   = N*log₂N - (N-1) ≈ N*log₂N
//   Sequential: N - 1 additions.  Ratio: N*log₂N / (N-1) ≈ log₂N.
//   NOT work-efficient — but achieves log₂N time steps with full parallelism.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef SECTION_SIZE
#define SECTION_SIZE 1024   // one thread per element; max block = 1024 threads
#endif

// ── Figure 11.3: Kogge-Stone inclusive scan ───────────────────────────────────
__global__ void Kogge_Stone_scan_inclusive_kernel(const float *X, float *Y,
                                                   unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    XY[threadIdx.x] = (i < N) ? X[i] : 0.0f;

    // Doubling-stride scan: after iteration k, XY[t] = sum of t-2^k+1..t
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp = XY[threadIdx.x];         // read old value into temp
        if (threadIdx.x >= stride)
            temp += XY[threadIdx.x - stride]; // accumulate from left
        __syncthreads();                       // all reads done before any write
        XY[threadIdx.x] = temp;
    }
    if (i < N) Y[i] = XY[threadIdx.x];
}

// ── Figure 11.4: Kogge-Stone exclusive scan ────────────────────────────────────
// To convert inclusive → exclusive: shift input right by 1, fill Y[0] = 0.
// Load X[i-1] into XY[threadIdx.x]; identity (0) at position 0.
__global__ void Kogge_Stone_scan_exclusive_kernel(const float *X, float *Y,
                                                   unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Shift right: XY[tx] = X[i-1] or 0 for the first position
    if (i < N && threadIdx.x != 0)
        XY[threadIdx.x] = X[i - 1];
    else
        XY[threadIdx.x] = 0.0f;

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp = XY[threadIdx.x];
        if (threadIdx.x >= stride)
            temp += XY[threadIdx.x - stride];
        __syncthreads();
        XY[threadIdx.x] = temp;
    }
    if (i < N) Y[i] = XY[threadIdx.x];
}

static void cpu_inclusive_scan(const float *X, float *Y, unsigned int N) {
    Y[0] = X[0];
    for (unsigned int i = 1; i < N; i++) Y[i] = Y[i-1] + X[i];
}

static bool verify(const float *ref, const float *gpu, unsigned int N) {
    for (unsigned int i = 0; i < N; i++) {
        float rel = fabsf(ref[i] - gpu[i]) / (fabsf(ref[i]) + 1.0f);
        if (rel > 1e-4f) {
            printf("  MISMATCH i=%u  ref=%.2f  gpu=%.2f\n", i, ref[i], gpu[i]);
            return false;
        }
    }
    return true;
}

int main(void) {
    const unsigned int N = SECTION_SIZE;

    float *X_h = (float *)malloc(N * sizeof(float));
    float *Y_h = (float *)malloc(N * sizeof(float));
    float *ref = (float *)malloc(N * sizeof(float));

    srand(42);
    for (unsigned int i = 0; i < N; i++) X_h[i] = (float)(rand() % 5 + 1);

    float *X_d, *Y_d;
    cudaMalloc(&X_d, N * sizeof(float));
    cudaMalloc(&Y_d, N * sizeof(float));
    cudaMemcpy(X_d, X_h, N * sizeof(float), cudaMemcpyHostToDevice);

    cpu_inclusive_scan(X_h, ref, N);

    // ── Inclusive scan ─────────────────────────────────────────────────────────
    Kogge_Stone_scan_inclusive_kernel<<<1, SECTION_SIZE>>>(X_d, Y_d, N);
    cudaDeviceSynchronize();
    cudaMemcpy(Y_h, Y_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Kogge-Stone inclusive (Fig 11.3): %s\n",
           verify(ref, Y_h, N) ? "PASS" : "FAIL");
    printf("  X[0..7]: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
           X_h[0],X_h[1],X_h[2],X_h[3],X_h[4],X_h[5],X_h[6],X_h[7]);
    printf("  Y[0..7]: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
           Y_h[0],Y_h[1],Y_h[2],Y_h[3],Y_h[4],Y_h[5],Y_h[6],Y_h[7]);

    // ── Exclusive scan ─────────────────────────────────────────────────────────
    float *excl_ref = (float *)malloc(N * sizeof(float));
    excl_ref[0] = 0.0f;
    for (unsigned int i = 1; i < N; i++) excl_ref[i] = ref[i-1];

    Kogge_Stone_scan_exclusive_kernel<<<1, SECTION_SIZE>>>(X_d, Y_d, N);
    cudaDeviceSynchronize();
    cudaMemcpy(Y_h, Y_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Kogge-Stone exclusive (Fig 11.4): %s\n",
           verify(excl_ref, Y_h, N) ? "PASS" : "FAIL");
    printf("  Y[0..7]: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f  (Y[0] must be 0)\n",
           Y_h[0],Y_h[1],Y_h[2],Y_h[3],Y_h[4],Y_h[5],Y_h[6],Y_h[7]);

    // ── Work efficiency analysis (§11.3) ─────────────────────────────────────
    printf("\nWork efficiency analysis (§11.3) for N=%u:\n", N);
    double ks_work = 0.0;
    for (unsigned int s = 1; s < N; s *= 2) ks_work += (N - s);
    printf("  Sequential:   %d additions  (O(N))\n", N - 1);
    printf("  Kogge-Stone:  %.0f additions  (O(N·log₂N))\n", ks_work);
    printf("  Extra work:   %.1fx more than sequential\n", ks_work / (N-1));
    printf("  Time steps:   %d  (log₂N with full parallelism)\n",
           (int)log2f((float)N));
    printf("  Useful for modest N (≤1024) with ample hardware resources.\n");

    free(X_h); free(Y_h); free(ref); free(excl_ref);
    cudaFree(X_d); cudaFree(Y_d);
    return 0;
}
