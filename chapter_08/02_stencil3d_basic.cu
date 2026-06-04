// §8.2 Parallel stencil: a basic algorithm
// Figure 8.6: 3D seven-point stencil sweep kernel.
//
// The 3D seven-point stencil (Fig 8.3C, order 1) computes the discrete
// Laplacian at each interior grid point:
//   out[i][j][k] = c0*in[i][j][k]
//                + c1*in[i][j][k-1] + c2*in[i][j][k+1]   (x neighbours)
//                + c3*in[i][j-1][k] + c4*in[i][j+1][k]   (y neighbours)
//                + c5*in[i-1][j][k] + c6*in[i+1][j][k]   (z neighbours)
//
// Thread assignment (Fig 8.6 lines 02-04):
//   i ← blockIdx.z * blockDim.z + threadIdx.z   (z / "i" axis)
//   j ← blockIdx.y * blockDim.y + threadIdx.y   (y / "j" axis)
//   k ← blockIdx.x * blockDim.x + threadIdx.x   (x / "k" axis — stride-1)
//
// Boundary conditions (§8.2 / Fig 8.5):
//   Boundary cells (i,j,k == 0 or N-1) hold initial conditions and are NOT
//   recalculated during a sweep.  They are copied to out unchanged.
//   Guard: i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1
//
// Arithmetic intensity:
//   7 input loads × 4 bytes = 28 bytes per output point
//   13 floating-point operations (7 muls + 6 adds) per output point
//   AI = 13 / 28 ≈ 0.46 OP/B  (severely memory-bound)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Laplacian coefficients for a 3D seven-point stencil
// (c0 = -6, c1..c6 = 1  gives the standard discrete Laplacian with h=1)
#define c0  (-6.0f)
#define c1   (1.0f)
#define c2   (1.0f)
#define c3   (1.0f)
#define c4   (1.0f)
#define c5   (1.0f)
#define c6   (1.0f)

// ── Figure 8.6 ────────────────────────────────────────────────────────────────
__global__ void stencil3d_basic_kernel(const float *in, float *out,
                                        unsigned int N) {
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    // Only interior points are updated; boundary cells hold BCs (§8.2)
    if (i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
        out[i*N*N + j*N + k] =
              c0 * in[i*N*N + j*N + k]
            + c1 * in[i*N*N + j*N + (k-1)]
            + c2 * in[i*N*N + j*N + (k+1)]
            + c3 * in[i*N*N + (j-1)*N + k]
            + c4 * in[i*N*N + (j+1)*N + k]
            + c5 * in[(i-1)*N*N + j*N + k]
            + c6 * in[(i+1)*N*N + j*N + k];
    }
}

// ── CPU reference ─────────────────────────────────────────────────────────────
static void cpu_stencil3d(const float *in, float *out, unsigned int N) {
    for (unsigned int i = 1; i < N-1; i++)
        for (unsigned int j = 1; j < N-1; j++)
            for (unsigned int k = 1; k < N-1; k++)
                out[i*N*N + j*N + k] =
                      c0 * in[i*N*N + j*N + k]
                    + c1 * in[i*N*N + j*N + (k-1)]
                    + c2 * in[i*N*N + j*N + (k+1)]
                    + c3 * in[i*N*N + (j-1)*N + k]
                    + c4 * in[i*N*N + (j+1)*N + k]
                    + c5 * in[(i-1)*N*N + j*N + k]
                    + c6 * in[(i+1)*N*N + j*N + k];
}

static int verify(const float *ref, const float *gpu, unsigned int n) {
    for (unsigned int idx = 0; idx < n; idx++) {
        float err = fabsf(ref[idx] - gpu[idx]);
        if (err > 1e-3f * (fabsf(ref[idx]) + 1.0f)) {
            printf("  MISMATCH idx=%u  ref=%.6f  gpu=%.6f\n", idx, ref[idx], gpu[idx]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    // Use N=64 for verification (fast), N=256 for benchmark timing
    const unsigned int NV = 64;    // verification grid
    const unsigned int NB = 256;   // benchmark grid

    // ── Correctness check (NV × NV × NV) ─────────────────────────────────────
    {
        unsigned long NE = (unsigned long)NV * NV * NV;
        float *in_h  = (float *)malloc(NE * sizeof(float));
        float *out_h = (float *)malloc(NE * sizeof(float));
        float *ref_h = (float *)malloc(NE * sizeof(float));

        // f(i,j,k) = i + j + k  →  Laplacian = 0 (linear function)
        for (unsigned int i = 0; i < NV; i++)
            for (unsigned int j = 0; j < NV; j++)
                for (unsigned int k = 0; k < NV; k++)
                    in_h[i*NV*NV + j*NV + k] = (float)(i + j + k);

        float *in_d, *out_d;
        cudaMalloc(&in_d,  NE * sizeof(float));
        cudaMalloc(&out_d, NE * sizeof(float));
        cudaMemcpy(in_d, in_h, NE * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(8, 8, 8);
        dim3 grid((NV + 7) / 8, (NV + 7) / 8, (NV + 7) / 8);

        stencil3d_basic_kernel<<<grid, block>>>(in_d, out_d, NV);
        cudaDeviceSynchronize();
        cudaMemcpy(out_h, out_d, NE * sizeof(float), cudaMemcpyDeviceToHost);

        cpu_stencil3d(in_h, ref_h, NV);
        printf("3D seven-point basic kernel: %s\n",
               verify(ref_h, out_h, (unsigned int)NE) ? "PASS" : "FAIL");

        // Spot-check: for f=i+j+k, Laplacian = c0*(i+j+k) + c1*(i+j+k-1) + ... = 0
        unsigned int ci = NV/2, cj = NV/2, ck = NV/2;
        printf("  Interior spot-check at (%u,%u,%u): GPU=%.4f  expected=0\n",
               ci, cj, ck, out_h[ci*NV*NV + cj*NV + ck]);

        free(in_h); free(out_h); free(ref_h);
        cudaFree(in_d); cudaFree(out_d);
    }

    // ── Performance benchmark (NB × NB × NB) ──────────────────────────────────
    {
        unsigned long NE = (unsigned long)NB * NB * NB;
        float *in_h  = (float *)malloc(NE * sizeof(float));
        float *in_d, *out_d;
        cudaMalloc(&in_d,  NE * sizeof(float));
        cudaMalloc(&out_d, NE * sizeof(float));

        srand(42);
        for (unsigned long e = 0; e < NE; e++) in_h[e] = (float)rand() / RAND_MAX;
        cudaMemcpy(in_d, in_h, NE * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(8, 8, 8);
        dim3 grid((NB + 7) / 8, (NB + 7) / 8, (NB + 7) / 8);

        // Warm-up
        stencil3d_basic_kernel<<<grid, block>>>(in_d, out_d, NB);
        cudaDeviceSynchronize();

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        stencil3d_basic_kernel<<<grid, block>>>(in_d, out_d, NB);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);

        long interior = (long)(NB-2) * (NB-2) * (NB-2);
        double flops   = 13.0 * interior;
        double bytes   = 7.0  * interior * sizeof(float);   // 7 loads per output
        printf("\nBenchmark grid: %u^3 (%lu interior points)\n", NB, interior);
        printf("Time: %.3f ms   GFLOPS: %.1f\n", ms, flops / (ms * 1e6));
        printf("Arithmetic intensity: %.2f OP/B  (= 13 / (7×4))\n",
               (float)(13.0 / (7.0 * 4.0)));
        printf("Effective bandwidth: %.1f GB/s\n", bytes / (ms * 1e6));

        free(in_h);
        cudaFree(in_d); cudaFree(out_d);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    return 0;
}
