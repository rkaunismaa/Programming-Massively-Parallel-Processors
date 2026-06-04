// §8.3 Shared memory tiling for stencil sweep
// Figure 8.8: 3D seven-point stencil with shared memory tiling.
//
// Design is almost identical to the tiled convolution in Chapter 7, with one
// important difference: stencil input tiles do NOT include corner grid points
// (only face-adjacent neighbours are needed), whereas convolution needs corners.
// This makes the arithmetic-to-memory ratio of stencil tiling lower than
// convolution tiling for the same tile size (§8.3).
//
// Tile dimensions:
//   IN_TILE_DIM  = 8   (block size per dimension — limited to 8 so that
//                       8×8×8 = 512 threads, staying under the 1024 limit)
//   OUT_TILE_DIM = 6   (= IN_TILE_DIM - 2)
//
// Each thread maps to an IN-tile element:
//   i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1
//   j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1
//   k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1
// (The -1 offset makes thread (0,0,0) cover the -1 halo layer.)
//
// Arithmetic intensity (§8.3):
//   AI = 13*(T-2)^3 / (4*T^3) = 13*(6)^3 / (4*(8)^3) ≈ 1.37 OP/B  for T=8
//   Upper bound as T→∞: 13/4 = 3.25 OP/B
//   Compare: 3D 7×7×7 convolution bound = 343*2/4 = 171.5 OP/B — much higher,
//   because convolution uses all corners; stencil only uses 6+1 points.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define c0  (-6.0f)
#define c1   (1.0f)
#define c2   (1.0f)
#define c3   (1.0f)
#define c4   (1.0f)
#define c5   (1.0f)
#define c6   (1.0f)

#define IN_TILE_DIM  8
#define OUT_TILE_DIM (IN_TILE_DIM - 2)     // = 6

// ── Figure 8.8 ────────────────────────────────────────────────────────────────
__global__ void stencil3d_shared_kernel(const float *in, float *out,
                                         unsigned int N) {
    // Map this thread to a global grid point (may be halo or ghost)
    int i = blockIdx.z * OUT_TILE_DIM + (int)threadIdx.z - 1;
    int j = blockIdx.y * OUT_TILE_DIM + (int)threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + (int)threadIdx.x - 1;

    // Collaboratively load the full IN_TILE_DIM^3 input tile
    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    // Guard: load only valid (non-ghost) grid points; ghost cells stay 0
    if (i >= 0 && i < (int)N && j >= 0 && j < (int)N && k >= 0 && k < (int)N)
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N + j*N + k];
    else
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // Only active (interior-of-tile) threads compute output.
    // Also enforce the global boundary condition: don't update boundary cells.
    if (i >= 1 && i < (int)N-1 && j >= 1 && j < (int)N-1 && k >= 1 && k < (int)N-1) {
        if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM-1 &&
            threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM-1 &&
            threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM-1) {
            out[i*N*N + j*N + k] =
                  c0 * in_s[threadIdx.z  ][threadIdx.y  ][threadIdx.x  ]
                + c1 * in_s[threadIdx.z  ][threadIdx.y  ][threadIdx.x-1]
                + c2 * in_s[threadIdx.z  ][threadIdx.y  ][threadIdx.x+1]
                + c3 * in_s[threadIdx.z  ][threadIdx.y-1][threadIdx.x  ]
                + c4 * in_s[threadIdx.z  ][threadIdx.y+1][threadIdx.x  ]
                + c5 * in_s[threadIdx.z-1][threadIdx.y  ][threadIdx.x  ]
                + c6 * in_s[threadIdx.z+1][threadIdx.y  ][threadIdx.x  ];
        }
    }
}

// ── Basic kernel for comparison ───────────────────────────────────────────────
__global__ void stencil3d_basic_kernel(const float *in, float *out,
                                        unsigned int N) {
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1)
        out[i*N*N + j*N + k] =
              c0*in[i*N*N + j*N + k]
            + c1*in[i*N*N + j*N + (k-1)]  + c2*in[i*N*N + j*N + (k+1)]
            + c3*in[i*N*N + (j-1)*N + k]  + c4*in[i*N*N + (j+1)*N + k]
            + c5*in[(i-1)*N*N + j*N + k]  + c6*in[(i+1)*N*N + j*N + k];
}

// ── CPU reference ─────────────────────────────────────────────────────────────
static void cpu_stencil3d(const float *in, float *out, unsigned int N) {
    for (unsigned int i = 1; i < N-1; i++)
        for (unsigned int j = 1; j < N-1; j++)
            for (unsigned int k = 1; k < N-1; k++)
                out[i*N*N + j*N + k] =
                      c0*in[i*N*N + j*N + k]
                    + c1*in[i*N*N + j*N + (k-1)] + c2*in[i*N*N + j*N + (k+1)]
                    + c3*in[i*N*N + (j-1)*N + k] + c4*in[i*N*N + (j+1)*N + k]
                    + c5*in[(i-1)*N*N + j*N + k] + c6*in[(i+1)*N*N + j*N + k];
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
    const unsigned int NV = 64;
    const unsigned int NB = 256;

    // ── Correctness check ──────────────────────────────────────────────────────
    {
        unsigned long NE = (unsigned long)NV * NV * NV;
        float *in_h  = (float *)malloc(NE * sizeof(float));
        float *out_h = (float *)malloc(NE * sizeof(float));
        float *ref_h = (float *)malloc(NE * sizeof(float));

        for (unsigned int i = 0; i < NV; i++)
            for (unsigned int j = 0; j < NV; j++)
                for (unsigned int k = 0; k < NV; k++)
                    in_h[i*NV*NV + j*NV + k] = (float)(i + j + k);

        float *in_d, *out_d;
        cudaMalloc(&in_d,  NE * sizeof(float));
        cudaMalloc(&out_d, NE * sizeof(float));
        cudaMemcpy(in_d, in_h, NE * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSM(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
        dim3 gridSM((NV + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                    (NV + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                    (NV + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

        stencil3d_shared_kernel<<<gridSM, blockSM>>>(in_d, out_d, NV);
        cudaDeviceSynchronize();
        cudaMemcpy(out_h, out_d, NE * sizeof(float), cudaMemcpyDeviceToHost);

        cpu_stencil3d(in_h, ref_h, NV);
        printf("3D shared-mem tiled stencil: %s\n",
               verify(ref_h, out_h, (unsigned int)NE) ? "PASS" : "FAIL");

        free(in_h); free(out_h); free(ref_h);
        cudaFree(in_d); cudaFree(out_d);
    }

    // ── Performance benchmark ──────────────────────────────────────────────────
    {
        unsigned long NE = (unsigned long)NB * NB * NB;
        float *in_h = (float *)malloc(NE * sizeof(float));
        float *in_d, *out_d;
        cudaMalloc(&in_d, NE * sizeof(float));
        cudaMalloc(&out_d, NE * sizeof(float));
        srand(42);
        for (unsigned long e = 0; e < NE; e++) in_h[e] = (float)rand() / RAND_MAX;
        cudaMemcpy(in_d, in_h, NE * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockB(8, 8, 8);
        dim3 gridB((NB + 7) / 8, (NB + 7) / 8, (NB + 7) / 8);

        dim3 blockSM(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
        dim3 gridSM((NB + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                    (NB + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                    (NB + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

        // Warm-up
        stencil3d_basic_kernel<<<gridB, blockB>>>(in_d, out_d, NB);
        stencil3d_shared_kernel<<<gridSM, blockSM>>>(in_d, out_d, NB);
        cudaDeviceSynchronize();

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        float ms_basic, ms_shared;

        cudaEventRecord(t0);
        stencil3d_basic_kernel<<<gridB, blockB>>>(in_d, out_d, NB);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_basic, t0, t1);

        cudaEventRecord(t0);
        stencil3d_shared_kernel<<<gridSM, blockSM>>>(in_d, out_d, NB);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_shared, t0, t1);

        long interior = (long)(NB-2) * (NB-2) * (NB-2);
        double flops = 13.0 * interior;

        // AI for shared memory tiling: 13*(T-2)^3 / (4*T^3)
        float T = IN_TILE_DIM;
        float ai_sm = 13.0f * (T-2)*(T-2)*(T-2) / (4.0f * T*T*T);

        printf("\nBenchmark grid: %u^3  IN_TILE=%d  OUT_TILE=%d\n",
               NB, IN_TILE_DIM, OUT_TILE_DIM);
        printf("%-32s  %6.3f ms  %5.1f GFLOPS  AI=%.2f OP/B\n",
               "Basic (global memory):", ms_basic,
               flops / (ms_basic * 1e6), 13.0f / (7.0f * 4.0f));
        printf("%-32s  %6.3f ms  %5.1f GFLOPS  AI=%.2f OP/B\n",
               "Shared mem tiled:", ms_shared,
               flops / (ms_shared * 1e6), ai_sm);
        printf("Speedup: %.2fx\n", ms_basic / ms_shared);

        printf("\nFig 8.3 arithmetic intensity bounds (seven-point stencil):\n");
        for (int Ti = 4; Ti <= 16; Ti += 2) {
            float ai = 13.0f * (float)(Ti-2)*(Ti-2)*(Ti-2) / (4.0f*(float)Ti*Ti*Ti);
            printf("  T=%-2d  OUT=%-2d  AI=%.2f OP/B\n", Ti, Ti-2, ai);
        }
        printf("  T=∞       → AI=%.2f OP/B (upper bound)\n", 13.0f / 4.0f);

        free(in_h);
        cudaFree(in_d); cudaFree(out_d);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    return 0;
}
