// §8.4 Thread coarsening
// Figure 8.10: 3D seven-point stencil with thread coarsening in the z direction.
//
// Problem with the 3D shared memory kernel (§8.3):
//   The block size is limited to T^3 threads.  For T=8, the 3D block has only
//   512 threads and the 8×8×8 shared memory tile holds 512 elements.  This small
//   tile gives poor reuse (AI ≈ 1.37 OP/B) and poor memory coalescing because
//   each warp spans four different rows of the tile.
//
// Solution — thread coarsening (§8.4 / Fig 8.9):
//   Use a 2D block of T×T threads (e.g. 32×32 = 1024 threads).
//   Each thread iterates through a column of z-planes, computing one x-y
//   plane's worth of output per iteration.
//   Three 2D shared memory arrays (inPrev_s, inCurr_s, inNext_s) serve as a
//   sliding window across the z dimension.
//
// Tile dimensions:
//   IN_TILE_DIM  = 32   (x-y block/tile dimension)
//   OUT_TILE_DIM = 30   (= IN_TILE_DIM - 2, active output columns per tile)
//
// Memory layout per iteration:
//   inPrev_s ← z-1 plane  (loaded before loop)
//   inCurr_s ← z   plane  (loaded before loop)
//   inNext_s ← z+1 plane  (loaded inside loop)
//   After computation: slide inPrev_s ← inCurr_s, inCurr_s ← inNext_s
//
// Arithmetic intensity with T=32:
//   AI = 13/4 * (1 - 2/32)^3 = 3.25 × 0.824 ≈ 2.68 OP/B
//   (significant improvement over the T=8 shared memory kernel's 1.37 OP/B)
//
// Shared memory per block: 3 × T^2 × 4 = 3 × 1024 × 4 = 12 KB   (vs T^3=2MB)

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

#define IN_TILE_DIM  32
#define OUT_TILE_DIM (IN_TILE_DIM - 2)    // = 30

// ── Figure 8.10: thread coarsening (z direction) ──────────────────────────────
__global__ void stencil3d_coarsened_kernel(const float *in, float *out,
                                            unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;   // starting z index for this block
    int j = blockIdx.y * OUT_TILE_DIM + (int)threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + (int)threadIdx.x - 1;

    // Three 2D shared memory planes sliding along z
    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];

    // ── Prime the sliding window: load z-1 and z planes before the loop ────────
    if (iStart - 1 >= 0 && iStart - 1 < (int)N &&
        j >= 0 && j < (int)N && k >= 0 && k < (int)N)
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart-1)*N*N + j*N + k];
    else
        inPrev_s[threadIdx.y][threadIdx.x] = 0.0f;

    if (iStart >= 0 && iStart < (int)N &&
        j >= 0 && j < (int)N && k >= 0 && k < (int)N)
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart*N*N + j*N + k];
    else
        inCurr_s[threadIdx.y][threadIdx.x] = 0.0f;

    // ── Iterate through z-planes ───────────────────────────────────────────────
    for (int i = iStart; i < iStart + OUT_TILE_DIM; i++) {
        // Load the next z-plane into inNext_s
        if (i+1 >= 0 && i+1 < (int)N &&
            j >= 0 && j < (int)N && k >= 0 && k < (int)N)
            inNext_s[threadIdx.y][threadIdx.x] = in[(i+1)*N*N + j*N + k];
        else
            inNext_s[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute output for interior grid points only
        if (i >= 1 && i < (int)N-1 &&
            j >= 1 && j < (int)N-1 &&
            k >= 1 && k < (int)N-1) {
            // Active threads: not the halo layer in x-y
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM-1 &&
                threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM-1) {
                out[i*N*N + j*N + k] =
                      c0 * inCurr_s[threadIdx.y  ][threadIdx.x  ]
                    + c1 * inCurr_s[threadIdx.y  ][threadIdx.x-1]
                    + c2 * inCurr_s[threadIdx.y  ][threadIdx.x+1]
                    + c3 * inCurr_s[threadIdx.y-1][threadIdx.x  ]
                    + c4 * inCurr_s[threadIdx.y+1][threadIdx.x  ]
                    + c5 * inPrev_s[threadIdx.y  ][threadIdx.x  ]
                    + c6 * inNext_s[threadIdx.y  ][threadIdx.x  ];
            }
        }

        __syncthreads();

        // ── Slide the window: prev ← curr, curr ← next ────────────────────────
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
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

        dim3 blockC(IN_TILE_DIM, IN_TILE_DIM);
        dim3 gridC((NV + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                   (NV + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                   (NV + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

        stencil3d_coarsened_kernel<<<gridC, blockC>>>(in_d, out_d, NV);
        cudaDeviceSynchronize();
        cudaMemcpy(out_h, out_d, NE * sizeof(float), cudaMemcpyDeviceToHost);

        cpu_stencil3d(in_h, ref_h, NV);
        printf("3D coarsened stencil (z direction): %s\n",
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

        dim3 blockC(IN_TILE_DIM, IN_TILE_DIM);
        dim3 gridC((NB + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                   (NB + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                   (NB + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

        // Warm-up
        stencil3d_basic_kernel<<<gridB, blockB>>>(in_d, out_d, NB);
        stencil3d_coarsened_kernel<<<gridC, blockC>>>(in_d, out_d, NB);
        cudaDeviceSynchronize();

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        float ms_basic, ms_coarse;

        cudaEventRecord(t0);
        stencil3d_basic_kernel<<<gridB, blockB>>>(in_d, out_d, NB);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_basic, t0, t1);

        cudaEventRecord(t0);
        stencil3d_coarsened_kernel<<<gridC, blockC>>>(in_d, out_d, NB);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_coarse, t0, t1);

        long interior = (long)(NB-2) * (NB-2) * (NB-2);
        double flops = 13.0 * interior;

        float T = IN_TILE_DIM;
        float ai_coarse = (13.0f / 4.0f) * powf((T-2.0f)/T, 3.0f);

        printf("\nBenchmark grid: %u^3\n", NB);
        printf("%-32s  %6.3f ms  %5.1f GFLOPS  AI=%.2f OP/B\n",
               "Basic (8x8x8 block):", ms_basic,
               flops/(ms_basic*1e6), 13.0f/(7.0f*4.0f));
        printf("%-32s  %6.3f ms  %5.1f GFLOPS  AI=%.2f OP/B\n",
               "Coarsened (32x32, z-loop):", ms_coarse,
               flops/(ms_coarse*1e6), ai_coarse);
        printf("Speedup: %.2fx\n", ms_basic / ms_coarse);

        printf("\nIN_TILE_DIM=%d  OUT_TILE_DIM=%d\n", IN_TILE_DIM, OUT_TILE_DIM);
        printf("Shared memory per block: 3 × %d^2 × 4 = %d KB\n",
               IN_TILE_DIM, 3 * IN_TILE_DIM * IN_TILE_DIM * 4 / 1024);
        printf("AI formula: 13/4 × (1 − 2/%d)^3 = %.2f OP/B\n",
               IN_TILE_DIM, ai_coarse);

        free(in_h);
        cudaFree(in_d); cudaFree(out_d);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    return 0;
}
