// §8.5 Register tiling
// Figure 8.12: Thread coarsening with register tiling for the z neighbours.
//
// Observation (§8.5):
//   In the coarsened kernel (Fig 8.10), inPrev_s and inNext_s hold the z-1 and
//   z+1 planes.  Each element of these planes is accessed by exactly ONE thread
//   (the thread whose (j,k) coordinates match).  Because the data is private to
//   one thread, it can live in that thread's registers rather than shared memory.
//   The x-y plane inCurr_s must stay in shared memory so that threads sharing
//   the same x-y row can access each other's current values.
//
// Changes from Fig 8.10 to Fig 8.12:
//   inPrev_s  → register  float inPrev   (§8.5 — z-1 value, private to thread)
//   inNext_s  → register  float inNext   (§8.5 — z+1 value, private to thread)
//   inCurr_s  stays shared memory        (x-y neighbours must be shared)
//   inCurr    register copy of inCurr_s[ty][tx] for the center contribution
//
// Benefits:
//   1. Shared memory reduced from 3×T^2 to 1×T^2 elements per block.
//      For T=32: 1×1024×4 = 4 KB vs 12 KB previously.
//   2. Reads from inPrev_s and inNext_s become register reads (zero latency).
//   3. Global memory access pattern and total AI are unchanged:
//      same 3*T^2 loads per z-plane, same AI ≈ 2.68 OP/B for T=32.
//
// All four kernels from this chapter run here for a final benchmark table.

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

// ── Figure 8.12: register tiling ──────────────────────────────────────────────
__global__ void stencil3d_regtile_kernel(const float *in, float *out,
                                          unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + (int)threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + (int)threadIdx.x - 1;

    // z neighbours stored in per-thread registers (private — accessed by one thread)
    float inPrev;
    float inCurr;
    float inNext;

    // Only the current x-y plane stays in shared memory (needed by x-y neighbours)
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];

    // ── Prime: load z-1 and z planes into registers and shared memory ──────────
    if (iStart-1 >= 0 && iStart-1 < (int)N &&
        j >= 0 && j < (int)N && k >= 0 && k < (int)N)
        inPrev = in[(iStart-1)*N*N + j*N + k];
    else
        inPrev = 0.0f;

    if (iStart >= 0 && iStart < (int)N &&
        j >= 0 && j < (int)N && k >= 0 && k < (int)N) {
        inCurr = in[iStart*N*N + j*N + k];
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    } else {
        inCurr = 0.0f;
        inCurr_s[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // ── Iterate through z-planes ───────────────────────────────────────────────
    for (int i = iStart; i < iStart + OUT_TILE_DIM; i++) {
        // Load next z-plane into register (not shared memory)
        if (i+1 >= 0 && i+1 < (int)N &&
            j >= 0 && j < (int)N && k >= 0 && k < (int)N)
            inNext = in[(i+1)*N*N + j*N + k];
        else
            inNext = 0.0f;

        // Wait for all threads to have inCurr_s ready for this iteration
        __syncthreads();

        if (i >= 1 && i < (int)N-1 &&
            j >= 1 && j < (int)N-1 &&
            k >= 1 && k < (int)N-1) {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM-1 &&
                threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM-1) {
                out[i*N*N + j*N + k] =
                      c0 * inCurr                              // center (register)
                    + c1 * inCurr_s[threadIdx.y  ][threadIdx.x-1]  // x-1
                    + c2 * inCurr_s[threadIdx.y  ][threadIdx.x+1]  // x+1
                    + c3 * inCurr_s[threadIdx.y-1][threadIdx.x  ]  // y-1
                    + c4 * inCurr_s[threadIdx.y+1][threadIdx.x  ]  // y+1
                    + c5 * inPrev                              // z-1 (register)
                    + c6 * inNext;                             // z+1 (register)
            }
        }

        // Ensure all threads done reading inCurr_s before we update it
        __syncthreads();

        // ── Slide registers, update shared memory for next iteration ───────────
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;  // keep x-y plane up-to-date
    }
}

// ── Fig 8.10: coarsened (3 shared memory planes) for comparison ───────────────
__global__ void stencil3d_coarsened_kernel(const float *in, float *out,
                                            unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + (int)threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + (int)threadIdx.x - 1;

    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];

    auto load = [&](float *arr, int zi) {
        if (zi >= 0 && zi < (int)N && j >= 0 && j < (int)N && k >= 0 && k < (int)N)
            arr[threadIdx.y * IN_TILE_DIM + threadIdx.x] = in[zi*N*N + j*N + k];
        else
            arr[threadIdx.y * IN_TILE_DIM + threadIdx.x] = 0.0f;
    };
    load((float*)inPrev_s, iStart-1);
    load((float*)inCurr_s, iStart);

    for (int i = iStart; i < iStart + OUT_TILE_DIM; i++) {
        load((float*)inNext_s, i+1);
        __syncthreads();
        if (i >= 1 && i < (int)N-1 && j >= 1 && j < (int)N-1 && k >= 1 && k < (int)N-1) {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM-1 &&
                threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM-1)
                out[i*N*N + j*N + k] =
                      c0*inCurr_s[threadIdx.y  ][threadIdx.x  ]
                    + c1*inCurr_s[threadIdx.y  ][threadIdx.x-1] + c2*inCurr_s[threadIdx.y  ][threadIdx.x+1]
                    + c3*inCurr_s[threadIdx.y-1][threadIdx.x  ] + c4*inCurr_s[threadIdx.y+1][threadIdx.x  ]
                    + c5*inPrev_s[threadIdx.y  ][threadIdx.x  ] + c6*inNext_s[threadIdx.y  ][threadIdx.x  ];
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
}

// ── Fig 8.8: 3D shared memory tiling for comparison ──────────────────────────
#define SM_IN_TILE 8
#define SM_OUT_TILE (SM_IN_TILE - 2)
__global__ void stencil3d_shared_kernel(const float *in, float *out,
                                         unsigned int N) {
    int i = blockIdx.z * SM_OUT_TILE + (int)threadIdx.z - 1;
    int j = blockIdx.y * SM_OUT_TILE + (int)threadIdx.y - 1;
    int k = blockIdx.x * SM_OUT_TILE + (int)threadIdx.x - 1;
    __shared__ float in_s[SM_IN_TILE][SM_IN_TILE][SM_IN_TILE];
    if (i >= 0 && i < (int)N && j >= 0 && j < (int)N && k >= 0 && k < (int)N)
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N + j*N + k];
    else
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();
    if (i >= 1 && i < (int)N-1 && j >= 1 && j < (int)N-1 && k >= 1 && k < (int)N-1) {
        if (threadIdx.z >= 1 && threadIdx.z < SM_IN_TILE-1 &&
            threadIdx.y >= 1 && threadIdx.y < SM_IN_TILE-1 &&
            threadIdx.x >= 1 && threadIdx.x < SM_IN_TILE-1)
            out[i*N*N + j*N + k] =
                  c0*in_s[threadIdx.z  ][threadIdx.y  ][threadIdx.x  ]
                + c1*in_s[threadIdx.z  ][threadIdx.y  ][threadIdx.x-1] + c2*in_s[threadIdx.z  ][threadIdx.y  ][threadIdx.x+1]
                + c3*in_s[threadIdx.z  ][threadIdx.y-1][threadIdx.x  ] + c4*in_s[threadIdx.z  ][threadIdx.y+1][threadIdx.x  ]
                + c5*in_s[threadIdx.z-1][threadIdx.y  ][threadIdx.x  ] + c6*in_s[threadIdx.z+1][threadIdx.y  ][threadIdx.x  ];
    }
}

// ── Fig 8.6: basic kernel ─────────────────────────────────────────────────────
__global__ void stencil3d_basic_kernel(const float *in, float *out, unsigned int N) {
    unsigned int i = blockIdx.z*blockDim.z+threadIdx.z;
    unsigned int j = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int k = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1)
        out[i*N*N+j*N+k] =
              c0*in[i*N*N+j*N+k]
            + c1*in[i*N*N+j*N+(k-1)]    + c2*in[i*N*N+j*N+(k+1)]
            + c3*in[i*N*N+(j-1)*N+k]   + c4*in[i*N*N+(j+1)*N+k]
            + c5*in[(i-1)*N*N+j*N+k]   + c6*in[(i+1)*N*N+j*N+k];
}

static void cpu_stencil3d(const float *in, float *out, unsigned int N) {
    for (unsigned int i = 1; i < N-1; i++)
        for (unsigned int j = 1; j < N-1; j++)
            for (unsigned int k = 1; k < N-1; k++)
                out[i*N*N+j*N+k] =
                      c0*in[i*N*N+j*N+k]
                    + c1*in[i*N*N+j*N+(k-1)] + c2*in[i*N*N+j*N+(k+1)]
                    + c3*in[i*N*N+(j-1)*N+k] + c4*in[i*N*N+(j+1)*N+k]
                    + c5*in[(i-1)*N*N+j*N+k] + c6*in[(i+1)*N*N+j*N+k];
}

static int verify(const float *ref, const float *gpu, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        float err = fabsf(ref[i] - gpu[i]);
        if (err > 1e-3f * (fabsf(ref[i]) + 1.0f)) {
            printf("  MISMATCH i=%u  ref=%.6f  gpu=%.6f\n", i, ref[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    const unsigned int NV = 64;
    const unsigned int NB = 256;

    // ── Correctness ────────────────────────────────────────────────────────────
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
        dim3 gridC((NV+OUT_TILE_DIM-1)/OUT_TILE_DIM,
                   (NV+OUT_TILE_DIM-1)/OUT_TILE_DIM,
                   (NV+OUT_TILE_DIM-1)/OUT_TILE_DIM);

        stencil3d_regtile_kernel<<<gridC, blockC>>>(in_d, out_d, NV);
        cudaDeviceSynchronize();
        cudaMemcpy(out_h, out_d, NE * sizeof(float), cudaMemcpyDeviceToHost);

        cpu_stencil3d(in_h, ref_h, NV);
        printf("Register-tiling kernel: %s\n\n",
               verify(ref_h, out_h, (unsigned int)NE) ? "PASS" : "FAIL");

        free(in_h); free(out_h); free(ref_h);
        cudaFree(in_d); cudaFree(out_d);
    }

    // ── Four-way benchmark ─────────────────────────────────────────────────────
    {
        unsigned long NE = (unsigned long)NB * NB * NB;
        float *in_h = (float *)malloc(NE * sizeof(float));
        float *in_d, *out_d;
        cudaMalloc(&in_d, NE * sizeof(float));
        cudaMalloc(&out_d, NE * sizeof(float));
        srand(42);
        for (unsigned long e = 0; e < NE; e++) in_h[e] = (float)rand() / RAND_MAX;
        cudaMemcpy(in_d, in_h, NE * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockB(8, 8, 8),  gridB((NB+7)/8, (NB+7)/8, (NB+7)/8);
        dim3 blockS(SM_IN_TILE, SM_IN_TILE, SM_IN_TILE),
             gridS((NB+SM_OUT_TILE-1)/SM_OUT_TILE, (NB+SM_OUT_TILE-1)/SM_OUT_TILE, (NB+SM_OUT_TILE-1)/SM_OUT_TILE);
        dim3 blockC(IN_TILE_DIM, IN_TILE_DIM),
             gridC((NB+OUT_TILE_DIM-1)/OUT_TILE_DIM, (NB+OUT_TILE_DIM-1)/OUT_TILE_DIM, (NB+OUT_TILE_DIM-1)/OUT_TILE_DIM);

        // Warm-up
        stencil3d_basic_kernel<<<gridB,blockB>>>(in_d, out_d, NB);
        stencil3d_shared_kernel<<<gridS,blockS>>>(in_d, out_d, NB);
        stencil3d_coarsened_kernel<<<gridC,blockC>>>(in_d, out_d, NB);
        stencil3d_regtile_kernel<<<gridC,blockC>>>(in_d, out_d, NB);
        cudaDeviceSynchronize();

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        float ms[4];
        #define TIME(k,g,b,idx) do { \
            cudaEventRecord(t0); k<<<g,b>>>(in_d,out_d,NB); \
            cudaEventRecord(t1); cudaEventSynchronize(t1); \
            cudaEventElapsedTime(&ms[idx], t0, t1); } while(0)
        TIME(stencil3d_basic_kernel,   gridB, blockB, 0);
        TIME(stencil3d_shared_kernel,  gridS, blockS, 1);
        TIME(stencil3d_coarsened_kernel, gridC, blockC, 2);
        TIME(stencil3d_regtile_kernel, gridC, blockC, 3);

        long interior = (long)(NB-2)*(NB-2)*(NB-2);
        double flops = 13.0 * interior;
        float T = IN_TILE_DIM;
        float ai[4] = {
            13.0f / (7.0f*4.0f),                              // basic
            13.0f*(float)(SM_IN_TILE-2)*(SM_IN_TILE-2)*(SM_IN_TILE-2) / (4.0f*(float)SM_IN_TILE*SM_IN_TILE*SM_IN_TILE),  // shared mem
            (13.0f/4.0f) * powf((T-2)/T, 3.0f),               // coarsened
            (13.0f/4.0f) * powf((T-2)/T, 3.0f),               // register (same AI)
        };
        const char *names[] = {
            "1. Basic (Fig 8.6):",
            "2. Shared mem 8^3 (Fig 8.8):",
            "3. Coarsened z (Fig 8.10):",
            "4. Register tiling (Fig 8.12):"
        };

        printf("Grid: %u^3   Filter: 7-point (order 1)\n\n", NB);
        for (int k = 0; k < 4; k++)
            printf("%-34s  %6.3f ms  %5.1f GFLOPS  AI=%.2f OP/B\n",
                   names[k], ms[k], flops/(ms[k]*1e6), ai[k]);
        printf("\nSpeedup over basic:  "
               "shmem=%.2fx  coarsened=%.2fx  regtile=%.2fx\n",
               ms[0]/ms[1], ms[0]/ms[2], ms[0]/ms[3]);

        printf("\nMemory per block (IN_TILE=%d):\n", IN_TILE_DIM);
        printf("  Fig 8.8  (3D tile):        %d KB\n",
               IN_TILE_DIM*IN_TILE_DIM*IN_TILE_DIM*4/1024);
        printf("  Fig 8.10 (3 shared planes): %d KB\n",
               3*IN_TILE_DIM*IN_TILE_DIM*4/1024);
        printf("  Fig 8.12 (1 shared plane):  %d KB  (+ 3 regs/thread)\n",
               IN_TILE_DIM*IN_TILE_DIM*4/1024);

        free(in_h);
        cudaFree(in_d); cudaFree(out_d);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    return 0;
}
