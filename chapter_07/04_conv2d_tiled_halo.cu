// §7.4 Tiled convolution with halo cells
// Figure 7.12: tiled 2D convolution — threads collaboratively load an input
// tile (including halo elements) into shared memory before computing output.
//
// Key design decisions (§7.4):
//   Thread organisation #1 (Fig 7.12):
//     Block size = IN_TILE_DIM × IN_TILE_DIM  (same as input tile)
//     Output tile = OUT_TILE_DIM × OUT_TILE_DIM = (IN-2r) × (IN-2r)
//     Grid launched based on output tile size.
//
//   Loading the input tile:
//     col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS
//     row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS
//     Ghost cells (col or row outside image) → N_s[ty][tx] = 0.0f
//
//   Computing output elements:
//     Active threads: tileCol ∈ [0,OUT_TILE_DIM) and tileRow ∈ [0,OUT_TILE_DIM)
//     These threads access N_s[tileRow+fRow][tileCol+fCol] for the filter loop.
//
// Arithmetic-to-global-memory access ratio (§7.4 / Fig 7.14):
//   OUT^2 * (2r+1)^2 * 2 FLOP
//   ─────────────────────────────  =  9.57 OP/B   for 5×5 filter, 28×28 output
//   IN^2 * 4 bytes

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef FILTER_RADIUS
#define FILTER_RADIUS 2
#endif
#define FILTER_DIM    (2 * FILTER_RADIUS + 1)
#define IN_TILE_DIM   32
#define OUT_TILE_DIM  (IN_TILE_DIM - 2 * FILTER_RADIUS)

// Compile-time assertion: OUT_TILE_DIM must be positive
#if OUT_TILE_DIM <= 0
#error "FILTER_RADIUS too large for IN_TILE_DIM=32"
#endif

// Filter in constant memory (§7.3) — assumed for all tiled variants
__constant__ float F_c[FILTER_DIM][FILTER_DIM];

// ── Fig 7.12 ─────────────────────────────────────────────────────────────────
__global__ void convolution_tiled_2D_const_mem_kernel(const float *N, float *P,
                                                        int width, int height) {
    // Global (col, row) of the N element this thread loads.
    // Offset by -FILTER_RADIUS so halo threads map to the border of the input tile.
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    // Collaborative load of the full input tile (including halo)
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < height && col >= 0 && col < width)
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    else
        N_s[threadIdx.y][threadIdx.x] = 0.0f;   // ghost cell
    __syncthreads();

    // Only the inner OUT_TILE_DIM × OUT_TILE_DIM threads compute output.
    // tileCol/tileRow is the thread's position relative to the output tile origin.
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    // Guard: thread must be (a) an active output tile thread and
    //        (b) its output element must be inside the image.
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM &&
            tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++)
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++)
                    Pvalue += F_c[fRow][fCol] *
                              N_s[tileRow + fRow][tileCol + fCol];
            P[row * width + col] = Pvalue;
        }
    }
}

// ── Constant-memory baseline (no tiling) ─────────────────────────────────────
__global__ void convolution_2D_const_mem_kernel(const float *N, float *P,
                                                 int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outCol >= width || outRow >= height) return;

    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < FILTER_DIM; fRow++)
        for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
            int inRow = outRow - FILTER_RADIUS + fRow;
            int inCol = outCol - FILTER_RADIUS + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                Pvalue += F_c[fRow][fCol] * N[inRow*width + inCol];
        }
    P[outRow*width + outCol] = Pvalue;
}

// ── CPU reference ─────────────────────────────────────────────────────────────
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

int main(void) {
    const int W  = 2048;
    const int H  = 2048;
    const int R  = FILTER_RADIUS;
    const int FD = FILTER_DIM;
    const long NE = (long)W * H;
    const long FE = (long)FD * FD;

    float *N_h   = (float *)malloc(NE * sizeof(float));
    float *F_h   = (float *)malloc(FE * sizeof(float));
    float *P_ref = (float *)malloc(NE * sizeof(float));
    float *P_gpu = (float *)malloc(NE * sizeof(float));

    srand(42);
    for (long i = 0; i < NE; i++) N_h[i] = (float)rand() / RAND_MAX;
    for (long f = 0; f < FE; f++) F_h[f] = 1.0f / FE;

    float *N_d, *P_d;
    cudaMalloc(&N_d, NE * sizeof(float));
    cudaMalloc(&P_d, NE * sizeof(float));
    cudaMemcpy(N_d, N_h, NE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F_c, F_h, FE * sizeof(float));

    // Grid for tiled kernel: based on output tile size
    dim3 blockTiled(IN_TILE_DIM, IN_TILE_DIM);
    dim3 gridTiled((W + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                   (H + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    // Grid for baseline (no tiling): one thread per output element
    dim3 blockBase(16, 16);
    dim3 gridBase((W + 15) / 16, (H + 15) / 16);

    // Warm-up
    convolution_tiled_2D_const_mem_kernel<<<gridTiled, blockTiled>>>(N_d, P_d, W, H);
    convolution_2D_const_mem_kernel<<<gridBase, blockBase>>>(N_d, P_d, W, H);
    cudaDeviceSynchronize();

    // Correctness
    conv2d_cpu(N_h, F_h, P_ref, R, W, H);
    convolution_tiled_2D_const_mem_kernel<<<gridTiled, blockTiled>>>(N_d, P_d, W, H);
    cudaDeviceSynchronize();
    cudaMemcpy(P_gpu, P_d, NE * sizeof(float), cudaMemcpyDeviceToHost);
    int ok = verify(P_ref, P_gpu, (int)NE);
    printf("Tiled halo kernel: %s\n", ok ? "PASS" : "FAIL");

    // Timing
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms_base, ms_tiled;

    cudaEventRecord(t0);
    convolution_2D_const_mem_kernel<<<gridBase, blockBase>>>(N_d, P_d, W, H);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_base, t0, t1);

    cudaEventRecord(t0);
    convolution_tiled_2D_const_mem_kernel<<<gridTiled, blockTiled>>>(N_d, P_d, W, H);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_tiled, t0, t1);

    double flops = 2.0 * (double)NE * FE;

    // Arithmetic intensity formula from §7.4:
    //   OUT^2 * FD^2 * 2  /  (IN^2 * 4)
    float ai_tiled = (float)OUT_TILE_DIM * OUT_TILE_DIM * FE * 2.0f /
                     ((float)IN_TILE_DIM * IN_TILE_DIM * 4.0f);
    float ai_const = (float)(2 * FE) / (float)(FE * 4);  // 0.5 OP/B

    printf("\nFilter radius: %d  filter: %dx%d\n", R, FD, FD);
    printf("IN_TILE_DIM=%d  OUT_TILE_DIM=%d\n", IN_TILE_DIM, OUT_TILE_DIM);
    printf("\n%-32s  %6.3f ms  %5.1f GFLOPS  AI=%.2f OP/B\n",
           "Const mem (no tiling):", ms_base, flops/(ms_base*1e6), ai_const);
    printf("%-32s  %6.3f ms  %5.1f GFLOPS  AI=%.2f OP/B\n",
           "Tiled halo (shared mem):", ms_tiled, flops/(ms_tiled*1e6), ai_tiled);
    printf("Speedup: %.2fx\n", ms_base / ms_tiled);

    printf("\nFig 7.14 — AI for 5×5 filter, varying IN_TILE_DIM:\n");
    printf("  IN_TILE_DIM=8  → OUT=4  → AI=%.2f OP/B\n",
           4.0f*4.0f*25.0f*2.0f / (8.0f*8.0f*4.0f));
    printf("  IN_TILE_DIM=16 → OUT=12 → AI=%.2f OP/B\n",
           12.0f*12.0f*25.0f*2.0f / (16.0f*16.0f*4.0f));
    printf("  IN_TILE_DIM=32 → OUT=28 → AI=%.2f OP/B\n",
           28.0f*28.0f*25.0f*2.0f / (32.0f*32.0f*4.0f));
    printf("  Asymptotic bound (OUT_TILE_DIM >> 2r): %.2f OP/B\n",
           (float)(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*2.0f/4.0f);

    free(N_h); free(F_h); free(P_ref); free(P_gpu);
    cudaFree(N_d); cudaFree(P_d);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ok ? 0 : 1;
}
