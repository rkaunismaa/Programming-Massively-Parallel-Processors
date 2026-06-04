// §16.3  Convolutional layer — basic CUDA inference kernel (Fig 16.15)
//
// Each output element Y[n, m, h, w] is computed by one thread.
// Thread organisation (§16.3, Fig 16.13):
//   Block : (TILE_WIDTH, TILE_WIDTH, 1) — 2-D tile of output pixels
//   Grid  : (M, T, N) where T = H_grid * W_grid (linearised tile index)
//
// Array layouts (row-major, §16.2):
//   X  [N, C, H,    W   ]   — input  feature maps
//   W  [M, C, K,    K   ]   — filter banks (M output maps, each C×K×K)
//   Y  [N, M, H_out, W_out] — output feature maps, H_out = H-K+1
//
// The innermost c-, p-, q-loops are kept serial to avoid atomics on Y
// accumulation (§16.3).  They are the "easy" parallelism left for a
// future tiled / shared-memory optimisation (exercise in the book).
//
// Reference: Fig 16.6  (sequential CPU), Fig 16.12 (minibatch CPU),
//            Fig 16.13 (host launch), Fig 16.15 (CUDA kernel).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 4      // 4×4 = 16 threads/block (small to keep register usage low)

// ── flat index helpers ────────────────────────────────────────────────────────
// Using #define instead of inline functions to keep close to the book's
// multidimensional indexing notation.
#define IDX_X(n,c,h,w)   ((n)*(C)*(H)*(W)   + (c)*(H)*(W)   + (h)*(W)   + (w))
#define IDX_W(m,c,p,q)   ((m)*(C)*(K)*(K)   + (c)*(K)*(K)   + (p)*(K)   + (q))
#define IDX_Y(n,m,h,w)   ((n)*(M)*(Ho)*(Wo) + (m)*(Ho)*(Wo) + (h)*(Wo)  + (w))

// ── CPU reference (Fig 16.12 — minibatch forward) ────────────────────────────
static void conv_forward_cpu(int N, int M, int C, int H, int W, int K,
                              const float *X, const float *Wf, float *Y) {
    int Ho = H - K + 1, Wo = W - K + 1;
    for (int n = 0; n < N; n++)
        for (int m = 0; m < M; m++)
            for (int h = 0; h < Ho; h++)
                for (int w = 0; w < Wo; w++) {
                    float acc = 0.f;
                    for (int c = 0; c < C; c++)
                        for (int p = 0; p < K; p++)
                            for (int q = 0; q < K; q++)
                                acc += X[n*C*H*W + c*H*W + (h+p)*W + (w+q)]
                                     * Wf[m*C*K*K + c*K*K + p*K + q];
                    Y[n*M*Ho*Wo + m*Ho*Wo + h*Wo + w] = acc;
                }
}

// ── CUDA kernel (Fig 16.15) ───────────────────────────────────────────────────
// Parameters passed by value to avoid repeated global-memory pointer chasing.
__global__ void ConvLayerForward_Kernel(int N, int M, int C, int H, int W, int K,
                                        int W_grid,
                                        const float * __restrict__ X,
                                        const float * __restrict__ Wf,
                                        float *Y) {
    int Ho = H - K + 1;
    int Wo = W - K + 1;

    // Output element addressed by this thread
    int m = blockIdx.x;                                          // output feature map
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;   // output row
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;   // output col
    int n = blockIdx.z;                                          // minibatch sample

    if (h >= Ho || w >= Wo) return;   // boundary guard for non-multiple sizes

    float acc = 0.f;
    for (int c = 0; c < C; c++)          // sum over input channels
        for (int p = 0; p < K; p++)      // filter row
            for (int q = 0; q < K; q++)  // filter col
                acc += X[n*C*H*W + c*H*W + (h+p)*W + (w+q)]
                     * Wf[m*C*K*K + c*K*K + p*K + q];

    Y[n*M*Ho*Wo + m*Ho*Wo + h*Wo + w] = acc;
}

static void launch_conv(int N, int M, int C, int H, int W, int K,
                        const float *d_X, const float *d_Wf, float *d_Y) {
    int Ho = H - K + 1;
    int Wo = W - K + 1;
    int H_grid = (Ho + TILE_WIDTH - 1) / TILE_WIDTH;
    int W_grid = (Wo + TILE_WIDTH - 1) / TILE_WIDTH;
    int T = H_grid * W_grid;   // tiles per output feature map (linearised into Y dim)

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(M, T, N);    // Fig 16.13
    ConvLayerForward_Kernel<<<gridDim, blockDim>>>(N, M, C, H, W, K, W_grid,
                                                   d_X, d_Wf, d_Y);
}

static int verify(const float *ref, const float *gpu, int n, float tol) {
    for (int i = 0; i < n; i++) {
        float err = fabsf(ref[i] - gpu[i]);
        if (err > tol * (1.f + fabsf(ref[i]))) {
            printf("  MISMATCH [%d]: ref=%.5f gpu=%.5f\n", i, ref[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    printf("=== Convolutional Layer: Basic CUDA Inference Kernel (§16.3) ===\n\n");

    // ── Test 1: tiny, hand-checkable ─────────────────────────────────────────
    {
        int N=1, M=1, C=1, H=5, W=5, K=3;
        int Ho=H-K+1, Wo=W-K+1;
        int szX=N*C*H*W, szWf=M*C*K*K, szY=N*M*Ho*Wo;

        float *h_X   = (float *)malloc(szX  * sizeof(float));
        float *h_Wf  = (float *)malloc(szWf * sizeof(float));
        float *h_Ycpu = (float *)malloc(szY  * sizeof(float));
        float *h_Ygpu = (float *)malloc(szY  * sizeof(float));

        for (int i = 0; i < szX;  i++) h_X[i]  = (float)i;   // 0..24
        for (int i = 0; i < szWf; i++) h_Wf[i] = 1.f;        // all-ones filter

        conv_forward_cpu(N, M, C, H, W, K, h_X, h_Wf, h_Ycpu);

        float *d_X, *d_Wf, *d_Y;
        cudaMalloc(&d_X,  szX  * sizeof(float));
        cudaMalloc(&d_Wf, szWf * sizeof(float));
        cudaMalloc(&d_Y,  szY  * sizeof(float));
        cudaMemcpy(d_X,  h_X,  szX  * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Wf, h_Wf, szWf * sizeof(float), cudaMemcpyHostToDevice);

        launch_conv(N, M, C, H, W, K, d_X, d_Wf, d_Y);
        cudaDeviceSynchronize();
        cudaMemcpy(h_Ygpu, d_Y, szY * sizeof(float), cudaMemcpyDeviceToHost);

        printf("Tiny test (N=1,M=1,C=1,H=5,W=5,K=3) — all-ones filter:\n");
        printf("  CPU output [%dx%d]: ", Ho, Wo);
        for (int i = 0; i < szY; i++) printf("%.0f ", h_Ycpu[i]);
        printf("\n  GPU output [%dx%d]: ", Ho, Wo);
        for (int i = 0; i < szY; i++) printf("%.0f ", h_Ygpu[i]);
        printf("\n  %s\n\n", verify(h_Ycpu, h_Ygpu, szY, 1e-5f) ? "PASS" : "FAIL");

        free(h_X); free(h_Wf); free(h_Ycpu); free(h_Ygpu);
        cudaFree(d_X); cudaFree(d_Wf); cudaFree(d_Y);
    }

    // ── Test 2: multi-channel, multi-map ─────────────────────────────────────
    {
        int N=4, M=6, C=3, H=10, W=10, K=3;
        int Ho=H-K+1, Wo=W-K+1;
        int szX=N*C*H*W, szWf=M*C*K*K, szY=N*M*Ho*Wo;

        float *h_X    = (float *)malloc(szX  * sizeof(float));
        float *h_Wf   = (float *)malloc(szWf * sizeof(float));
        float *h_Ycpu = (float *)malloc(szY  * sizeof(float));
        float *h_Ygpu = (float *)malloc(szY  * sizeof(float));

        srand(42);
        for (int i = 0; i < szX;  i++) h_X[i]  = (rand()/(float)RAND_MAX)*2.f - 1.f;
        for (int i = 0; i < szWf; i++) h_Wf[i] = (rand()/(float)RAND_MAX)*0.4f - 0.2f;

        conv_forward_cpu(N, M, C, H, W, K, h_X, h_Wf, h_Ycpu);

        float *d_X, *d_Wf, *d_Y;
        cudaMalloc(&d_X,  szX  * sizeof(float));
        cudaMalloc(&d_Wf, szWf * sizeof(float));
        cudaMalloc(&d_Y,  szY  * sizeof(float));
        cudaMemcpy(d_X,  h_X,  szX  * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Wf, h_Wf, szWf * sizeof(float), cudaMemcpyHostToDevice);

        launch_conv(N, M, C, H, W, K, d_X, d_Wf, d_Y);
        cudaDeviceSynchronize();
        cudaMemcpy(h_Ygpu, d_Y, szY * sizeof(float), cudaMemcpyDeviceToHost);

        printf("Multi-channel test (N=%d,M=%d,C=%d,H=%d,W=%d,K=%d):\n",
               N, M, C, H, W, K);
        printf("  %s\n\n", verify(h_Ycpu, h_Ygpu, szY, 1e-4f) ? "PASS" : "FAIL");

        free(h_X); free(h_Wf); free(h_Ycpu); free(h_Ygpu);
        cudaFree(d_X); cudaFree(d_Wf); cudaFree(d_Y);
    }

    // ── Test 3: larger — timing ───────────────────────────────────────────────
    {
        // Comparable to LeNet C1: 6 maps, 5×5 filters on 32×32 input
        int N=8, M=16, C=3, H=32, W=32, K=5;
        int Ho=H-K+1, Wo=W-K+1;
        int szX=N*C*H*W, szWf=M*C*K*K, szY=N*M*Ho*Wo;

        float *h_X    = (float *)malloc(szX  * sizeof(float));
        float *h_Wf   = (float *)malloc(szWf * sizeof(float));
        float *h_Ycpu = (float *)malloc(szY  * sizeof(float));
        float *h_Ygpu = (float *)malloc(szY  * sizeof(float));

        srand(7);
        for (int i = 0; i < szX;  i++) h_X[i]  = (rand()/(float)RAND_MAX)*2.f - 1.f;
        for (int i = 0; i < szWf; i++) h_Wf[i] = (rand()/(float)RAND_MAX)*0.2f - 0.1f;

        conv_forward_cpu(N, M, C, H, W, K, h_X, h_Wf, h_Ycpu);

        float *d_X, *d_Wf, *d_Y;
        cudaMalloc(&d_X,  szX  * sizeof(float));
        cudaMalloc(&d_Wf, szWf * sizeof(float));
        cudaMalloc(&d_Y,  szY  * sizeof(float));
        cudaMemcpy(d_X,  h_X,  szX  * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Wf, h_Wf, szWf * sizeof(float), cudaMemcpyHostToDevice);

        // Warm-up
        launch_conv(N, M, C, H, W, K, d_X, d_Wf, d_Y);
        cudaDeviceSynchronize();

        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0);
        for (int r = 0; r < 20; r++)
            launch_conv(N, M, C, H, W, K, d_X, d_Wf, d_Y);
        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        float ms; cudaEventElapsedTime(&ms, ev0, ev1);

        cudaMemcpy(h_Ygpu, d_Y, szY * sizeof(float), cudaMemcpyDeviceToHost);

        long long ops = 2LL * N * M * C * (long long)Ho * Wo * K * K;
        float gflops  = (float)ops / (ms / 20.f * 1e6f);

        printf("Larger test (N=%d,M=%d,C=%d,H=%d,W=%d,K=%d):\n",
               N, M, C, H, W, K);
        printf("  Output per sample: %d maps × %d×%d pixels\n", M, Ho, Wo);
        printf("  Avg time: %.3f ms   GFLOPS: %.2f\n", ms / 20.f, gflops);
        printf("  %s\n\n", verify(h_Ycpu, h_Ygpu, szY, 1e-4f) ? "PASS" : "FAIL");

        printf("Kernel design notes (§16.3):\n");
        printf("  Grid (M=%d, T=%d, N=%d) — one block per output tile per map per sample\n",
               M, ((Ho+TILE_WIDTH-1)/TILE_WIDTH) * ((Wo+TILE_WIDTH-1)/TILE_WIDTH), N);
        printf("  Four outer loops (n,m,h,w) parallelised; c,p,q kept serial\n");
        printf("  − Memory bandwidth limited: W and X re-read from global memory each time\n");
        printf("  → Tiled version (exercise) uses constant/shared memory like Chapter 7\n");

        free(h_X); free(h_Wf); free(h_Ycpu); free(h_Ygpu);
        cudaFree(d_X); cudaFree(d_Wf); cudaFree(d_Y);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    }
    return 0;
}
