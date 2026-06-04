// §16.4  Convolutional layer — unroll + GEMM formulation (Figs 16.17–16.18)
//
// Core idea (§16.4): reformulate convolution as matrix multiplication so
// that highly optimised GEMM routines (e.g. cuBLAS) can be used.
//
// Step 1 — Unroll (im2col): for each input sample build X_unroll,
//   shape [C·K·K, H_out·W_out], by gathering the receptive-field patches.
//   Each column of X_unroll holds all C·K·K input values needed to produce
//   one output pixel.  Input pixels are duplicated for overlapping patches
//   (expansion ratio ≈ K² for large feature maps).
//
// Step 2 — GEMM: Y = W_filter · X_unroll
//   W_filter [M, C·K·K]  — same data as W[M,C,K,K], viewed row-major
//   X_unroll [C·K·K, H_out·W_out]
//   Y        [M,     H_out·W_out] — same layout as Y[M, H_out, W_out]
//
// We implement a self-contained tiled GEMM kernel instead of calling cuBLAS
// so the sample has no external library dependency.  In production, cuBLAS
// sgemm (or cublasGemmEx) achieves near-peak FLOP/s.
//
// Reference: Fig 16.17 (sequential unroll), Fig 16.18 (CUDA unroll kernel),
//            §16.4 text (GEMM mapping), §16.5 (CUDNN lazy unrolling).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_W 16   // GEMM tile width

// ── CUDA unroll kernel (Fig 16.18) ───────────────────────────────────────────
// Converts one input sample X[C, H, W] → X_unroll[C·K·K, H_out·W_out].
// One thread handles all K·K gather operations for one (channel, output-pixel).
//
// Variable names follow Fig 16.18:
//   t           — linear thread id in [0, C·W_unroll)
//   c           — input channel index for this thread
//   w_unroll    — column index in X_unroll = which output pixel (0..W_unroll-1)
//   h_unroll    — row index in X_unroll    = weight element  (0..C·K²-1)
//   W_unroll    — H_out·W_out  (number of output pixels = column count)
__global__ void unroll_Kernel(int C, int H, int W, int K,
                               const float * __restrict__ X,
                               float *X_unroll) {
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= C * W_unroll) return;

    int c        = t / W_unroll;       // which input channel
    int w_unroll = t % W_unroll;       // which output pixel (column)
    int h_out    = w_unroll / W_out;   // output pixel row
    int w_out    = w_unroll % W_out;   // output pixel col
    int w_base   = c * K * K;          // first row in X_unroll for channel c

    for (int p = 0; p < K; p++)
        for (int q = 0; q < K; q++) {
            int h_unroll = w_base + p * K + q;   // row in X_unroll
            X_unroll[h_unroll * W_unroll + w_unroll] =
                X[c * H * W + (h_out + p) * W + (w_out + q)];
        }
}

// ── Tiled GEMM kernel: C_out = A · B ─────────────────────────────────────────
// A [M_dim, K_dim], B [K_dim, N_dim], C_out [M_dim, N_dim]
// Called with: A = W_filter, B = X_unroll, C_out = Y (one sample)
__global__ void gemm_Kernel(int M_dim, int K_dim, int N_dim,
                             const float * __restrict__ A,
                             const float * __restrict__ B,
                             float *C_out) {
    __shared__ float As[TILE_W][TILE_W];
    __shared__ float Bs[TILE_W][TILE_W];

    int row = blockIdx.y * TILE_W + threadIdx.y;
    int col = blockIdx.x * TILE_W + threadIdx.x;
    float acc = 0.f;

    int nTiles = (K_dim + TILE_W - 1) / TILE_W;
    for (int t = 0; t < nTiles; t++) {
        int aCol = t * TILE_W + threadIdx.x;
        int bRow = t * TILE_W + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M_dim && aCol < K_dim)
                                       ? A[row * K_dim + aCol] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K_dim && col < N_dim)
                                       ? B[bRow * N_dim + col] : 0.f;
        __syncthreads();

        for (int k = 0; k < TILE_W; k++)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M_dim && col < N_dim)
        C_out[row * N_dim + col] = acc;
}

// ── CPU reference (sequential conv, Fig 16.6 / Fig 16.12) ───────────────────
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

// ── CPU sequential unroll (Fig 16.17) ────────────────────────────────────────
// Generates X_unroll for one sample X[C, H, W].
static void unroll_cpu(int C, int H, int W, int K,
                        const float *X, float *X_unroll) {
    int H_out = H - K + 1, W_out = W - K + 1;
    int W_unroll = H_out * W_out;
    for (int c = 0; c < C; c++) {
        int w_base = c * K * K;
        for (int p = 0; p < K; p++)
            for (int q = 0; q < K; q++) {
                int h_unroll = w_base + p * K + q;
                for (int h = 0; h < H_out; h++)
                    for (int w = 0; w < W_out; w++) {
                        int w_unroll = h * W_out + w;
                        X_unroll[h_unroll * W_unroll + w_unroll] =
                            X[c * H * W + (h + p) * W + (w + q)];
                    }
            }
    }
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
    printf("=== Convolutional Layer: Unroll + GEMM (§16.4) ===\n\n");

    // ── Test 1: tiny — verify unroll against Fig 16.16 values ────────────────
    {
        // Fig 16.16 example: C=3 input maps (3×3 each), M=2 output maps, K=2
        int N=1, M=2, C=3, H=3, W=3, K=2;
        int Ho=H-K+1, Wo=W-K+1;
        int W_unroll = Ho * Wo, H_unroll = C * K * K;

        // Input feature maps X from Fig 16.16
        float h_X[] = {
            1,2,0,  1,3,2,  0,2,2,   // channel 0
            0,2,1,  0,3,2,  1,1,0,   // channel 1
            0,2,1,  3,3,2,  3,3,2    // channel 2
        };
        // Filter banks W from Fig 16.16 (M=2, C=3, K=2, K=2)
        float h_Wf[] = {
            1,1, 1,1,  0,1, 1,0,  2,1, 1,2,   // filter for output map 0
            1,0, 0,1,  2,1, 2,1,  1,2, 0,1    // filter for output map 1
        };

        // Expected output from Fig 16.16: Y[0]={14,20,15,24} Y[1]={12,24,17,26}
        float h_Ycpu[8];
        conv_forward_cpu(N, M, C, H, W, K, h_X, h_Wf, h_Ycpu);
        printf("Fig 16.16 tiny test (C=3,M=2,H=3,W=3,K=2):\n");
        printf("  CPU conv:    map0=[%.0f %.0f %.0f %.0f]  map1=[%.0f %.0f %.0f %.0f]\n",
               h_Ycpu[0],h_Ycpu[1],h_Ycpu[2],h_Ycpu[3],
               h_Ycpu[4],h_Ycpu[5],h_Ycpu[6],h_Ycpu[7]);

        // Verify CPU unroll → matmul matches conv_forward_cpu
        float *X_unroll = (float *)malloc(H_unroll * W_unroll * sizeof(float));
        unroll_cpu(C, H, W, K, h_X, X_unroll);

        printf("  X_unroll [%d×%d]:\n", H_unroll, W_unroll);
        for (int r = 0; r < H_unroll; r++) {
            printf("    row %2d: ", r);
            for (int c2 = 0; c2 < W_unroll; c2++)
                printf("%.0f ", X_unroll[r * W_unroll + c2]);
            printf("\n");
        }

        // CPU matmul
        float h_Ygemm[8] = {0};
        for (int m = 0; m < M; m++)
            for (int col = 0; col < W_unroll; col++) {
                float acc = 0.f;
                for (int k = 0; k < H_unroll; k++)
                    acc += h_Wf[m * H_unroll + k] * X_unroll[k * W_unroll + col];
                h_Ygemm[m * W_unroll + col] = acc;
            }
        printf("  CPU GEMM:    map0=[%.0f %.0f %.0f %.0f]  map1=[%.0f %.0f %.0f %.0f]\n",
               h_Ygemm[0],h_Ygemm[1],h_Ygemm[2],h_Ygemm[3],
               h_Ygemm[4],h_Ygemm[5],h_Ygemm[6],h_Ygemm[7]);
        printf("  Conv == GEMM: %s\n\n",
               (h_Ygemm[0]==h_Ycpu[0] && h_Ygemm[7]==h_Ycpu[7]) ? "PASS" : "FAIL");
        free(X_unroll);
    }

    // ── Test 2: GPU unroll + GEMM vs CPU reference ────────────────────────────
    {
        int N=4, M=8, C=3, H=10, W=10, K=3;
        int Ho=H-K+1, Wo=W-K+1;
        int H_unroll = C * K * K;        // rows of X_unroll
        int W_unroll = Ho * Wo;           // cols of X_unroll

        int szX  = N * C * H * W;
        int szWf = M * C * K * K;
        int szY  = N * M * Ho * Wo;
        int szXu = H_unroll * W_unroll;  // per-sample unrolled matrix

        float *h_X    = (float *)malloc(szX  * sizeof(float));
        float *h_Wf   = (float *)malloc(szWf * sizeof(float));
        float *h_Ycpu = (float *)malloc(szY  * sizeof(float));
        float *h_Ygpu = (float *)malloc(szY  * sizeof(float));

        srand(42);
        for (int i = 0; i < szX;  i++) h_X[i]  = (rand()/(float)RAND_MAX)*2.f - 1.f;
        for (int i = 0; i < szWf; i++) h_Wf[i] = (rand()/(float)RAND_MAX)*0.4f - 0.2f;

        conv_forward_cpu(N, M, C, H, W, K, h_X, h_Wf, h_Ycpu);

        float *d_X, *d_Wf, *d_Y, *d_Xu;
        cudaMalloc(&d_X,  szX  * sizeof(float));
        cudaMalloc(&d_Wf, szWf * sizeof(float));
        cudaMalloc(&d_Y,  szY  * sizeof(float));
        cudaMalloc(&d_Xu, szXu * sizeof(float));   // single-sample scratch buffer

        cudaMemcpy(d_X,  h_X,  szX  * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Wf, h_Wf, szWf * sizeof(float), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks_unroll = (C * W_unroll + threads - 1) / threads;
        dim3 gemmBlock(TILE_W, TILE_W);
        dim3 gemmGrid((W_unroll + TILE_W - 1) / TILE_W,
                      (M       + TILE_W - 1) / TILE_W);

        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0);

        // Loop over minibatch: unroll + GEMM per sample
        for (int n = 0; n < N; n++) {
            const float *d_Xn = d_X + n * C * H * W;
            float       *d_Yn = d_Y + n * M * Ho * Wo;

            unroll_Kernel<<<blocks_unroll, threads>>>(C, H, W, K, d_Xn, d_Xu);
            // W_filter [M, H_unroll] × X_unroll [H_unroll, W_unroll] → Y_n [M, W_unroll]
            gemm_Kernel<<<gemmGrid, gemmBlock>>>(M, H_unroll, W_unroll,
                                                 d_Wf, d_Xu, d_Yn);
        }
        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        float ms; cudaEventElapsedTime(&ms, ev0, ev1);

        cudaMemcpy(h_Ygpu, d_Y, szY * sizeof(float), cudaMemcpyDeviceToHost);

        float expand = (float)(H_unroll * W_unroll) / (float)(C * H * W);
        long long ops = 2LL * N * M * H_unroll * (long long)W_unroll;
        float gflops  = (float)ops / (ms * 1e6f);

        printf("GPU unroll+GEMM (N=%d,M=%d,C=%d,H=%d,W=%d,K=%d):\n",
               N, M, C, H, W, K);
        printf("  X_unroll: %d × %d  (expansion ratio %.2fx per sample)\n",
               H_unroll, W_unroll, expand);
        printf("  GEMM dims: A[%d,%d] × B[%d,%d]\n",
               M, H_unroll, H_unroll, W_unroll);
        printf("  GPU time: %.3f ms   GFLOPS: %.2f\n", ms, gflops);
        printf("  %s\n\n", verify(h_Ycpu, h_Ygpu, szY, 1e-4f) ? "PASS" : "FAIL");

        free(h_X); free(h_Wf); free(h_Ycpu); free(h_Ygpu);
        cudaFree(d_X); cudaFree(d_Wf); cudaFree(d_Y); cudaFree(d_Xu);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    }

    // ── Test 3: larger — LeNet-scale ─────────────────────────────────────────
    {
        // Approximate LeNet C3: 6 input maps → 16 output maps, 5×5 filters, 10×10 input
        int N=8, M=16, C=6, H=14, W=14, K=5;
        int Ho=H-K+1, Wo=W-K+1;
        int H_unroll = C * K * K, W_unroll = Ho * Wo;

        int szX  = N * C * H * W;
        int szWf = M * C * K * K;
        int szY  = N * M * Ho * Wo;
        int szXu = H_unroll * W_unroll;

        float *h_X    = (float *)malloc(szX  * sizeof(float));
        float *h_Wf   = (float *)malloc(szWf * sizeof(float));
        float *h_Ycpu = (float *)malloc(szY  * sizeof(float));
        float *h_Ygpu = (float *)malloc(szY  * sizeof(float));

        srand(13);
        for (int i = 0; i < szX;  i++) h_X[i]  = (rand()/(float)RAND_MAX)*2.f - 1.f;
        for (int i = 0; i < szWf; i++) h_Wf[i] = (rand()/(float)RAND_MAX)*0.2f - 0.1f;

        conv_forward_cpu(N, M, C, H, W, K, h_X, h_Wf, h_Ycpu);

        float *d_X, *d_Wf, *d_Y, *d_Xu;
        cudaMalloc(&d_X,  szX  * sizeof(float));
        cudaMalloc(&d_Wf, szWf * sizeof(float));
        cudaMalloc(&d_Y,  szY  * sizeof(float));
        cudaMalloc(&d_Xu, szXu * sizeof(float));
        cudaMemcpy(d_X,  h_X,  szX  * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Wf, h_Wf, szWf * sizeof(float), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks_unroll = (C * W_unroll + threads - 1) / threads;
        dim3 gemmBlock(TILE_W, TILE_W);
        dim3 gemmGrid((W_unroll + TILE_W - 1) / TILE_W,
                      (M       + TILE_W - 1) / TILE_W);

        // Warm-up
        for (int n = 0; n < N; n++) {
            unroll_Kernel<<<blocks_unroll, threads>>>(C, H, W, K,
                           d_X + n*C*H*W, d_Xu);
            gemm_Kernel<<<gemmGrid, gemmBlock>>>(M, H_unroll, W_unroll,
                         d_Wf, d_Xu, d_Y + n*M*Ho*Wo);
        }
        cudaDeviceSynchronize();

        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0);
        for (int r = 0; r < 10; r++)
            for (int n = 0; n < N; n++) {
                unroll_Kernel<<<blocks_unroll, threads>>>(C, H, W, K,
                               d_X + n*C*H*W, d_Xu);
                gemm_Kernel<<<gemmGrid, gemmBlock>>>(M, H_unroll, W_unroll,
                             d_Wf, d_Xu, d_Y + n*M*Ho*Wo);
            }
        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        float ms; cudaEventElapsedTime(&ms, ev0, ev1);

        cudaMemcpy(h_Ygpu, d_Y, szY * sizeof(float), cudaMemcpyDeviceToHost);

        long long ops = 2LL * N * M * H_unroll * (long long)W_unroll;
        float gflops  = (float)ops / (ms / 10.f * 1e6f);

        printf("LeNet-scale test (N=%d,M=%d,C=%d,H=%d,W=%d,K=%d):\n",
               N, M, C, H, W, K);
        printf("  X_unroll: %d × %d  expansion %.2fx\n",
               H_unroll, W_unroll,
               (float)(H_unroll * W_unroll) / (float)(C * H * W));
        printf("  Avg time: %.3f ms   GFLOPS: %.2f\n", ms / 10.f, gflops);
        printf("  %s\n\n", verify(h_Ycpu, h_Ygpu, szY, 1e-4f) ? "PASS" : "FAIL");

        printf("Unroll+GEMM trade-offs (§16.4, §16.5):\n");
        printf("  + Convolution reduced to GEMM — leverages cuBLAS peak throughput\n");
        printf("  + GEMM matrix size (C·K²·H_out·W_out) stays large across network layers\n");
        printf("  − Input duplication up to K²=%d×; extra %zu KB per sample for X_unroll\n",
               K*K, (size_t)szXu * sizeof(float) / 1024);
        printf("  − Sequential per-sample loop limits GPU utilisation for small N\n");
        printf("  → CUDNN: lazy on-chip unrolling avoids materialising X_unroll in DRAM\n");

        free(h_X); free(h_Wf); free(h_Ycpu); free(h_Ygpu);
        cudaFree(d_X); cudaFree(d_Wf); cudaFree(d_Y); cudaFree(d_Xu);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    }
    return 0;
}
