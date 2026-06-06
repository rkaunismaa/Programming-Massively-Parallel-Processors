/*
 * Chapter 6 — Section 6.4: A Checklist of Optimizations (Table 6.1)
 *
 * This file benchmarks a progression of matrix multiplication kernels,
 * one for each major optimization from Table 6.1, so you can see the
 * cumulative performance impact of applying each technique.
 *
 * Table 6.1 optimizations demonstrated:
 *
 *  1. Baseline (Chapter 3)   — naïve global memory access, no optimizations
 *  2. Tiling / data reuse    — shared memory tiles (Chapter 5, Section 5.3-5.4)
 *  3. Coalesced accesses     — the Ch5 tiled kernel already achieves this
 *  4. Thread coarsening      — COARSE_FACTOR threads share M tile loads (Ch6 §6.3)
 *  5. Occupancy tuning       — vary TILE_WIDTH to show occupancy vs performance
 *
 * The sixth optimization in Table 6.1 — privatization — is not applicable
 * to matmul (it targets output-update atomics) and is demonstrated in Ch9.
 *
 * Build:
 *   nvcc -O2 -arch=sm_89 -o opt_progression 04_optimization_progression.cu -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define W         1024   /* matrix width — multiple of all tile sizes */
#define N_RUNS    10
#define TILE      16
#define TILE32    32
#define CF        4      /* coarsening factor */

/* ── Kernel 1: Naïve (Chapter 3, Figure 3.11) ─────────────────────── */
__global__
void k1_naive(const float* M, const float* N, float* P, int w) {
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if (row<w && col<w) {
        float v=0.0f;
        for (int k=0; k<w; ++k) v += M[row*w+k]*N[k*w+col];
        P[row*w+col]=v;
    }
}

/* ── Kernel 2: Tiled 16×16 (Chapter 5, Figure 5.9) ──────────────── */
__global__
void k2_tiled16(const float* M, const float* N, float* P, int w) {
    __shared__ float Mds[TILE][TILE];
    __shared__ float Nds[TILE][TILE];
    int bx=blockIdx.x,by=blockIdx.y,tx=threadIdx.x,ty=threadIdx.y;
    int Row=by*TILE+ty, Col=bx*TILE+tx;
    float Pv=0.0f;
    for (int ph=0;ph<w/TILE;++ph) {
        Mds[ty][tx]=M[Row*w+ph*TILE+tx];
        Nds[ty][tx]=N[(ph*TILE+ty)*w+Col];
        __syncthreads();
        for (int k=0;k<TILE;++k) Pv+=Mds[ty][k]*Nds[k][tx];
        __syncthreads();
    }
    P[Row*w+Col]=Pv;
}

/* ── Kernel 3: Tiled 32×32 (larger tile → higher arithmetic intensity) */
__global__
void k3_tiled32(const float* M, const float* N, float* P, int w) {
    __shared__ float Mds[TILE32][TILE32];
    __shared__ float Nds[TILE32][TILE32];
    int bx=blockIdx.x,by=blockIdx.y,tx=threadIdx.x,ty=threadIdx.y;
    int Row=by*TILE32+ty, Col=bx*TILE32+tx;
    float Pv=0.0f;
    for (int ph=0;ph<w/TILE32;++ph) {
        Mds[ty][tx]=M[Row*w+ph*TILE32+tx];
        Nds[ty][tx]=N[(ph*TILE32+ty)*w+Col];
        __syncthreads();
        for (int k=0;k<TILE32;++k) Pv+=Mds[ty][k]*Nds[k][tx];
        __syncthreads();
    }
    P[Row*w+Col]=Pv;
}

/* ── Kernel 4: Thread-coarsened 32×32 tile + CF=4 (Chapter 6, Fig 6.13) */
__global__
void k4_coarsened(const float* M, const float* N, float* P, int w) {
    __shared__ float Mds[TILE32][TILE32];
    __shared__ float Nds[TILE32][TILE32];
    int bx=blockIdx.x,by=blockIdx.y,tx=threadIdx.x,ty=threadIdx.y;
    int row=by*TILE32+ty;
    int colStart=bx*TILE32*CF+tx;
    float Pvalue[CF]; for (int c=0;c<CF;++c) Pvalue[c]=0.0f;

    for (int ph=0;ph<w/TILE32;++ph) {
        Mds[ty][tx]=M[row*w+ph*TILE32+tx];
        for (int c=0;c<CF;++c) {
            int col=colStart+c*TILE32;
            Nds[ty][tx]=N[(ph*TILE32+ty)*w+col];
            __syncthreads();
            for (int k=0;k<TILE32;++k) Pvalue[c]+=Mds[ty][k]*Nds[k][tx];
            __syncthreads();
        }
    }
    for (int c=0;c<CF;++c) P[row*w + colStart+c*TILE32]=Pvalue[c];
}

/* ── Timing ──────────────────────────────────────────────────────── */
/* Variadic macro needed: CUDA <<<g,b>>> contains commas the preprocessor
   would otherwise interpret as argument separators in a single-arg macro. */
#define TIME(...)  do {                                                    \
    __VA_ARGS__; cudaDeviceSynchronize();                                  \
    cudaEventRecord(t0);                                                   \
    for (int _r=0;_r<N_RUNS;_r++) { __VA_ARGS__; }                        \
    cudaEventRecord(t1); cudaEventSynchronize(t1);                         \
    cudaEventElapsedTime(&ms, t0, t1); ms /= N_RUNS;                      \
} while(0)

int main() {
    size_t bytes = (size_t)W*W*sizeof(float);
    float *h_M=(float*)malloc(bytes), *h_N=(float*)malloc(bytes);
    float *h_P=(float*)malloc(bytes), *h_ref=(float*)malloc(bytes);
    float *d_M, *d_N, *d_P;

    srand(42);
    for (int i=0;i<W*W;i++) { h_M[i]=(float)rand()/RAND_MAX; h_N[i]=(float)rand()/RAND_MAX; }

    cudaMalloc((void**)&d_M,bytes); cudaMalloc((void**)&d_N,bytes); cudaMalloc((void**)&d_P,bytes);
    cudaMemcpy(d_M,h_M,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_N,h_N,bytes,cudaMemcpyHostToDevice);

    /* Grids */
    dim3 b16(TILE,TILE), g16(W/TILE,W/TILE);
    dim3 b32(TILE32,TILE32), g32(W/TILE32,W/TILE32);
    dim3 g_cf(W/(TILE32*CF), W/TILE32);

    cudaEvent_t t0,t1; float ms;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    double flops = 2.0*W*W*W;

    printf("=== Optimization Progression — Table 6.1 (%d×%d matrix) ===\n\n", W, W);
    printf("%-35s  %7s  %8s  %s\n", "Kernel", "ms", "GFLOPS", "Optimization applied");
    printf("%-35s  %7s  %8s  %s\n", "------", "--", "------", "--------------------");

    auto report = [&](const char* name, float t, const char* opt) {
        printf("%-35s  %7.2f  %8.1f  %s\n", name, t, (flops/t)/1e6, opt);
    };

    TIME(k1_naive<<<g16,b16>>>(d_M,d_N,d_P,W));
    float ms1=ms; report("1. Naive (Ch3)",                ms1, "none");

    TIME(k2_tiled16<<<g16,b16>>>(d_M,d_N,d_P,W));
    float ms2=ms; report("2. Tiled 16×16 (Ch5)",          ms2, "tiling / data reuse");

    TIME(k3_tiled32<<<g32,b32>>>(d_M,d_N,d_P,W));
    float ms3=ms; report("3. Tiled 32×32 (Ch5)",          ms3, "larger tile → more reuse");

    TIME(k4_coarsened<<<g_cf,b32>>>(d_M,d_N,d_P,W));
    float ms4=ms; report("4. Coarsened 32×32×CF4 (Ch6)", ms4, "+ thread coarsening");

    printf("\nCumulative speedups over naive:\n");
    printf("  Tiled 16×16:    %.1fx\n", ms1/ms2);
    printf("  Tiled 32×32:    %.1fx\n", ms1/ms3);
    printf("  + Coarsening:   %.1fx\n", ms1/ms4);

    printf("\nTable 6.1 checklist status for these kernels:\n");
    printf("  ✓ Maximize occupancy      — block sizes are multiples of 32\n");
    printf("  ✓ Coalesced global access — row-major tile loads, stride 1\n");
    printf("  ~ Minimize control div.   — only boundary warp diverges (none here)\n");
    printf("  ✓ Tiling of reused data   — shared memory Mds/Nds\n");
    printf("  ✓ Thread coarsening       — COARSE_FACTOR=%d in kernel 4\n", CF);
    printf("  — Privatization           — not applicable to matmul (see Ch9)\n");

    free(h_M); free(h_N); free(h_P); free(h_ref);
    cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
