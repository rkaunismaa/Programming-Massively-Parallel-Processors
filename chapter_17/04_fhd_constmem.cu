// §17.3 Steps 2–3 — F^H D kernel: constant memory + AoS + hardware trig
//                   (Figs 17.12, 17.15, 17.16, 17.17)
//
// After the register optimisation, the remaining 5 global-memory accesses
// per inner-loop iteration are: kx[m], ky[m], kz[m], rMu[m], iMu[m].
// The k-space coordinate arrays kx, ky, kz are ideal candidates for
// constant memory (§17.3 Step 2 constant memory discussion):
//   - All threads in a warp access the *same* index m each iteration
//     (m is the loop variable, not threadIdx), so every warp broadcast
//     is served by a single cache-line fetch — 96%+ cache hit rate.
//   - kx/ky/kz are read-only and not modified by the kernel.
//
// Limitation: constant memory capacity is 64 KB on CUDA devices.
// Solution: chunk the k-space data (Fig 17.12):
//   Loop over chunks of CHUNK_SIZE samples, copying each chunk to
//   constant memory before invoking the kernel for that chunk.
//
// AoS layout improvement (Fig 17.14 → 17.15 → 17.16):
//   Storing kx, ky, kz as separate arrays means each warp needs 3
//   separate cache-line fetches per iteration (one per array).
//   Using a struct { float x, y, z; } packs x, y, z for the same
//   sample index m into one cache line, so a single constant-cache
//   fetch serves all three values for all 32 warp threads.
//
// Hardware trig (Fig 17.17 → Step 3):
//   Replace cosf()/sinf() with __cosf()/__sinf() — hardware SFU
//   instructions with lower latency and higher throughput, at the
//   cost of ~0.5 ULP less precision (acceptable for MRI per §17.3).
//
// Reference: Figs 17.12 (chunking), 17.15 (AoS struct),
//            17.16 (AoS kernel), 17.17 (hardware trig).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MU_THREADS_PER_BLOCK  1024
#define FHD_THREADS_PER_BLOCK 1024
#define TWO_PI 6.2831853071795864769f
#define CHUNK_SIZE 2048   // k-space samples per constant-memory chunk
                          // 2048 * 12 bytes = 24 KB (fits in 64 KB)

// ── AoS struct for k-space coordinates (Fig 17.15) ───────────────────────────
struct kdata { float x, y, z; };

// k-space chunk in constant memory (broadcast access pattern: §17.3)
__constant__ struct kdata k_c[CHUNK_SIZE];

// ── cmpMu kernel (Fig 17.7) ──────────────────────────────────────────────────
__global__ void cmpMu(const float *rPhi, const float *iPhi,
                      const float *rD,   const float *iD,
                      float *rMu, float *iMu) {
    int m = blockIdx.x * MU_THREADS_PER_BLOCK + threadIdx.x;
    rMu[m] = rPhi[m]*rD[m] + iPhi[m]*iD[m];
    iMu[m] = rPhi[m]*iD[m] - iPhi[m]*rD[m];
}

// ── cmpFhD constant-memory + AoS + hardware-trig kernel (Figs 17.16, 17.17) ─
// k_c[] holds one chunk of CHUNK_SIZE k-space samples from constant memory.
// rMu/iMu are offset by the chunk's starting index so k_c[m] corresponds
// to rMu[chunkOffset+m] / iMu[chunkOffset+m].
__global__ void cmpFhD_constmem(const float *rMu_chunk, const float *iMu_chunk,
                                  float *rFhD, float *iFhD,
                                  float *x, float *y, float *z,
                                  int chunk_size) {
    int n = blockIdx.x * FHD_THREADS_PER_BLOCK + threadIdx.x;
    float xn = x[n], yn = y[n], zn = z[n];
    float rFn = rFhD[n], iFn = iFhD[n];
    for (int m = 0; m < chunk_size; m++) {
        // k_c[m].x/y/z accessed from constant cache — broadcast to warp
        float expFhD = TWO_PI * (k_c[m].x*xn + k_c[m].y*yn + k_c[m].z*zn);
        float cArg = __cosf(expFhD);   // hardware SFU trig (Fig 17.17)
        float sArg = __sinf(expFhD);
        rFn += rMu_chunk[m]*cArg - iMu_chunk[m]*sArg;
        iFn += iMu_chunk[m]*cArg + rMu_chunk[m]*sArg;
    }
    rFhD[n] = rFn;
    iFhD[n] = iFn;
}

// ── Register kernel (Fig 17.11) for timing comparison ────────────────────────
__global__ void cmpFhD_register(const float *kx, const float *ky, const float *kz,
                                 const float *x,  const float *y,  const float *z,
                                 const float *rMu, const float *iMu,
                                 float *rFhD, float *iFhD, int M) {
    int n = blockIdx.x * FHD_THREADS_PER_BLOCK + threadIdx.x;
    float xn=x[n], yn=y[n], zn=z[n];
    float rFn=rFhD[n], iFn=iFhD[n];
    for (int m=0;m<M;m++){
        float e=TWO_PI*(kx[m]*xn+ky[m]*yn+kz[m]*zn);
        float c=cosf(e),s=sinf(e);
        rFn+=rMu[m]*c-iMu[m]*s; iFn+=iMu[m]*c+rMu[m]*s;
    }
    rFhD[n]=rFn; iFhD[n]=iFn;
}

static void fhd_cpu(const float *rPhi,const float *iPhi,
                    const float *rD,const float *iD,
                    const float *kx,const float *ky,const float *kz,
                    const float *x,const float *y,const float *z,
                    float *rFhD,float *iFhD,int M,int N) {
    float *rMu=(float*)malloc(M*sizeof(float)),*iMu=(float*)malloc(M*sizeof(float));
    for(int m=0;m<M;m++){rMu[m]=rPhi[m]*rD[m]+iPhi[m]*iD[m];iMu[m]=rPhi[m]*iD[m]-iPhi[m]*rD[m];}
    for(int n=0;n<N;n++){
        rFhD[n]=iFhD[n]=0.f;
        for(int m=0;m<M;m++){float e=TWO_PI*(kx[m]*x[n]+ky[m]*y[n]+kz[m]*z[n]);float c=cosf(e),s=sinf(e);rFhD[n]+=rMu[m]*c-iMu[m]*s;iFhD[n]+=iMu[m]*c+rMu[m]*s;}
    }
    free(rMu);free(iMu);
}

static float max_rel_err(const float *a,const float *b,int n){float mx=0.f;for(int i=0;i<n;i++){float e=fabsf(a[i]-b[i])/(1.f+fabsf(b[i]));if(e>mx)mx=e;}return mx;}

int main(void) {
    printf("=== F^H D: Constant Memory + AoS + Hardware Trig (§17.3, Figs 17.16/17.17) ===\n\n");

    int M = 4096;   // must be a multiple of CHUNK_SIZE (2048)
    int N = 4096;
    int nChunks = M / CHUNK_SIZE;   // 2 chunks

    size_t szM=M*sizeof(float), szN=N*sizeof(float);

    float *h_rPhi=(float*)malloc(szM),*h_iPhi=(float*)malloc(szM);
    float *h_rD  =(float*)malloc(szM),*h_iD  =(float*)malloc(szM);
    float *h_kx  =(float*)malloc(szM),*h_ky  =(float*)malloc(szM),*h_kz=(float*)malloc(szM);
    float *h_x   =(float*)malloc(szN),*h_y   =(float*)malloc(szN),*h_z =(float*)malloc(szN);
    struct kdata *h_k = (struct kdata*)malloc(M*sizeof(struct kdata));
    float *h_rFcpu=(float*)calloc(N,sizeof(float)),*h_iFcpu=(float*)calloc(N,sizeof(float));
    float *h_rFgpu=(float*)calloc(N,sizeof(float)),*h_iFgpu=(float*)calloc(N,sizeof(float));

    srand(42);
    for(int m=0;m<M;m++){
        h_rPhi[m]=(rand()/(float)RAND_MAX)-.5f;h_iPhi[m]=(rand()/(float)RAND_MAX)-.5f;
        h_rD[m]=(rand()/(float)RAND_MAX)-.5f;h_iD[m]=(rand()/(float)RAND_MAX)-.5f;
        h_kx[m]=(rand()/(float)RAND_MAX)*.5f-.25f;
        h_ky[m]=(rand()/(float)RAND_MAX)*.5f-.25f;
        h_kz[m]=(rand()/(float)RAND_MAX)*.5f-.25f;
        h_k[m].x=h_kx[m]; h_k[m].y=h_ky[m]; h_k[m].z=h_kz[m];
    }
    for(int n=0;n<N;n++){h_x[n]=(rand()/(float)RAND_MAX)*2.f-1.f;h_y[n]=(rand()/(float)RAND_MAX)*2.f-1.f;h_z[n]=(rand()/(float)RAND_MAX)*2.f-1.f;}

    fhd_cpu(h_rPhi,h_iPhi,h_rD,h_iD,h_kx,h_ky,h_kz,h_x,h_y,h_z,h_rFcpu,h_iFcpu,M,N);

    float *d_rPhi,*d_iPhi,*d_rD,*d_iD,*d_kx,*d_ky,*d_kz;
    float *d_x,*d_y,*d_z,*d_rMu,*d_iMu,*d_rFhD,*d_iFhD;
    cudaMalloc(&d_rPhi,szM);cudaMalloc(&d_iPhi,szM);
    cudaMalloc(&d_rD,szM);cudaMalloc(&d_iD,szM);
    cudaMalloc(&d_kx,szM);cudaMalloc(&d_ky,szM);cudaMalloc(&d_kz,szM);
    cudaMalloc(&d_x,szN);cudaMalloc(&d_y,szN);cudaMalloc(&d_z,szN);
    cudaMalloc(&d_rMu,szM);cudaMalloc(&d_iMu,szM);
    cudaMalloc(&d_rFhD,szN);cudaMalloc(&d_iFhD,szN);

    cudaMemcpy(d_rPhi,h_rPhi,szM,cudaMemcpyHostToDevice);
    cudaMemcpy(d_iPhi,h_iPhi,szM,cudaMemcpyHostToDevice);
    cudaMemcpy(d_rD,h_rD,szM,cudaMemcpyHostToDevice);cudaMemcpy(d_iD,h_iD,szM,cudaMemcpyHostToDevice);
    cudaMemcpy(d_kx,h_kx,szM,cudaMemcpyHostToDevice);cudaMemcpy(d_ky,h_ky,szM,cudaMemcpyHostToDevice);
    cudaMemcpy(d_kz,h_kz,szM,cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,h_x,szN,cudaMemcpyHostToDevice);cudaMemcpy(d_y,h_y,szN,cudaMemcpyHostToDevice);
    cudaMemcpy(d_z,h_z,szN,cudaMemcpyHostToDevice);

    cmpMu<<<M/MU_THREADS_PER_BLOCK,MU_THREADS_PER_BLOCK>>>(d_rPhi,d_iPhi,d_rD,d_iD,d_rMu,d_iMu);
    cudaDeviceSynchronize();

    cudaEvent_t t0,t1;
    cudaEventCreate(&t0);cudaEventCreate(&t1);

    // ── Timing: register baseline (reset before each run) ────────────────────
    cudaEventRecord(t0);
    for (int r=0;r<10;r++) {
        cudaMemset(d_rFhD,0,szN);cudaMemset(d_iFhD,0,szN);
        cmpFhD_register<<<N/FHD_THREADS_PER_BLOCK,FHD_THREADS_PER_BLOCK>>>(
            d_kx,d_ky,d_kz,d_x,d_y,d_z,d_rMu,d_iMu,d_rFhD,d_iFhD,M);
    }
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    float ms_reg; cudaEventElapsedTime(&ms_reg,t0,t1);

    // ── Timing: constant memory + AoS + hardware trig ─────────────────────────
    // Host loop chunking k-space data into constant memory (Fig 17.12)
    cudaEventRecord(t0);
    for (int r=0;r<10;r++) {
        cudaMemset(d_rFhD,0,szN);cudaMemset(d_iFhD,0,szN);
        for (int chunk=0; chunk<nChunks; chunk++) {
            int offset = chunk * CHUNK_SIZE;
            // Transfer chunk of k-space AoS struct to constant memory (Fig 17.15)
            cudaMemcpyToSymbol(k_c, &h_k[offset],
                               CHUNK_SIZE * sizeof(struct kdata), 0,
                               cudaMemcpyHostToDevice);
            // Launch kernel with Mu pointers offset to this chunk
            cmpFhD_constmem<<<N/FHD_THREADS_PER_BLOCK, FHD_THREADS_PER_BLOCK>>>(
                d_rMu + offset, d_iMu + offset,
                d_rFhD, d_iFhD, d_x, d_y, d_z, CHUNK_SIZE);
        }
    }
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    float ms_cm; cudaEventElapsedTime(&ms_cm,t0,t1);

    cudaMemcpy(h_rFgpu,d_rFhD,szN,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_iFgpu,d_iFhD,szN,cudaMemcpyDeviceToHost);

    float er=max_rel_err(h_rFgpu,h_rFcpu,N);
    float ei=max_rel_err(h_iFgpu,h_iFcpu,N);
    printf("M=%d (%d chunks of %d), N=%d (10-run avg)\n",M,nChunks,CHUNK_SIZE,N);
    printf("Register (Fig 17.11):          %.3f ms/run\n", ms_reg/10.f);
    printf("Const mem+AoS+HW trig (Fig 17.16/17.17): %.3f ms/run  speedup=%.2fx\n",
           ms_cm/10.f, ms_reg/ms_cm);
    printf("Max rel error: rFhD=%.2e  iFhD=%.2e  %s\n",
           er,ei,(er<1e-3f&&ei<1e-3f)?"PASS":"FAIL");
    printf("\nOptimisations applied:\n");
    printf("  + Constant memory for k-space: warp broadcasts k_c[m] (Fig 17.12)\n");
    printf("  + AoS struct kdata{x,y,z}: x,y,z in one cache line (Fig 17.15)\n");
    printf("  + __cosf/__sinf: hardware SFU trig, ~2x throughput (Fig 17.17)\n");
    printf("  + Chunking (%d×%d): fits 64 KB constant memory limit\n",nChunks,CHUNK_SIZE);

    free(h_rPhi);free(h_iPhi);free(h_rD);free(h_iD);
    free(h_kx);free(h_ky);free(h_kz);free(h_k);free(h_x);free(h_y);free(h_z);
    free(h_rFcpu);free(h_iFcpu);free(h_rFgpu);free(h_iFgpu);
    cudaFree(d_rPhi);cudaFree(d_iPhi);cudaFree(d_rD);cudaFree(d_iD);
    cudaFree(d_kx);cudaFree(d_ky);cudaFree(d_kz);
    cudaFree(d_x);cudaFree(d_y);cudaFree(d_z);
    cudaFree(d_rMu);cudaFree(d_iMu);cudaFree(d_rFhD);cudaFree(d_iFhD);
    cudaEventDestroy(t0);cudaEventDestroy(t1);
    return 0;
}
