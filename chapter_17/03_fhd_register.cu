// §17.3 Step 2 — F^H D kernel: register optimisation (Fig 17.11)
//
// Problem with the basic gather kernel (Fig 17.10):
//   Each inner iteration reads x[n], y[n], z[n] from global memory (same
//   values for every iteration of the m-loop), AND reads/writes rFhD[n]
//   and iFhD[n] (accumulate through global memory on every iteration).
//   Compute-to-memory ratio ≈ 0.23 OP/B — bandwidth-bound.
//
// Register optimisation (Fig 17.11):
//   Before the m-loop: load x[n], y[n], z[n] into automatic variables
//   (registers).  Also accumulate rFhD and iFhD in register variables
//   rFhDn_r and iFhDn_r, writing back to global memory only once after
//   the loop ends.
//
// Effect: the 14 global-memory accesses per iteration drop to 5
//   (kx[m], ky[m], kz[m], rMu[m], iMu[m]) — all indexed by m, so
//   they change every iteration and cannot be cached further in registers.
//   Compute-to-memory ratio improves from 0.23 to ~0.46 OP/B.
//
// Reference: Fig 17.11 (register kernel), §17.3 "Step 2: Getting around
//            the memory bandwidth limitation".

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MU_THREADS_PER_BLOCK  1024
#define FHD_THREADS_PER_BLOCK 1024
#define TWO_PI 6.2831853071795864769f

__global__ void cmpMu(const float *rPhi, const float *iPhi,
                      const float *rD,   const float *iD,
                      float *rMu, float *iMu) {
    int m = blockIdx.x * MU_THREADS_PER_BLOCK + threadIdx.x;
    rMu[m] = rPhi[m]*rD[m] + iPhi[m]*iD[m];
    iMu[m] = rPhi[m]*iD[m] - iPhi[m]*rD[m];
}

// ── cmpFhD register kernel (Fig 17.11) ───────────────────────────────────────
// Like the gather kernel but promotes frequently-used values into registers:
//   xn_r, yn_r, zn_r   — loaded once, reused for all M iterations
//   rFhDn_r, iFhDn_r   — accumulate in register, one global write at the end
__global__ void cmpFhD_register(const float *kx, const float *ky, const float *kz,
                                 const float *x,  const float *y,  const float *z,
                                 const float *rMu, const float *iMu,
                                 float *rFhD, float *iFhD, int M) {
    int n = blockIdx.x * FHD_THREADS_PER_BLOCK + threadIdx.x;

    // Assign frequently accessed coordinate and output elements into registers
    float xn_r  = x[n];      float yn_r  = y[n];    float zn_r  = z[n];
    float rFhDn_r = rFhD[n]; float iFhDn_r = iFhD[n];  // (initially 0)

    for (int m = 0; m < M; m++) {
        float expFhD = TWO_PI * (kx[m]*xn_r + ky[m]*yn_r + kz[m]*zn_r);
        float cArg = cosf(expFhD);
        float sArg = sinf(expFhD);
        rFhDn_r += rMu[m]*cArg - iMu[m]*sArg;
        iFhDn_r += iMu[m]*cArg + rMu[m]*sArg;
    }
    rFhD[n] = rFhDn_r;   // one global write
    iFhD[n] = iFhDn_r;
}

// ── Basic gather kernel (Fig 17.10) for timing comparison ────────────────────
__global__ void cmpFhD_gather(const float *kx, const float *ky, const float *kz,
                               const float *x,  const float *y,  const float *z,
                               const float *rMu, const float *iMu,
                               float *rFhD, float *iFhD, int M) {
    int n = blockIdx.x * FHD_THREADS_PER_BLOCK + threadIdx.x;
    float rF = rFhD[n], iF = iFhD[n];
    for (int m = 0; m < M; m++) {
        float e = TWO_PI * (kx[m]*x[n] + ky[m]*y[n] + kz[m]*z[n]);
        float c = cosf(e), s = sinf(e);
        rF += rMu[m]*c - iMu[m]*s;
        iF += iMu[m]*c + rMu[m]*s;
    }
    rFhD[n] = rF; iFhD[n] = iF;
}

static void fhd_cpu(const float *rPhi,const float *iPhi,
                    const float *rD,  const float *iD,
                    const float *kx,  const float *ky,const float *kz,
                    const float *x,   const float *y, const float *z,
                    float *rFhD,float *iFhD,int M,int N) {
    float *rMu=(float*)malloc(M*sizeof(float)),*iMu=(float*)malloc(M*sizeof(float));
    for (int m=0;m<M;m++){rMu[m]=rPhi[m]*rD[m]+iPhi[m]*iD[m];iMu[m]=rPhi[m]*iD[m]-iPhi[m]*rD[m];}
    for (int n=0;n<N;n++){
        rFhD[n]=iFhD[n]=0.f;
        for (int m=0;m<M;m++){float e=TWO_PI*(kx[m]*x[n]+ky[m]*y[n]+kz[m]*z[n]);float c=cosf(e),s=sinf(e);rFhD[n]+=rMu[m]*c-iMu[m]*s;iFhD[n]+=iMu[m]*c+rMu[m]*s;}
    }
    free(rMu);free(iMu);
}

static float max_rel_err(const float *a,const float *b,int n){float mx=0.f;for(int i=0;i<n;i++){float e=fabsf(a[i]-b[i])/(1.f+fabsf(b[i]));if(e>mx)mx=e;}return mx;}

int main(void) {
    printf("=== F^H D: Register Optimisation (§17.3, Fig 17.11) ===\n\n");

    int M=2048, N=4096;
    size_t szM=M*sizeof(float), szN=N*sizeof(float);

    float *h_rPhi=(float*)malloc(szM),*h_iPhi=(float*)malloc(szM);
    float *h_rD  =(float*)malloc(szM),*h_iD  =(float*)malloc(szM);
    float *h_kx  =(float*)malloc(szM),*h_ky  =(float*)malloc(szM),*h_kz=(float*)malloc(szM);
    float *h_x   =(float*)malloc(szN),*h_y   =(float*)malloc(szN),*h_z =(float*)malloc(szN);
    float *h_rFcpu=(float*)calloc(N,sizeof(float)),*h_iFcpu=(float*)calloc(N,sizeof(float));
    float *h_rFgpu=(float*)calloc(N,sizeof(float)),*h_iFgpu=(float*)calloc(N,sizeof(float));

    srand(42);
    for (int m=0;m<M;m++){
        h_rPhi[m]=(rand()/(float)RAND_MAX)-.5f;h_iPhi[m]=(rand()/(float)RAND_MAX)-.5f;
        h_rD[m]=(rand()/(float)RAND_MAX)-.5f;h_iD[m]=(rand()/(float)RAND_MAX)-.5f;
        h_kx[m]=(rand()/(float)RAND_MAX)*.5f-.25f;h_ky[m]=(rand()/(float)RAND_MAX)*.5f-.25f;
        h_kz[m]=(rand()/(float)RAND_MAX)*.5f-.25f;
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

    cudaEvent_t t0,t1;
    cudaEventCreate(&t0);cudaEventCreate(&t1);

    // ── Time the basic gather kernel (reset before each run) ─────────────────
    cmpMu<<<M/MU_THREADS_PER_BLOCK,MU_THREADS_PER_BLOCK>>>(d_rPhi,d_iPhi,d_rD,d_iD,d_rMu,d_iMu);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int r=0;r<10;r++){
        cudaMemset(d_rFhD,0,szN);cudaMemset(d_iFhD,0,szN);
        cmpFhD_gather<<<N/FHD_THREADS_PER_BLOCK,FHD_THREADS_PER_BLOCK>>>(d_kx,d_ky,d_kz,d_x,d_y,d_z,d_rMu,d_iMu,d_rFhD,d_iFhD,M);
    }
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    float ms_gather; cudaEventElapsedTime(&ms_gather,t0,t1);

    // ── Time the register kernel ──────────────────────────────────────────────
    cudaEventRecord(t0);
    for (int r=0;r<10;r++){
        cudaMemset(d_rFhD,0,szN);cudaMemset(d_iFhD,0,szN);
        cmpFhD_register<<<N/FHD_THREADS_PER_BLOCK,FHD_THREADS_PER_BLOCK>>>(d_kx,d_ky,d_kz,d_x,d_y,d_z,d_rMu,d_iMu,d_rFhD,d_iFhD,M);
    }
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    float ms_reg; cudaEventElapsedTime(&ms_reg,t0,t1);

    cudaMemcpy(h_rFgpu,d_rFhD,szN,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_iFgpu,d_iFhD,szN,cudaMemcpyDeviceToHost);

    float er=max_rel_err(h_rFgpu,h_rFcpu,N);
    float ei=max_rel_err(h_iFgpu,h_iFcpu,N);
    printf("M=%d, N=%d (10-run avg)\n",M,N);
    printf("Gather (Fig 17.10): %.3f ms/run\n", ms_gather/10.f);
    printf("Register (Fig 17.11): %.3f ms/run  speedup=%.2fx\n",
           ms_reg/10.f, ms_gather/ms_reg);
    printf("Max rel error: rFhD=%.2e  iFhD=%.2e  %s\n",
           er,ei,(er<1e-3f&&ei<1e-3f)?"PASS":"FAIL");
    printf("\nRegister optimisation (§17.3 Step 2):\n");
    printf("  x[n],y[n],z[n] cached in registers → no re-read per iteration\n");
    printf("  rFhD[n],iFhD[n] accumulated in registers → 1 global write vs M\n");
    printf("  Memory accesses per m-iteration: 14 → 5 (kx,ky,kz,rMu,iMu)\n");
    printf("  Compute-to-memory ratio: 0.23 → ~0.46 OP/B\n");
    printf("  → Constant memory (file 04) caches kx/ky/kz for all threads\n");

    free(h_rPhi);free(h_iPhi);free(h_rD);free(h_iD);
    free(h_kx);free(h_ky);free(h_kz);free(h_x);free(h_y);free(h_z);
    free(h_rFcpu);free(h_iFcpu);free(h_rFgpu);free(h_iFgpu);
    cudaFree(d_rPhi);cudaFree(d_iPhi);cudaFree(d_rD);cudaFree(d_iD);
    cudaFree(d_kx);cudaFree(d_ky);cudaFree(d_kz);
    cudaFree(d_x);cudaFree(d_y);cudaFree(d_z);
    cudaFree(d_rMu);cudaFree(d_iMu);cudaFree(d_rFhD);cudaFree(d_iFhD);
    cudaEventDestroy(t0);cudaEventDestroy(t1);
    return 0;
}
