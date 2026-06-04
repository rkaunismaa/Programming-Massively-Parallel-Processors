// §17.3 Step 1 — F^H D kernel: scatter approach (Fig 17.5)
//
// Problem: compute F^H D[n] for all N image voxels, given M k-space samples.
//
//   Mu[m]    = Phi[m] * D[m]           — complex weighting
//   FhD[n]   = Σ_m  Mu[m] * exp(j·2π·(kx[m]·x[n]+ky[m]·y[n]+kz[m]·z[n]))
//
// Scatter strategy (Fig 17.5):
//   Assign one thread per k-space sample m.
//   Each thread scatters its contribution to *all* N voxels via atomicAdd.
//
// Problem with scatter (§17.3 discussion):
//   Every thread writes to every rFhD[n]/iFhD[n].  With M threads all
//   contending on the same N output elements, atomicAdd serialises all M
//   updates on each element — massive contention that kills parallelism.
//
// Reference: Fig 17.4 (sequential), Fig 17.6 (loop fission),
//            Fig 17.7 (cmpMu kernel), Fig 17.5 (scatter kernel).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MU_THREADS_PER_BLOCK  1024
#define FHD_THREADS_PER_BLOCK 1024
#define TWO_PI 6.2831853071795864769f

// ── cmpMu kernel (Fig 17.7) ──────────────────────────────────────────────────
// One thread per k-space sample; no conflicts (each writes its own element).
__global__ void cmpMu(const float *rPhi, const float *iPhi,
                      const float *rD,   const float *iD,
                      float *rMu, float *iMu) {
    int m = blockIdx.x * MU_THREADS_PER_BLOCK + threadIdx.x;
    rMu[m] = rPhi[m]*rD[m] + iPhi[m]*iD[m];
    iMu[m] = rPhi[m]*iD[m] - iPhi[m]*rD[m];
}

// ── cmpFhD scatter kernel (Fig 17.5) ─────────────────────────────────────────
// One thread per k-space sample m; loops over all N voxels.
// atomicAdd is required because M threads all write to the same N elements.
__global__ void cmpFhD_scatter(const float *kx, const float *ky, const float *kz,
                                const float *x,  const float *y,  const float *z,
                                const float *rMu, const float *iMu,
                                float *rFhD, float *iFhD, int N) {
    int m = blockIdx.x * FHD_THREADS_PER_BLOCK + threadIdx.x;
    float rMu_m = rMu[m], iMu_m = iMu[m];
    float kx_m  = kx[m],  ky_m  = ky[m], kz_m = kz[m];
    for (int n = 0; n < N; n++) {
        float expFhD = TWO_PI * (kx_m*x[n] + ky_m*y[n] + kz_m*z[n]);
        float cArg = cosf(expFhD);
        float sArg = sinf(expFhD);
        atomicAdd(&rFhD[n], rMu_m*cArg - iMu_m*sArg);  // contention here
        atomicAdd(&iFhD[n], iMu_m*cArg + rMu_m*sArg);  // contention here
    }
}

// ── CPU reference (Fig 17.6 loop fission; Fig 17.9 loop interchange) ─────────
static void fhd_cpu(const float *rPhi, const float *iPhi,
                    const float *rD,   const float *iD,
                    const float *kx,   const float *ky, const float *kz,
                    const float *x,    const float *y,  const float *z,
                    float *rFhD, float *iFhD, int M, int N) {
    float *rMu = (float *)malloc(M * sizeof(float));
    float *iMu = (float *)malloc(M * sizeof(float));
    for (int m = 0; m < M; m++) {
        rMu[m] = rPhi[m]*rD[m] + iPhi[m]*iD[m];
        iMu[m] = rPhi[m]*iD[m] - iPhi[m]*rD[m];
    }
    for (int n = 0; n < N; n++) {
        rFhD[n] = iFhD[n] = 0.f;
        for (int m = 0; m < M; m++) {
            float e = TWO_PI * (kx[m]*x[n] + ky[m]*y[n] + kz[m]*z[n]);
            float c = cosf(e), s = sinf(e);
            rFhD[n] += rMu[m]*c - iMu[m]*s;
            iFhD[n] += iMu[m]*c + rMu[m]*s;
        }
    }
    free(rMu); free(iMu);
}

static float max_rel_err(const float *a, const float *b, int n) {
    float mx = 0.f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(a[i]-b[i]) / (1.f + fabsf(b[i]));
        if (e > mx) mx = e;
    }
    return mx;
}

int main(void) {
    printf("=== F^H D: Scatter Approach (§17.3, Fig 17.5) ===\n\n");

    int M = 2048;   // k-space samples (divisible by 1024)
    int N = 4096;   // image voxels   (divisible by 1024)

    size_t szM = M * sizeof(float);
    size_t szN = N * sizeof(float);

    float *h_rPhi=(float*)malloc(szM), *h_iPhi=(float*)malloc(szM);
    float *h_rD  =(float*)malloc(szM), *h_iD  =(float*)malloc(szM);
    float *h_kx  =(float*)malloc(szM), *h_ky  =(float*)malloc(szM);
    float *h_kz  =(float*)malloc(szM);
    float *h_x   =(float*)malloc(szN), *h_y   =(float*)malloc(szN);
    float *h_z   =(float*)malloc(szN);
    float *h_rFcpu=(float*)calloc(N,sizeof(float));
    float *h_iFcpu=(float*)calloc(N,sizeof(float));
    float *h_rFgpu=(float*)calloc(N,sizeof(float));
    float *h_iFgpu=(float*)calloc(N,sizeof(float));

    srand(42);
    for (int m=0;m<M;m++) {
        h_rPhi[m]=(rand()/(float)RAND_MAX)-0.5f; h_iPhi[m]=(rand()/(float)RAND_MAX)-0.5f;
        h_rD[m]  =(rand()/(float)RAND_MAX)-0.5f; h_iD[m]  =(rand()/(float)RAND_MAX)-0.5f;
        h_kx[m]  =(rand()/(float)RAND_MAX)*0.5f-0.25f;
        h_ky[m]  =(rand()/(float)RAND_MAX)*0.5f-0.25f;
        h_kz[m]  =(rand()/(float)RAND_MAX)*0.5f-0.25f;
    }
    for (int n=0;n<N;n++) {
        h_x[n]=(rand()/(float)RAND_MAX)*2.f-1.f;
        h_y[n]=(rand()/(float)RAND_MAX)*2.f-1.f;
        h_z[n]=(rand()/(float)RAND_MAX)*2.f-1.f;
    }

    fhd_cpu(h_rPhi,h_iPhi,h_rD,h_iD,h_kx,h_ky,h_kz,
            h_x,h_y,h_z, h_rFcpu,h_iFcpu, M,N);

    float *d_rPhi,*d_iPhi,*d_rD,*d_iD,*d_kx,*d_ky,*d_kz;
    float *d_x,*d_y,*d_z,*d_rMu,*d_iMu,*d_rFhD,*d_iFhD;
    cudaMalloc(&d_rPhi,szM); cudaMalloc(&d_iPhi,szM);
    cudaMalloc(&d_rD,  szM); cudaMalloc(&d_iD,  szM);
    cudaMalloc(&d_kx,  szM); cudaMalloc(&d_ky,  szM); cudaMalloc(&d_kz,szM);
    cudaMalloc(&d_x,   szN); cudaMalloc(&d_y,   szN); cudaMalloc(&d_z, szN);
    cudaMalloc(&d_rMu, szM); cudaMalloc(&d_iMu, szM);
    cudaMalloc(&d_rFhD,szN); cudaMalloc(&d_iFhD,szN);

    cudaMemcpy(d_rPhi,h_rPhi,szM,cudaMemcpyHostToDevice);
    cudaMemcpy(d_iPhi,h_iPhi,szM,cudaMemcpyHostToDevice);
    cudaMemcpy(d_rD,  h_rD,  szM,cudaMemcpyHostToDevice);
    cudaMemcpy(d_iD,  h_iD,  szM,cudaMemcpyHostToDevice);
    cudaMemcpy(d_kx,  h_kx,  szM,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ky,  h_ky,  szM,cudaMemcpyHostToDevice);
    cudaMemcpy(d_kz,  h_kz,  szM,cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,   h_x,   szN,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,   h_y,   szN,cudaMemcpyHostToDevice);
    cudaMemcpy(d_z,   h_z,   szN,cudaMemcpyHostToDevice);
    cudaMemset(d_rFhD,0,szN); cudaMemset(d_iFhD,0,szN);

    cudaEvent_t t0,t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    cmpMu<<<M/MU_THREADS_PER_BLOCK, MU_THREADS_PER_BLOCK>>>(
        d_rPhi,d_iPhi,d_rD,d_iD,d_rMu,d_iMu);
    cmpFhD_scatter<<<M/FHD_THREADS_PER_BLOCK, FHD_THREADS_PER_BLOCK>>>(
        d_kx,d_ky,d_kz, d_x,d_y,d_z, d_rMu,d_iMu, d_rFhD,d_iFhD, N);

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms,t0,t1);

    cudaMemcpy(h_rFgpu,d_rFhD,szN,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_iFgpu,d_iFhD,szN,cudaMemcpyDeviceToHost);

    float er=max_rel_err(h_rFgpu,h_rFcpu,N);
    float ei=max_rel_err(h_iFgpu,h_iFcpu,N);
    printf("M=%d, N=%d\n", M, N);
    printf("GPU time: %.3f ms\n", ms);
    printf("Max rel error: rFhD=%.2e  iFhD=%.2e  %s\n",
           er, ei, (er<1e-3f && ei<1e-3f) ? "PASS" : "FAIL");
    printf("\nScatter trade-offs (§17.3 Step 1):\n");
    printf("  Threads:   M=%d (one per k-space sample)\n", M);
    printf("  Per thread: %d atomicAdd pairs — contends with all %d threads\n", N, M);
    printf("  − atomicAdd on N=%d locations → heavy serialisation\n", N);
    printf("  → Gather approach (file 02): loop interchange removes atomics\n");

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
