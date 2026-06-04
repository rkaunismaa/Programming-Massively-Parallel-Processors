// §14.5  SpMV with the hybrid ELL-COO format — Figure 14.13
//
// Problem with ELL: when a few rows have far more nonzeros than the rest,
// padding every row to maxNnzPerRow wastes memory and bandwidth.
//
// Fix: choose a threshold T.
//   ELL part: store the first T nonzeros of every row in ELL (column-major).
//   COO part: collect all overflow nonzeros (beyond T) in COO.
//
// SpMV = SpMV/ELL on ELL part  (coalesced, no atomics)
//       + SpMV/COO on COO part  (coalesced reads, atomicAdd for overflow rows)
//
// T is typically chosen to minimise total storage:
//   total = numRows * T * 2 + numOverflowNZ * 3   (two ELL arrays + three COO)
//
// vs. ELL alone: much less padding for skewed nnz distributions.
// vs. COO alone: the majority of work is handled with coalesced, atomic-free ELL.
//
// Fig 14.13 illustrates: rows with exceedingly large nonzero counts "overflow"
// into the COO part, reducing the ELL column count and hence the padding.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// ── ELL part kernel ───────────────────────────────────────────────────────────
// Identical to §14.4 kernel, but iterates exactly ellMaxNnz times.
__global__ void spmv_ell_part_kernel(const int *ellCol, const float *ellVal,
                                      const float *x, float *y,
                                      int numRows, int ellMaxNnz) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        for (int t = 0; t < ellMaxNnz; t++) {
            int   i   = t * numRows + row;
            int   col = ellCol[i];
            float val = ellVal[i];
            sum += x[col] * val;
        }
        y[row] = sum;
    }
}

// ── COO overflow part kernel ──────────────────────────────────────────────────
// atomicAdd required: overflow nonzeros from the same row may be handled
// by different threads. (Same as §14.2 kernel.)
__global__ void spmv_coo_part_kernel(const int *rowIdx, const int *colIdx,
                                      const float *value, const float *x, float *y,
                                      int numNZ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNZ) {
        unsigned int row = rowIdx[i];
        unsigned int col = colIdx[i];
        atomicAdd(&y[row], x[col] * value[i]);
    }
}

// ── Build hybrid ELL-COO from COO (sorted by row) ────────────────────────────
static void coo_to_ell_coo(const int *row, const int *col, const float *val,
                             int numNZ, int numRows, int threshold,
                             int **ellCol, float **ellVal,       // ELL arrays
                             int **cooRow, int **cooCol, float **cooVal,
                             int *cooNZ) {                        // COO overflow
    // ELL part: column-major, numRows × threshold elements (padded with 0)
    int ellSize = numRows * threshold;
    *ellCol = (int   *)calloc(ellSize, sizeof(int));
    *ellVal = (float *)calloc(ellSize, sizeof(float));

    // COO overflow: first pass — count overflow NZ
    int *nnz = (int *)calloc(numRows, sizeof(int));
    int overflow = 0;
    for (int i = 0; i < numNZ; i++) {
        int r = row[i];
        if (nnz[r] >= threshold) overflow++;
        nnz[r]++;
    }
    *cooNZ  = overflow;
    *cooRow = (int   *)malloc(overflow * sizeof(int));
    *cooCol = (int   *)malloc(overflow * sizeof(int));
    *cooVal = (float *)malloc(overflow * sizeof(float));

    // Second pass — fill ELL and COO
    int *fill = (int *)calloc(numRows, sizeof(int));
    int  ki   = 0;
    for (int i = 0; i < numNZ; i++) {
        int r = row[i];
        int t = fill[r]++;
        if (t < threshold) {
            // ELL slot: column-major index
            (*ellCol)[t * numRows + r] = col[i];
            (*ellVal)[t * numRows + r] = val[i];
        } else {
            // COO overflow
            (*cooRow)[ki] = row[i];
            (*cooCol)[ki] = col[i];
            (*cooVal)[ki] = val[i];
            ki++;
        }
    }
    free(nnz); free(fill);
}

// ── CPU reference (direct COO spmv) ──────────────────────────────────────────
static void spmv_coo_cpu(const int *row, const int *col, const float *val,
                          const float *x, float *y, int numNZ) {
    for (int i = 0; i < numNZ; i++)
        y[row[i]] += x[col[i]] * val[i];
}

static int verify(const float *ref, const float *gpu, int n, float tol) {
    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - gpu[i]);
        if (diff > tol * (1.0f + fabsf(ref[i]))) {
            printf("  MISMATCH i=%d ref=%.4f gpu=%.4f\n", i, ref[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

static void gen_coo(int NR, int NC, int minNnz, int maxNnz,
                    int **pRow, int **pCol, float **pVal, int *pNZ) {
    srand(42);
    int *nnz = (int *)malloc(NR * sizeof(int));
    int total = 0;
    for (int r = 0; r < NR; r++) {
        nnz[r] = (rand() % 5 == 0)
                 ? maxNnz / 2 + rand() % (maxNnz / 2 + 1)
                 : minNnz     + rand() % (minNnz + 1);
        if (nnz[r] > NC) nnz[r] = NC;
        total += nnz[r];
    }
    *pRow = (int   *)malloc(total * sizeof(int));
    *pCol = (int   *)malloc(total * sizeof(int));
    *pVal = (float *)malloc(total * sizeof(float));
    *pNZ  = total;
    int k = 0;
    for (int r = 0; r < NR; r++)
        for (int j = 0; j < nnz[r]; j++) {
            (*pRow)[k] = r;
            (*pCol)[k] = (r * 17 + j * 31 + 7) % NC;
            (*pVal)[k] = 0.1f + 0.9f * (rand() / (float)RAND_MAX);
            k++;
        }
    free(nnz);
}

int main(void) {
    printf("=== SpMV / Hybrid ELL-COO Format (§14.5, Fig 14.13) ===\n\n");

    // ── Small test: Fig 14.1, threshold = 2 ──────────────────────────────────
    {
        int   coo_row[] = {0, 0, 1, 1, 1, 2, 2, 3};
        int   coo_col[] = {0, 1, 0, 2, 3, 2, 3, 3};
        float coo_val[] = {1, 7, 5, 3, 9, 2, 8, 6};
        int NZ = 8, NR = 4, T = 2;
        float h_x[] = {1, 2, 3, 4};

        int *ellCol, *cooRow, *cooCol; float *ellVal, *cooVal; int cooNZ;
        coo_to_ell_coo(coo_row, coo_col, coo_val, NZ, NR, T,
                       &ellCol, &ellVal, &cooRow, &cooCol, &cooVal, &cooNZ);

        printf("ELL threshold T=%d: ELL size=%d×%d=%d, COO overflow NZ=%d\n",
               T, NR, T, NR*T, cooNZ);
        printf("  (row 1 has 3 NZ; its 3rd NZ overflows to COO)\n\n");

        int *d_ecol, *d_crow, *d_ccol;
        float *d_eval, *d_cval, *d_x, *d_y;
        cudaMalloc(&d_ecol, NR*T*sizeof(int));    cudaMalloc(&d_eval, NR*T*sizeof(float));
        cudaMalloc(&d_crow, cooNZ*sizeof(int));   cudaMalloc(&d_ccol, cooNZ*sizeof(int));
        cudaMalloc(&d_cval, cooNZ*sizeof(float));
        cudaMalloc(&d_x,   NR*sizeof(float));     cudaMalloc(&d_y,   NR*sizeof(float));
        cudaMemcpy(d_ecol, ellCol, NR*T*sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_eval, ellVal, NR*T*sizeof(float),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_crow, cooRow, cooNZ*sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_ccol, cooCol, cooNZ*sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_cval, cooVal, cooNZ*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x,    h_x,   NR*sizeof(float),    cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, NR*sizeof(float));

        int nb_ell = (NR    + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int nb_coo = (cooNZ + BLOCK_SIZE - 1) / BLOCK_SIZE;
        spmv_ell_part_kernel<<<nb_ell, BLOCK_SIZE>>>(d_ecol, d_eval, d_x, d_y, NR, T);
        if (cooNZ > 0)
            spmv_coo_part_kernel<<<nb_coo, BLOCK_SIZE>>>(d_crow, d_ccol, d_cval,
                                                          d_x, d_y, cooNZ);
        cudaDeviceSynchronize();

        float h_y[4];
        cudaMemcpy(h_y, d_y, NR*sizeof(float), cudaMemcpyDeviceToHost);
        printf("Fig 14.1 (T=%d): y = [%.0f %.0f %.0f %.0f]  expected [15 50 38 24]  %s\n\n",
               T, h_y[0], h_y[1], h_y[2], h_y[3],
               (h_y[0]==15 && h_y[1]==50 && h_y[2]==38 && h_y[3]==24) ? "PASS":"FAIL");

        free(ellCol); free(ellVal); free(cooRow); free(cooCol); free(cooVal);
        cudaFree(d_ecol); cudaFree(d_eval); cudaFree(d_crow);
        cudaFree(d_ccol); cudaFree(d_cval); cudaFree(d_x); cudaFree(d_y);
    }

    // ── Large test: compare ELL-only vs. hybrid ───────────────────────────────
    {
        int NR = 4096, NC = 4096;
        int *coo_row, *coo_col; float *coo_val; int NZ;
        gen_coo(NR, NC, 4, 32, &coo_row, &coo_col, &coo_val, &NZ);
        printf("Large test: %d × %d  NZ=%d  avg=%.1f/row\n", NR, NC, NZ, (float)NZ/NR);

        // CPU reference
        float *h_x   = (float *)malloc(NC * sizeof(float));
        float *y_ref = (float *)calloc(NR, sizeof(float));
        for (int i = 0; i < NC; i++) h_x[i] = 1.0f / (i + 1);
        spmv_coo_cpu(coo_row, coo_col, coo_val, h_x, y_ref, NZ);

        // Find maxNnzPerRow (what ELL alone would need)
        int *nnzCount = (int *)calloc(NR, sizeof(int));
        for (int i = 0; i < NZ; i++) nnzCount[coo_row[i]]++;
        int maxNnz = 0;
        for (int r = 0; r < NR; r++) if (nnzCount[r] > maxNnz) maxNnz = nnzCount[r];
        int ell_only_size = NR * maxNnz;

        int T = 8;  // threshold: median nnz per row
        int *ellCol, *cooRow, *cooCol; float *ellVal, *cooVal; int cooNZ;
        coo_to_ell_coo(coo_row, coo_col, coo_val, NZ, NR, T,
                       &ellCol, &ellVal, &cooRow, &cooCol, &cooVal, &cooNZ);
        int hybrid_size = NR * T + cooNZ * 3;

        printf("  ELL-only:  %d elements (%d rows × %d max NZ)\n",
               ell_only_size, NR, maxNnz);
        printf("  Hybrid T=%d: %d ELL + %d COO overflow = %d total elements\n",
               T, NR*T, cooNZ*3, hybrid_size);
        printf("  Space saving: %.1f%%\n\n",
               100.0f * (ell_only_size - hybrid_size) / ell_only_size);

        int *d_ecol, *d_crow, *d_ccol;
        float *d_eval, *d_cval, *d_x, *d_y;
        cudaMalloc(&d_ecol, NR*T*sizeof(int));    cudaMalloc(&d_eval, NR*T*sizeof(float));
        cudaMalloc(&d_crow, cooNZ*sizeof(int));   cudaMalloc(&d_ccol, cooNZ*sizeof(int));
        cudaMalloc(&d_cval, cooNZ*sizeof(float));
        cudaMalloc(&d_x,   NC*sizeof(float));     cudaMalloc(&d_y,   NR*sizeof(float));
        cudaMemcpy(d_ecol, ellCol, NR*T*sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_eval, ellVal, NR*T*sizeof(float),  cudaMemcpyHostToDevice);
        if (cooNZ > 0) {
            cudaMemcpy(d_crow, cooRow, cooNZ*sizeof(int),  cudaMemcpyHostToDevice);
            cudaMemcpy(d_ccol, cooCol, cooNZ*sizeof(int),  cudaMemcpyHostToDevice);
            cudaMemcpy(d_cval, cooVal, cooNZ*sizeof(float),cudaMemcpyHostToDevice);
        }
        cudaMemcpy(d_x, h_x, NC*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, NR*sizeof(float));

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        int nb_ell = (NR    + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int nb_coo = (cooNZ + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaEventRecord(t0);
        spmv_ell_part_kernel<<<nb_ell, BLOCK_SIZE>>>(d_ecol, d_eval, d_x, d_y, NR, T);
        if (cooNZ > 0)
            spmv_coo_part_kernel<<<nb_coo, BLOCK_SIZE>>>(d_crow, d_ccol, d_cval,
                                                          d_x, d_y, cooNZ);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);

        float *y_gpu = (float *)malloc(NR * sizeof(float));
        cudaMemcpy(y_gpu, d_y, NR*sizeof(float), cudaMemcpyDeviceToHost);
        printf("  Hybrid GPU time: %.3f ms  %s\n\n", ms,
               verify(y_ref, y_gpu, NR, 1e-3f) ? "PASS" : "FAIL");

        printf("Hybrid ELL-COO trade-offs (§14.5):\n");
        printf("  + ELL part (T=%d): coalesced reads, no atomics, handles %.1f%% of NZ\n",
               T, 100.0f * (NZ - cooNZ) / NZ);
        printf("  + COO part: handles remaining %.1f%% overflow NZ with atomicAdd\n",
               100.0f * cooNZ / NZ);
        printf("  + Reduced padding vs. ELL-only (%.1f%% space saving shown above)\n",
               100.0f * (ell_only_size - hybrid_size) / ell_only_size);
        printf("  − Threshold T must be chosen wisely (too low → more COO atomics)\n");

        free(nnzCount); free(ellCol); free(ellVal); free(cooRow); free(cooCol); free(cooVal);
        free(coo_row); free(coo_col); free(coo_val); free(h_x); free(y_ref); free(y_gpu);
        cudaFree(d_ecol); cudaFree(d_eval); cudaFree(d_crow);
        cudaFree(d_ccol); cudaFree(d_cval); cudaFree(d_x); cudaFree(d_y);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    return 0;
}
