// §14.4  SpMV with the ELL format — Figures 14.10–14.12
//
// ELL (from the ELLPACK sparse matrix package) achieves coalesced memory
// accesses by padding and transposing the CSR data layout.
//
// Construction (Fig 14.10):
//   1. Find maxNnzPerRow across all rows.
//   2. Pad shorter rows with (colIdx=0, value=0) entries to maxNnzPerRow.
//   3. Transpose the padded rectangle to COLUMN-MAJOR order.
//
// Column-major index formula: i = t * numRows + row
//   → for a fixed iteration t, consecutive threads (row=0,1,2,...)
//     access colIdx[t*numRows+0], colIdx[t*numRows+1], ...
//     which are CONSECUTIVE in memory → fully coalesced.
//
// Fig 14.1 example (maxNnzPerRow = 3):
//   t=0 colIdx:[0,0,2,3]  value:[1,5,2,6]   (rows 0-3, iteration 0)
//   t=1 colIdx:[1,2,3,0]  value:[7,3,8,0]   (row 3 is padding: val=0)
//   t=2 colIdx:[0,3,0,0]  value:[0,9,0,0]   (rows 0,2,3 are padding)
//
// Kernel (Fig 14.12): one thread per row.
//   All threads execute exactly maxNnzPerRow iterations (no control divergence).
//   Padding entries (value=0) contribute zero to the sum silently.
//
// Trade-offs vs. CSR (§14.4):
//   + Coalesced reads (column-major layout).
//   + No control divergence (uniform iteration count).
//   + No atomicAdd.
//   − Padding wastes memory and bandwidth when nnz-per-row varies widely.
//   − Space: numRows × maxNnzPerRow elements vs. numNZ for CSR.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// ── Kernel (Fig 14.12): one thread per row ────────────────────────────────────
// Iterates exactly maxNnzPerRow times; padding values are 0 → no effect on sum.
__global__ void spmv_ell_kernel(const int *colIdx, const float *value,
                                 const float *x, float *y,
                                 int numRows, int maxNnzPerRow) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        for (int t = 0; t < maxNnzPerRow; t++) {
            int   i   = t * numRows + row;   // column-major index
            int   col = colIdx[i];
            float val = value[i];
            sum += x[col] * val;             // padding: val=0, no contribution
        }
        y[row] = sum;
    }
}

// ── Build ELL from COO (sorted by row) ───────────────────────────────────────
// Returns column-major colIdx[] and value[] arrays of size numRows*maxNnz.
static void coo_to_ell(const int *row, const int *col, const float *val,
                        int numNZ, int numRows,
                        int **ellCol, float **ellVal, int *maxNnz,
                        int **nnzPerRow) {
    *nnzPerRow = (int *)calloc(numRows, sizeof(int));
    for (int i = 0; i < numNZ; i++) (*nnzPerRow)[row[i]]++;
    *maxNnz = 0;
    for (int r = 0; r < numRows; r++)
        if ((*nnzPerRow)[r] > *maxNnz) *maxNnz = (*nnzPerRow)[r];

    int size = numRows * (*maxNnz);
    *ellCol = (int   *)calloc(size, sizeof(int));
    *ellVal = (float *)calloc(size, sizeof(float));

    int *fill = (int *)calloc(numRows, sizeof(int));
    for (int i = 0; i < numNZ; i++) {
        int r   = row[i];
        int t   = fill[r]++;
        int idx = t * numRows + r;    // column-major
        (*ellCol)[idx] = col[i];
        (*ellVal)[idx] = val[i];
    }
    free(fill);
}

// ── CPU reference ─────────────────────────────────────────────────────────────
static void spmv_ell_cpu(const int *ellCol, const float *ellVal,
                          const float *x, float *y,
                          int numRows, int maxNnz) {
    for (int r = 0; r < numRows; r++) {
        float sum = 0.0f;
        for (int t = 0; t < maxNnz; t++)
            sum += x[ellCol[t*numRows+r]] * ellVal[t*numRows+r];
        y[r] = sum;
    }
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
    printf("=== SpMV / ELL Format (§14.4, Figs 14.10–14.12) ===\n\n");

    // ── Small test: Fig 14.1 ──────────────────────────────────────────────────
    {
        int   coo_row[] = {0, 0, 1, 1, 1, 2, 2, 3};
        int   coo_col[] = {0, 1, 0, 2, 3, 2, 3, 3};
        float coo_val[] = {1, 7, 5, 3, 9, 2, 8, 6};
        int NZ = 8, NR = 4;
        float h_x[] = {1, 2, 3, 4};

        int *ellCol, *nnzPerRow; float *ellVal; int maxNnz;
        coo_to_ell(coo_row, coo_col, coo_val, NZ, NR,
                   &ellCol, &ellVal, &maxNnz, &nnzPerRow);

        printf("ELL layout (maxNnzPerRow=%d):\n", maxNnz);
        for (int t = 0; t < maxNnz; t++) {
            printf("  t=%d colIdx:[", t);
            for (int r = 0; r < NR; r++) printf("%d%s", ellCol[t*NR+r], r<NR-1?",":"]");
            printf(" value:[");
            for (int r = 0; r < NR; r++) printf("%.0f%s", ellVal[t*NR+r], r<NR-1?",":"]");
            printf("\n");
        }

        int *d_col; float *d_val, *d_x, *d_y;
        int sz = NR * maxNnz;
        cudaMalloc(&d_col, sz*sizeof(int));    cudaMalloc(&d_val, sz*sizeof(float));
        cudaMalloc(&d_x,   NR*sizeof(float));  cudaMalloc(&d_y,   NR*sizeof(float));
        cudaMemcpy(d_col, ellCol, sz*sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, ellVal, sz*sizeof(float),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_x,   h_x,   NR*sizeof(float),  cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, NR*sizeof(float));

        spmv_ell_kernel<<<1, BLOCK_SIZE>>>(d_col, d_val, d_x, d_y, NR, maxNnz);
        cudaDeviceSynchronize();

        float h_y[4];
        cudaMemcpy(h_y, d_y, NR*sizeof(float), cudaMemcpyDeviceToHost);
        printf("Fig 14.1: y = [%.0f %.0f %.0f %.0f]  expected [15 50 38 24]  %s\n\n",
               h_y[0], h_y[1], h_y[2], h_y[3],
               (h_y[0]==15 && h_y[1]==50 && h_y[2]==38 && h_y[3]==24) ? "PASS":"FAIL");

        free(ellCol); free(ellVal); free(nnzPerRow);
        cudaFree(d_col); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
    }

    // ── Large test ────────────────────────────────────────────────────────────
    {
        int NR = 4096, NC = 4096;
        int *coo_row, *coo_col; float *coo_val; int NZ;
        gen_coo(NR, NC, 4, 32, &coo_row, &coo_col, &coo_val, &NZ);
        printf("Large test: %d × %d  NZ=%d  avg=%.1f/row\n", NR, NC, NZ, (float)NZ/NR);

        int *ellCol, *nnzPerRow; float *ellVal; int maxNnz;
        coo_to_ell(coo_row, coo_col, coo_val, NZ, NR,
                   &ellCol, &ellVal, &maxNnz, &nnzPerRow);

        float *h_x   = (float *)malloc(NC * sizeof(float));
        float *y_ref = (float *)calloc(NR, sizeof(float));
        for (int i = 0; i < NC; i++) h_x[i] = 1.0f / (i + 1);
        spmv_ell_cpu(ellCol, ellVal, h_x, y_ref, NR, maxNnz);

        int sz = NR * maxNnz;
        int *d_col; float *d_val, *d_x, *d_y;
        cudaMalloc(&d_col, sz*sizeof(int));   cudaMalloc(&d_val, sz*sizeof(float));
        cudaMalloc(&d_x,   NC*sizeof(float)); cudaMalloc(&d_y,   NR*sizeof(float));
        cudaMemcpy(d_col, ellCol, sz*sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, ellVal, sz*sizeof(float),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_x,   h_x,   NC*sizeof(float),  cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, NR*sizeof(float));

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        int nb = (NR + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaEventRecord(t0);
        spmv_ell_kernel<<<nb, BLOCK_SIZE>>>(d_col, d_val, d_x, d_y, NR, maxNnz);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);

        float *y_gpu = (float *)malloc(NR * sizeof(float));
        cudaMemcpy(y_gpu, d_y, NR*sizeof(float), cudaMemcpyDeviceToHost);
        printf("  maxNnzPerRow=%d  ELL array size=%d  (%.1f× larger than NZ=%d)\n",
               maxNnz, sz, (float)sz / NZ, NZ);
        printf("  GPU time: %.3f ms  %s\n\n", ms,
               verify(y_ref, y_gpu, NR, 1e-3f) ? "PASS" : "FAIL");

        printf("ELL trade-offs (§14.4):\n");
        printf("  + Coalesced: index i=t*numRows+row → consecutive threads read\n");
        printf("    colIdx[t*%d+0], colIdx[t*%d+1], ... (stride-1 access)\n", NR, NR);
        printf("  + No atomicAdd, no control divergence (fixed %d iterations)\n", maxNnz);
        printf("  − Padding: %d allocated vs. %d actual NZ (%.1f%% overhead)\n",
               sz, NZ, 100.0f*(sz-NZ)/NZ);
        printf("  − One outlier row with many NZ forces all rows to pad to %d\n",
               maxNnz);
        printf("    → §14.5 hybrid ELL-COO addresses this\n");

        free(coo_row); free(coo_col); free(coo_val);
        free(ellCol); free(ellVal); free(nnzPerRow);
        free(h_x); free(y_ref); free(y_gpu);
        cudaFree(d_col); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    return 0;
}
