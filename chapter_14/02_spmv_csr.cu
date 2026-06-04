// §14.3  SpMV with the CSR format — Figures 14.7–14.9
//
// CSR (compressed sparse row) replaces the per-nonzero rowIdx array of COO
// with a compact rowPtrs array of length numRows+1.
// rowPtrs[r] = index in colIdx/value of the first nonzero in row r.
// rowPtrs[numRows] = numNZ  (sentinel for easy end-of-row detection).
//
// Fig 14.1 example:
//   rowPtrs: 0  2  5  7  8
//   colIdx:  0 1 | 0 2 3 | 2 3 | 3
//   value:   1 7 | 5 3 9 | 2 8 | 6
//
// Kernel (Fig 14.9): one thread per row.
//   No atomicAdd: each thread owns its entire row, writes y[row] once.
//   Non-coalesced reads: thread r reads value[rowPtrs[r]], value[rowPtrs[r]+1],...
//   In iteration 0, threads 0,1,2,3 access value[0], value[2], value[5], value[7]
//   — addresses that are NOT consecutive in memory.
//   Control divergence: threads in a warp iterate different numbers of times
//   depending on how many nonzeros their row contains.
//
// Trade-offs vs. COO (§14.3):
//   + No atomicAdd (one thread owns each row).
//   + Space: rowPtrs costs numRows+1 ints vs. numNZ ints for rowIdx.
//   − Non-coalesced reads for colIdx and value arrays.
//   − Control divergence when row lengths vary across a warp.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// ── Kernel (Fig 14.9): one thread per row ─────────────────────────────────────
__global__ void spmv_csr_kernel(const int *rowPtrs, const int *colIdx,
                                 const float *value, const float *x, float *y,
                                 int numRows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        for (int i = rowPtrs[row]; i < rowPtrs[row + 1]; i++)
            sum += x[colIdx[i]] * value[i];
        y[row] = sum;
    }
}

// ── Build CSR from COO (sorted by row) ───────────────────────────────────────
static void coo_to_csr(const int *row, const int *col, const float *val,
                        int numNZ, int numRows,
                        int **rowPtrs, int **csrCol, float **csrVal) {
    *rowPtrs = (int   *)calloc(numRows + 1, sizeof(int));
    *csrCol  = (int   *)malloc(numNZ * sizeof(int));
    *csrVal  = (float *)malloc(numNZ * sizeof(float));
    // Count NZ per row → rowPtrs[r+1]
    for (int i = 0; i < numNZ; i++) (*rowPtrs)[row[i] + 1]++;
    // Exclusive prefix sum
    for (int r = 1; r <= numRows; r++) (*rowPtrs)[r] += (*rowPtrs)[r - 1];
    // Fill colIdx / value using a fill-position counter
    int *pos = (int *)calloc(numRows, sizeof(int));
    for (int i = 0; i < numNZ; i++) {
        int r   = row[i];
        int dst = (*rowPtrs)[r] + pos[r]++;
        (*csrCol)[dst] = col[i];
        (*csrVal)[dst] = val[i];
    }
    free(pos);
}

// ── CPU reference ─────────────────────────────────────────────────────────────
static void spmv_csr_cpu(const int *rowPtrs, const int *colIdx, const float *value,
                          const float *x, float *y, int numRows) {
    for (int r = 0; r < numRows; r++) {
        float sum = 0.0f;
        for (int i = rowPtrs[r]; i < rowPtrs[r + 1]; i++)
            sum += x[colIdx[i]] * value[i];
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
    printf("=== SpMV / CSR Format (§14.3, Figs 14.7–14.9) ===\n\n");

    // ── Small test: Fig 14.1 ──────────────────────────────────────────────────
    {
        // COO source (same as file 01)
        int   coo_row[] = {0, 0, 1, 1, 1, 2, 2, 3};
        int   coo_col[] = {0, 1, 0, 2, 3, 2, 3, 3};
        float coo_val[] = {1, 7, 5, 3, 9, 2, 8, 6};
        int NZ = 8, NR = 4;
        float h_x[] = {1, 2, 3, 4};

        int *rowPtrs, *csrCol; float *csrVal;
        coo_to_csr(coo_row, coo_col, coo_val, NZ, NR, &rowPtrs, &csrCol, &csrVal);
        // CSR: rowPtrs = [0,2,5,7,8]

        int *d_rp, *d_col; float *d_val, *d_x, *d_y;
        cudaMalloc(&d_rp,  (NR+1)*sizeof(int));
        cudaMalloc(&d_col, NZ*sizeof(int));
        cudaMalloc(&d_val, NZ*sizeof(float));
        cudaMalloc(&d_x,   NR*sizeof(float));
        cudaMalloc(&d_y,   NR*sizeof(float));
        cudaMemcpy(d_rp,  rowPtrs, (NR+1)*sizeof(int),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_col, csrCol,  NZ*sizeof(int),       cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, csrVal,  NZ*sizeof(float),     cudaMemcpyHostToDevice);
        cudaMemcpy(d_x,   h_x,    NR*sizeof(float),      cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, NR*sizeof(float));

        spmv_csr_kernel<<<1, BLOCK_SIZE>>>(d_rp, d_col, d_val, d_x, d_y, NR);
        cudaDeviceSynchronize();

        float h_y[4];
        cudaMemcpy(h_y, d_y, NR*sizeof(float), cudaMemcpyDeviceToHost);
        printf("Fig 14.1: y = [%.0f %.0f %.0f %.0f]  expected [15 50 38 24]  %s\n",
               h_y[0], h_y[1], h_y[2], h_y[3],
               (h_y[0]==15 && h_y[1]==50 && h_y[2]==38 && h_y[3]==24) ? "PASS":"FAIL");
        printf("  rowPtrs = [%d %d %d %d %d]\n\n",
               rowPtrs[0], rowPtrs[1], rowPtrs[2], rowPtrs[3], rowPtrs[4]);

        free(rowPtrs); free(csrCol); free(csrVal);
        cudaFree(d_rp); cudaFree(d_col); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
    }

    // ── Large test ────────────────────────────────────────────────────────────
    {
        int NR = 4096, NC = 4096;
        int *coo_row, *coo_col; float *coo_val; int NZ;
        gen_coo(NR, NC, 4, 32, &coo_row, &coo_col, &coo_val, &NZ);
        printf("Large test: %d × %d  NZ=%d  avg=%.1f/row\n", NR, NC, NZ, (float)NZ/NR);

        int *rowPtrs, *csrCol; float *csrVal;
        coo_to_csr(coo_row, coo_col, coo_val, NZ, NR, &rowPtrs, &csrCol, &csrVal);

        float *h_x   = (float *)malloc(NC * sizeof(float));
        float *y_ref = (float *)calloc(NR, sizeof(float));
        for (int i = 0; i < NC; i++) h_x[i] = 1.0f / (i + 1);
        spmv_csr_cpu(rowPtrs, csrCol, csrVal, h_x, y_ref, NR);

        int *d_rp, *d_col; float *d_val, *d_x, *d_y;
        cudaMalloc(&d_rp,  (NR+1)*sizeof(int));
        cudaMalloc(&d_col, NZ*sizeof(int));
        cudaMalloc(&d_val, NZ*sizeof(float));
        cudaMalloc(&d_x,   NC*sizeof(float));
        cudaMalloc(&d_y,   NR*sizeof(float));
        cudaMemcpy(d_rp,  rowPtrs, (NR+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col, csrCol,  NZ*sizeof(int),      cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, csrVal,  NZ*sizeof(float),    cudaMemcpyHostToDevice);
        cudaMemcpy(d_x,   h_x,    NC*sizeof(float),     cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, NR*sizeof(float));

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        int nb = (NR + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaEventRecord(t0);
        spmv_csr_kernel<<<nb, BLOCK_SIZE>>>(d_rp, d_col, d_val, d_x, d_y, NR);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);

        float *y_gpu = (float *)malloc(NR * sizeof(float));
        cudaMemcpy(y_gpu, d_y, NR*sizeof(float), cudaMemcpyDeviceToHost);
        printf("  GPU time: %.3f ms  %s\n\n", ms,
               verify(y_ref, y_gpu, NR, 1e-3f) ? "PASS" : "FAIL");

        printf("CSR trade-offs (§14.3):\n");
        printf("  + No atomicAdd: each thread owns its row entirely\n");
        printf("  + rowPtrs costs %d ints; rowIdx in COO would cost %d ints\n",
               NR + 1, NZ);
        printf("  − Non-coalesced: threads access value at offsets rowPtrs[0..3]=");
        printf(" [%d,%d,%d,%d] — not consecutive\n",
               rowPtrs[0], rowPtrs[1], rowPtrs[2], rowPtrs[3]);
        printf("  − Control divergence: loop counts differ per row (1..%d NZ)\n",
               rowPtrs[NR] - rowPtrs[NR-1] > rowPtrs[1]-rowPtrs[0]
               ? rowPtrs[NR] - rowPtrs[NR-1] : rowPtrs[1]-rowPtrs[0]);

        free(coo_row); free(coo_col); free(coo_val);
        free(rowPtrs); free(csrCol); free(csrVal);
        free(h_x); free(y_ref); free(y_gpu);
        cudaFree(d_rp); cudaFree(d_col); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    return 0;
}
