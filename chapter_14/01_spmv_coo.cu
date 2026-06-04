// §14.2  SpMV with the COO format — Figures 14.3–14.5
//
// COO (coordinate list) stores every nonzero as a (rowIdx, colIdx, value)
// triplet.  Three parallel arrays of length numNZ hold all the data.
//
// Fig 14.1 example (4 × 4 matrix):
//   rowIdx: 0 0 1 1 1 2 2 3
//   colIdx: 0 1 0 2 3 2 3 3
//   value:  1 7 5 3 9 2 8 6
//
// Kernel (Fig 14.5): one thread per nonzero.
//   Coalesced global reads: consecutive threads read consecutive positions
//   in rowIdx, colIdx, value (physical view of Fig 14.4).
//   atomicAdd required: two or more nonzeros may share a row index and must
//   both update the same y[row] element safely.
//
// Trade-offs (§14.2):
//   + Coalesced reads for all three arrays.
//   + Parallelism over numNZ elements (more than numRows).
//   + Flexible: nonzeros can be stored in any order.
//   − atomicAdd causes serialisation for rows with many nonzeros.
//   − rowIdx array uses extra memory vs. CSR's compact rowPtrs.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// ── Kernel (Fig 14.5) ─────────────────────────────────────────────────────────
__global__ void spmv_coo_kernel(const int *rowIdx, const int *colIdx,
                                 const float *value, const float *x, float *y,
                                 int numNZ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNZ) {
        unsigned int row = rowIdx[i];
        unsigned int col = colIdx[i];
        atomicAdd(&y[row], x[col] * value[i]);
    }
}

// ── CPU reference ─────────────────────────────────────────────────────────────
static void spmv_coo_cpu(const int *rowIdx, const int *colIdx, const float *value,
                          const float *x, float *y, int numNZ) {
    for (int i = 0; i < numNZ; i++)
        y[rowIdx[i]] += x[colIdx[i]] * value[i];
}

// ── Helpers ───────────────────────────────────────────────────────────────────
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

// Skewed random COO matrix: 80 % of rows have minNnz..2*minNnz nonzeros,
// 20 % have maxNnz/2..maxNnz nonzeros.  Sorted by row, duplicates avoided.
static void gen_coo(int NR, int NC, int minNnz, int maxNnz,
                    int **pRow, int **pCol, float **pVal, int *pNZ) {
    srand(42);
    int *nnz = (int *)malloc(NR * sizeof(int));
    int total = 0;
    for (int r = 0; r < NR; r++) {
        nnz[r] = (rand() % 5 == 0)
                 ? maxNnz / 2 + rand() % (maxNnz / 2 + 1)
                 : minNnz    + rand() % (minNnz + 1);
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
    printf("=== SpMV / COO Format (§14.2, Figs 14.3–14.5) ===\n\n");

    // ── Small test: Fig 14.1 ──────────────────────────────────────────────────
    {
        int   h_row[] = {0, 0, 1, 1, 1, 2, 2, 3};
        int   h_col[] = {0, 1, 0, 2, 3, 2, 3, 3};
        float h_val[] = {1, 7, 5, 3, 9, 2, 8, 6};
        float h_x[]   = {1, 2, 3, 4};
        int NZ = 8, NR = 4;

        int *d_row, *d_col; float *d_val, *d_x, *d_y;
        cudaMalloc(&d_row, NZ*sizeof(int));   cudaMalloc(&d_col, NZ*sizeof(int));
        cudaMalloc(&d_val, NZ*sizeof(float)); cudaMalloc(&d_x,   NR*sizeof(float));
        cudaMalloc(&d_y,   NR*sizeof(float));
        cudaMemcpy(d_row, h_row, NZ*sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_col, h_col, NZ*sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, h_val, NZ*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x,   h_x,  NR*sizeof(float),  cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, NR*sizeof(float));

        spmv_coo_kernel<<<1, BLOCK_SIZE>>>(d_row, d_col, d_val, d_x, d_y, NZ);
        cudaDeviceSynchronize();

        float h_y[4];
        cudaMemcpy(h_y, d_y, NR*sizeof(float), cudaMemcpyDeviceToHost);
        printf("Fig 14.1: y = [%.0f %.0f %.0f %.0f]  expected [15 50 38 24]  %s\n\n",
               h_y[0], h_y[1], h_y[2], h_y[3],
               (h_y[0]==15 && h_y[1]==50 && h_y[2]==38 && h_y[3]==24) ? "PASS":"FAIL");

        cudaFree(d_row); cudaFree(d_col); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
    }

    // ── Large test ────────────────────────────────────────────────────────────
    {
        int NR = 4096, NC = 4096;
        int *h_row, *h_col; float *h_val; int NZ;
        gen_coo(NR, NC, 4, 32, &h_row, &h_col, &h_val, &NZ);
        printf("Large test: %d × %d  NZ=%d  avg=%.1f/row\n", NR, NC, NZ, (float)NZ/NR);

        float *h_x   = (float *)calloc(NC, sizeof(float));
        float *y_ref = (float *)calloc(NR, sizeof(float));
        for (int i = 0; i < NC; i++) h_x[i] = 1.0f / (i + 1);
        spmv_coo_cpu(h_row, h_col, h_val, h_x, y_ref, NZ);

        int *d_row, *d_col; float *d_val, *d_x, *d_y;
        cudaMalloc(&d_row, NZ*sizeof(int));    cudaMalloc(&d_col, NZ*sizeof(int));
        cudaMalloc(&d_val, NZ*sizeof(float));  cudaMalloc(&d_x,   NC*sizeof(float));
        cudaMalloc(&d_y,   NR*sizeof(float));
        cudaMemcpy(d_row, h_row, NZ*sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_col, h_col, NZ*sizeof(int),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, h_val, NZ*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x,   h_x,  NC*sizeof(float),  cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, NR*sizeof(float));

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        int nb = (NZ + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaEventRecord(t0);
        spmv_coo_kernel<<<nb, BLOCK_SIZE>>>(d_row, d_col, d_val, d_x, d_y, NZ);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);

        float *y_gpu = (float *)malloc(NR*sizeof(float));
        cudaMemcpy(y_gpu, d_y, NR*sizeof(float), cudaMemcpyDeviceToHost);
        printf("  GPU time: %.3f ms  %s\n\n", ms,
               verify(y_ref, y_gpu, NR, 1e-3f) ? "PASS" : "FAIL");

        printf("COO trade-offs (§14.2):\n");
        printf("  + %d threads (one/NZ) → more parallelism than %d rows\n", NZ, NR);
        printf("  + Coalesced reads: rowIdx, colIdx, value all accessed stride-1\n");
        printf("  − atomicAdd: rows with many NZ serialise partial sums\n");
        printf("  − rowIdx costs %d ints; CSR rowPtrs costs only %d ints\n",
               NZ, NR + 1);

        free(h_row); free(h_col); free(h_val); free(h_x); free(y_ref); free(y_gpu);
        cudaFree(d_row); cudaFree(d_col); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    return 0;
}
