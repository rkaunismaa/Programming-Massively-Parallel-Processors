// §14.6  SpMV with the JDS format — Figures 14.14–14.15
//
// JDS (jagged diagonal storage) achieves BOTH coalesced memory accesses AND
// reduced control divergence without the padding overhead of ELL.
//
// Construction (Fig 14.14):
//   1. Sort rows by DECREASING nonzero count.  Keep row[] to map sorted
//      index back to the original row index.
//   2. Store the resulting jagged matrix in column-major order.
//   3. Build iterPtr[]: iterPtr[t] = start of iteration t in colIdx/value.
//      At iteration t, the active sorted rows are those with nnz > t.
//      Count of active rows at t = iterPtr[t+1] - iterPtr[t].
//
// Access pattern at iteration t (Fig 14.15):
//   Thread r reads colIdx[iterPtr[t] + r]  and  value[iterPtr[t] + r].
//   For a fixed t, consecutive r values read consecutive array locations
//   → fully coalesced, same as ELL.
//
// Reduced divergence: because rows are sorted, threads in the same warp
// cover rows with similar nnz counts.  They tend to exhaust their nonzeros
// at nearly the same iteration step, minimising idle lanes within a warp.
//
// vs. ELL:      no padding (smaller arrays), same coalescing, less divergence.
// vs. CSR:      coalesced, less divergence (but harder to add nonzeros).
// vs. ELL-COO:  no threshold needed; handles all rows uniformly.
//
// Kernel based on the description in §14.6 and Fig 14.15.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// ── Kernel (§14.6, Fig 14.15): one thread per sorted row ─────────────────────
// Thread r works on sorted row r.  At iteration t it accesses element
// iterPtr[t]+r as long as r is still an active row (r < active count at t).
// Once inactive it breaks — rows are sorted so this is contiguous within warp.
__global__ void spmv_jds_kernel(const int *iterPtr, const int *colIdx,
                                  const float *value, const int *rowMap,
                                  const float *x, float *y,
                                  int numSortedRows, int numIter) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= numSortedRows) return;
    float sum = 0.0f;
    for (int t = 0; t < numIter; t++) {
        int active = iterPtr[t + 1] - iterPtr[t];
        if (r >= active) break;              // row r has no nonzero at step t
        int i = iterPtr[t] + r;
        sum  += x[colIdx[i]] * value[i];
    }
    y[rowMap[r]] = sum;
}

// ── Build JDS from COO (sorted by row) ───────────────────────────────────────
// Returns iterPtr[numIter+1], colIdx[numNZ], value[numNZ], rowMap[numRows].
typedef struct {
    int   *iterPtr;   // [numIter+1]
    int   *colIdx;    // [numNZ]
    float *value;     // [numNZ]
    int   *rowMap;    // [numRows]  rowMap[sorted_idx] = original row index
    int    numNZ;
    int    numRows;
    int    numIter;   // = max nnz per row after sorting
} JDSMatrix;

static int g_nnz[65536];  // scratch for qsort comparator (max numRows)

static int cmp_desc(const void *a, const void *b) {
    return g_nnz[*(int *)b] - g_nnz[*(int *)a];
}

static JDSMatrix coo_to_jds(const int *row, const int *col, const float *val,
                              int numNZ, int numRows) {
    // Step 1: count NZ per row
    int *nnz = (int *)calloc(numRows, sizeof(int));
    for (int i = 0; i < numNZ; i++) nnz[row[i]]++;

    // Step 2: sort rows by decreasing NZ count
    int *sortedRows = (int *)malloc(numRows * sizeof(int));
    for (int r = 0; r < numRows; r++) sortedRows[r] = r;
    memcpy(g_nnz, nnz, numRows * sizeof(int));
    qsort(sortedRows, numRows, sizeof(int), cmp_desc);

    // Step 3: build iterPtr
    int maxNnz = nnz[sortedRows[0]];
    int *iterPtr = (int *)malloc((maxNnz + 1) * sizeof(int));
    iterPtr[0] = 0;
    for (int t = 0; t < maxNnz; t++) {
        // Count sorted rows with nnz > t (they're in descending order so stop early)
        int count = 0;
        for (int k = 0; k < numRows; k++) {
            if (nnz[sortedRows[k]] > t) count++;
            else break;
        }
        iterPtr[t + 1] = iterPtr[t] + count;
    }

    // Step 4: build colIdx and value (column-major in sorted-row order)
    // Use a temporary CSR to access each row's nonzeros by iteration index.
    int *rowStart = (int *)calloc(numRows + 1, sizeof(int));
    for (int i = 0; i < numNZ; i++) rowStart[row[i] + 1]++;
    for (int r = 1; r <= numRows; r++) rowStart[r] += rowStart[r - 1];
    int   *rowCol = (int   *)malloc(numNZ * sizeof(int));
    float *rowVal = (float *)malloc(numNZ * sizeof(float));
    int *fill = (int *)calloc(numRows, sizeof(int));
    for (int i = 0; i < numNZ; i++) {
        int r   = row[i];
        int pos = rowStart[r] + fill[r]++;
        rowCol[pos] = col[i];
        rowVal[pos] = val[i];
    }

    JDSMatrix m;
    m.numNZ    = numNZ;
    m.numRows  = numRows;
    m.numIter  = maxNnz;
    m.iterPtr  = iterPtr;
    m.rowMap   = sortedRows;
    m.colIdx   = (int   *)malloc(numNZ * sizeof(int));
    m.value    = (float *)malloc(numNZ * sizeof(float));

    int idx = 0;
    for (int t = 0; t < maxNnz; t++) {
        int count = iterPtr[t + 1] - iterPtr[t];
        for (int k = 0; k < count; k++) {
            int r = sortedRows[k];
            m.colIdx[idx] = rowCol[rowStart[r] + t];
            m.value [idx] = rowVal[rowStart[r] + t];
            idx++;
        }
    }

    free(nnz); free(rowStart); free(rowCol); free(rowVal); free(fill);
    return m;
}

static void free_jds(JDSMatrix *m) {
    free(m->iterPtr); free(m->colIdx); free(m->value); free(m->rowMap);
}

// ── CPU reference ─────────────────────────────────────────────────────────────
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
    printf("=== SpMV / JDS Format (§14.6, Figs 14.14–14.15) ===\n\n");

    // ── Small test: Fig 14.1 ──────────────────────────────────────────────────
    {
        int   coo_row[] = {0, 0, 1, 1, 1, 2, 2, 3};
        int   coo_col[] = {0, 1, 0, 2, 3, 2, 3, 3};
        float coo_val[] = {1, 7, 5, 3, 9, 2, 8, 6};
        int NZ = 8, NR = 4;
        float h_x[] = {1, 2, 3, 4};

        JDSMatrix jds = coo_to_jds(coo_row, coo_col, coo_val, NZ, NR);
        printf("JDS layout (numIter=%d):\n", jds.numIter);
        printf("  rowMap (sorted→original): [");
        for (int k = 0; k < NR; k++) printf("%d%s", jds.rowMap[k], k<NR-1?",":"]\n");
        printf("  iterPtr: [");
        for (int t = 0; t <= jds.numIter; t++)
            printf("%d%s", jds.iterPtr[t], t<jds.numIter?",":"]\n");
        printf("  colIdx: [");
        for (int i = 0; i < NZ; i++) printf("%d%s", jds.colIdx[i], i<NZ-1?",":"]\n");
        printf("\n");

        int *d_iptr, *d_col, *d_rmap;
        float *d_val, *d_x, *d_y;
        cudaMalloc(&d_iptr, (jds.numIter+1)*sizeof(int));
        cudaMalloc(&d_col,  NZ*sizeof(int));
        cudaMalloc(&d_val,  NZ*sizeof(float));
        cudaMalloc(&d_rmap, NR*sizeof(int));
        cudaMalloc(&d_x,    NR*sizeof(float));
        cudaMalloc(&d_y,    NR*sizeof(float));
        cudaMemcpy(d_iptr, jds.iterPtr, (jds.numIter+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col,  jds.colIdx,  NZ*sizeof(int),               cudaMemcpyHostToDevice);
        cudaMemcpy(d_val,  jds.value,   NZ*sizeof(float),              cudaMemcpyHostToDevice);
        cudaMemcpy(d_rmap, jds.rowMap,  NR*sizeof(int),               cudaMemcpyHostToDevice);
        cudaMemcpy(d_x,    h_x,         NR*sizeof(float),              cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, NR*sizeof(float));

        spmv_jds_kernel<<<1, BLOCK_SIZE>>>(d_iptr, d_col, d_val, d_rmap,
                                            d_x, d_y, NR, jds.numIter);
        cudaDeviceSynchronize();

        float h_y[4];
        cudaMemcpy(h_y, d_y, NR*sizeof(float), cudaMemcpyDeviceToHost);
        printf("Fig 14.1: y = [%.0f %.0f %.0f %.0f]  expected [15 50 38 24]  %s\n\n",
               h_y[0], h_y[1], h_y[2], h_y[3],
               (h_y[0]==15 && h_y[1]==50 && h_y[2]==38 && h_y[3]==24) ? "PASS":"FAIL");

        free_jds(&jds);
        cudaFree(d_iptr); cudaFree(d_col); cudaFree(d_val);
        cudaFree(d_rmap); cudaFree(d_x); cudaFree(d_y);
    }

    // ── Large test ────────────────────────────────────────────────────────────
    {
        int NR = 4096, NC = 4096;
        int *coo_row, *coo_col; float *coo_val; int NZ;
        gen_coo(NR, NC, 4, 32, &coo_row, &coo_col, &coo_val, &NZ);
        printf("Large test: %d × %d  NZ=%d  avg=%.1f/row\n", NR, NC, NZ, (float)NZ/NR);

        JDSMatrix jds = coo_to_jds(coo_row, coo_col, coo_val, NZ, NR);
        printf("  JDS: numIter=%d  NZ storage=%d (no padding; ELL-only would need ≥%d)\n",
               jds.numIter, NZ, NR * jds.numIter);

        float *h_x   = (float *)malloc(NC * sizeof(float));
        float *y_ref = (float *)calloc(NR, sizeof(float));
        for (int i = 0; i < NC; i++) h_x[i] = 1.0f / (i + 1);
        spmv_coo_cpu(coo_row, coo_col, coo_val, h_x, y_ref, NZ);

        int *d_iptr, *d_col, *d_rmap;
        float *d_val, *d_x, *d_y;
        cudaMalloc(&d_iptr, (jds.numIter+1)*sizeof(int));
        cudaMalloc(&d_col,  NZ*sizeof(int));
        cudaMalloc(&d_val,  NZ*sizeof(float));
        cudaMalloc(&d_rmap, NR*sizeof(int));
        cudaMalloc(&d_x,    NC*sizeof(float));
        cudaMalloc(&d_y,    NR*sizeof(float));
        cudaMemcpy(d_iptr, jds.iterPtr, (jds.numIter+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col,  jds.colIdx,  NZ*sizeof(int),               cudaMemcpyHostToDevice);
        cudaMemcpy(d_val,  jds.value,   NZ*sizeof(float),              cudaMemcpyHostToDevice);
        cudaMemcpy(d_rmap, jds.rowMap,  NR*sizeof(int),               cudaMemcpyHostToDevice);
        cudaMemcpy(d_x,    h_x,         NC*sizeof(float),              cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, NR*sizeof(float));

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        int nb = (NR + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaEventRecord(t0);
        spmv_jds_kernel<<<nb, BLOCK_SIZE>>>(d_iptr, d_col, d_val, d_rmap,
                                             d_x, d_y, NR, jds.numIter);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);

        float *y_gpu = (float *)malloc(NR * sizeof(float));
        cudaMemcpy(y_gpu, d_y, NR*sizeof(float), cudaMemcpyDeviceToHost);
        printf("  GPU time: %.3f ms  %s\n\n", ms,
               verify(y_ref, y_gpu, NR, 1e-3f) ? "PASS" : "FAIL");

        printf("JDS trade-offs (§14.6):\n");
        printf("  + No padding: stores exactly %d NZ (ELL-only would store %d)\n",
               NZ, NR * jds.numIter);
        printf("  + Coalesced: at iteration t, threads read colIdx[iterPtr[t]+0],\n");
        printf("    colIdx[iterPtr[t]+1], ... (consecutive addresses)\n");
        printf("  + Reduced divergence: sorted rows → warp threads share similar nnz\n");
        printf("    counts → they exhaust at nearly the same step\n");
        printf("  − Harder to add nonzeros (requires re-sorting)\n");
        printf("  − iterPtr[] cannot force architecturally aligned starts per iter\n");
        printf("    (slight disadvantage vs. ELL which can be padded to 64-byte boundaries)\n");

        free(coo_row); free(coo_col); free(coo_val);
        free(h_x); free(y_ref); free(y_gpu);
        free_jds(&jds);
        cudaFree(d_iptr); cudaFree(d_col); cudaFree(d_val);
        cudaFree(d_rmap); cudaFree(d_x); cudaFree(d_y);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    return 0;
}
