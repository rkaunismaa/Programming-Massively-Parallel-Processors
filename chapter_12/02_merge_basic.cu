// §12.3–12.5 Co-rank function and basic parallel merge kernel
//
// Parallelization approach (§12.3, Siebert & Traff 2012):
//   Each thread owns a contiguous output subarray C[k_curr..k_next-1].
//   The co-rank function identifies the A and B start indices (i, j) such
//   that C[0..k-1] = merge(A[0..i-1], B[0..j-1]).  Then each thread
//   independently runs the sequential merge on its private subarrays.
//
// Co-rank function (§12.4, Figure 12.5):
//   Binary search: O(log(max(m,n))) per call.
//   Invariant: i + j = k throughout; final i satisfies
//     A[i-1] <= B[j] AND B[j-1] < A[i].
//
// Basic kernel (§12.5, Figure 12.9):
//   - ceil((m+n)/gridDim.x * blockDim.x) output elements per thread
//   - Two co_rank calls (start and end of output subarray)
//   - One merge_sequential call per thread
//   - No shared memory: all accesses to global A and B — not coalesced

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ── Figure 12.5: co-rank function (device callable) ──────────────────────────
// Returns i (co-rank of k in A) such that j = k - i is the co-rank in B.
// Constraint: A[i-1] <= B[j]  and  B[j-1] < A[i]
__device__ __host__ int co_rank(int k, int *A, int m, int *B, int n) {
    int i = (k < m) ? k : m;          // i = min(k, m)
    int j = k - i;
    int i_low = (0 > (k - n)) ? 0 : k - n;  // max(0, k-n)
    int j_low = (0 > (k - m)) ? 0 : k - m;  // max(0, k-m)
    bool active = true;

    while (active) {
        if (i > 0 && j < n && A[i-1] > B[j]) {
            int delta = ((i - i_low + 1) >> 1);  // ceil((i - i_low) / 2)
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (j > 0 && i < m && B[j-1] >= A[i]) {
            int delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        } else {
            active = false;
        }
    }
    return i;
}

// ── Figure 12.2: sequential merge (used by each thread) ──────────────────────
__device__ __host__ void merge_sequential(int *A, int m, int *B, int n, int *C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] <= B[j]) C[k++] = A[i++];
        else               C[k++] = B[j++];
    }
    while (j < n) C[k++] = B[j++];
    while (i < m) C[k++] = A[i++];
}

// ── Figure 12.9: basic parallel merge kernel ──────────────────────────────────
// Each thread handles ceil((m+n)/(gridDim.x*blockDim.x)) output elements.
// Two co_rank calls + one sequential merge per thread.
// Global memory accesses in co_rank are uncoalesced (§12.6 motivation).
__global__ void merge_basic_kernel(int *A, int m, int *B, int n, int *C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elementsPerThread = (int)ceilf((float)(m + n) /
                                       (float)(blockDim.x * gridDim.x));

    int k_curr = tid * elementsPerThread;
    int k_next = min((tid + 1) * elementsPerThread, m + n);

    if (k_curr >= m + n) return;

    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);

    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    merge_sequential(&A[i_curr], i_next - i_curr,
                     &B[j_curr], j_next - j_curr,
                     &C[k_curr]);
}

// ── CPU reference ──────────────────────────────────────────────────────────────
static void cpu_merge(int *A, int m, int *B, int n, int *C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] <= B[j]) C[k++] = A[i++];
        else               C[k++] = B[j++];
    }
    while (j < n) C[k++] = B[j++];
    while (i < m) C[k++] = A[i++];
}

static bool verify(int *ref, int *gpu, int N) {
    for (int i = 0; i < N; i++) {
        if (ref[i] != gpu[i]) {
            printf("  MISMATCH i=%d  ref=%d  gpu=%d\n", i, ref[i], gpu[i]);
            return false;
        }
    }
    return true;
}

// ── Co-rank unit test on host ─────────────────────────────────────────────────
static void test_corank_host(void) {
    // Fig. 12.1 example: A=[1,7,8,9,10], B=[7,10,10,12]
    int A[] = {1, 7, 8, 9, 10};
    int B[] = {7, 10, 10, 12};
    int m = 5, n = 4;

    printf("Co-rank unit test (Fig 12.3 / Fig 12.8 example):\n");
    // C[4] comes from A[3]=9 → i=3, j=1 (book Fig 12.3A: k=4, i=3, j=1)
    int i = co_rank(4, A, m, B, n);
    printf("  co_rank(k=4):  i=%d  j=%d  (expected i=3, j=1) %s\n",
           i, 4 - i, (i == 3) ? "PASS" : "FAIL");
    // C[6] comes from B[1]=10 → k=6, i=4, j=2 (book Fig 12.3B: k=6, i=4, j=2... wait)
    // Actually book says k=6, i=4, j=1 for case B (C[6] from B[1])
    i = co_rank(6, A, m, B, n);
    printf("  co_rank(k=6):  i=%d  j=%d  (expected i=4, j=2) %s\n",
           i, 6 - i, (i == 4) ? "PASS" : "FAIL");
    // k=9 (end): all A and B exhausted
    i = co_rank(9, A, m, B, n);
    printf("  co_rank(k=9):  i=%d  j=%d  (expected i=5, j=4) %s\n\n",
           i, 9 - i, (i == 5) ? "PASS" : "FAIL");
}

int main(void) {
    test_corank_host();

    // ── Small example: Fig 12.1 ────────────────────────────────────────────────
    printf("=== Basic parallel merge kernel (Fig 12.9) ===\n\n");

    {
        int A_h[] = {1, 7, 8, 9, 10};
        int B_h[] = {7, 10, 10, 12};
        int m = 5, n = 4;
        int C_exp[] = {1, 7, 7, 8, 9, 10, 10, 10, 12};
        int C_h[9];

        int *A_d, *B_d, *C_d;
        cudaMalloc(&A_d, m * sizeof(int));
        cudaMalloc(&B_d, n * sizeof(int));
        cudaMalloc(&C_d, (m + n) * sizeof(int));
        cudaMemcpy(A_d, A_h, m * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h, n * sizeof(int), cudaMemcpyHostToDevice);

        merge_basic_kernel<<<1, 9>>>(A_d, m, B_d, n, C_d);
        cudaDeviceSynchronize();
        cudaMemcpy(C_h, C_d, (m + n) * sizeof(int), cudaMemcpyDeviceToHost);

        printf("A: [1 7 8 9 10]  B: [7 10 10 12]\n");
        printf("C: [");
        for (int i = 0; i < m + n; i++) printf("%d%s", C_h[i], i < m+n-1 ? " " : "");
        printf("]\n");
        printf("Fig 12.1 result: %s\n\n", verify(C_exp, C_h, m + n) ? "PASS" : "FAIL");

        cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    }

    // ── Large random test ──────────────────────────────────────────────────────
    {
        int M = 100000, N = 80000;
        int total = M + N;
        int *A_h = (int *)malloc(M * sizeof(int));
        int *B_h = (int *)malloc(N * sizeof(int));
        int *C_h = (int *)malloc(total * sizeof(int));
        int *ref  = (int *)malloc(total * sizeof(int));

        srand(42);
        A_h[0] = rand() % 3;
        for (int i = 1; i < M; i++) A_h[i] = A_h[i-1] + rand() % 5;
        B_h[0] = rand() % 3;
        for (int i = 1; i < N; i++) B_h[i] = B_h[i-1] + rand() % 5;
        cpu_merge(A_h, M, B_h, N, ref);

        int *A_d, *B_d, *C_d;
        cudaMalloc(&A_d, M * sizeof(int));
        cudaMalloc(&B_d, N * sizeof(int));
        cudaMalloc(&C_d, total * sizeof(int));
        cudaMemcpy(A_d, A_h, M * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h, N * sizeof(int), cudaMemcpyHostToDevice);

        int blockDim = 256;
        int gridDim  = (total + blockDim - 1) / blockDim;
        merge_basic_kernel<<<gridDim, blockDim>>>(A_d, M, B_d, N, C_d);
        cudaDeviceSynchronize();
        cudaMemcpy(C_h, C_d, total * sizeof(int), cudaMemcpyDeviceToHost);

        printf("Large merge: M=%d, N=%d, total=%d\n", M, N, total);
        printf("Result: %s\n", verify(ref, C_h, total) ? "PASS" : "FAIL");

        free(A_h); free(B_h); free(C_h); free(ref);
        cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    }

    return 0;
}
