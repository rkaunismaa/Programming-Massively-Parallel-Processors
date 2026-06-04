// §12.6 A tiled merge kernel to improve coalescing — Figures 12.11–12.13
//
// Problem with the basic kernel (§12.5):
//   - co_rank accesses A and B from global memory with irregular patterns → not coalesced
//   - merge_sequential reads A and B elements with non-adjacent thread access → not coalesced
//
// Tiled approach (§12.6):
//   1. Each block finds its block-level output subarray (C_curr..C_next).
//   2. One thread calls co_rank twice to find the block's A and B subarrays.
//   3. All threads load tile_size A elements + tile_size B elements into shared
//      memory A_S / B_S in a coalesced fashion (consecutive threads load
//      consecutive elements).
//   4. Each thread merges its portion of the tile using co_rank on shared memory.
//   5. Repeat until the block's entire output subarray is produced.
//
// Shared memory layout: A_S and B_S are the first and second halves of a
// single extern shared array of size 2*tile_size.
//
// Deficiency: only half the loaded data is used per iteration (the other half
// remains unused as the next tile overwrites it). §12.7 fixes this with a
// circular buffer.

#ifndef TILE_SIZE
#define TILE_SIZE 1024
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ── Co-rank on arbitrary memory (host or device) ──────────────────────────────
__device__ __host__ int co_rank(int k, int *A, int m, int *B, int n) {
    int i     = (k < m) ? k : m;
    int j     = k - i;
    int i_low = (0 > k - n) ? 0 : k - n;
    int j_low = (0 > k - m) ? 0 : k - m;
    bool active = true;
    while (active) {
        if (i > 0 && j < n && A[i-1] > B[j]) {
            int delta = ((i - i_low + 1) >> 1);
            j_low = j;  j = j + delta;  i = i - delta;
        } else if (j > 0 && i < m && B[j-1] >= A[i]) {
            int delta = ((j - j_low + 1) >> 1);
            i_low = i;  i = i + delta;  j = j - delta;
        } else {
            active = false;
        }
    }
    return i;
}

__device__ __host__ void merge_sequential(int *A, int m, int *B, int n, int *C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] <= B[j]) C[k++] = A[i++];
        else               C[k++] = B[j++];
    }
    while (j < n) C[k++] = B[j++];
    while (i < m) C[k++] = A[i++];
}

// ── Figure 12.11 / 12.12 / 12.13: tiled merge kernel ─────────────────────────
__global__ void merge_tiled_kernel(int *A, int m, int *B, int n, int *C,
                                   int tile_size) {
    extern __shared__ int shareAB[];
    int *A_S = &shareAB[0];           // first half: tile_size ints for A tile
    int *B_S = &shareAB[tile_size];   // second half: tile_size ints for B tile

    // ── Part 1 (Fig 12.11): block-level output and input subarrays ────────────
    int C_curr = blockIdx.x * (int)ceilf((float)(m + n) / gridDim.x);
    int C_next = min((blockIdx.x + 1) * (int)ceilf((float)(m + n) / gridDim.x),
                     m + n);

    // One thread computes block-level co-rank values into shared memory
    if (threadIdx.x == 0) {
        A_S[0] = co_rank(C_curr, A, m, B, n);  // A start for this block
        A_S[1] = co_rank(C_next, A, m, B, n);  // A start for next block
    }
    __syncthreads();

    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads();

    // ── Part 2 (Fig 12.12): iterative tiled merge ─────────────────────────────
    int counter       = 0;
    int C_length      = C_next - C_curr;
    int A_length      = A_next - A_curr;
    int B_length      = B_next - B_curr;
    int total_iter    = (int)ceilf((float)C_length / tile_size);
    int C_completed   = 0;
    int A_consumed    = 0;
    int B_consumed    = 0;

    while (counter < total_iter) {
        // Load tile_size A elements cooperatively (coalesced)
        for (int i = 0; i < tile_size; i += blockDim.x) {
            if (i + threadIdx.x < A_length - A_consumed) {
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }
        // Load tile_size B elements cooperatively (coalesced)
        for (int i = 0; i < tile_size; i += blockDim.x) {
            if (i + threadIdx.x < B_length - B_consumed) {
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        // ── Part 3 (Fig 12.13): each thread merges its output section ─────────
        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);

        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;

        // co-rank within shared-memory tile
        int a_curr = co_rank(c_curr, A_S, min(tile_size, A_length - A_consumed),
                             B_S, min(tile_size, B_length - B_consumed));
        int a_next = co_rank(c_next, A_S, min(tile_size, A_length - A_consumed),
                             B_S, min(tile_size, B_length - B_consumed));
        int b_curr = c_curr - a_curr;
        int b_next = c_next - a_next;

        // merge from shared memory into global C
        merge_sequential(A_S + a_curr, a_next - a_curr,
                         B_S + b_curr, b_next - b_curr,
                         C + C_curr + C_completed + c_curr);

        counter++;
        C_completed += tile_size;
        // Update A_consumed: how many A elements were used this iteration
        A_consumed += co_rank(tile_size, A_S,
                              min(tile_size, A_length - (A_consumed)),
                              B_S,
                              min(tile_size, B_length - (B_consumed)));
        B_consumed = C_completed - A_consumed;
        __syncthreads();
    }
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

int main(void) {
    printf("=== Tiled Merge Kernel (§12.6, Figs 12.11-12.13) ===\n\n");

    // ── Small example ──────────────────────────────────────────────────────────
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

        int ts = 8;  // small tile for test
        merge_tiled_kernel<<<1, 4, 2 * ts * sizeof(int)>>>(A_d, m, B_d, n, C_d, ts);
        cudaDeviceSynchronize();
        cudaMemcpy(C_h, C_d, (m + n) * sizeof(int), cudaMemcpyDeviceToHost);

        printf("A: [1 7 8 9 10]  B: [7 10 10 12]\n");
        printf("C: [");
        for (int i = 0; i < m + n; i++) printf("%d%s", C_h[i], i < m+n-1 ? " " : "");
        printf("]\n");
        printf("Small test: %s\n\n", verify(C_exp, C_h, m + n) ? "PASS" : "FAIL");

        cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    }

    // ── Large test ─────────────────────────────────────────────────────────────
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

        int blockDim = 128;
        int gridDim  = 128;
        int ts       = TILE_SIZE;
        size_t smem  = 2 * ts * sizeof(int);

        merge_tiled_kernel<<<gridDim, blockDim, smem>>>(A_d, M, B_d, N, C_d, ts);
        cudaDeviceSynchronize();
        cudaMemcpy(C_h, C_d, total * sizeof(int), cudaMemcpyDeviceToHost);

        printf("Tiled merge: M=%d, N=%d, tile_size=%d, blocks=%d, threads=%d\n",
               M, N, ts, gridDim, blockDim);
        printf("Result: %s\n\n", verify(ref, C_h, total) ? "PASS" : "FAIL");

        printf("Coalescing analysis (§12.6):\n");
        printf("  Global loads:  coalesced — threads load A_S[i..i+blockDim-1]\n");
        printf("  co_rank:       runs on shared memory → no uncoalesced global accesses\n");
        printf("  Deficiency:    only half of 2*tile_size loaded data is used per iter\n");
        printf("  → §12.7 circular buffer fixes this wasted bandwidth\n");

        free(A_h); free(B_h); free(C_h); free(ref);
        cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    }

    return 0;
}
