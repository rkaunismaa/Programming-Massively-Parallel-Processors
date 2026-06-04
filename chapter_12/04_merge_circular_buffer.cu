// §12.7 A circular buffer merge kernel — Figures 12.16, 12.18–12.20
//
// Problem with tiled kernel (§12.6):
//   Each iteration reloads the entire A and B tiles, even though some elements
//   from the previous tile were not consumed. Only ~half the loaded data is
//   used → 50% bandwidth wasted.
//
// Circular buffer solution (§12.7):
//   Track A_S_start and B_S_start pointers into the shared memory arrays.
//   After each iteration, only refill the consumed portion of each tile.
//   Remaining (unconsumed) elements stay in place; new elements wrap around.
//   Update: A_S_start = (A_S_start + A_S_consumed) % tile_size
//
// Simplified model (Fig 12.17 / §12.7):
//   co_rank_circular and merge_sequential_circular use virtual 0-based offsets
//   and apply (start + offset) % tile_size internally. The binary search logic
//   is identical to the flat co_rank — only the index computation differs.
//
// Thread coarsening (§12.8):
//   Each thread handles tile_size/blockDim.x output elements per iteration,
//   amortizing one O(log N) binary search across multiple outputs. Without
//   coarsening every output would need its own binary search.

#ifndef TILE_SIZE
#define TILE_SIZE 512
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ── Figure 12.19: co_rank for circular buffers ────────────────────────────────
// Uses virtual 0-based offsets into A_S / B_S; wraps via modulo internally.
__device__ int co_rank_circular(int k, int *A_S, int m, int *B_S, int n,
                                 int A_S_start, int B_S_start, int tile_size) {
    int i     = (k < m) ? k : m;
    int j     = k - i;
    int i_low = (0 > k - n) ? 0 : k - n;
    int j_low = (0 > k - m) ? 0 : k - m;
    bool active = true;

    while (active) {
        int i_cir     = (A_S_start + i)     % tile_size;
        int i_m_1_cir = (A_S_start + i - 1 + tile_size) % tile_size;
        int j_cir     = (B_S_start + j)     % tile_size;
        int j_m_1_cir = (B_S_start + j - 1 + tile_size) % tile_size;

        if (i > 0 && j < n && A_S[i_m_1_cir] > B_S[j_cir]) {
            int delta = ((i - i_low + 1) >> 1);
            j_low = j;  j = j + delta;  i = i - delta;
        } else if (j > 0 && i < m && B_S[j_m_1_cir] >= A_S[i_cir]) {
            int delta = ((j - j_low + 1) >> 1);
            i_low = i;  i = i + delta;  j = j - delta;
        } else {
            active = false;
        }
    }
    return i;
}

// ── Figure 12.20: merge_sequential for circular buffers ───────────────────────
// A_S and B_S are circular; A_S_start / B_S_start mark the logical element 0.
// Output C is linear global memory (not circular).
__device__ void merge_sequential_circular(int *A_S, int m, int *B_S, int n,
                                           int *C, int A_S_start, int B_S_start,
                                           int tile_size) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        int ia = (A_S_start + i) % tile_size;
        int jb = (B_S_start + j) % tile_size;
        if (A_S[ia] <= B_S[jb]) { C[k++] = A_S[ia]; i++; }
        else                     { C[k++] = B_S[jb]; j++; }
    }
    if (i == m)
        for (; j < n; j++) C[k++] = B_S[(B_S_start + j) % tile_size];
    else
        for (; i < m; i++) C[k++] = A_S[(A_S_start + i) % tile_size];
}

// ── Helper: plain co_rank on global memory (for block-level setup) ────────────
__device__ int co_rank_global(int k, int *A, int m, int *B, int n) {
    int i     = (k < m) ? k : m;
    int j     = k - i;
    int i_low = (0 > k - n) ? 0 : k - n;
    int j_low = (0 > k - m) ? 0 : k - m;
    bool active = true;
    while (active) {
        if (i > 0 && j < n && A[i-1] > B[j]) {
            int d = ((i - i_low + 1) >> 1);
            j_low = j;  j += d;  i -= d;
        } else if (j > 0 && i < m && B[j-1] >= A[i]) {
            int d = ((j - j_low + 1) >> 1);
            i_low = i;  i += d;  j -= d;
        } else { active = false; }
    }
    return i;
}

// ── Circular buffer merge kernel (Figs 12.11/12.16/12.18) ────────────────────
__global__ void merge_circular_buffer_kernel(int *A, int m, int *B, int n,
                                              int *C, int tile_size) {
    extern __shared__ int shareAB[];
    int *A_S = &shareAB[0];
    int *B_S = &shareAB[tile_size];

    // ── Part 1: block-level subarrays (same as Fig 12.11) ────────────────────
    int C_curr = blockIdx.x * (int)ceilf((float)(m + n) / gridDim.x);
    int C_next = min((blockIdx.x + 1) * (int)ceilf((float)(m + n) / gridDim.x),
                     m + n);

    if (threadIdx.x == 0) {
        A_S[0] = co_rank_global(C_curr, A, m, B, n);
        A_S[1] = co_rank_global(C_next, A, m, B, n);
    }
    __syncthreads();

    int A_curr   = A_S[0];
    int A_next   = A_S[1];
    int B_curr   = C_curr - A_curr;
    int B_next   = C_next - A_next;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int C_length = C_next - C_curr;
    __syncthreads();

    // ── Part 2 (Fig 12.16): circular-buffer iterative tile loading ────────────
    int A_S_start    = 0;
    int B_S_start    = 0;
    int A_S_consumed = tile_size;   // first iteration: fill entire tile
    int B_S_consumed = tile_size;

    int counter     = 0;
    int C_completed = 0;
    int A_consumed  = 0;   // A elements output to C so far (tracks consumption)
    int B_consumed  = 0;   // B elements output to C so far
    int A_loaded    = 0;   // A elements loaded into shared memory so far
    int B_loaded    = 0;   // B elements loaded into shared memory so far
    int total_iter  = (int)ceilf((float)C_length / tile_size);

    while (counter < total_iter) {
        // Refill A_S_consumed fresh A elements (replace the consumed slots).
        // Source starts at A[A_curr + A_loaded], NOT A_consumed — these differ
        // because A_loaded = tile_size + prior_consumed while A_consumed = prior_consumed.
        for (int i = 0; i < A_S_consumed; i += blockDim.x) {
            if (i + threadIdx.x < A_S_consumed &&
                A_loaded + i + threadIdx.x < A_length) {
                int dst = (A_S_start + (tile_size - A_S_consumed) + i + threadIdx.x)
                          % tile_size;
                A_S[dst] = A[A_curr + A_loaded + i + threadIdx.x];
            }
        }
        A_loaded += A_S_consumed;   // advance the "next to load" pointer

        // Refill B_S_consumed fresh B elements (coalesced)
        for (int i = 0; i < B_S_consumed; i += blockDim.x) {
            if (i + threadIdx.x < B_S_consumed &&
                B_loaded + i + threadIdx.x < B_length) {
                int dst = (B_S_start + (tile_size - B_S_consumed) + i + threadIdx.x)
                          % tile_size;
                B_S[dst] = B[B_curr + B_loaded + i + threadIdx.x];
            }
        }
        B_loaded += B_S_consumed;
        __syncthreads();

        // ── Part 3 (Fig 12.18): thread-level merge using circular co_rank ─────
        int c_curr_t = threadIdx.x * (tile_size / blockDim.x);
        int c_next_t = (threadIdx.x + 1) * (tile_size / blockDim.x);
        c_curr_t = min(c_curr_t, C_length - C_completed);
        c_next_t = min(c_next_t, C_length - C_completed);

        int tile_A = min(tile_size, A_length - A_consumed);
        int tile_B = min(tile_size, B_length - B_consumed);

        int a_curr_s = co_rank_circular(c_curr_t, A_S, tile_A, B_S, tile_B,
                                        A_S_start, B_S_start, tile_size);
        int a_next_s = co_rank_circular(c_next_t, A_S, tile_A, B_S, tile_B,
                                        A_S_start, B_S_start, tile_size);
        int b_curr_s = c_curr_t - a_curr_s;
        int b_next_s = c_next_t - a_next_s;

        merge_sequential_circular(
            A_S, a_next_s - a_curr_s,
            B_S, b_next_s - b_curr_s,
            C + C_curr + C_completed + c_curr_t,
            (A_S_start + a_curr_s) % tile_size,
            (B_S_start + b_curr_s) % tile_size,
            tile_size);

        // ── End-of-iteration bookkeeping (Fig 12.18 lines 49-57) ─────────────
        // Compute actual elements produced this iteration BEFORE advancing counter
        int c_this_iter = min(tile_size, C_length - C_completed);

        // Count A elements consumed this iteration via co_rank on full tile output
        A_S_consumed = co_rank_circular(c_this_iter, A_S, tile_A, B_S, tile_B,
                                        A_S_start, B_S_start, tile_size);
        B_S_consumed = c_this_iter - A_S_consumed;

        A_consumed  += A_S_consumed;
        C_completed += c_this_iter;
        B_consumed   = C_completed - A_consumed;

        // Advance start pointers past the consumed portion
        A_S_start = (A_S_start + A_S_consumed) % tile_size;
        B_S_start = (B_S_start + B_S_consumed) % tile_size;

        counter++;
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
    printf("=== Circular Buffer Merge Kernel (§12.7, Figs 12.16/12.18-12.20) ===\n\n");

    // ── Small example ──────────────────────────────────────────────────────────
    {
        int A_h[] = {1, 7, 8, 9, 10};
        int B_h[] = {7, 10, 10, 12};
        int m = 5, n = 4;
        int C_exp[] = {1, 7, 7, 8, 9, 10, 10, 10, 12};
        int C_h[9] = {};

        int *A_d, *B_d, *C_d;
        cudaMalloc(&A_d, m * sizeof(int));
        cudaMalloc(&B_d, n * sizeof(int));
        cudaMalloc(&C_d, (m + n) * sizeof(int));
        cudaMemcpy(A_d, A_h, m * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h, n * sizeof(int), cudaMemcpyHostToDevice);

        int ts = 8;
        merge_circular_buffer_kernel<<<1, 4, 2 * ts * sizeof(int)>>>(
            A_d, m, B_d, n, C_d, ts);
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

        merge_circular_buffer_kernel<<<gridDim, blockDim, smem>>>(
            A_d, M, B_d, N, C_d, ts);
        cudaDeviceSynchronize();
        cudaMemcpy(C_h, C_d, total * sizeof(int), cudaMemcpyDeviceToHost);

        printf("Circular buffer merge: M=%d, N=%d, tile_size=%d\n", M, N, ts);
        printf("Result: %s\n\n", verify(ref, C_h, total) ? "PASS" : "FAIL");

        printf("Bandwidth improvement over tiled kernel (§12.7):\n");
        printf("  Tiled:           loads 2*tile_size per iter, uses ~tile_size → ~50%% util\n");
        printf("  Circular buffer: refills only consumed elements → ~100%% util\n");
        printf("  Trade-off:       more register usage + code complexity\n\n");
        printf("Thread coarsening (§12.8):\n");
        printf("  Each thread handles tile_size/blockDim = %d/%d = %d output elements\n",
               ts, blockDim, ts / blockDim);
        printf("  Binary search cost amortized across %d elements per thread\n",
               ts / blockDim);

        free(A_h); free(B_h); free(C_h); free(ref);
        cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    }

    return 0;
}
