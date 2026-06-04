// §13.7  Parallel merge sort — Figure 13.11
//
// Merge sort divides the input into segments, sorts each independently, then
// repeatedly merges adjacent sorted pairs until all data is in one sorted run.
//
// At each stage the independent merge operations can run in parallel, and
// within each merge operation multiple thread blocks can collaborate (as in
// Chapter 12).  This exposes two levels of parallelism (Fig 13.11):
//   - Early stages: many independent merges → more blocks, each merge small
//   - Late stages:  fewer, larger merges   → fewer blocks, each merge large
//
// Two-phase implementation:
//
//   Phase 1  bitonic_sort_kernel
//     Each block sorts one SEG_SIZE-element segment in shared memory using
//     bitonic sort.  Requires SEG_SIZE to be a power of 2 and ≤ 1024.
//
//   Phase 2  iterative merge (host loop)
//     Each round doubles the sorted segment size.  For each adjacent pair of
//     sorted segments the tiled co-rank merge kernel from Chapter 12 is
//     launched with enough blocks to cover the merged output.
//     A ping-pong buffer avoids read/write aliasing.
//
// The co_rank and merge kernels are included here (adapted from ch12 for
// unsigned int) so this file is self-contained.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef SEG_SIZE
#define SEG_SIZE 256     // initial sorted segment size (power of 2, ≤ 1024)
#endif
#ifndef TILE_SIZE
#define TILE_SIZE 256    // tile size for the co-rank merge kernel
#endif
#define MERGE_BLOCK 128  // threads per block for the merge kernel

// ── Phase 1: bitonic sort of each SEG_SIZE segment ───────────────────────────
__global__ void bitonic_sort_kernel(unsigned int *data, int N) {
    extern __shared__ unsigned int s[];
    int tid       = threadIdx.x;
    int seg_start = blockIdx.x * SEG_SIZE;
    int global_i  = seg_start + tid;

    s[tid] = (global_i < N) ? data[global_i] : ~0u;
    __syncthreads();

    for (int k = 2; k <= SEG_SIZE; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                bool ascending = ((tid & k) == 0);
                if ((ascending  && s[tid] > s[ixj]) ||
                    (!ascending && s[tid] < s[ixj])) {
                    unsigned int tmp = s[tid]; s[tid] = s[ixj]; s[ixj] = tmp;
                }
            }
            __syncthreads();
        }
    }
    if (global_i < N) data[global_i] = s[tid];
}

// ── Phase 2: tiled co-rank merge (adapted from §12.6) ────────────────────────

__device__ __host__ int co_rank_u(int k,
                                   const unsigned int *A, int m,
                                   const unsigned int *B, int n) {
    int i     = (k < m) ? k : m;
    int j     = k - i;
    int i_low = (0 > k - n) ? 0 : k - n;
    int j_low = (0 > k - m) ? 0 : k - m;
    bool active = true;
    while (active) {
        if (i > 0 && j < n && A[i-1] > B[j]) {
            int delta = (i - i_low + 1) >> 1;
            j_low = j;  j += delta;  i -= delta;
        } else if (j > 0 && i < m && B[j-1] >= A[i]) {
            int delta = (j - j_low + 1) >> 1;
            i_low = i;  i += delta;  j -= delta;
        } else {
            active = false;
        }
    }
    return i;
}

__device__ __host__ void merge_seq_u(const unsigned int *A, int m,
                                      const unsigned int *B, int n,
                                      unsigned int *C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) C[k++] = (A[i] <= B[j]) ? A[i++] : B[j++];
    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}

__global__ void merge_tiled_kernel_u(const unsigned int *A, int m,
                                      const unsigned int *B, int n,
                                      unsigned int *C, int tile_size) {
    extern __shared__ unsigned int shareAB[];
    unsigned int *A_S = shareAB;
    unsigned int *B_S = shareAB + tile_size;

    int C_curr = blockIdx.x * (int)ceilf((float)(m + n) / gridDim.x);
    int C_next = min((blockIdx.x + 1) * (int)ceilf((float)(m + n) / gridDim.x),
                     m + n);

    if (threadIdx.x == 0) {
        A_S[0] = co_rank_u(C_curr, A, m, B, n);
        A_S[1] = co_rank_u(C_next, A, m, B, n);
    }
    __syncthreads();

    int A_curr = A_S[0], A_next = A_S[1];
    int B_curr = C_curr - A_curr, B_next = C_next - A_next;
    __syncthreads();

    int C_length   = C_next - C_curr;
    int A_length   = A_next - A_curr;
    int B_length   = B_next - B_curr;
    int total_iter = (int)ceilf((float)C_length / tile_size);
    int C_done = 0, A_done = 0, B_done = 0;

    for (int counter = 0; counter < total_iter; counter++) {
        for (int x = 0; x < tile_size; x += blockDim.x)
            if (x + threadIdx.x < A_length - A_done)
                A_S[x + threadIdx.x] = A[A_curr + A_done + x + threadIdx.x];
        for (int x = 0; x < tile_size; x += blockDim.x)
            if (x + threadIdx.x < B_length - B_done)
                B_S[x + threadIdx.x] = B[B_curr + B_done + x + threadIdx.x];
        __syncthreads();

        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
        c_curr = min(c_curr, C_length - C_done);
        c_next = min(c_next, C_length - C_done);

        int a_avail = min(tile_size, A_length - A_done);
        int b_avail = min(tile_size, B_length - B_done);
        int a_curr = co_rank_u(c_curr, A_S, a_avail, B_S, b_avail);
        int a_next = co_rank_u(c_next, A_S, a_avail, B_S, b_avail);
        int b_curr = c_curr - a_curr, b_next = c_next - a_next;

        merge_seq_u(A_S + a_curr, a_next - a_curr,
                    B_S + b_curr, b_next - b_curr,
                    C + C_curr + C_done + c_curr);

        C_done += tile_size;
        A_done += co_rank_u(tile_size, A_S, a_avail, B_S, b_avail);
        B_done  = C_done - A_done;
        __syncthreads();
    }
}

// ── Host helpers ──────────────────────────────────────────────────────────────
static int cmp_uint(const void *a, const void *b) {
    unsigned int x = *(unsigned int *)a, y = *(unsigned int *)b;
    return (x > y) - (x < y);
}
static int arrays_eq_u(const unsigned int *a, const unsigned int *b, int n) {
    for (int i = 0; i < n; i++) if (a[i] != b[i]) return 0;
    return 1;
}

// Merge all adjacent pairs at the current segment_size level.
// Reads from d_src, writes to d_dst, N total elements.
static void merge_round(const unsigned int *d_src, unsigned int *d_dst,
                         int N, int seg_size) {
    int ts = TILE_SIZE;
    for (int base = 0; base < N; base += 2 * seg_size) {
        int left_start  = base;
        int left_size   = min(seg_size, N - left_start);
        int right_start = left_start + left_size;
        int right_size  = min(seg_size, N - right_start);
        if (right_size <= 0) {
            // Odd segment: copy unchanged
            cudaMemcpy(d_dst + left_start, d_src + left_start,
                       left_size * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
            continue;
        }
        int merged_len = left_size + right_size;
        int blocks     = max(1, (merged_len + ts - 1) / ts);
        merge_tiled_kernel_u<<<blocks, MERGE_BLOCK, 2 * ts * sizeof(unsigned int)>>>(
            d_src + left_start,  left_size,
            d_src + right_start, right_size,
            d_dst + left_start,  ts);
    }
    cudaDeviceSynchronize();
}

int main(void) {
    printf("=== Parallel Merge Sort (§13.7, Fig 13.11) ===\n");
    printf("    SEG_SIZE=%d, TILE_SIZE=%d, merge block=%d threads\n\n",
           SEG_SIZE, TILE_SIZE, MERGE_BLOCK);

    // ── Small test ────────────────────────────────────────────────────────────
    {
        unsigned int h[] = {9,3,7,1,8,2,6,4,5,0,11,10,15,13,14,12};
        int N = 16;

        // Pad to multiple of SEG_SIZE for Phase 1
        int padded = ((N + SEG_SIZE - 1) / SEG_SIZE) * SEG_SIZE;
        unsigned int *h_pad = (unsigned int *)malloc(padded * sizeof(unsigned int));
        memcpy(h_pad, h, N * sizeof(unsigned int));
        for (int i = N; i < padded; i++) h_pad[i] = ~0u;

        unsigned int *d_a, *d_b;
        cudaMalloc(&d_a, padded * sizeof(unsigned int));
        cudaMalloc(&d_b, padded * sizeof(unsigned int));
        cudaMemcpy(d_a, h_pad, padded * sizeof(unsigned int), cudaMemcpyHostToDevice);

        int num_segs = padded / SEG_SIZE;
        bitonic_sort_kernel<<<num_segs, SEG_SIZE, SEG_SIZE * sizeof(unsigned int)>>>(
            d_a, padded);
        cudaDeviceSynchronize();

        // Iterative merge
        unsigned int *src = d_a, *dst = d_b;
        for (int seg = SEG_SIZE; seg < padded; seg <<= 1) {
            merge_round(src, dst, padded, seg);
            unsigned int *t = src; src = dst; dst = t;
        }

        cudaMemcpy(h_pad, src, padded * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);

        printf("Input:  ");
        for (int i = 0; i < N; i++) printf("%u ", h[i]);
        printf("\nSorted: ");
        for (int i = 0; i < N; i++) printf("%u ", h_pad[i]);
        unsigned int ref[16]; memcpy(ref, h, N * sizeof(unsigned int));
        qsort(ref, N, sizeof(unsigned int), cmp_uint);
        printf("\nSmall test: %s\n\n",
               arrays_eq_u(h_pad, ref, N) ? "PASS" : "FAIL");

        free(h_pad);
        cudaFree(d_a); cudaFree(d_b);
    }

    // ── Large test ────────────────────────────────────────────────────────────
    {
        int N = 1 << 20;  // 1 M
        int padded = ((N + SEG_SIZE - 1) / SEG_SIZE) * SEG_SIZE;
        size_t nbytes = N * sizeof(unsigned int);
        size_t pbytes = padded * sizeof(unsigned int);

        unsigned int *h_keys = (unsigned int *)malloc(nbytes);
        unsigned int *h_ref  = (unsigned int *)malloc(nbytes);
        unsigned int *h_pad  = (unsigned int *)malloc(pbytes);
        srand(42);
        for (int i = 0; i < N; i++) h_keys[i] = h_ref[i] = (unsigned int)rand();
        qsort(h_ref, N, sizeof(unsigned int), cmp_uint);
        memcpy(h_pad, h_keys, nbytes);
        for (int i = N; i < padded; i++) h_pad[i] = ~0u;

        unsigned int *d_a, *d_b;
        cudaMalloc(&d_a, pbytes); cudaMalloc(&d_b, pbytes);
        cudaMemcpy(d_a, h_pad, pbytes, cudaMemcpyHostToDevice);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);

        // Phase 1: sort segments
        int num_segs = padded / SEG_SIZE;
        bitonic_sort_kernel<<<num_segs, SEG_SIZE, SEG_SIZE * sizeof(unsigned int)>>>(
            d_a, padded);
        cudaDeviceSynchronize();

        // Phase 2: iterative merge (Fig 13.11)
        unsigned int *src = d_a, *dst = d_b;
        int rounds = 0;
        for (int seg = SEG_SIZE; seg < padded; seg <<= 1, rounds++) {
            merge_round(src, dst, padded, seg);
            unsigned int *t = src; src = dst; dst = t;
        }

        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);

        cudaMemcpy(h_pad, src, pbytes, cudaMemcpyDeviceToHost);
        int ok = arrays_eq_u(h_pad, h_ref, N);
        printf("Large test: N=1M  GPU time=%.2f ms  %s\n\n", ms,
               ok ? "PASS" : "FAIL");

        printf("Merge sort parallelism (§13.7, Fig 13.11):\n");
        printf("  Phase 1: %d bitonic sort blocks (%d keys/block, fully parallel)\n",
               num_segs, SEG_SIZE);
        printf("  Phase 2: %d merge rounds (segment size doubles each round)\n", rounds);
        printf("    Round 1: %d independent merges, each %d+%d keys\n",
               padded / (2*SEG_SIZE), SEG_SIZE, SEG_SIZE);
        printf("    Round %d: 1 merge of %d keys\n", rounds, padded);
        printf("  Within each merge: co-rank tiled kernel (§12.6) assigns\n");
        printf("    one output tile per block, all blocks run in parallel.\n");

        free(h_keys); free(h_ref); free(h_pad);
        cudaFree(d_a); cudaFree(d_b);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    return 0;
}
