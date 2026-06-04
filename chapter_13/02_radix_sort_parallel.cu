// §13.3  Parallel radix sort — one thread per key, Figure 13.4
//
// Each thread is responsible for one key.  Per bit-iteration:
//   1. Each thread extracts the current bit of its key.
//   2. A grid-wide exclusive prefix sum (scan) on the bits array gives each
//      thread the number of 1-bits before its position (ones_before).
//   3. Each thread computes its output destination:
//        dest(bit=0) = key_index − ones_before           (§13.3)
//        dest(bit=1) = N − ones_total + ones_before
//      and scatters its key there.
//
// The book (Fig 13.4) folds the scan call into the kernel.  In practice,
// a single-kernel grid-wide scan requires cooperative groups; here we use
// three separate kernel launches per iteration instead:
//   launch 1  extract_bits_kernel         → bits[N]
//   launch 2  hierarchical exclusive scan → ones_prefix[N]
//   launch 3  scatter_kernel              → output[N]
//
// Exclusive scan implementation (two-level Kogge-Stone):
//   phase 1  block_scan_kernel       — local exclusive scan per block + block sums
//   phase 2  scan_block_sums_kernel  — exclusive scan of the block sums (single block)
//   phase 3  add_block_offsets_kernel— add each block's prefix to its elements
// This correctly handles N ≤ BLOCK_SIZE² (up to ~262 K with BLOCK_SIZE=512).
//
// Limitation noted in §13.4: the scatter writes to global memory are
// not coalesced because consecutive threads may map to non-adjacent output
// indices.  The next file (03_radix_sort_coalesced.cu) addresses this.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512   // threads per block for sort and scan kernels

// ── Kernel 1: extract bit `iter` of each input key ───────────────────────────
__global__ void extract_bits_kernel(const unsigned int *input,
                                    unsigned int *bits, int N, int iter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) bits[i] = (input[i] >> iter) & 1u;
}

// ── Hierarchical two-level exclusive scan ─────────────────────────────────────

// Phase 1: Kogge-Stone inclusive scan within each block; write exclusive
//          result back to data[] and the inclusive block total to block_sums[].
__global__ void block_scan_kernel(unsigned int *data, unsigned int *block_sums,
                                  int N) {
    extern __shared__ unsigned int s[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    s[tid] = (i < N) ? data[i] : 0u;
    __syncthreads();

    for (int stride = 1; stride < (int)blockDim.x; stride <<= 1) {
        unsigned int v = (tid >= stride) ? s[tid - stride] : 0u;
        __syncthreads();
        s[tid] += v;
        __syncthreads();
    }
    unsigned int inclusive_val = s[tid];
    if (i < N) data[i] = (tid > 0) ? s[tid - 1] : 0u;
    // Block sum = inclusive value of the last thread in this block
    if (tid == blockDim.x - 1) block_sums[blockIdx.x] = inclusive_val;
}

// Phase 2: exclusive scan of block_sums[] with a single block.
//          Requires num_blocks ≤ blockDim.x.
__global__ void scan_block_sums_kernel(unsigned int *block_sums, int num_blocks) {
    extern __shared__ unsigned int s[];
    int tid = threadIdx.x;
    s[tid] = (tid < num_blocks) ? block_sums[tid] : 0u;
    __syncthreads();
    for (int stride = 1; stride < (int)blockDim.x; stride <<= 1) {
        unsigned int v = (tid >= stride) ? s[tid - stride] : 0u;
        __syncthreads();
        s[tid] += v;
        __syncthreads();
    }
    if (tid < num_blocks) block_sums[tid] = (tid > 0) ? s[tid - 1] : 0u;
}

// Phase 3: add each block's accumulated prefix to its locally scanned elements.
__global__ void add_block_offsets_kernel(unsigned int *data,
                                         const unsigned int *block_sums, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] += block_sums[blockIdx.x];
}

// Full in-place exclusive scan of d_data[N].
// Requires num_blocks ≤ BLOCK_SIZE (i.e., N ≤ BLOCK_SIZE²).
static void exclusive_scan(unsigned int *d_data, int N) {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int *d_bsums;
    cudaMalloc(&d_bsums, num_blocks * sizeof(unsigned int));

    block_scan_kernel<<<num_blocks, BLOCK_SIZE,
                        BLOCK_SIZE * sizeof(unsigned int)>>>(d_data, d_bsums, N);

    if (num_blocks > 1) {
        // Round up to next power-of-2 so KS covers all block sums
        int sb = 1;
        while (sb < num_blocks) sb <<= 1;
        scan_block_sums_kernel<<<1, sb, sb * sizeof(unsigned int)>>>(d_bsums, num_blocks);
        add_block_offsets_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, d_bsums, N);
    }
    cudaFree(d_bsums);
}

// ── Kernel 3: scatter keys to their computed output positions ─────────────────
// §13.3: dest(bit=0) = i - ones_before
//        dest(bit=1) = N - ones_total + ones_before
__global__ void scatter_kernel(const unsigned int *input, unsigned int *output,
                               const unsigned int *bits,
                               const unsigned int *ones_prefix,
                               unsigned int ones_total, int N, int iter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    unsigned int key          = input[i];
    unsigned int bit          = bits[i];
    unsigned int ones_before  = ones_prefix[i];
    unsigned int dst = (bit == 0) ? (unsigned int)(i - ones_before)
                                  : (N - ones_total + ones_before);
    output[dst] = key;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
static int is_sorted_u(const unsigned int *a, int n) {
    for (int i = 1; i < n; i++) if (a[i] < a[i-1]) return 0;
    return 1;
}
static int arrays_equal_u(const unsigned int *a, const unsigned int *b, int n) {
    for (int i = 0; i < n; i++) if (a[i] != b[i]) return 0;
    return 1;
}
static int cmp_uint(const void *a, const void *b) {
    unsigned int x = *(unsigned int *)a, y = *(unsigned int *)b;
    return (x > y) - (x < y);
}

// Run all 32 radix iterations on d_in; result ends up in d_in (pointer is
// updated by the ping-pong swap).
static void gpu_radix_sort(unsigned int **d_in, unsigned int *d_out,
                           unsigned int *d_bits, unsigned int *d_ones_prefix,
                           int N) {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t bytes   = N * sizeof(unsigned int);

    for (int iter = 0; iter < 32; iter++) {
        // Step 1: extract bits
        extract_bits_kernel<<<num_blocks, BLOCK_SIZE>>>(*d_in, d_bits, N, iter);

        // Step 2: exclusive scan on a copy of bits
        cudaMemcpy(d_ones_prefix, d_bits, bytes, cudaMemcpyDeviceToDevice);
        exclusive_scan(d_ones_prefix, N);

        // Compute ones_total = ones_prefix[N-1] + bits[N-1]
        unsigned int last_prefix, last_bit;
        cudaMemcpy(&last_prefix, d_ones_prefix + (N-1), sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_bit,    d_bits        + (N-1), sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
        unsigned int ones_total = last_prefix + last_bit;

        // Step 3: scatter
        scatter_kernel<<<num_blocks, BLOCK_SIZE>>>(*d_in, d_out, d_bits,
                                                    d_ones_prefix, ones_total,
                                                    N, iter);
        cudaDeviceSynchronize();

        // Ping-pong buffers
        unsigned int *tmp = *d_in; *d_in = d_out; d_out = tmp;
    }
}

int main(void) {
    printf("=== Parallel Radix Sort — One Thread per Key (§13.3, Fig 13.4) ===\n\n");

    // ── Small test ────────────────────────────────────────────────────────────
    {
        unsigned int h[] = {0xC, 0x3, 0x6, 0x9, 0xF, 0x8, 0x5, 0xA,
                            0xA, 0x6, 0xB, 0xD, 0x4, 0xA, 0x7, 0x0};
        int N = 16;
        size_t bytes = N * sizeof(unsigned int);

        unsigned int *d_in, *d_out, *d_bits, *d_ones_prefix;
        cudaMalloc(&d_in,          bytes);
        cudaMalloc(&d_out,         bytes);
        cudaMalloc(&d_bits,        bytes);
        cudaMalloc(&d_ones_prefix, bytes);
        cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice);

        gpu_radix_sort(&d_in, d_out, d_bits, d_ones_prefix, N);

        unsigned int result[16];
        cudaMemcpy(result, d_in, bytes, cudaMemcpyDeviceToHost);

        printf("Input:  ");
        for (int i = 0; i < N; i++) printf("%X ", h[i]);
        printf("\nSorted: ");
        for (int i = 0; i < N; i++) printf("%X ", result[i]);
        printf("\nSmall test: %s\n\n", is_sorted_u(result, N) ? "PASS" : "FAIL");

        cudaFree(d_in); cudaFree(d_out); cudaFree(d_bits); cudaFree(d_ones_prefix);
    }

    // ── Large test ────────────────────────────────────────────────────────────
    {
        int N = 1 << 18;  // 256 K  (≤ BLOCK_SIZE² = 262 144)
        size_t bytes = N * sizeof(unsigned int);

        unsigned int *h_keys = (unsigned int *)malloc(bytes);
        unsigned int *h_ref  = (unsigned int *)malloc(bytes);
        srand(42);
        for (int i = 0; i < N; i++) h_keys[i] = h_ref[i] = (unsigned int)rand();
        qsort(h_ref, N, sizeof(unsigned int), cmp_uint);

        unsigned int *d_in, *d_out, *d_bits, *d_ones_prefix;
        cudaMalloc(&d_in,          bytes);
        cudaMalloc(&d_out,         bytes);
        cudaMalloc(&d_bits,        bytes);
        cudaMalloc(&d_ones_prefix, bytes);
        cudaMemcpy(d_in, h_keys, bytes, cudaMemcpyHostToDevice);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);

        gpu_radix_sort(&d_in, d_out, d_bits, d_ones_prefix, N);

        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);

        cudaMemcpy(h_keys, d_in, bytes, cudaMemcpyDeviceToHost);
        int ok = arrays_equal_u(h_keys, h_ref, N);
        printf("Large test: N=%d  GPU time=%.2f ms  %s\n\n", N, ms,
               ok ? "PASS" : "FAIL");

        printf("Per-iteration breakdown (§13.3, Fig 13.4):\n");
        printf("  1. extract_bits_kernel  : 1 launch, 1 thread/key\n");
        printf("  2. exclusive_scan       : 3 launches, 2-level Kogge-Stone\n");
        printf("  3. scatter_kernel       : 1 launch, 1 thread/key\n");
        printf("  Total: 32 iterations × 5 launches = 160 kernel launches\n\n");
        printf("Issue (§13.4): scatter writes are non-coalesced because\n");
        printf("  consecutive threads map to non-adjacent output addresses.\n");
        printf("  => see 03_radix_sort_coalesced.cu\n");

        free(h_keys); free(h_ref);
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_bits); cudaFree(d_ones_prefix);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    return 0;
}
