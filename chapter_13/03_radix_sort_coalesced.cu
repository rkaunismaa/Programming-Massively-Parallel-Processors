// §13.4  Radix sort with shared-memory local buckets — Figures 13.5–13.6
//
// Problem with §13.3: threads with consecutive indices scatter to
// non-consecutive output addresses (poor memory coalescing).
//
// Fix (§13.4, Fig 13.5): each block performs a LOCAL radix sort in shared
// memory, then writes its local 0-bucket and 1-bucket sequentially to global
// memory.  Consecutive threads within the same local bucket write to
// consecutive addresses → fully coalesced stores.
//
// Pipeline per bit-iteration (Fig 13.6):
//
//   Kernel A  local_sort_count
//     Each block:
//       1. Loads its SECTION_SIZE keys into shared memory.
//       2. Kogge-Stone exclusive scan on extracted bits → ones_before[tid].
//       3. Scatters each key to its local destination:
//            local_dst(0-bit) = tid - ones_before
//            local_dst(1-bit) = zeros_total + ones_before
//          so that sorted_s[] holds the block's 0-bucket first, then 1-bucket.
//       4. Writes locally sorted keys to local_sorted_d[] (global memory).
//       5. Stores per-block bucket counts:
//            bucket_counts[0*num_blocks + bx] = zeros_total
//            bucket_counts[1*num_blocks + bx] = ones_total
//
//   Host   exclusive_scan on bucket_counts[2*num_blocks]
//     After scan (Fig 13.6):
//       bucket_starts[0*num_blocks + bx] = global start of block bx's 0-bucket
//       bucket_starts[1*num_blocks + bx] = global start of block bx's 1-bucket
//
//   Kernel B  coalesced_scatter
//     Each block writes local_sorted_d[bx*SECTION .. bx*SECTION+SECTION-1]
//     to the correct global positions using its bucket_starts entry.
//     Threads within each local bucket write to consecutive global addresses
//     → coalesced (unlike §13.3).
//
// Note: N must be a multiple of SECTION_SIZE; main() pads with UINT_MAX
// (which sorts last) and trims the result.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define SECTION_SIZE 256   // keys per block = threads per block
#define SCAN_BLOCK   512   // threads for the global scan kernels

// ── Shared-memory layout for local_sort_count_kernel ─────────────────────────
// smem[0          .. S-1  ]  keys_s   — input keys for this block
// smem[S          .. 2S-1 ]  bits_s   — extracted bits (reused for KS scan)
// smem[2S         .. 3S-1 ]  sorted_s — locally sorted output
// smem[3S]                   ones_total scratch (1 uint)
// Total: 3*S + 1 uints  =  3*256*4 + 4  =  3076 bytes per block

#define SMEM_KEYS   0
#define SMEM_BITS   SECTION_SIZE
#define SMEM_SORTED (2*SECTION_SIZE)
#define SMEM_TOT    (3*SECTION_SIZE)

// ── Kernel A: local sort in shared memory + count bucket sizes ────────────────
__global__ void local_sort_count_kernel(const unsigned int *input,
                                         unsigned int *local_sorted,
                                         unsigned int *bucket_counts,
                                         int N, int iter) {
    extern __shared__ unsigned int smem[];
    int bx  = blockIdx.x;
    int tid = threadIdx.x;
    int i   = bx * SECTION_SIZE + tid;

    // Load keys; threads beyond N see UINT_MAX (sort to end, trimmed later)
    unsigned int key = (i < N) ? input[i] : ~0u;
    smem[SMEM_KEYS + tid] = key;
    smem[SMEM_BITS + tid] = (i < N) ? ((key >> iter) & 1u) : 1u;
    __syncthreads();

    // Kogge-Stone inclusive scan on bits
    for (int stride = 1; stride < SECTION_SIZE; stride <<= 1) {
        unsigned int v = (tid >= stride) ? smem[SMEM_BITS + tid - stride] : 0u;
        __syncthreads();
        smem[SMEM_BITS + tid] += v;
        __syncthreads();
    }
    // smem[SMEM_BITS + tid] now holds inclusive prefix (sum of bits[0..tid])

    // Broadcast ones_total via shared scratch
    if (tid == SECTION_SIZE - 1) smem[SMEM_TOT] = smem[SMEM_BITS + tid];
    __syncthreads();
    unsigned int ones_total  = smem[SMEM_TOT];
    unsigned int zeros_total = (unsigned int)SECTION_SIZE - ones_total;

    // Derive exclusive prefix and original bit from inclusive scan values
    unsigned int ones_before = (tid > 0) ? smem[SMEM_BITS + tid - 1] : 0u;
    unsigned int bit = smem[SMEM_BITS + tid] - ones_before; // = original bit

    // Local destination within the block's sorted array
    unsigned int local_dst = (bit == 0) ? (unsigned int)(tid - ones_before)
                                        : (zeros_total + ones_before);
    smem[SMEM_SORTED + local_dst] = smem[SMEM_KEYS + tid];
    __syncthreads();

    // Write locally sorted segment and bucket counts
    local_sorted[i] = smem[SMEM_SORTED + tid];
    if (tid == 0) {
        int nb = gridDim.x;
        bucket_counts[0 * nb + bx] = zeros_total;
        bucket_counts[1 * nb + bx] = ones_total;
    }
}

// ── Kernel B: coalesced scatter using global bucket start positions ───────────
// Each block writes its local 0-bucket then 1-bucket sequentially, so all
// threads in a local bucket write to consecutive global addresses.
__global__ void coalesced_scatter_kernel(const unsigned int *local_sorted,
                                          unsigned int *output,
                                          const unsigned int *bucket_counts,
                                          const unsigned int *bucket_starts,
                                          int N) {
    int bx   = blockIdx.x;
    int tid  = threadIdx.x;
    int base = bx * SECTION_SIZE;

    unsigned int zeros_count = bucket_counts [0 * gridDim.x + bx];
    unsigned int zero_start  = bucket_starts [0 * gridDim.x + bx];
    unsigned int one_start   = bucket_starts [1 * gridDim.x + bx];

    unsigned int key = local_sorted[base + tid];
    unsigned int dst;
    // Threads 0..zeros_count-1 belong to the 0-bucket (consecutive → coalesced)
    // Threads zeros_count..SECTION-1 belong to the 1-bucket (also consecutive)
    if ((unsigned int)tid < zeros_count)
        dst = zero_start + tid;
    else
        dst = one_start  + (tid - zeros_count);

    if (base + tid < N) output[dst] = key;
}

// ── Two-level exclusive scan (same as file 02) ───────────────────────────────
__global__ void bs_kernel(unsigned int *data, unsigned int *bsums, int N) {
    extern __shared__ unsigned int s[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    s[tid]  = (i < N) ? data[i] : 0u;
    __syncthreads();
    for (int st = 1; st < (int)blockDim.x; st <<= 1) {
        unsigned int v = (tid >= st) ? s[tid - st] : 0u;
        __syncthreads(); s[tid] += v; __syncthreads();
    }
    unsigned int inc = s[tid];
    if (i < N) data[i] = (tid > 0) ? s[tid - 1] : 0u;
    if (tid == blockDim.x - 1) bsums[blockIdx.x] = inc;
}
__global__ void sbs_kernel(unsigned int *bsums, int nb) {
    extern __shared__ unsigned int s[];
    int tid = threadIdx.x;
    s[tid]  = (tid < nb) ? bsums[tid] : 0u;
    __syncthreads();
    for (int st = 1; st < (int)blockDim.x; st <<= 1) {
        unsigned int v = (tid >= st) ? s[tid - st] : 0u;
        __syncthreads(); s[tid] += v; __syncthreads();
    }
    if (tid < nb) bsums[tid] = (tid > 0) ? s[tid - 1] : 0u;
}
__global__ void abo_kernel(unsigned int *data, const unsigned int *bsums, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] += bsums[blockIdx.x];
}
static void exclusive_scan(unsigned int *d_data, int N) {
    int nb = (N + SCAN_BLOCK - 1) / SCAN_BLOCK;
    unsigned int *d_bs; cudaMalloc(&d_bs, nb * sizeof(unsigned int));
    bs_kernel<<<nb, SCAN_BLOCK, SCAN_BLOCK * sizeof(unsigned int)>>>(d_data, d_bs, N);
    if (nb > 1) {
        int sb = 1; while (sb < nb) sb <<= 1;
        sbs_kernel<<<1, sb, sb * sizeof(unsigned int)>>>(d_bs, nb);
        abo_kernel<<<nb, SCAN_BLOCK>>>(d_data, d_bs, N);
    }
    cudaFree(d_bs);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
static int is_sorted_u(const unsigned int *a, int n) {
    for (int i = 1; i < n; i++) if (a[i] < a[i-1]) return 0;
    return 1;
}
static int cmp_uint(const void *a, const void *b) {
    unsigned int x = *(unsigned int *)a, y = *(unsigned int *)b;
    return (x > y) - (x < y);
}

// One full radix sort pass over a padded_N-element array (padded_N % SECTION_SIZE == 0).
// Ping-pong between d_in and d_tmp; result pointer returned in *pp_in.
static void radix_sort_coalesced(unsigned int **pp_in, unsigned int *d_tmp,
                                  unsigned int *d_local, unsigned int *d_bcounts,
                                  unsigned int *d_bstarts, int padded_N,
                                  int num_blocks) {
    size_t smem  = (3 * SECTION_SIZE + 1) * sizeof(unsigned int);

    for (int iter = 0; iter < 32; iter++) {
        // Kernel A: local sort + count
        local_sort_count_kernel<<<num_blocks, SECTION_SIZE, smem>>>(
            *pp_in, d_local, d_bcounts, padded_N, iter);

        // Copy bucket_counts → bucket_starts, then scan in place
        cudaMemcpy(d_bstarts, d_bcounts,
                   2 * num_blocks * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
        exclusive_scan(d_bstarts, 2 * num_blocks);

        // Kernel B: coalesced scatter
        coalesced_scatter_kernel<<<num_blocks, SECTION_SIZE>>>(
            d_local, d_tmp, d_bcounts, d_bstarts, padded_N);
        cudaDeviceSynchronize();

        // Ping-pong
        unsigned int *t = *pp_in; *pp_in = d_tmp; d_tmp = t;
    }
}

int main(void) {
    printf("=== Coalesced Radix Sort — Shared-Memory Local Buckets "
           "(§13.4, Figs 13.5–13.6) ===\n\n");

    // ── Small test ────────────────────────────────────────────────────────────
    {
        unsigned int h[] = {0xC, 0x3, 0x6, 0x9, 0xF, 0x8, 0x5, 0xA,
                            0xA, 0x6, 0xB, 0xD, 0x4, 0xA, 0x7, 0x0};
        int N = 16;
        // Pad to SECTION_SIZE
        int padded_N = SECTION_SIZE;
        int nb = padded_N / SECTION_SIZE;  // = 1
        size_t pbytes = padded_N * sizeof(unsigned int);

        unsigned int *h_pad = (unsigned int *)malloc(pbytes);
        memcpy(h_pad, h, N * sizeof(unsigned int));
        for (int i = N; i < padded_N; i++) h_pad[i] = ~0u;

        unsigned int *d_in, *d_tmp, *d_local, *d_bcounts, *d_bstarts;
        cudaMalloc(&d_in,     pbytes);
        cudaMalloc(&d_tmp,    pbytes);
        cudaMalloc(&d_local,  pbytes);
        cudaMalloc(&d_bcounts, 2 * nb * sizeof(unsigned int));
        cudaMalloc(&d_bstarts, 2 * nb * sizeof(unsigned int));
        cudaMemcpy(d_in, h_pad, pbytes, cudaMemcpyHostToDevice);

        radix_sort_coalesced(&d_in, d_tmp, d_local, d_bcounts, d_bstarts,
                             padded_N, nb);

        unsigned int result[256];
        cudaMemcpy(result, d_in, pbytes, cudaMemcpyDeviceToHost);

        printf("Input:  ");
        for (int i = 0; i < N; i++) printf("%X ", h[i]);
        printf("\nSorted: ");
        for (int i = 0; i < N; i++) printf("%X ", result[i]);
        printf("\nSmall test: %s\n\n", is_sorted_u(result, N) ? "PASS" : "FAIL");

        free(h_pad);
        cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_local);
        cudaFree(d_bcounts); cudaFree(d_bstarts);
    }

    // ── Large test with timing ────────────────────────────────────────────────
    {
        int N       = 1 << 20;   // 1 M keys
        int padded_N = ((N + SECTION_SIZE - 1) / SECTION_SIZE) * SECTION_SIZE;
        int nb      = padded_N / SECTION_SIZE;
        size_t nbytes  = N        * sizeof(unsigned int);
        size_t pbytes  = padded_N * sizeof(unsigned int);

        unsigned int *h_keys = (unsigned int *)malloc(nbytes);
        unsigned int *h_ref  = (unsigned int *)malloc(nbytes);
        unsigned int *h_pad  = (unsigned int *)calloc(padded_N, sizeof(unsigned int));
        srand(42);
        for (int i = 0; i < N; i++) h_keys[i] = h_ref[i] = (unsigned int)rand();
        qsort(h_ref, N, sizeof(unsigned int), cmp_uint);
        memcpy(h_pad, h_keys, nbytes);
        for (int i = N; i < padded_N; i++) h_pad[i] = ~0u;

        unsigned int *d_in, *d_tmp, *d_local, *d_bcounts, *d_bstarts;
        cudaMalloc(&d_in,      pbytes);
        cudaMalloc(&d_tmp,     pbytes);
        cudaMalloc(&d_local,   pbytes);
        cudaMalloc(&d_bcounts, 2 * nb * sizeof(unsigned int));
        cudaMalloc(&d_bstarts, 2 * nb * sizeof(unsigned int));
        cudaMemcpy(d_in, h_pad, pbytes, cudaMemcpyHostToDevice);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        radix_sort_coalesced(&d_in, d_tmp, d_local, d_bcounts, d_bstarts,
                             padded_N, nb);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);

        cudaMemcpy(h_pad, d_in, pbytes, cudaMemcpyDeviceToHost);
        int ok = 1;
        for (int i = 0; i < N; i++) if (h_pad[i] != h_ref[i]) { ok = 0; break; }

        printf("Large test: N=1M (padded to %d)  GPU time=%.2f ms  %s\n\n",
               padded_N, ms, ok ? "PASS" : "FAIL");

        printf("Coalescing improvement (§13.4, Figs 13.5–13.6):\n");
        printf("  §13.3 scatter: thread tid writes to output[i - ones_before] or\n");
        printf("    output[N - ones_total + ones_before]  → irregular, non-coalesced\n");
        printf("  §13.4 scatter: within each local bucket, consecutive threads write\n");
        printf("    to consecutive global addresses → fully coalesced stores\n");
        printf("  Key insight: shared-memory local sort rearranges keys into\n");
        printf("    bucket order before writing to global memory\n");

        free(h_keys); free(h_ref); free(h_pad);
        cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_local);
        cudaFree(d_bcounts); cudaFree(d_bstarts);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    return 0;
}
