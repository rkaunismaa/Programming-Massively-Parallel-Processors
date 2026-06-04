// §9.6 Aggregation — Figure 9.15
//
// Problem (§9.6):
//   For data with large runs of identical values (e.g., sky images with many
//   pixels of the same luminance), many consecutive thread iterations map to
//   the same histogram bin.  Each such update still costs one atomicAdd, even
//   when the bin index hasn't changed.  Heavy repetition means heavy contention.
//
// Aggregation idea (§9.6 / Fig 9.15 / Merrill 2015):
//   Each thread maintains two extra variables:
//     accumulator  — count of updates buffered for the current bin streak
//     prevBinIdx   — the bin being aggregated
//   While the bin index stays the same, the thread increments accumulator
//   (cheap, register operation) rather than calling atomicAdd.
//   When the bin index changes, the thread flushes the accumulated count
//   with a single atomicAdd before switching to the new bin.
//
// Benefits:
//   - Reduces atomicAdd calls proportionally to the average streak length.
//   - For a perfectly uniform random input, no savings (every call changes bin).
//   - For a fully biased input (all same bin), reduces to 1 atomicAdd per thread.
//
// Caveat:
//   More code, more registers → extra overhead when contention is low.
//   Aggregation pays off only when contention is the dominant bottleneck.
//
// This file runs all five kernel variants on two datasets:
//   uniform  — random lowercase letters
//   biased   — 90% 'm' (bin 3), rest random

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define NUM_BINS   7
#define BLOCK_SIZE 256
#ifndef CFACTOR
#define CFACTOR    4
#endif

// ── Figure 9.15: aggregated histogram kernel ──────────────────────────────────
// Uses interleaved partitioning (coalesced) + shared memory privatization
// + accumulator-based aggregation.
__global__ void histo_aggregated_kernel(const char *data, unsigned int length,
                                         unsigned int *histo) {
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        histo_s[bin] = 0u;
    __syncthreads();

    // Histogram with aggregation
    unsigned int accumulator = 0;
    int          prevBinIdx  = -1;    // -1 → no active streak yet

    unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = tid; i < length; i += stride) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            int bin = alphabet_position / 4;
            if (bin == prevBinIdx) {
                // Continue the streak — no atomic needed yet
                ++accumulator;
            } else {
                // Bin changed: flush the buffered count for the previous bin
                if (accumulator > 0)
                    atomicAdd(&histo_s[prevBinIdx], accumulator);
                accumulator = 1;
                prevBinIdx  = bin;
            }
        }
    }
    // Flush the final streak after the loop
    if (accumulator > 0)
        atomicAdd(&histo_s[prevBinIdx], accumulator);

    __syncthreads();

    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int bv = histo_s[bin];
        if (bv > 0) atomicAdd(&histo[bin], bv);
    }
}

// ── Interleaved coarsened (no aggregation) for comparison ────────────────────
__global__ void histo_interleaved_kernel(const char *data, unsigned int length,
                                          unsigned int *histo) {
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        histo_s[bin] = 0u;
    __syncthreads();
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x) {
        int ap = data[i] - 'a';
        if (ap >= 0 && ap < 26) atomicAdd(&histo_s[ap / 4], 1);
    }
    __syncthreads();
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int bv = histo_s[bin];
        if (bv > 0) atomicAdd(&histo[bin], bv);
    }
}

// ── Basic atomic (Fig 9.6) ────────────────────────────────────────────────────
__global__ void histo_basic_kernel(const char *data, unsigned int length,
                                    unsigned int *histo) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int ap = data[i] - 'a';
        if (ap >= 0 && ap < 26) atomicAdd(&histo[ap / 4], 1);
    }
}

// ── Privatized shared (Fig 9.10) ──────────────────────────────────────────────
__global__ void histo_private_shared_kernel(const char *data, unsigned int length,
                                             unsigned int *histo) {
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        histo_s[bin] = 0u;
    __syncthreads();
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int ap = data[i] - 'a';
        if (ap >= 0 && ap < 26) atomicAdd(&histo_s[ap / 4], 1);
    }
    __syncthreads();
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int bv = histo_s[bin];
        if (bv > 0) atomicAdd(&histo[bin], bv);
    }
}

static void histogram_cpu(const char *data, unsigned int length, unsigned int *histo) {
    memset(histo, 0, NUM_BINS * sizeof(unsigned int));
    for (unsigned int i = 0; i < length; i++) {
        int ap = data[i] - 'a';
        if (ap >= 0 && ap < 26) histo[ap / 4]++;
    }
}

static bool equal(const unsigned int *a, const unsigned int *b, int n) {
    for (int i = 0; i < n; i++) if (a[i] != b[i]) return false;
    return true;
}

static void run_dataset(const char *label, const char *data_h, unsigned int length,
                        const char *data_d, unsigned int *histo_d) {
    printf("\n── %s (%u chars) ─────────────────────────────────────────\n",
           label, length);

    unsigned int cpu_histo[NUM_BINS];
    histogram_cpu(data_h, length, cpu_histo);

    dim3 block(BLOCK_SIZE);
    dim3 gridFull((length + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 gridCoarse((length + (long)BLOCK_SIZE * CFACTOR - 1) /
                    ((long)BLOCK_SIZE * CFACTOR));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    unsigned int gpu_histo[NUM_BINS];
    float ms;

    struct { const char *name; float ms; } results[4];
    int ri = 0;

#define TIME_KERNEL(kname, g) \
    cudaMemset(histo_d, 0, NUM_BINS * sizeof(unsigned int)); \
    cudaEventRecord(t0); \
    kname<<<g, block>>>(data_d, length, histo_d); \
    cudaEventRecord(t1); cudaEventSynchronize(t1); \
    cudaEventElapsedTime(&ms, t0, t1); \
    cudaMemcpy(gpu_histo, histo_d, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost); \
    results[ri++] = {#kname, ms}; \
    printf("  %-36s  %s  %.3f ms\n", #kname, equal(cpu_histo, gpu_histo, NUM_BINS) ? "PASS" : "FAIL", ms)

    TIME_KERNEL(histo_basic_kernel,          gridFull);
    TIME_KERNEL(histo_private_shared_kernel, gridFull);
    TIME_KERNEL(histo_interleaved_kernel,    gridCoarse);
    TIME_KERNEL(histo_aggregated_kernel,     gridCoarse);

    printf("  Speedup aggregated vs basic:       %.2fx\n", results[0].ms / results[3].ms);
    printf("  Speedup aggregated vs interleaved: %.2fx\n", results[2].ms / results[3].ms);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
#undef TIME_KERNEL
}

int main(void) {
    const unsigned int LEN = 1 << 23;   // 8 M characters
    char *uniform_h = (char *)malloc(LEN);
    char *biased_h  = (char *)malloc(LEN);

    srand(42);
    // Uniform: random lowercase letters
    for (unsigned int i = 0; i < LEN; i++)
        uniform_h[i] = 'a' + (rand() % 26);

    // Biased: 90% 'm' (bin 3), 10% random — simulates hot-spot dataset
    for (unsigned int i = 0; i < LEN; i++) {
        if (rand() % 10 < 9)
            biased_h[i] = 'm';     // bin 3 — "m-p"
        else
            biased_h[i] = 'a' + (rand() % 26);
    }

    char         *uniform_d, *biased_d;
    unsigned int *histo_d;
    cudaMalloc(&uniform_d, LEN * sizeof(char));
    cudaMalloc(&biased_d,  LEN * sizeof(char));
    cudaMalloc(&histo_d,   NUM_BINS * sizeof(unsigned int));
    cudaMemcpy(uniform_d, uniform_h, LEN * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(biased_d,  biased_h,  LEN * sizeof(char), cudaMemcpyHostToDevice);

    printf("CFACTOR=%d  BLOCK_SIZE=%d  NUM_BINS=%d\n", CFACTOR, BLOCK_SIZE, NUM_BINS);

    run_dataset("Uniform random distribution", uniform_h, LEN, uniform_d, histo_d);
    run_dataset("Biased (90% 'm', bin 3)", biased_h, LEN, biased_d, histo_d);

    printf("\nNote: aggregation benefit is most pronounced on the biased dataset\n");
    printf("      because consecutive 'm' chars create long single-bin streaks,\n");
    printf("      replacing many atomicAdds with one register increment per streak.\n");

    free(uniform_h); free(biased_h);
    cudaFree(uniform_d); cudaFree(biased_d); cudaFree(histo_d);
    return 0;
}
