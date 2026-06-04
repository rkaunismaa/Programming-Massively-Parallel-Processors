// §9.4 Privatization — Figure 9.10: per-block private copies in shared memory
//
// Improvement over global-memory privatization (Fig 9.9):
//   The private histogram is stored in __shared__ memory instead of global.
//   Shared memory has ~10-100× lower latency than DRAM (a few cycles vs
//   hundreds).  Lower latency → dramatically higher atomic throughput for
//   the within-block updates.
//
// Three-phase structure (Fig 9.10):
//   Phase 1 — Init:    each thread zeroes one or more histo_s bins in parallel
//   Phase 2 — Update:  each thread updates histo_s[bin] with atomicAdd
//   Phase 3 — Commit:  each thread commits one or more histo_s bins to the
//                      global histo[] with atomicAdd (only if bin is non-zero)
//
// The __syncthreads() between phases 1 and 2 ensures all bins are zeroed
// before any thread starts updating them.
// The __syncthreads() between phases 2 and 3 ensures all within-block updates
// are visible before the commit reads histo_s.
//
// When NUM_BINS is small (≤ 1024), this approach uses negligible shared memory:
//   7 bins × 4 bytes = 28 bytes.  Essentially free.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define NUM_BINS   7
#define BLOCK_SIZE 256

// ── Figure 9.10: privatization in shared memory ──────────────────────────────
__global__ void histo_private_shared_kernel(const char *data, unsigned int length,
                                             unsigned int *histo) {
    // ── Phase 1: initialize private histogram bins to 0 ───────────────────────
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        histo_s[bin] = 0u;
    __syncthreads();

    // ── Phase 2: accumulate into the private (shared memory) histogram ─────────
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26)
            atomicAdd(&histo_s[alphabet_position / 4], 1);
    }
    __syncthreads();

    // ── Phase 3: commit non-zero bins to the global output histogram ───────────
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if (binValue > 0)
            atomicAdd(&histo[bin], binValue);
    }
}

// ── Basic atomic kernel for comparison ────────────────────────────────────────
__global__ void histo_basic_kernel(const char *data, unsigned int length,
                                    unsigned int *histo) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int ap = data[i] - 'a';
        if (ap >= 0 && ap < 26) atomicAdd(&histo[ap / 4], 1);
    }
}

// ── CPU reference ─────────────────────────────────────────────────────────────
static void histogram_cpu(const char *data, unsigned int length,
                           unsigned int *histo) {
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

static float run_kernel(void (*launch)(dim3,dim3,const char*,unsigned int,unsigned int*),
                        const char *data_d, unsigned int length,
                        unsigned int *histo_d, unsigned int num_bins,
                        dim3 grid, dim3 block) {
    cudaMemset(histo_d, 0, num_bins * sizeof(unsigned int));
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    launch(grid, block, data_d, length, histo_d);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

int main(void) {
    const char *phrase = "programmingmassiveleparallelprocessors";
    const unsigned int REPS = 1 << 14;
    unsigned int length = (unsigned int)(strlen(phrase) * REPS);

    char *data_h = (char *)malloc(length);
    for (unsigned int r = 0; r < REPS; r++)
        memcpy(data_h + r * strlen(phrase), phrase, strlen(phrase));

    unsigned int cpu_histo[NUM_BINS];
    histogram_cpu(data_h, length, cpu_histo);

    char         *data_d;
    unsigned int *histo_d;
    cudaMalloc(&data_d, length * sizeof(char));
    cudaMalloc(&histo_d, NUM_BINS * sizeof(unsigned int));
    cudaMemcpy(data_d, data_h, length * sizeof(char), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((length + BLOCK_SIZE - 1) / BLOCK_SIZE);

    unsigned int gpu_histo[NUM_BINS];
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms;

    // ── Basic atomic ───────────────────────────────────────────────────────────
    cudaMemset(histo_d, 0, NUM_BINS * sizeof(unsigned int));
    cudaEventRecord(t0);
    histo_basic_kernel<<<grid, block>>>(data_d, length, histo_d);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_basic = 0.0f;
    cudaEventElapsedTime(&ms_basic, t0, t1);
    cudaMemcpy(gpu_histo, histo_d, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Basic atomic:             %s  %.3f ms\n",
           equal(cpu_histo, gpu_histo, NUM_BINS) ? "PASS" : "FAIL", ms_basic);

    // ── Warm-up shared privatization ──────────────────────────────────────────
    cudaMemset(histo_d, 0, NUM_BINS * sizeof(unsigned int));
    histo_private_shared_kernel<<<grid, block>>>(data_d, length, histo_d);
    cudaDeviceSynchronize();

    // ── Shared-memory privatization (Fig 9.10) ─────────────────────────────────
    cudaMemset(histo_d, 0, NUM_BINS * sizeof(unsigned int));
    cudaEventRecord(t0);
    histo_private_shared_kernel<<<grid, block>>>(data_d, length, histo_d);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_priv = 0.0f;
    cudaEventElapsedTime(&ms_priv, t0, t1);
    cudaMemcpy(gpu_histo, histo_d, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Privatized shared mem:    %s  %.3f ms  (speedup=%.2fx)\n",
           equal(cpu_histo, gpu_histo, NUM_BINS) ? "PASS" : "FAIL",
           ms_priv, ms_basic / ms_priv);

    printf("\nInput: %u chars   Blocks: %u   NUM_BINS: %d\n",
           length, grid.x, NUM_BINS);
    printf("Shared memory per block: %d bins × 4 bytes = %d bytes\n",
           NUM_BINS, NUM_BINS * 4);
    printf("Within-block atomic uses shared memory → few-cycle latency\n");
    printf("Global commit: at most NUM_BINS=%d atomics per block (vs %u input per block)\n",
           NUM_BINS, BLOCK_SIZE);

    free(data_h);
    cudaFree(data_d); cudaFree(histo_d);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
