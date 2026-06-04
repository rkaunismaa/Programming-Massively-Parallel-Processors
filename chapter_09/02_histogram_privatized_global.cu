// §9.4 Privatization — Figure 9.9: per-block private copies in global memory
//
// Problem with the basic atomic kernel (§9.3):
//   All threads across all blocks update the SAME 7 histogram elements.
//   Heavy contention → throughput limited by serialisation latency.
//
// Privatization idea (§9.4 / Fig 9.8):
//   Give each block its own private copy of the histogram so that
//   contention is limited to threads within the SAME block.
//   After all blocks finish, merge the private copies into a single result.
//
// Figure 9.9 — global-memory private copies:
//   Host allocates gridDim.x × NUM_BINS elements.
//   Each thread offsets its bin index by  blockIdx.x * NUM_BINS.
//   After the data pass, all blocks merge into block 0's copy.
//
// Contention reduction:
//   With B active blocks across all SMs, contention per bin drops by ~B.
//   Orders of magnitude speedup for heavily contended datasets.
//
// Note on merge cost:
//   The merge (lines 09-17 of Fig 9.9) is done inside the same kernel by
//   block 0 after a __syncthreads() barrier.  All other blocks commit to
//   block 0's copy using atomic operations (still needed because multiple
//   blocks update the same bin simultaneously during the merge phase).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define NUM_BINS   7
#define BLOCK_SIZE 256

// ── Figure 9.9: privatization in global memory ───────────────────────────────
// histo must be allocated with gridDim.x * NUM_BINS elements by the host.
// The final histogram is in histo[0 .. NUM_BINS-1].
__global__ void histo_private_global_kernel(const char *data, unsigned int length,
                                             unsigned int *histo) {
    // ── Per-block histogram pass ───────────────────────────────────────────────
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26)
            // Offset into this block's private copy
            atomicAdd(&histo[blockIdx.x * NUM_BINS + alphabet_position / 4], 1);
    }

    // ── Merge: every block except block 0 commits its copy to block 0 ─────────
    if (blockIdx.x > 0) {
        __syncthreads();
        // Each thread is responsible for one or more bins (strided loop)
        for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
            unsigned int binValue = histo[blockIdx.x * NUM_BINS + bin];
            if (binValue > 0)
                atomicAdd(&histo[bin], binValue);
        }
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
    unsigned int *histo_basic_d, *histo_priv_d;
    dim3 block(BLOCK_SIZE);
    dim3 grid((length + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaMalloc(&data_d, length * sizeof(char));
    cudaMalloc(&histo_basic_d, NUM_BINS * sizeof(unsigned int));
    // Private version needs gridDim.x × NUM_BINS elements
    cudaMalloc(&histo_priv_d, grid.x * NUM_BINS * sizeof(unsigned int));
    cudaMemcpy(data_d, data_h, length * sizeof(char), cudaMemcpyHostToDevice);

    unsigned int gpu_histo[NUM_BINS];
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms_basic, ms_priv;

    // ── Warm-up ────────────────────────────────────────────────────────────────
    cudaMemset(histo_basic_d, 0, NUM_BINS * sizeof(unsigned int));
    histo_basic_kernel<<<grid, block>>>(data_d, length, histo_basic_d);
    cudaMemset(histo_priv_d, 0, grid.x * NUM_BINS * sizeof(unsigned int));
    histo_private_global_kernel<<<grid, block>>>(data_d, length, histo_priv_d);
    cudaDeviceSynchronize();

    // ── Basic atomic ───────────────────────────────────────────────────────────
    cudaMemset(histo_basic_d, 0, NUM_BINS * sizeof(unsigned int));
    cudaEventRecord(t0);
    histo_basic_kernel<<<grid, block>>>(data_d, length, histo_basic_d);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_basic, t0, t1);
    cudaMemcpy(gpu_histo, histo_basic_d, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Basic atomic:            %s  %.3f ms\n",
           equal(cpu_histo, gpu_histo, NUM_BINS) ? "PASS" : "FAIL", ms_basic);

    // ── Global-memory privatization (Fig 9.9) ─────────────────────────────────
    cudaMemset(histo_priv_d, 0, grid.x * NUM_BINS * sizeof(unsigned int));
    cudaEventRecord(t0);
    histo_private_global_kernel<<<grid, block>>>(data_d, length, histo_priv_d);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_priv, t0, t1);
    cudaMemcpy(gpu_histo, histo_priv_d, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Privatized global mem:   %s  %.3f ms  (speedup=%.2fx)\n",
           equal(cpu_histo, gpu_histo, NUM_BINS) ? "PASS" : "FAIL",
           ms_priv, ms_basic / ms_priv);

    printf("\nInput: %u chars  Blocks: %u  NUM_BINS: %d\n", length, grid.x, NUM_BINS);
    printf("Private global allocation: %u × %d × 4 = %u KB\n",
           grid.x, NUM_BINS, grid.x * NUM_BINS * 4 / 1024);
    printf("Contention reduction: ~%ux (one private copy per block)\n", grid.x);

    free(data_h);
    cudaFree(data_d); cudaFree(histo_basic_d); cudaFree(histo_priv_d);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
