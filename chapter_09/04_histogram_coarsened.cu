// §9.5 Coarsening — Figures 9.12 and 9.14
//
// Motivation (§9.5):
//   Privatization's overhead (commit phase) occurs once per thread block.
//   If we launch more blocks than the hardware can run simultaneously, the
//   extra blocks are serialized by the scheduler, paying the commit overhead
//   with no additional parallelism benefit.  The solution is thread coarsening:
//   use fewer, heavier blocks so each block is genuinely running in parallel.
//
// Two input-partitioning strategies (§9.5 / Figs 9.11–9.14):
//
// 1. CONTIGUOUS partitioning (Fig 9.12 / Fig 9.11):
//    Each thread takes a contiguous segment of CFACTOR consecutive elements.
//    Thread tid processes indices [tid*CFACTOR, min((tid+1)*CFACTOR, length)).
//    Simple to implement.  On GPUs, adjacent threads in a warp access
//    non-adjacent locations → poor coalescing.
//
// 2. INTERLEAVED partitioning (Fig 9.14 / Fig 9.13):
//    Each thread strides through the array by (blockDim.x × gridDim.x).
//    Thread tid processes indices tid, tid + stride, tid + 2*stride, …
//    Adjacent threads access adjacent locations each iteration → COALESCED.
//    Preferred on GPUs for memory bandwidth efficiency.
//
// Both variants use shared-memory privatization as the inner mechanism.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define NUM_BINS   7
#define BLOCK_SIZE 256
#ifndef CFACTOR
#define CFACTOR    4
#endif

// ── Figure 9.12: coarsening with CONTIGUOUS partitioning ──────────────────────
__global__ void histo_contiguous_kernel(const char *data, unsigned int length,
                                         unsigned int *histo) {
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        histo_s[bin] = 0u;
    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Process CFACTOR consecutive elements — contiguous segment per thread
    for (unsigned int i = tid * CFACTOR;
         i < (unsigned int)min((int)((tid+1) * CFACTOR), (int)length);
         i++) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26)
            atomicAdd(&histo_s[alphabet_position / 4], 1);
    }
    __syncthreads();

    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if (binValue > 0) atomicAdd(&histo[bin], binValue);
    }
}

// ── Figure 9.14: coarsening with INTERLEAVED partitioning ────────────────────
// Each thread strides across the array — stride = total number of threads.
// Adjacent threads access adjacent locations → COALESCED memory access.
__global__ void histo_interleaved_kernel(const char *data, unsigned int length,
                                          unsigned int *histo) {
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        histo_s[bin] = 0u;
    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;    // total threads
    for (unsigned int i = tid; i < length; i += stride) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26)
            atomicAdd(&histo_s[alphabet_position / 4], 1);
    }
    __syncthreads();

    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if (binValue > 0) atomicAdd(&histo[bin], binValue);
    }
}

// ── Non-coarsened privatized shared kernel for baseline ───────────────────────
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
    unsigned int *histo_d;
    cudaMalloc(&data_d, length * sizeof(char));
    cudaMalloc(&histo_d, NUM_BINS * sizeof(unsigned int));
    cudaMemcpy(data_d, data_h, length * sizeof(char), cudaMemcpyHostToDevice);

    // Grids for the three kernels
    dim3 block(BLOCK_SIZE);
    dim3 gridFull((length + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // Coarsened: CFACTOR elements per thread → fewer blocks
    dim3 gridCoarse((length + (long)BLOCK_SIZE * CFACTOR - 1) /
                    ((long)BLOCK_SIZE * CFACTOR));

    unsigned int gpu_histo[NUM_BINS];
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms;

    // Helper lambda for timing
    auto time_it = [&](auto kernel, dim3 g) -> float {
        cudaMemset(histo_d, 0, NUM_BINS * sizeof(unsigned int));
        cudaEventRecord(t0);
        kernel<<<g, block>>>(data_d, length, histo_d);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float t = 0.0f;
        cudaEventElapsedTime(&t, t0, t1);
        return t;
    };

    // Warm-up
    histo_private_shared_kernel<<<gridFull, block>>>(data_d, length, histo_d);
    histo_contiguous_kernel<<<gridCoarse, block>>>(data_d, length, histo_d);
    histo_interleaved_kernel<<<gridCoarse, block>>>(data_d, length, histo_d);
    cudaDeviceSynchronize();

    float ms_base  = time_it(histo_private_shared_kernel, gridFull);
    cudaMemcpy(gpu_histo, histo_d, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Privatized shared (no coarsening): %s  %.3f ms  (%u blocks)\n",
           equal(cpu_histo, gpu_histo, NUM_BINS) ? "PASS" : "FAIL",
           ms_base, gridFull.x);

    float ms_cont  = time_it(histo_contiguous_kernel, gridCoarse);
    cudaMemcpy(gpu_histo, histo_d, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Coarsened contiguous  CF=%d:        %s  %.3f ms  (%u blocks, speedup=%.2fx)\n",
           CFACTOR, equal(cpu_histo, gpu_histo, NUM_BINS) ? "PASS" : "FAIL",
           ms_cont, gridCoarse.x, ms_base / ms_cont);

    float ms_inter = time_it(histo_interleaved_kernel, gridCoarse);
    cudaMemcpy(gpu_histo, histo_d, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Coarsened interleaved CF=%d:        %s  %.3f ms  (%u blocks, speedup=%.2fx)\n",
           CFACTOR, equal(cpu_histo, gpu_histo, NUM_BINS) ? "PASS" : "FAIL",
           ms_inter, gridCoarse.x, ms_base / ms_inter);

    printf("\nInput: %u chars   CFACTOR=%d\n", length, CFACTOR);
    printf("Contiguous: thread tid reads data[tid*CF .. (tid+1)*CF-1]  (non-coalesced)\n");
    printf("Interleaved: thread tid reads data[tid], data[tid+stride], ... (COALESCED)\n");
    printf("Interleaved preferred on GPU for better memory bandwidth utilization.\n");

    free(data_h);
    cudaFree(data_d); cudaFree(histo_d);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
