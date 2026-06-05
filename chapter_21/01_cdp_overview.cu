// §21.2  CUDA Dynamic Parallelism — overview
//        Figs 21.4 (without CDP) and 21.5 (with CDP)
//
// Pattern: each outer element i owns a variable-length inner range
// [start[i], end[i]).  Without CDP the parent kernel serialises that inner
// loop.  With CDP the parent thread launches a child grid sized to the
// range, exposing the inner iterations as independent parallel threads.
//
// Concrete work: out[i] = someData[i]*2 + Σ moreData[j] for j in [start,end).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N_ELEMENTS  64     // outer elements (parent-grid threads)
#define INNER_MAX  1024    // maximum inner-loop iterations per element
#define CHILD_BLOCK 256    // threads per child block

// ── Fig 21.4: Without CDP ─────────────────────────────────────────────────────
// Each thread serialises its own inner loop. Threads with long ranges stall
// threads with short ones in the same warp → control divergence.
__global__ void kernel_no_cdp(const unsigned int *start, const unsigned int *end,
                               const float *someData, const float *moreData,
                               float *out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_ELEMENTS) return;

    out[i] = someData[i] * 2.0f;                    // doSomeWork (Fig 21.4 line 05)

    for (unsigned int j = start[i]; j < end[i]; ++j) // Fig 21.4 lines 07-09
        out[i] += moreData[j];                        // doMoreWork
}

// ── Fig 21.5: With CDP — child kernel ────────────────────────────────────────
// The child does the inner-loop body for one contiguous range.
// Multiple child threads write to the same out[i] → need atomicAdd.
__global__ void kernel_child(unsigned int j_start, unsigned int j_end,
                             const float *moreData, float *out, unsigned int i) {
    unsigned int j = j_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (j < j_end)
        atomicAdd(&out[i], moreData[j]);
}

// ── Fig 21.5: With CDP — parent kernel ───────────────────────────────────────
// Instead of looping, each parent thread launches a child grid whose size
// matches the per-element work. Control divergence is eliminated because
// every parent thread executes one kernel launch (variable-size, but no
// branch divergence in the parent).
__global__ void kernel_parent(const unsigned int *start, const unsigned int *end,
                              const float *someData, const float *moreData,
                              float *out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_ELEMENTS) return;

    out[i] = someData[i] * 2.0f;                     // doSomeWork (Fig 21.5 line 05)

    unsigned int nwork = end[i] - start[i];
    if (nwork == 0) return;
    unsigned int nblocks = (nwork + CHILD_BLOCK - 1) / CHILD_BLOCK;
    kernel_child<<<nblocks, CHILD_BLOCK>>>(           // Fig 21.5 lines 07-08
        start[i], end[i], moreData, out, i);
}

int main(void) {
    printf("=== CUDA Dynamic Parallelism: Overview (§21.2, Figs 21.4/21.5) ===\n\n");

    srand(42);

    // Variable-length inner ranges; build them contiguously in moreData[].
    unsigned int h_start[N_ELEMENTS], h_end[N_ELEMENTS];
    unsigned int totalWork = 0;
    for (int i = 0; i < N_ELEMENTS; i++) {
        h_start[i] = totalWork;
        unsigned int wlen = (rand() % INNER_MAX) + 1;
        h_end[i] = totalWork + wlen;
        totalWork += wlen;
    }

    float *h_someData = (float *)malloc(N_ELEMENTS * sizeof(float));
    float *h_moreData = (float *)malloc(totalWork   * sizeof(float));
    for (int i = 0; i < N_ELEMENTS; i++)  h_someData[i] = (float)(i + 1);
    for (unsigned int j = 0; j < totalWork; j++) h_moreData[j] = 1.0f;

    // CPU reference
    float h_ref[N_ELEMENTS];
    for (int i = 0; i < N_ELEMENTS; i++) {
        h_ref[i] = h_someData[i] * 2.0f;
        for (unsigned int j = h_start[i]; j < h_end[i]; j++)
            h_ref[i] += h_moreData[j];
    }

    unsigned int *d_start, *d_end;
    float *d_someData, *d_moreData, *d_out;
    cudaMalloc(&d_start,    N_ELEMENTS * sizeof(unsigned int));
    cudaMalloc(&d_end,      N_ELEMENTS * sizeof(unsigned int));
    cudaMalloc(&d_someData, N_ELEMENTS * sizeof(float));
    cudaMalloc(&d_moreData, totalWork  * sizeof(float));
    cudaMalloc(&d_out,      N_ELEMENTS * sizeof(float));

    cudaMemcpy(d_start,    h_start,    N_ELEMENTS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_end,      h_end,      N_ELEMENTS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_someData, h_someData, N_ELEMENTS * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_moreData, h_moreData, totalWork  * sizeof(float),        cudaMemcpyHostToDevice);

    float h_out[N_ELEMENTS];
    dim3 block(64), grid((N_ELEMENTS + 63) / 64);

    // ── Without CDP ──────────────────────────────────────────────────────────
    cudaMemset(d_out, 0, N_ELEMENTS * sizeof(float));
    kernel_no_cdp<<<grid, block>>>(d_start, d_end, d_someData, d_moreData, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);

    int ok = 1;
    for (int i = 0; i < N_ELEMENTS; i++)
        if (fabsf(h_out[i] - h_ref[i]) > 1e-3f) ok = 0;
    printf("Without CDP: %s\n", ok ? "PASS" : "FAIL");

    // ── With CDP ─────────────────────────────────────────────────────────────
    cudaMemset(d_out, 0, N_ELEMENTS * sizeof(float));
    kernel_parent<<<grid, block>>>(d_start, d_end, d_someData, d_moreData, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);

    ok = 1;
    for (int i = 0; i < N_ELEMENTS; i++)
        if (fabsf(h_out[i] - h_ref[i]) > 1e-3f) ok = 0;
    printf("With CDP:    %s\n", ok ? "PASS" : "FAIL");

    printf("\nCDP benefit: each parent thread launches a child grid sized exactly\n");
    printf("  to its work range.  Inner iterations run in parallel; divergence\n");
    printf("  from variable ranges is eliminated at the parent level.\n");

    cudaFree(d_start); cudaFree(d_end);
    cudaFree(d_someData); cudaFree(d_moreData); cudaFree(d_out);
    free(h_someData); free(h_moreData);
    return 0;
}
