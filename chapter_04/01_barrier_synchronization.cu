/*
 * Chapter 4 — Section 4.3: Synchronization and Transparent Scalability
 *             Figures 4.3, 4.4
 *
 * __syncthreads() is a CUDA barrier: every thread in the block must reach
 * the call before any thread may pass it.  This guarantees that all
 * shared-memory writes made before the barrier are visible to all threads
 * after it.
 *
 * RULES (Section 4.3):
 *   1. __syncthreads() must be reached by ALL threads in the block.
 *   2. If it appears inside an if-else, either ALL threads take the branch
 *      that contains it or NONE do — mixing paths causes a deadlock.
 *   3. Threads in DIFFERENT blocks cannot synchronise with __syncthreads().
 *      (Cooperative Groups adds limited cross-block sync; see Chapter 21.)
 *
 * Programs in this file:
 *   A) In-place shared-memory array reversal — correct barrier usage.
 *   B) Parallel prefix sum within a block (scan) — shows multi-phase barriers.
 *   C) The INCORRECT deadlock pattern from Figure 4.4 (compile-time disabled).
 *
 * Build:
 *   nvcc -O2 -arch=sm_89 -o barrier_sync 01_barrier_synchronization.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* -----------------------------------------------------------------------
 * A) In-place reversal using shared memory
 *
 * Phase 1: every thread loads one element into shared memory (s[i])
 *          __syncthreads() — all loads must complete before any store
 * Phase 2: every thread writes s[N-1-i] back to the output array
 *
 * Without the barrier, some threads would read s[] before other threads
 * had stored their values there — a classic read-before-write hazard.
 * ----------------------------------------------------------------------- */
#define BLOCK 256

__global__
void reverseKernel(float* in, float* out, int n) {
    __shared__ float s[BLOCK];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;

    /* Phase 1: cooperative load into shared memory */
    if (i < n)
        s[t] = in[i];

    /* Barrier: every thread in this block must finish Phase 1 before
     * any thread starts Phase 2.  ALL threads reach this point because
     * the __syncthreads() is not inside a conditional branch. */
    __syncthreads();

    /* Phase 2: write in reversed order within the block */
    if (i < n)
        out[blockIdx.x * blockDim.x + (blockDim.x - 1 - t)] = s[t];
}

/* -----------------------------------------------------------------------
 * B) Parallel prefix sum (inclusive scan) within a block
 *
 * Uses the naive O(n log n) algorithm.  Each of the log2(blockDim) passes
 * requires a barrier before the next pass can begin, because every thread
 * reads values that threads from the previous pass wrote.
 *
 * This is a preview of the scan pattern covered in Chapter 11.
 * ----------------------------------------------------------------------- */
__global__
void blockScanKernel(float* in, float* out, int n) {
    __shared__ float s[BLOCK];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;

    s[t] = (i < n) ? in[i] : 0.0f;
    __syncthreads();  /* all threads loaded */

    /* Log2(BLOCK) reduction passes */
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float addend = (t >= stride) ? s[t - stride] : 0.0f;
        __syncthreads();  /* wait: everyone must read before anyone writes */
        s[t] += addend;
        __syncthreads();  /* wait: writes done before next read pass */
    }

    if (i < n) out[i] = s[t];
}

/* -----------------------------------------------------------------------
 * C) The INCORRECT barrier pattern from Figure 4.4
 *
 * This function is intentionally NOT compiled (wrapped in #if 0) because
 * it results in deadlock or undefined behaviour.
 *
 * The two __syncthreads() calls define TWO DIFFERENT barrier points.
 * Even-indexed threads reach barrier_1; odd-indexed threads reach barrier_2.
 * Neither set of threads ever reaches the other's barrier, so BOTH barriers
 * wait forever → deadlock.
 * ----------------------------------------------------------------------- */
#if 0
__global__
void INCORRECT_barrier_example(int n) {
    if (threadIdx.x % 2 == 0) {
        /* … even-thread work … */
        __syncthreads();  /* barrier_1 — odd threads never reach this */
    } else {
        /* … odd-thread work … */
        __syncthreads();  /* barrier_2 — even threads never reach this */
    }
    /* Execution never reaches here: deadlock */
}
#endif

/* -----------------------------------------------------------------------
 * Helpers
 * ----------------------------------------------------------------------- */
static void cpu_reverse_blocks(float* in, float* out, int n) {
    /* Reverse within each BLOCK-sized chunk, matching the kernel */
    for (int b = 0; b < (n + BLOCK - 1) / BLOCK; b++) {
        for (int t = 0; t < BLOCK && b * BLOCK + t < n; t++) {
            int src = b * BLOCK + t;
            int dst = b * BLOCK + (BLOCK - 1 - t);
            if (dst < n) out[dst] = in[src];
        }
    }
}

static void cpu_scan(float* in, float* out, int n) {
    /* Inclusive prefix sum within each BLOCK-sized chunk */
    for (int b = 0; b < (n + BLOCK - 1) / BLOCK; b++) {
        float sum = 0.0f;
        for (int t = 0; t < BLOCK && b * BLOCK + t < n; t++) {
            sum += in[b * BLOCK + t];
            out[b * BLOCK + t] = sum;
        }
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    float *d_in, *d_out;

    cudaMalloc((void**)&d_in,  bytes);
    cudaMalloc((void**)&d_out, bytes);

    /* ── A) Reversal ──────────────────────────────────────────────── */
    for (int i = 0; i < N; i++) h_in[i] = (float)i;
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    reverseKernel<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cpu_reverse_blocks(h_in, h_ref, N);
    int ok = 1;
    for (int i = 0; i < N; i++) ok &= (h_out[i] == h_ref[i]);
    printf("A) Block-reverse (barrier between load and store): %s\n",
           ok ? "PASSED" : "FAILED");

    /* ── B) Scan ──────────────────────────────────────────────────── */
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;   /* scan of 1s → index+1 */
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    blockScanKernel<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cpu_scan(h_in, h_ref, N);
    ok = 1;
    for (int i = 0; i < N; i++) ok &= (h_out[i] == h_ref[i]);
    printf("B) Block prefix-sum (multi-phase barriers):        %s\n",
           ok ? "PASSED" : "FAILED");

    /* ── Transparent scalability demo (Section 4.3) ──────────────── */
    printf("\nTransparent scalability:\n");
    printf("  Blocks can execute on SMs in any order.\n");
    printf("  No barriers exist between different blocks.\n");
    printf("  This is what allows the same code to run on\n");
    printf("  both a 2-SM and a 128-SM GPU without modification.\n");

    free(h_in); free(h_out); free(h_ref);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
