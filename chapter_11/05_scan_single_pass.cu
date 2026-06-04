// §11.7 Single-pass scan for memory access efficiency
//
// Problem with the three-kernel hierarchical scan (§11.6):
//   Between kernels, S[] is written to global memory and then reloaded.
//   These stores and loads are not overlapped with computation, adding
//   latency that can dominate the total execution time for large inputs.
//
// Domino-style (single-pass) scan (§11.7):
//   One kernel processes the entire input.  Adjacent thread blocks communicate
//   through a "scan_value" array + atomic flags:
//     - Block i completes its local scan → stores cumulative sum → sets flag[i].
//     - Block i+1 spin-waits on flag[i], reads the cumulative sum, then
//       updates its own elements and flags the next block.
//   This "dominoes" the prefix sum forward block by block, all in one kernel.
//
// Dynamic block index (§11.7):
//   Block scheduling on GPUs is not guaranteed to be in blockIdx order.
//   Block i+N may run before block i-1, causing spin-wait deadlocks.
//   Fix: each block acquires a dynamic block ID (bid) by atomically
//   incrementing a global counter.  This guarantees block bid has been
//   scheduled AFTER bid-1 (because bid-1 incremented the counter first).
//
// Adjacent synchronisation (§11.7):
//   - Leader thread (threadIdx.x == 0) of block bid waits for flags[bid-1].
//   - atomicAdd(&flags[bid], 0) is used as a cached read-with-acquire.
//   - __threadfence() ensures scan_value is visible before the flag is set.
//
// Single-pass advantages over three-kernel:
//   - Eliminates the global memory round-trip for S[].
//   - Phase 1 and phase 3 of each block overlap in time with other blocks.
//
// Benchmark: compare Kogge-Stone, Brent-Kung, hierarchical, and single-pass.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef SECTION_SIZE
#define SECTION_SIZE 2048
#endif
#define BLOCK_DIM (SECTION_SIZE / 2)

// ── §11.7 Single-pass (domino-style) scan kernel ──────────────────────────────
__global__ void single_pass_scan_kernel(const float *X, float *Y,
                                         float *scan_value, int *flags,
                                         int *blockCounter, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    __shared__ int   bid_s;
    __shared__ float prev_sum_s;

    // ── Dynamic block index assignment (§11.7) ────────────────────────────────
    if (threadIdx.x == 0)
        bid_s = atomicAdd(blockCounter, 1);
    __syncthreads();
    int bid = bid_s;

    unsigned int i = 2 * bid * blockDim.x + threadIdx.x;

    // ── Load input ────────────────────────────────────────────────────────────
    XY[threadIdx.x]              = (i < N)              ? X[i]              : 0.0f;
    XY[threadIdx.x + blockDim.x] = (i + blockDim.x < N) ? X[i + blockDim.x] : 0.0f;

    // ── Phase 1: local Brent-Kung scan ────────────────────────────────────────
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int idx = (threadIdx.x + 1) * 2 * stride - 1;
        if (idx < SECTION_SIZE) XY[idx] += XY[idx - stride];
    }
    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int idx = (threadIdx.x + 1) * stride * 2 - 1;
        if (idx + stride < SECTION_SIZE) XY[idx + stride] += XY[idx];
    }
    __syncthreads();

    // Local sum = XY[SECTION_SIZE - 1]
    float local_sum = XY[SECTION_SIZE - 1];

    // ── Adjacent synchronisation (§11.7) ─────────────────────────────────────
    if (threadIdx.x == 0) {
        if (bid == 0) {
            // Block 0: no predecessor; cumulative sum = local sum
            scan_value[0] = local_sum;
            __threadfence();                    // ensure value visible before flag
            atomicAdd(&flags[0], 1);            // signal ready
            prev_sum_s = 0.0f;
        } else {
            // Wait for predecessor to set its flag (spin on cached load)
            while (atomicAdd(&flags[bid - 1], 0) == 0) { /* spin */ }
            float prev = scan_value[bid - 1];   // predecessor's cumulative sum
            prev_sum_s = prev;
            scan_value[bid] = prev + local_sum; // our cumulative sum
            __threadfence();
            atomicAdd(&flags[bid], 1);
        }
    }
    __syncthreads();

    // ── Phase 2: add predecessor cumulative sum to all local elements ─────────
    float prev = prev_sum_s;
    if (i < N)              Y[i]              = XY[threadIdx.x]              + prev;
    if (i + blockDim.x < N) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x] + prev;
}

// ── Hierarchical three-kernel scan for comparison ─────────────────────────────
__global__ void scan_local_k1(const float *X, float *Y, float *S, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    XY[threadIdx.x]              = (i < N)              ? X[i]              : 0.0f;
    XY[threadIdx.x + blockDim.x] = (i + blockDim.x < N) ? X[i + blockDim.x] : 0.0f;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int idx = (threadIdx.x + 1) * 2 * stride - 1;
        if (idx < SECTION_SIZE) XY[idx] += XY[idx - stride];
    }
    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int idx = (threadIdx.x + 1) * stride * 2 - 1;
        if (idx + stride < SECTION_SIZE) XY[idx + stride] += XY[idx];
    }
    __syncthreads();
    if (i < N)              Y[i]              = XY[threadIdx.x];
    if (i + blockDim.x < N) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
    if (threadIdx.x == blockDim.x - 1) S[blockIdx.x] = XY[SECTION_SIZE - 1];
}
__global__ void scan_S_k2(float *S, unsigned int nblk) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = threadIdx.x;
    XY[i]              = (i < nblk)              ? S[i]              : 0.0f;
    XY[i + blockDim.x] = (i + blockDim.x < nblk) ? S[i + blockDim.x] : 0.0f;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int idx = (threadIdx.x + 1) * 2 * stride - 1;
        if (idx < SECTION_SIZE) XY[idx] += XY[idx - stride];
    }
    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int idx = (threadIdx.x + 1) * stride * 2 - 1;
        if (idx + stride < SECTION_SIZE) XY[idx + stride] += XY[idx];
    }
    __syncthreads();
    if (i < nblk)              S[i]              = XY[i];
    if (i + blockDim.x < nblk) S[i + blockDim.x] = XY[i + blockDim.x];
}
__global__ void add_S_k3(float *Y, const float *S, unsigned int N) {
    if (blockIdx.x == 0) return;
    unsigned int i = blockIdx.x * SECTION_SIZE + threadIdx.x;
    float off = S[blockIdx.x - 1];
    if (i < N)              Y[i]              += off;
    if (i + blockDim.x < N) Y[i + blockDim.x] += off;
}

static void cpu_scan(const float *X, float *Y, unsigned int N) {
    Y[0] = X[0];
    for (unsigned int i = 1; i < N; i++) Y[i] = Y[i-1] + X[i];
}

static bool verify(const float *ref, const float *gpu, unsigned int N) {
    for (unsigned int i = 0; i < N; i++) {
        float rel = fabsf(ref[i] - gpu[i]) / (fabsf(ref[i]) + 1.0f);
        if (rel > 1e-4f) {
            printf("  MISMATCH i=%u  ref=%.4f  gpu=%.4f\n", i, ref[i], gpu[i]);
            return false;
        }
    }
    return true;
}

int main(void) {
    const unsigned int N = 1 << 22;   // 4M elements
    unsigned int num_blocks = (N + SECTION_SIZE - 1) / SECTION_SIZE;

    float *X_h = (float *)malloc(N * sizeof(float));
    float *Y_h = (float *)malloc(N * sizeof(float));
    float *ref = (float *)malloc(N * sizeof(float));
    for (unsigned int i = 0; i < N; i++) X_h[i] = 1.0f;
    cpu_scan(X_h, ref, N);

    float *X_d, *Y_d, *S_d, *sv_d;
    int   *flags_d, *bc_d;
    cudaMalloc(&X_d,     N * sizeof(float));
    cudaMalloc(&Y_d,     N * sizeof(float));
    cudaMalloc(&S_d,     num_blocks * sizeof(float));
    cudaMalloc(&sv_d,    num_blocks * sizeof(float));  // scan_value for single-pass
    cudaMalloc(&flags_d, num_blocks * sizeof(int));
    cudaMalloc(&bc_d,    sizeof(int));
    cudaMemcpy(X_d, X_h, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms;

    dim3 block(BLOCK_DIM), grid(num_blocks);

    // ── Hierarchical (3 kernels) ───────────────────────────────────────────────
    cudaMemset(Y_d, 0, N * sizeof(float));
    cudaEventRecord(t0);
    scan_local_k1<<<grid, block>>>(X_d, Y_d, S_d, N);
    scan_S_k2<<<1, BLOCK_DIM>>>(S_d, num_blocks);
    add_S_k3<<<grid, block>>>(Y_d, S_d, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    cudaMemcpy(Y_h, Y_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Hierarchical (3 kernels):  %s  %.3f ms\n",
           verify(ref, Y_h, N) ? "PASS" : "FAIL", ms);
    float ms_hier = ms;

    // ── Single-pass (domino) ───────────────────────────────────────────────────
    cudaMemset(Y_d,     0, N * sizeof(float));
    cudaMemset(sv_d,    0, num_blocks * sizeof(float));
    cudaMemset(flags_d, 0, num_blocks * sizeof(int));
    cudaMemset(bc_d,    0, sizeof(int));
    cudaEventRecord(t0);
    single_pass_scan_kernel<<<grid, block>>>(X_d, Y_d, sv_d, flags_d, bc_d, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    cudaMemcpy(Y_h, Y_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Single-pass (domino §11.7): %s  %.3f ms  (%.2fx speedup over hierarchical)\n",
           verify(ref, Y_h, N) ? "PASS" : "FAIL", ms, ms_hier / ms);

    printf("\nN=%u  blocks=%u  SECTION_SIZE=%d\n", N, num_blocks, SECTION_SIZE);
    printf("Single-pass design notes (§11.7):\n");
    printf("  - Dynamic bid via atomicAdd(&blockCounter, 1) prevents deadlock\n");
    printf("  - atomicAdd(&flags[bid-1], 0) acts as a cached spin-wait\n");
    printf("  - __threadfence() ensures scan_value visible before flag is set\n");
    printf("  - Eliminates S[] global memory round-trip between kernel calls\n");

    free(X_h); free(Y_h); free(ref);
    cudaFree(X_d); cudaFree(Y_d); cudaFree(S_d);
    cudaFree(sv_d); cudaFree(flags_d); cudaFree(bc_d);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
