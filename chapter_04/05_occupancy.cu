/*
 * Chapter 4 — Section 4.7: Resource Partitioning and Occupancy
 *
 * Occupancy = (warps assigned to SM) / (max warps SM can support)
 *
 * Higher occupancy gives the SM more warps to switch to when one warp
 * stalls on a long-latency operation (Section 4.6, latency tolerance).
 *
 * Three resources can independently cap occupancy (Section 4.7):
 *   1. Thread block slots  — max blocks/SM (hardware limit)
 *   2. Thread slots        — max threads/SM (hardware limit)
 *   3. Register file       — max registers/SM divided by registers/thread
 *   4. Shared memory       — max shared mem/SM divided by shared mem/block
 *      (shared memory is covered in Chapter 5)
 *
 * CUDA provides a runtime helper:
 *   cudaOccupancyMaxActiveBlocksPerMultiprocessor(
 *       &blocks, kernel, blockSize, dynamicSharedMem)
 * This accounts for ALL resource constraints simultaneously.
 *
 * Programs:
 *   A) Manual occupancy calculation for several block sizes
 *   B) API-based occupancy query with cudaOccupancyMaxActiveBlocksPerMultiprocessor
 *   C) The "performance cliff": adding one register can halve occupancy
 *   D) Finding the optimal block size with cudaOccupancyMaxPotentialBlockSize
 *
 * Build:
 *   nvcc -O2 -arch=sm_89 -o occupancy 05_occupancy.cu
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* -----------------------------------------------------------------------
 * Sample kernels — different register pressures to demonstrate occupancy
 * ----------------------------------------------------------------------- */

/* Low-register kernel (NVCC should use ~10-15 registers) */
__global__
void lowRegKernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

/* Heavier kernel — unrolled arithmetic forces more registers */
__global__
void heavyRegKernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v0 = a[i], v1 = b[i];
        float r0  = v0 * v1;
        float r1  = r0 + v0 * 1.1f;
        float r2  = r1 - v1 * 2.2f;
        float r3  = r2 * r0 + 0.5f;
        float r4  = r3 / (r2 + 1.0f);
        float r5  = r4 * r4 - r3;
        float r6  = r5 + v0 * r4;
        float r7  = r6 * r1 + r0;
        float r8  = r7 - r5 * 0.3f;
        float r9  = r8 * r9 + r6;   /* intentional self-reference to prevent CSE */
        c[i] = r9;
        (void)r9;
    }
}

/* Kernel with explicit shared memory to show shared-mem occupancy limits */
template <int SHMEM_BYTES>
__global__
void shmemKernel(float* a, float* c, int n) {
    __shared__ char buf[SHMEM_BYTES];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    /* Touch shared memory so the compiler cannot eliminate it */
    buf[threadIdx.x % SHMEM_BYTES] = (char)i;
    __syncthreads();
    if (i < n) c[i] = a[i] + (float)buf[threadIdx.x % SHMEM_BYTES] * 0.0f;
}

/* -----------------------------------------------------------------------
 * Helper: print occupancy info for one (kernel, blockSize) combo
 * ----------------------------------------------------------------------- */
template <typename KernelFunc>
static void print_occupancy(const char* name, KernelFunc kernel,
                             int blockSize, int dynamicShmem,
                             int maxThreadsPerSM) {
    int blocksPerSM;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM, kernel, blockSize, dynamicShmem);
    if (err != cudaSuccess) {
        printf("  %-30s blockSize=%4d  ERROR: %s\n",
               name, blockSize, cudaGetErrorString(err));
        return;
    }
    int threadsPerSM = blocksPerSM * blockSize;
    float occ = 100.0f * threadsPerSM / maxThreadsPerSM;
    printf("  %-30s blockSize=%4d  blocks/SM=%2d  threads/SM=%4d  occ=%.1f%%\n",
           name, blockSize, blocksPerSM, threadsPerSM, occ);
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    printf("Device: %s  (sm_%d%d)\n", p.name, p.major, p.minor);
    printf("  Max threads/SM:  %d\n", p.maxThreadsPerMultiProcessor);
    printf("  Max blocks/SM:   %d\n", p.maxBlocksPerMultiProcessor);
    printf("  Registers/SM:    %d\n", p.regsPerMultiprocessor);
    printf("  Shared mem/SM:   %zu KiB\n\n",
           p.sharedMemPerMultiprocessor / 1024);

    int maxT = p.maxThreadsPerMultiProcessor;

    /* ── A) Manual thread-slot-only calculation ──────────────────── */
    printf("=== A) Manual occupancy (thread slots only, Section 4.7) ===\n");
    {
        int maxBlk = p.maxBlocksPerMultiProcessor;
        int sizes[] = {32, 64, 128, 256, 512, 1024};
        printf("  %-12s %-14s %-14s %s\n",
               "Block size", "Blocks/SM", "Threads/SM", "Occupancy");
        for (int s = 0; s < 6; s++) {
            int bs = sizes[s];
            if (bs > p.maxThreadsPerBlock) continue;
            int blocks = maxT / bs;
            if (blocks > maxBlk) blocks = maxBlk;
            float occ = 100.0f * blocks * bs / maxT;
            printf("  %-12d %-14d %-14d %.1f%%\n", bs, blocks, blocks*bs, occ);
        }
    }

    /* ── B) API-based occupancy — accounts for register limits ──── */
    printf("\n=== B) API occupancy (cudaOccupancyMaxActiveBlocksPerMultiprocessor) ===\n");
    int sizes[] = {32, 64, 128, 256, 512, 1024};
    printf("  lowRegKernel:\n");
    for (int s = 0; s < 6; s++) {
        int bs = sizes[s];
        if (bs > p.maxThreadsPerBlock) continue;
        print_occupancy("lowRegKernel", lowRegKernel, bs, 0, maxT);
    }

    /* ── C) Performance cliff (Section 4.7) ─────────────────────── */
    printf("\n=== C) Performance cliff demo ===\n");
    printf("  The book's example: Volta SM with 65536 regs, 2048 max threads.\n");
    printf("  At block size 512:\n");
    {
        int regs_sm   = 65536;  /* Volta/Ampere */
        int threads_sm = 2048;
        int bs         = 512;
        int max_blk    = 4;     /* 2048/512 */

        printf("  %-14s %-14s %-14s %s\n",
               "Regs/thread", "Threads/SM", "Blocks/SM", "Occupancy");
        for (int r = 29; r <= 35; r++) {
            int threads_by_regs = (regs_sm / r / 32) * 32; /* warp-aligned */
            int blocks_by_regs  = threads_by_regs / bs;
            if (blocks_by_regs > max_blk) blocks_by_regs = max_blk;
            int active = blocks_by_regs * bs;
            float occ = 100.0f * active / threads_sm;
            const char* flag = (r == 33 && active < threads_sm)
                               ? "  ← cliff!" : "";
            printf("  %-14d %-14d %-14d %.0f%%%s\n",
                   r, active, blocks_by_regs, occ, flag);
        }
        printf("  Going from 32→33 regs/thread: 3 blocks fit → 1536 threads → 75%% occ\n");
    }

    /* ── D) Optimal block size ───────────────────────────────────── */
    printf("\n=== D) Optimal block size (cudaOccupancyMaxPotentialBlockSize) ===\n");
    {
        int minGridSize, blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                           lowRegKernel, 0, 0);
        printf("  lowRegKernel:   recommended blockSize=%d, minGridSize=%d\n",
               blockSize, minGridSize);

        /* Occupancy at recommended size */
        int blocksPerSM;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerSM, lowRegKernel, blockSize, 0);
        float occ = 100.0f * blocksPerSM * blockSize / maxT;
        printf("  Occupancy at recommended blockSize: %.1f%%\n\n", occ);

        printf("  Practical advice (Section 4.7):\n");
        printf("    • Block size should be a multiple of %d (warp size)\n",
               p.warpSize);
        printf("    • Larger blocks give the SM more warps for latency hiding\n");
        printf("    • But more registers/thread with larger blocks can lower occupancy\n");
        printf("    • Use cudaOccupancyMaxPotentialBlockSize to find the sweet spot\n");
    }

    return 0;
}
