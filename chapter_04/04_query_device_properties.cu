/*
 * Chapter 4 — Section 4.8: Querying Device Properties
 *
 * CUDA provides two API functions for discovering hardware resources at
 * runtime (Section 4.8):
 *
 *   cudaGetDeviceCount(&devCount)
 *     Returns the number of CUDA-capable devices in the system.
 *     Modern PCs often have 2+: a discrete GPU and an integrated GPU.
 *
 *   cudaGetDeviceProperties(&devProp, deviceId)
 *     Fills a cudaDeviceProp struct with the properties of device deviceId.
 *
 * Selected cudaDeviceProp fields discussed in Section 4.8:
 *
 *   name                  — device name string
 *   multiProcessorCount   — number of SMs
 *   maxThreadsPerBlock    — maximum threads allowed per block (usually 1024)
 *   maxThreadsDim[3]      — maximum threads per dimension
 *   maxGridSize[3]        — maximum grid size per dimension
 *   warpSize              — threads per warp (32 on all current hardware)
 *   regsPerBlock          — total 32-bit registers per SM
 *   sharedMemPerBlock     — shared memory per block (bytes)
 *   clockRate             — SM clock rate in kHz
 *   major, minor          — compute capability (e.g. 8, 9 = sm_89)
 *   maxThreadsPerMultiProcessor — max threads an SM can accommodate
 *
 * This program prints all resource-relevant fields and computes the
 * theoretical maximum blocks/warps per SM under various block sizes,
 * demonstrating the dynamic-partitioning discussion from Section 4.7.
 *
 * Build:
 *   nvcc -O2 -arch=sm_89 -o query_device 04_query_device_properties.cu
 */

#include <stdio.h>
#include <cuda_runtime.h>

static void print_separator(void) {
    printf("  %s\n", "──────────────────────────────────────────────");
}

int main() {
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA devices found: %d\n\n", devCount);

    for (int d = 0; d < devCount; d++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);

        printf("╔══ Device %d: %s ══╗\n", d, p.name);
        print_separator();

        /* ── Architecture ────────────────────────────────────────── */
        printf("  Compute capability:        %d.%d  (sm_%d%d)\n",
               p.major, p.minor, p.major, p.minor);
        printf("  Streaming Multiprocessors: %d\n",  p.multiProcessorCount);
        printf("  Warp size:                 %d threads\n", p.warpSize);
        printf("  Clock rate:                %.0f MHz\n",
               p.clockRate / 1000.0f);
        print_separator();

        /* ── Thread / block limits ───────────────────────────────── */
        printf("  Max threads per block:     %d\n",   p.maxThreadsPerBlock);
        printf("  Max threads per SM:        %d\n",   p.maxThreadsPerMultiProcessor);
        printf("  Max warps per SM:          %d\n",
               p.maxThreadsPerMultiProcessor / p.warpSize);
        printf("  Max block dims (x,y,z):    (%d, %d, %d)\n",
               p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
        printf("  Max grid  dims (x,y,z):    (%d, %d, %d)\n",
               p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
        print_separator();

        /* ── Memory resources ────────────────────────────────────── */
        printf("  Global memory:             %.0f MiB\n",
               p.totalGlobalMem / (1024.0 * 1024.0));
        printf("  Shared memory per block:   %zu bytes (%zu KiB)\n",
               p.sharedMemPerBlock, p.sharedMemPerBlock / 1024);
        printf("  Registers per SM:          %d\n",   p.regsPerMultiprocessor);
        printf("  Registers per block:       %d\n",   p.regsPerBlock);
        printf("  Max registers per thread:  %d\n",
               p.regsPerBlock / p.maxThreadsPerMultiProcessor);
        printf("  L2 cache size:             %d KiB\n", p.l2CacheSize / 1024);
        printf("  Memory clock rate:         %.0f MHz\n",
               p.memoryClockRate / 1000.0f);
        printf("  Memory bus width:          %d bits\n", p.memoryBusWidth);
        printf("  Peak memory bandwidth:     %.1f GB/s\n",
               2.0 * p.memoryClockRate * 1000.0
                   * (p.memoryBusWidth / 8.0) / 1.0e9);
        print_separator();

        /* ── Block-slot limits (Section 4.7 dynamic partitioning) ── */
        printf("  Max concurrent blocks/SM:  %d\n",   p.maxBlocksPerMultiProcessor);
        print_separator();

        /* ── Occupancy analysis for several block sizes (Section 4.7) ── */
        printf("\n  Block-size occupancy table (thread-slot limited):\n");
        printf("  %-12s  %-12s  %-12s  %-10s\n",
               "Block size", "Blocks/SM", "Threads/SM", "Occupancy");
        printf("  %-12s  %-12s  %-12s  %-10s\n",
               "----------", "---------", "----------", "---------");

        int max_threads = p.maxThreadsPerMultiProcessor;
        int max_blocks  = p.maxBlocksPerMultiProcessor;
        int warp        = p.warpSize;

        int sizes[] = {32, 64, 128, 256, 512, 1024};
        for (int s = 0; s < 6; s++) {
            int bs = sizes[s];
            if (bs > p.maxThreadsPerBlock) continue;
            /* Block must be warp-aligned; no padding issue here since all
             * test sizes are multiples of 32 */
            int blocks_by_threads = max_threads / bs;
            int blocks = blocks_by_threads < max_blocks
                       ? blocks_by_threads : max_blocks;
            int threads = blocks * bs;
            float occ = 100.0f * threads / max_threads;
            printf("  %-12d  %-12d  %-12d  %.1f%%\n",
                   bs, blocks, threads, occ);
        }

        /* ── Performance cliff example (Section 4.7) ─────────────── */
        printf("\n  Register-limited occupancy (Section 4.7 example):\n");
        printf("  Max registers/SM: %d\n", p.regsPerMultiprocessor);
        {
            int regs_sm = p.regsPerMultiprocessor;
            int threads_sm = p.maxThreadsPerMultiProcessor;
            int bs = 512;
            int blocks_by_t = threads_sm / bs;
            int blocks_by_b = p.maxBlocksPerMultiProcessor;
            int max_blk = blocks_by_t < blocks_by_b ? blocks_by_t : blocks_by_b;

            /* At R registers/thread, how many threads fit? */
            for (int R = 30; R <= 36; R++) {
                int threads_by_regs = regs_sm / R;
                /* Round down to warp boundary */
                threads_by_regs = (threads_by_regs / warp) * warp;
                int blocks_by_regs = threads_by_regs / bs;
                int actual_blocks = blocks_by_regs < max_blk
                                  ? blocks_by_regs : max_blk;
                int active_threads = actual_blocks * bs;
                float occ = 100.0f * active_threads / threads_sm;
                printf("  R=%2d regs/thread → %d threads/SM → %.0f%% occupancy%s\n",
                       R, active_threads, occ,
                       (R == 32 && occ < 99.0f) ? "  ← performance cliff!" : "");
            }
        }

        printf("\n");
    }

    return 0;
}
