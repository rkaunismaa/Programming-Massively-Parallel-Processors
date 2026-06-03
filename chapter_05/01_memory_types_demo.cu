/*
 * Chapter 5 — Section 5.2: CUDA Memory Types
 *             Table 5.1: Variable declaration qualifiers, memory, scope, lifetime
 *
 * A CUDA device has five categories of memory.  Each is declared with a
 * different qualifier and has a different scope and lifetime (Table 5.1):
 *
 * ┌──────────────────────────────────┬──────────┬────────┬─────────────┐
 * │ Declaration                      │ Memory   │ Scope  │ Lifetime    │
 * ├──────────────────────────────────┼──────────┼────────┼─────────────┤
 * │ Automatic scalar (int x)         │ Register │ Thread │ Grid        │
 * │ Automatic array (int a[10])      │ Local    │ Thread │ Grid        │
 * │ __device__ __shared__ int s      │ Shared   │ Block  │ Grid        │
 * │ __device__ int g                 │ Global   │ Grid   │ Application │
 * │ __device__ __constant__ int c    │ Constant │ Grid   │ Application │
 * └──────────────────────────────────┴──────────┴────────┴─────────────┘
 *
 * Key properties:
 *   Registers  — fastest, private per-thread, limited supply (see Ch4 occupancy)
 *   Local      — thread-private arrays; physically in global memory (slow!)
 *   Shared     — on-chip, low-latency, shared within a block; the key to tiling
 *   Global     — large off-chip DRAM; accessible by all threads and host
 *   Constant   — read-only from device, cached, very fast for broadcast reads;
 *                limited to 65 536 bytes total
 *
 * Build:
 *   nvcc -O2 -arch=sm_89 -o memory_types 01_memory_types_demo.cu
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* ── Global memory variable (Section 5.2) ───────────────────────────────
 * Declared outside any function with __device__.
 * Scope: all threads of all grids.  Lifetime: entire application.        */
__device__ int d_global_counter;

/* ── Constant memory variable (Section 5.2) ─────────────────────────────
 * Read-only from device, cached for broadcast access.
 * Scope: all threads of all grids.  Lifetime: entire application.
 * Total constant memory limit: 65 536 bytes.                              */
__constant__ float d_scale_factor;

/* -----------------------------------------------------------------------
 * Kernel: exercises every memory type mentioned in Table 5.1
 * ----------------------------------------------------------------------- */
__global__
void memoryTypesKernel(float* in, float* out, int n) {
    /* ── Registers: automatic scalar variables (Table 5.1 row 1) ──────
     * The compiler allocates these in the register file.
     * Each thread gets its own private copy — one million threads →
     * one million independent copies of each register variable.         */
    int   i      = blockIdx.x * blockDim.x + threadIdx.x;  /* register */
    float val    = 0.0f;                                     /* register */
    float result = 0.0f;                                     /* register */

    /* ── Local memory: automatic array variable (Table 5.1 row 2) ─────
     * Arrays that cannot fit in registers spill to "local memory",
     * which is physically located in global memory.
     * Avoid large automatic arrays in kernels — they are slow.
     * Compilers may promote small constant-indexed arrays to registers. */
    float scratch[4];           /* may live in local (global) memory */

    /* ── Shared memory: __shared__ variable (Table 5.1 row 3) ─────────
     * On-chip, very fast, visible to all threads in this block.
     * One version per block; threads must synchronise to use it safely. */
    __shared__ float s_tile[256];

    if (i < n) {
        val = in[i];

        /* Use scratch (local memory) — one copy per thread */
        for (int j = 0; j < 4; j++) scratch[j] = val * (j + 1);

        /* Cooperative load into shared memory */
        s_tile[threadIdx.x] = val;
        __syncthreads();   /* ensure all threads have written before reads */

        /* Read a neighbour from shared memory — no global memory access */
        int neighbour = (threadIdx.x + 1) % blockDim.x;
        result = s_tile[neighbour] + scratch[2];

        /* Apply the constant-memory scale factor — broadcast read, very fast */
        result *= d_scale_factor;

        out[i] = result;

        /* Increment global memory counter atomically (one per active thread) */
        atomicAdd(&d_global_counter, 1);
    }
}

/* -----------------------------------------------------------------------
 * Host: shows how each memory type is accessed from host code
 * ----------------------------------------------------------------------- */
int main() {
    printf("=== CUDA Memory Types — Table 5.1 ===\n\n");

    const int N = 1024;
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in,  N * sizeof(float));
    cudaMalloc((void**)&d_out, N * sizeof(float));

    /* ── Initialise global memory via cudaMemset ──────────────────── */
    cudaMemset(d_in, 0, N * sizeof(float));

    /* ── Initialise device global variable via cudaMemcpyToSymbol ─── */
    int zero = 0;
    cudaMemcpyToSymbol(d_global_counter, &zero, sizeof(int));

    /* ── Initialise constant memory via cudaMemcpyToSymbol ───────────
     * This is the host-side API for writing to __constant__ variables.
     * The device cannot write to constant memory.                      */
    float scale = 2.5f;
    cudaMemcpyToSymbol(d_scale_factor, &scale, sizeof(float));
    printf("Constant memory d_scale_factor set to %.1f\n", scale);

    /* ── Launch kernel ────────────────────────────────────────────── */
    memoryTypesKernel<<<N / 256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    /* ── Read back device global variable ────────────────────────── */
    int counter;
    cudaMemcpyFromSymbol(&counter, d_global_counter, sizeof(int));
    printf("d_global_counter after kernel: %d (expected %d)\n", counter, N);

    /* ── Summary ─────────────────────────────────────────────────── */
    printf("\nMemory type summary (Table 5.1):\n");
    printf("  %-10s  %-12s  %-8s  %-12s  %s\n",
           "Type", "Location", "Scope", "Lifetime", "Declared as");
    printf("  %-10s  %-12s  %-8s  %-12s  %s\n",
           "Register",  "On-chip",  "Thread", "Grid",        "auto scalar");
    printf("  %-10s  %-12s  %-8s  %-12s  %s\n",
           "Local",     "Off-chip", "Thread", "Grid",        "auto array");
    printf("  %-10s  %-12s  %-8s  %-12s  %s\n",
           "Shared",    "On-chip",  "Block",  "Grid",        "__shared__");
    printf("  %-10s  %-12s  %-8s  %-12s  %s\n",
           "Global",    "Off-chip", "Grid",   "Application", "__device__");
    printf("  %-10s  %-12s  %-8s  %-12s  %s\n",
           "Constant",  "Off-chip*","Grid",   "Application", "__constant__");
    printf("  (* cached on-chip for read-only broadcast access)\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
