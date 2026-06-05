// §22.1  Zero-copy memory
//
// Zero-copy (CUDA 2.2+) maps pinned host memory into the device virtual
// address space so a kernel can access host data directly over the PCIe bus
// without an explicit cudaMemcpy (§22.1).
//
// Key API:
//   cudaHostAlloc(&ptr, size, cudaHostAllocMapped)   — allocate pinned, mapped
//   cudaHostGetDevicePointer(&dptr, ptr, 0)           — get device VA for ptr
//
// §22.1 notes that zero-copy access suffers from PCIe bandwidth (<10% of
// device DRAM bandwidth) and high latency, so it is only advantageous for
// data that a kernel accesses occasionally or sparsely.  For dense, repeated
// accesses, copying data to device memory first is faster.
//
// This file demonstrates the use case described in §22.1:
//
//   A read-only coefficient table that is used for irregular lookups by a
//   kernel.  The pattern is:
//       out[i] = table[hash_a(i)] + table[hash_b(i)]
//   where hash_a / hash_b produce irregular (non-coalesceable) table indices.
//
// Three variants are compared:
//   Device memory:   table copied to device before the kernel runs
//   Zero-copy:       table stays in pinned host memory; kernel reads over PCIe
//   Unified memory:  table allocated with cudaMallocManaged (§22.1 comparison)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N          (1 << 20)    // 1 M output elements
#define TABLE_SIZE (1 << 12)    // 4 K table entries — sparse access pattern
#define BLOCK      256

__global__ void lookup(const float *__restrict__ table, int table_sz,
                        float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Irregular table access: two pseudo-random indices derived from i
    int ia = (i * 2654435761u) % (unsigned)table_sz;   // hash A
    int ib = (i * 2246822519u) % (unsigned)table_sz;   // hash B
    out[i] = table[ia] + table[ib];
}

int main(void) {
    printf("=== Zero-Copy Memory (§22.1) ===\n\n");

    // cudaSetDeviceFlags must be called before any context-creating API
    cudaSetDeviceFlags(cudaDeviceMapHost);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!prop.canMapHostMemory) {
        printf("Device does not support mapped host memory — skipping zero-copy path.\n");
        return 0;
    }

    // Build a reference table on the host
    float *h_table_ref = (float *)malloc(TABLE_SIZE * sizeof(float));
    for (int i = 0; i < TABLE_SIZE; i++) h_table_ref[i] = (float)i * 0.5f;

    float *h_out_dev  = (float *)malloc(N * sizeof(float));
    float *h_out_zero = (float *)malloc(N * sizeof(float));
    float *h_out_um   = (float *)malloc(N * sizeof(float));

    int grid = (N + BLOCK - 1) / BLOCK;

    // ── Variant 1: copy table to device memory ───────────────────────────────
    {
        float *d_table, *d_out;
        cudaMalloc(&d_table, TABLE_SIZE * sizeof(float));
        cudaMalloc(&d_out,   N          * sizeof(float));
        cudaMemcpy(d_table, h_table_ref, TABLE_SIZE * sizeof(float),
                   cudaMemcpyHostToDevice);

        lookup<<<grid, BLOCK>>>(d_table, TABLE_SIZE, d_out, N);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out_dev, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_table); cudaFree(d_out);
    }

    // ── Variant 2: zero-copy — kernel reads table over PCIe (§22.1) ──────────
    {
        float *h_pinned, *d_table_ptr, *d_out;
        // Allocate pinned, device-mapped host memory
        cudaHostAlloc((void **)&h_pinned, TABLE_SIZE * sizeof(float),
                      cudaHostAllocMapped);
        for (int i = 0; i < TABLE_SIZE; i++) h_pinned[i] = h_table_ref[i];

        // Obtain the device-side pointer for the mapped host allocation
        cudaHostGetDevicePointer((void **)&d_table_ptr, h_pinned, 0);

        cudaMalloc(&d_out, N * sizeof(float));

        // Kernel reads table directly from host memory over PCIe — no cudaMemcpy
        lookup<<<grid, BLOCK>>>(d_table_ptr, TABLE_SIZE, d_out, N);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out_zero, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFreeHost(h_pinned); cudaFree(d_out);
    }

    // ── Variant 3: unified memory — table allocated with cudaMallocManaged ───
    {
        float *um_table, *d_out;
        cudaMallocManaged(&um_table, TABLE_SIZE * sizeof(float));
        for (int i = 0; i < TABLE_SIZE; i++) um_table[i] = h_table_ref[i];

        cudaMalloc(&d_out, N * sizeof(float));

        lookup<<<grid, BLOCK>>>(um_table, TABLE_SIZE, d_out, N);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out_um, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(um_table); cudaFree(d_out);
    }

    // Verify all three give the same results
    int fail_zero = 0, fail_um = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_out_dev[i]  - h_out_zero[i]) > 1e-5f) { fail_zero++; break; }
    }
    for (int i = 0; i < N; i++) {
        if (fabsf(h_out_dev[i] - h_out_um[i])   > 1e-5f) { fail_um++;   break; }
    }

    printf("Device-copy vs zero-copy:     %s\n", fail_zero == 0 ? "PASS" : "FAIL");
    printf("Device-copy vs unified mem:   %s\n\n", fail_um   == 0 ? "PASS" : "FAIL");

    printf("Zero-copy notes (§22.1):\n");
    printf("  • cudaHostAlloc(cudaHostAllocMapped) pins and maps host memory\n");
    printf("  • cudaHostGetDevicePointer() returns the device-side VA\n");
    printf("  • With UVA (CUDA 4+) the host and device VAs are identical,\n");
    printf("    so the host pointer can often be passed directly\n");
    printf("  • PCIe bandwidth is ~10%% of device DRAM bandwidth (§22.1):\n");
    printf("    zero-copy suits sparse/occasional access; device memory for dense\n");

    free(h_table_ref); free(h_out_dev); free(h_out_zero); free(h_out_um);
    return 0;
}
