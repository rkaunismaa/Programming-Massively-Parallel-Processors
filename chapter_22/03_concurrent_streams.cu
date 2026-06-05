// §22.2  Kernel execution control — simultaneous execution of multiple grids
//
// Early CUDA systems executed at most one grid at a time.  Fermi and later
// GPUs can execute multiple grids from the same application simultaneously
// (§22.2).  CUDA streams are the programmer-facing mechanism:
//
//   • Kernels in the same stream execute in issue order (serialised).
//   • Kernels in different streams may overlap, subject to resource limits.
//
// §22.2 (HPC cluster context) also discusses pipelining: by issuing
// H→D copy, compute, and D→H copy for successive data segments into
// separate streams, the three operations can overlap across segments.
// This is the same technique described in Chapter 20 (§20.5) for
// overlapping MPI communication with GPU computation.
//
// This file demonstrates three patterns for N_SEGS independent work segments:
//
//   1. Serial        — all kernels in the default stream (guaranteed serial).
//   2. Concurrent    — each kernel in its own stream (may overlap).
//   3. Pipelined     — H→D + compute + D→H per segment, all in their own
//                      streams, so the three stages overlap across segments.
//
// Pinned (page-locked) host memory is required for cudaMemcpyAsync to be
// truly asynchronous with respect to host execution.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N_PER_SEG  (1 << 22)    // 4 M floats per segment (~16 MB)
#define BLOCK      256
#define ITERS      256          // arithmetic iterations per element
#define N_SEGS     4            // number of independent work segments

__global__ void compute(float *data, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = data[i];
    // Enough compute to make execution time measurable
    for (int it = 0; it < iters; it++)
        v = v * 1.0001f + 0.0001f;
    data[i] = v;
}

static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return (double)ms;
}

int main(void) {
    printf("=== Concurrent Streams (§22.2) ===\n\n");
    printf("N_SEGS=%d  N_PER_SEG=%d  ITERS=%d\n\n", N_SEGS, N_PER_SEG, ITERS);

    // Pinned host memory — required for async memcpy
    float *h_data;
    cudaMallocHost(&h_data, (size_t)N_SEGS * N_PER_SEG * sizeof(float));
    for (int i = 0; i < N_SEGS * N_PER_SEG; i++) h_data[i] = 1.0f;

    float *d_data[N_SEGS];
    for (int s = 0; s < N_SEGS; s++)
        cudaMalloc(&d_data[s], N_PER_SEG * sizeof(float));

    cudaStream_t streams[N_SEGS];
    for (int s = 0; s < N_SEGS; s++) cudaStreamCreate(&streams[s]);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start); cudaEventCreate(&ev_stop);

    int grid = (N_PER_SEG + BLOCK - 1) / BLOCK;

    // ── 1. Serial — default stream ────────────────────────────────────────────
    for (int s = 0; s < N_SEGS; s++)
        cudaMemcpy(d_data[s], h_data + (size_t)s*N_PER_SEG,
                   N_PER_SEG*sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(ev_start);
    for (int s = 0; s < N_SEGS; s++)
        compute<<<grid, BLOCK>>>(d_data[s], N_PER_SEG, ITERS);
    cudaEventRecord(ev_stop);
    cudaDeviceSynchronize();
    double t_serial = elapsed_ms(ev_start, ev_stop);
    printf("1. Serial (default stream, %d kernels):          %7.2f ms\n",
           N_SEGS, t_serial);

    // ── 2. Concurrent — each kernel in its own stream ────────────────────────
    for (int s = 0; s < N_SEGS; s++)
        cudaMemcpy(d_data[s], h_data + (size_t)s*N_PER_SEG,
                   N_PER_SEG*sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(ev_start);
    for (int s = 0; s < N_SEGS; s++)
        compute<<<grid, BLOCK, 0, streams[s]>>>(d_data[s], N_PER_SEG, ITERS);
    cudaEventRecord(ev_stop);
    cudaDeviceSynchronize();
    double t_concurrent = elapsed_ms(ev_start, ev_stop);
    printf("2. Concurrent (%d streams, compute only):        %7.2f ms  (%4.1fx)\n",
           N_SEGS, t_concurrent, t_serial / t_concurrent);

    // ── 3a. Sequential H→D + compute + D→H (baseline for pipeline) ───────────
    // Reset h_data so the H→D source is clean
    for (int i = 0; i < N_SEGS * N_PER_SEG; i++) h_data[i] = 1.0f;

    cudaEventRecord(ev_start);
    for (int s = 0; s < N_SEGS; s++) {
        cudaMemcpy(d_data[s], h_data + (size_t)s*N_PER_SEG,
                   N_PER_SEG*sizeof(float), cudaMemcpyHostToDevice);
        compute<<<grid, BLOCK>>>(d_data[s], N_PER_SEG, ITERS);
        cudaMemcpy(h_data + (size_t)s*N_PER_SEG, d_data[s],
                   N_PER_SEG*sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(ev_stop);
    cudaDeviceSynchronize();
    double t_seq_pipe = elapsed_ms(ev_start, ev_stop);
    printf("3a. Sequential H→D+compute+D→H (%d segs):       %7.2f ms\n",
           N_SEGS, t_seq_pipe);

    // ── 3b. Pipelined — H→D + compute + D→H overlapped across segments ───────
    // Reset h_data to original values (overwritten by 3a D→H)
    for (int i = 0; i < N_SEGS * N_PER_SEG; i++) h_data[i] = 1.0f;

    // Ideal timeline (3 stages, N_SEGS segments):
    //   t0: H→D(0)
    //   t1: H→D(1)  | compute(0)
    //   t2: H→D(2)  | compute(1)  | D→H(0)
    //   t3: H→D(3)  | compute(2)  | D→H(1)
    //   t4:           compute(3)  | D→H(2)
    //   t5:                         D→H(3)
    // Wall time ≈ max(N_SEGS*H2D, N_SEGS*compute, N_SEGS*D2H)
    //           + 2 * max(H2D, compute, D2H)   [pipeline ramp]
    cudaEventRecord(ev_start);
    for (int s = 0; s < N_SEGS; s++) {
        float *h_seg = h_data + (size_t)s * N_PER_SEG;
        cudaMemcpyAsync(d_data[s], h_seg, N_PER_SEG*sizeof(float),
                        cudaMemcpyHostToDevice, streams[s]);
        compute<<<grid, BLOCK, 0, streams[s]>>>(d_data[s], N_PER_SEG, ITERS);
        cudaMemcpyAsync(h_seg, d_data[s], N_PER_SEG*sizeof(float),
                        cudaMemcpyDeviceToHost, streams[s]);
    }
    cudaEventRecord(ev_stop);
    cudaDeviceSynchronize();
    double t_pipeline = elapsed_ms(ev_start, ev_stop);
    printf("3b. Pipelined H→D+compute+D→H (%d streams):     %7.2f ms  (%4.1fx)\n",
           N_SEGS, t_pipeline, t_seq_pipe / t_pipeline);

    printf("\nNotes (§22.2):\n");
    printf("  • Kernels in different streams may execute simultaneously\n");
    printf("    (Fermi architecture and later) subject to device resources\n");
    printf("  • cudaMemcpyAsync requires pinned (page-locked) host memory\n");
    printf("  • Pipeline overlap is maximised when H→D, compute, and D→H\n");
    printf("    have similar durations across segments\n");
    printf("  • Speedup over serial depends on how much the device can\n");
    printf("    fill from multiple concurrent grids simultaneously\n");

    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    for (int s = 0; s < N_SEGS; s++) {
        cudaStreamDestroy(streams[s]);
        cudaFree(d_data[s]);
    }
    cudaFreeHost(h_data);
    return 0;
}
