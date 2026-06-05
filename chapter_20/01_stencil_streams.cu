// §20.5  Overlapping computation and communication
//
// Chapter 20's central CUDA concept: use two streams to overlap the
// computation of interior grid points with the asynchronous device-to-host
// transfer of boundary data that would be sent to neighbouring MPI ranks.
//
// This file runs entirely on one GPU (no MPI required).  It simulates one
// compute node's partition of a 3-D heat-transfer stencil.
//
// Two-stage strategy (Fig. 20.12):
//
//   Stage 1 (stream 0):  compute HALO boundary slices on each side.
//                        These are the slices that neighbours need as halo
//                        cells in the next iteration.
//
//   Stage 2 runs TWO activities in parallel:
//     stream 1:  compute the interior slices (the bulk of the work)
//     stream 0:  async D→H copy of the freshly computed boundary slices
//                (the future MPI send buffer)
//
//   After cudaStreamSynchronize(stream0):
//     host:      simulate MPI_Sendrecv — on a real cluster this is a
//                blocking network transfer; here it is a host memcpy
//     stream 0:  async H→D copy of received halo data
//
//   cudaDeviceSynchronize() then swap d_in ↔ d_out for next iteration.
//
// Key APIs demonstrated:
//   cudaHostAlloc(cudaHostAllocDefault)   — pinned memory for async copies
//   cudaMemcpyAsync                       — non-blocking PCIe transfer
//   cudaStreamSynchronize / cudaDeviceSynchronize
//
// Note: the book uses a 25-point stencil with HALO=4 (four neighbour slices
// in each direction).  This file uses a 7-point stencil with HALO=1 so the
// compute kernel stays short.  The stream-overlap pattern is identical.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// ── Grid dimensions ───────────────────────────────────────────────────────────
#define DIMX  64      // x extent of one node's partition
#define DIMY  64      // y extent of one node's partition
#define DIMZ  512     // z extent of one node's partition (excluding halos)
#define HALO  1       // halo slices on each side (book §20.5 uses HALO=4)
#define NITER 20      // Jacobi iterations

#define NZ_TOT (DIMZ + 2*HALO)   // total z including halos

// ── Block size ────────────────────────────────────────────────────────────────
#define BX 8
#define BY 8
#define BZ 4

// ── 7-point stencil kernel ────────────────────────────────────────────────────
// Processes z-slices in [z_start, z_end).
// Grid layout: x innermost (stride 1), then y (stride nx), then z (stride nx*ny).
__global__ void stencil7(const float * __restrict__ in,
                                float * __restrict__ out,
                          int nx, int ny, int nz,
                          int z_start, int z_end)
{
    int ix = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int iy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int iz = z_start + (int)(blockIdx.z * blockDim.z + threadIdx.z);

    if (ix < 1 || ix >= nx-1) return;
    if (iy < 1 || iy >= ny-1) return;
    if (iz < z_start || iz >= z_end) return;
    if (iz < 1 || iz >= nz-1) return;

    int s  = nx * ny;
    int id = iz*s + iy*nx + ix;

    out[id] = 0.25f * in[id]
            + 0.125f * (in[id - 1]  + in[id + 1]    // ±x
                      + in[id - nx] + in[id + nx]    // ±y
                      + in[id - s]  + in[id + s]);   // ±z
}

static void launch_stencil(float *d_out, const float *d_in,
                            int nx, int ny, int nz,
                            int z_start, int z_end,
                            cudaStream_t stream)
{
    int nz_range = z_end - z_start;
    if (nz_range <= 0) return;
    dim3 block(BX, BY, BZ);
    dim3 grid((nx + BX-1)/BX, (ny + BY-1)/BY, (nz_range + BZ-1)/BZ);
    stencil7<<<grid, block, 0, stream>>>(d_in, d_out, nx, ny, nz, z_start, z_end);
}

static double elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms; cudaEventElapsedTime(&ms, a, b); return (double)ms;
}

int main(void)
{
    printf("=== §20.5  Overlapping Computation with Communication ===\n\n");
    printf("Partition: %dx%dx%d  HALO=%d  NITER=%d\n\n",
           DIMX, DIMY, DIMZ, HALO, NITER);

    const int    nx = DIMX, ny = DIMY, nz = NZ_TOT;
    const size_t vol       = (size_t)nx * ny * nz;
    const size_t vol_bytes = vol * sizeof(float);

    // Each halo region: HALO contiguous z-slices
    const int    halo_pts   = HALO * nx * ny;
    const size_t halo_bytes = halo_pts * sizeof(float);

    // ── Device buffers (double-buffered for Jacobi) ───────────────────────────
    float *d_in, *d_out;
    cudaMalloc(&d_in,  vol_bytes);
    cudaMalloc(&d_out, vol_bytes);

    // Initialise to simple ramp
    {
        float *h_tmp = (float *)malloc(vol_bytes);
        for (size_t i = 0; i < vol; i++) h_tmp[i] = (float)(i % 1000) * 0.001f;
        cudaMemcpy(d_in,  h_tmp, vol_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out, h_tmp, vol_bytes, cudaMemcpyHostToDevice);
        free(h_tmp);
    }

    // ── Pinned host bounce buffers (§20.5, Fig. 20.11, lines 21–24) ──────────
    // Required so cudaMemcpyAsync is genuinely non-blocking.
    float *h_left_bnd,  *h_right_bnd;   // boundary → send to neighbours
    float *h_left_halo, *h_right_halo;  // received halo from neighbours
    cudaHostAlloc(&h_left_bnd,   halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_right_bnd,  halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_left_halo,  halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_right_halo, halo_bytes, cudaHostAllocDefault);
    memset(h_left_halo,  0, halo_bytes);
    memset(h_right_halo, 0, halo_bytes);

    // ── Z-slice offsets (Fig. 20.14) ─────────────────────────────────────────
    //  [  left halo  |  left bnd  |  interior  |  right bnd  |  right halo  ]
    //  0          HALO         2*HALO      nz-2*HALO       nz-HALO         nz
    const int left_halo_z  = 0;
    const int left_bnd_z   = HALO;
    const int interior_z   = 2 * HALO;
    const int right_bnd_z  = nz - 2 * HALO;
    const int right_halo_z = nz - HALO;

    cudaStream_t s0, s1;
    cudaStreamCreate(&s0);
    cudaStreamCreate(&s1);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);

    // ═════════════════════════════════════════════════════════════════════════
    // 1. SERIAL baseline
    //    All slices computed, then synchronous D↔H, then synchronous H↔D.
    // ═════════════════════════════════════════════════════════════════════════
    cudaEventRecord(ev0);
    for (int it = 0; it < NITER; it++) {
        // Compute entire partition in default stream
        launch_stencil(d_out, d_in, nx, ny, nz, HALO, nz-HALO, 0);
        cudaDeviceSynchronize();

        // D→H boundary (synchronous)
        cudaMemcpy(h_left_bnd,
                   d_out + (size_t)left_bnd_z * nx*ny,
                   halo_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_right_bnd,
                   d_out + (size_t)right_bnd_z * nx*ny,
                   halo_bytes, cudaMemcpyDeviceToHost);

        // Simulate MPI_Sendrecv (network exchange)
        memcpy(h_left_halo,  h_left_bnd,  halo_bytes);
        memcpy(h_right_halo, h_right_bnd, halo_bytes);

        // H→D halo (synchronous)
        cudaMemcpy(d_out + (size_t)left_halo_z * nx*ny,
                   h_left_halo,  halo_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out + (size_t)right_halo_z * nx*ny,
                   h_right_halo, halo_bytes, cudaMemcpyHostToDevice);

        float *tmp = d_in; d_in = d_out; d_out = tmp;
    }
    cudaEventRecord(ev1);
    cudaDeviceSynchronize();
    double t_serial = elapsed_ms(ev0, ev1);

    // ═════════════════════════════════════════════════════════════════════════
    // 2. OVERLAPPED  §20.5 two-stage strategy (Fig. 20.12 / Fig. 20.13–20.15)
    // ═════════════════════════════════════════════════════════════════════════
    // Reset
    {
        float *h_tmp = (float *)malloc(vol_bytes);
        for (size_t i = 0; i < vol; i++) h_tmp[i] = (float)(i % 1000) * 0.001f;
        cudaMemcpy(d_in,  h_tmp, vol_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out, h_tmp, vol_bytes, cudaMemcpyHostToDevice);
        free(h_tmp);
    }

    cudaEventRecord(ev0);
    for (int it = 0; it < NITER; it++) {
        // ── Stage 1  (Fig. 20.13, lines 39–40) ───────────────────────────────
        // Compute boundary slices in stream 0 first; neighbours need these.
        launch_stencil(d_out, d_in, nx, ny, nz,
                       left_bnd_z, interior_z, s0);    // left boundary
        launch_stencil(d_out, d_in, nx, ny, nz,
                       right_bnd_z, right_halo_z, s0); // right boundary

        // ── Stage 2  (Fig. 20.13, line 41 + Fig. 20.15, lines 42–43) ─────────
        // Interior kernel in stream 1 runs concurrently with D→H in stream 0.
        launch_stencil(d_out, d_in, nx, ny, nz,
                       interior_z, right_bnd_z, s1);   // interior (stream 1)

        // Async D→H of boundary data — stream 0 (Fig. 20.15, lines 42–43)
        cudaMemcpyAsync(h_left_bnd,
                        d_out + (size_t)left_bnd_z * nx*ny,
                        halo_bytes, cudaMemcpyDeviceToHost, s0);
        cudaMemcpyAsync(h_right_bnd,
                        d_out + (size_t)right_bnd_z * nx*ny,
                        halo_bytes, cudaMemcpyDeviceToHost, s0);

        // Wait for D→H before MPI exchange (Fig. 20.15, line 44)
        cudaStreamSynchronize(s0);

        // Simulate MPI_Sendrecv (Fig. 20.15, lines 45–46)
        memcpy(h_left_halo,  h_left_bnd,  halo_bytes);
        memcpy(h_right_halo, h_right_bnd, halo_bytes);

        // Async H→D of received halos — stream 0 (Fig. 20.15, lines 47–48)
        cudaMemcpyAsync(d_out + (size_t)left_halo_z * nx*ny,
                        h_left_halo,  halo_bytes, cudaMemcpyHostToDevice, s0);
        cudaMemcpyAsync(d_out + (size_t)right_halo_z * nx*ny,
                        h_right_halo, halo_bytes, cudaMemcpyHostToDevice, s0);

        // Wait for all device activity (Fig. 20.15, line 49)
        cudaDeviceSynchronize();

        // Swap double buffers (Fig. 20.15, lines 50–51)
        float *tmp = d_in; d_in = d_out; d_out = tmp;
    }
    cudaEventRecord(ev1);
    cudaDeviceSynchronize();
    double t_overlap = elapsed_ms(ev0, ev1);

    // ── Results ───────────────────────────────────────────────────────────────
    printf("Serial (sync D↔H after compute):   %7.2f ms  (%5.3f ms/iter)\n",
           t_serial,  t_serial  / NITER);
    printf("Overlapped §20.5 (two streams):    %7.2f ms  (%5.3f ms/iter)"
           "  %.2fx\n\n",
           t_overlap, t_overlap / NITER, t_serial / t_overlap);

    printf("Timeline per iteration (overlapped):\n");
    printf("  stream 0: [left bnd kernel] [right bnd kernel] [D→H bnd async]"
           " [H→D halo async]\n");
    printf("  stream 1:                                       [interior kernel]\n");
    printf("  host:                                           (after stream0 sync)"
           " MPI_Sendrecv\n\n");

    printf("Notes (§20.5):\n");
    printf("  • cudaHostAlloc (pinned) is required for cudaMemcpyAsync\n");
    printf("  • Interior kernel in stream 1 overlaps with D→H in stream 0\n");
    printf("  • On a real cluster the host memcpy models MPI_Sendrecv latency\n");
    printf("  • Speedup grows with partition z-depth (larger interior = "
           "more to hide)\n");
    printf("  • Book §20.5 uses HALO=4 (25-pt stencil); this demo uses HALO=%d\n",
           HALO);

    cudaStreamDestroy(s0); cudaStreamDestroy(s1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d_in); cudaFree(d_out);
    cudaFreeHost(h_left_bnd);  cudaFreeHost(h_right_bnd);
    cudaFreeHost(h_left_halo); cudaFreeHost(h_right_halo);
    return 0;
}
