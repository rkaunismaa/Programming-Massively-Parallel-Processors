// §20.3–20.7  MPI + CUDA stencil — heterogeneous cluster
//
// Full implementation of the running example from Chapter 20 (Figs. 20.6,
// 20.9–20.19).  The problem is a 3-D heat-transfer Jacobi iteration over a
// rectangular grid partitioned along the z dimension across MPI ranks.
//
// Process layout (Fig. 20.6):
//   ranks 0 .. np-2  compute processes  (one GPU each)
//   rank  np-1        data server        (CPU only, no GPU required)
//
// Compile:
//   mpicxx -x cu -arch=sm_89 -O2 -o mpi_cuda_stencil 02_mpi_cuda_stencil.cu \
//          -lcudart -I/usr/local/cuda/include -L/usr/local/cuda/lib64
//   OR use the Makefile target:  make SM_ARCH=sm_89 mpi_cuda_stencil
//
// Run (minimum 3 MPI ranks: at least 2 compute + 1 data server):
//   mpirun -np 5 ./mpi_cuda_stencil
//
// §20.7 CUDA-aware MPI:
//   Build with -DCUDA_AWARE_MPI and use an MPI library that supports
//   device-pointer arguments (MVAPICH2, IBM Platform MPI, Open MPI 4+).
//   This removes the need for host bounce buffers and async H↔D copies.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>

// ── Grid dimensions ───────────────────────────────────────────────────────────
// Book uses dimx=480+pad, dimy=480, dimz=400.  We use a smaller default.
#define DIMX   64
#define DIMY   64
#define DIMZ   256    // total z slices (distributed across compute ranks)
#define NREPS  10     // Jacobi iterations

// Halo depth: 1 slice for a 7-pt stencil; book uses 4 for 25-pt stencil.
#define HALO   1

// ── Stencil block size ────────────────────────────────────────────────────────
#define BX 8
#define BY 8
#define BZ 4

// ─────────────────────────────────────────────────────────────────────────────
// Stencil kernel — 7-point, processes z in [z_start, z_end)
// ─────────────────────────────────────────────────────────────────────────────
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
            + 0.125f * (in[id-1] + in[id+1]
                      + in[id-nx] + in[id+nx]
                      + in[id-s]  + in[id+s]);
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

// ─────────────────────────────────────────────────────────────────────────────
// Data server (rank np-1)  — Figs. 20.9 and 20.18
// ─────────────────────────────────────────────────────────────────────────────
static void data_server(int dimx, int dimy, int dimz, int nreps)
{
    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    int num_comp = np - 1;         // compute ranks
    int num_pts  = dimx * dimy * dimz;

    // Allocate and initialise input data
    float *input  = (float *)malloc(num_pts * sizeof(float));
    float *output = (float *)malloc(num_pts * sizeof(float));
    if (!input || !output) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    for (int i = 0; i < num_pts; i++) input[i] = (float)(i % 1000) * 0.001f;

    // Number of z-slices per compute rank
    int slices_per = dimz / num_comp;

    // Points per rank including HALO on each side
    // Edge ranks need only one side of halo; for simplicity send the same
    // amount to all ranks (edge ranks ignore the unused halo region).
    int pts_edge     = dimx * dimy * (slices_per + HALO);      // edge rank
    int pts_internal = dimx * dimy * (slices_per + 2 * HALO);  // internal rank

    // Send partitions (Fig. 20.9, lines 18–23)
    float *send_ptr = input;
    for (int r = 0; r < num_comp; r++) {
        int send_count = (r == 0 || r == num_comp-1) ? pts_edge : pts_internal;
        // For internal ranks, pull back by HALO slices so they receive the
        // left neighbour's boundary as their left halo.
        float *adjusted = (r == 0) ? send_ptr : send_ptr - HALO * dimx * dimy;
        MPI_Send(adjusted, send_count, MPI_FLOAT, r, 0, MPI_COMM_WORLD);
        send_ptr += slices_per * dimx * dimy;
    }

    // Wait for all compute ranks to finish (Fig. 20.18, line 24)
    MPI_Barrier(MPI_COMM_WORLD);

    // Collect results (Fig. 20.18, lines 26–27)
    MPI_Status status;
    float *recv_ptr = output;
    for (int r = 0; r < num_comp; r++) {
        MPI_Recv(recv_ptr, slices_per * dimx * dimy,
                 MPI_FLOAT, r, 0, MPI_COMM_WORLD, &status);
        recv_ptr += slices_per * dimx * dimy;
    }

    // Verify: all output values should be finite
    int bad = 0;
    for (int i = 0; i < num_pts; i++)
        if (!isfinite(output[i])) bad++;
    printf("[data server] Output collected: %s  (%d bad values)\n",
           bad == 0 ? "PASS" : "FAIL", bad);

    free(input); free(output);
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute process (ranks 0..np-2)  — Figs. 20.10–20.17 and 20.19
// ─────────────────────────────────────────────────────────────────────────────
static void compute_process(int dimx, int dimy, int dimz_per, int nreps)
{
    int pid, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int num_comp     = np - 1;
    int server       = np - 1;
    int is_edge_left  = (pid == 0);
    int is_edge_right = (pid == num_comp - 1);

    // Total z including halos on both sides
    int nz_tot = dimz_per + 2 * HALO;

    int    num_pts  = dimx * dimy * nz_tot;
    size_t num_bytes = (size_t)num_pts * sizeof(float);
    int    halo_pts  = HALO * dimx * dimy;
    size_t halo_bytes = (size_t)halo_pts * sizeof(float);

    // ── Allocate host input buffer (Fig. 20.10, lines 10) ────────────────────
    float *h_input = (float *)malloc(num_bytes);
    if (!h_input) { MPI_Abort(MPI_COMM_WORLD, 1); return; }

    // Receive address: edge-left process skips left halo area in host buffer
    // (data server sends no left halo for rank 0)
    float *recv_addr = h_input + (is_edge_left ? halo_pts : 0);
    int    recv_count = num_pts - (is_edge_left ? halo_pts : 0)
                                - (is_edge_right ? halo_pts : 0);

    MPI_Status status;
    MPI_Recv(recv_addr, recv_count, MPI_FLOAT, server, 0, MPI_COMM_WORLD, &status);

    // Zero the unused halo areas at the edges
    if (is_edge_left)  memset(h_input, 0, halo_bytes);
    if (is_edge_right) memset(h_input + num_pts - halo_pts, 0, halo_bytes);

    // ── Device memory (Fig. 20.10, lines 11–12) ───────────────────────────────
    float *d_input = NULL, *d_output = NULL;
    cudaMalloc(&d_input,  num_bytes);
    cudaMalloc(&d_output, num_bytes);
    cudaMemcpy(d_input, h_input, num_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, num_bytes);

    // ── Host output buffer ────────────────────────────────────────────────────
    float *h_output = (float *)malloc(num_bytes);

    // ── Pinned bounce buffers for halo exchange (Fig. 20.11, lines 21–24) ─────
    float *h_left_bnd  = NULL, *h_right_bnd  = NULL;  // outgoing boundaries
    float *h_left_halo = NULL, *h_right_halo = NULL;  // incoming halos
    cudaHostAlloc(&h_left_bnd,   halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_right_bnd,  halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_left_halo,  halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_right_halo, halo_bytes, cudaHostAllocDefault);

    // ── CUDA streams (Fig. 20.11, lines 25–27) ───────────────────────────────
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    // ── Neighbour ranks (Fig. 20.13, lines 29–30) ─────────────────────────────
    int left_nbr  = is_edge_left  ? MPI_PROC_NULL : pid - 1;
    int right_nbr = is_edge_right ? MPI_PROC_NULL : pid + 1;

    // ── Z-slice offsets (Fig. 20.14) ──────────────────────────────────────────
    //  [left halo | left bnd | interior | right bnd | right halo]
    const int left_halo_z  = 0;
    const int left_bnd_z   = HALO;
    const int interior_z   = 2 * HALO;
    const int right_bnd_z  = nz_tot - 2 * HALO;
    const int right_halo_z = nz_tot - HALO;

    // ── MPI_Barrier: all ranks ready (Fig. 20.13, line 37) ───────────────────
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Iteration loop (Fig. 20.13, line 38) ──────────────────────────────────
    for (int rep = 0; rep < nreps; rep++) {

        // Stage 1: compute boundary slices in stream 0 (lines 39–40)
        launch_stencil(d_output, d_input, dimx, dimy, nz_tot,
                       left_bnd_z,  interior_z,   stream0);  // left boundary
        launch_stencil(d_output, d_input, dimx, dimy, nz_tot,
                       right_bnd_z, right_halo_z, stream0);  // right boundary

        // Stage 2a: interior in stream 1 (line 41) — concurrent with below
        launch_stencil(d_output, d_input, dimx, dimy, nz_tot,
                       interior_z, right_bnd_z, stream1);    // interior

#ifndef CUDA_AWARE_MPI
        // Stage 2b: async D→H of boundary data (lines 42–43)
        cudaMemcpyAsync(h_left_bnd,
                        d_output + (size_t)left_bnd_z * dimx*dimy,
                        halo_bytes, cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(h_right_bnd,
                        d_output + (size_t)right_bnd_z * dimx*dimy,
                        halo_bytes, cudaMemcpyDeviceToHost, stream0);

        // Wait for D→H before MPI (line 44)
        cudaStreamSynchronize(stream0);

        // Halo exchange: send right boundary → left neighbour's halo,
        //                send left boundary  → right neighbour's halo
        // (lines 45–46, using MPI_Sendrecv Fig. 20.16)
        MPI_Sendrecv(h_right_bnd, halo_pts, MPI_FLOAT, right_nbr, rep,
                     h_left_halo, halo_pts, MPI_FLOAT, left_nbr,  rep,
                     MPI_COMM_WORLD, &status);
        MPI_Sendrecv(h_left_bnd,  halo_pts, MPI_FLOAT, left_nbr,  rep,
                     h_right_halo,halo_pts, MPI_FLOAT, right_nbr, rep,
                     MPI_COMM_WORLD, &status);

        // H→D received halos (lines 47–48)
        cudaMemcpyAsync(d_output + (size_t)left_halo_z  * dimx*dimy,
                        h_left_halo,  halo_bytes, cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_output + (size_t)right_halo_z * dimx*dimy,
                        h_right_halo, halo_bytes, cudaMemcpyHostToDevice, stream0);
#else
        // §20.7 CUDA-aware MPI: pass device pointers directly (Fig. 20.19)
        // No host bounce buffers or async copies needed.
        cudaStreamSynchronize(stream0);   // still need boundary kernels done

        MPI_Sendrecv(d_output + (size_t)right_bnd_z  * dimx*dimy,
                     halo_pts, MPI_FLOAT, right_nbr, rep,
                     d_output + (size_t)left_halo_z  * dimx*dimy,
                     halo_pts, MPI_FLOAT, left_nbr,  rep,
                     MPI_COMM_WORLD, &status);
        MPI_Sendrecv(d_output + (size_t)left_bnd_z   * dimx*dimy,
                     halo_pts, MPI_FLOAT, left_nbr,  rep,
                     d_output + (size_t)right_halo_z * dimx*dimy,
                     halo_pts, MPI_FLOAT, right_nbr, rep,
                     MPI_COMM_WORLD, &status);
#endif

        // Wait for all device work (line 49)
        cudaDeviceSynchronize();

        // Swap double buffers (lines 50–51)
        float *tmp = d_input; d_input = d_output; d_output = tmp;
    }

    // ── Send results back to data server (Fig. 20.17, lines 57–58) ───────────
    cudaMemcpy(h_output, d_input, num_bytes, cudaMemcpyDeviceToHost);

    // Barrier: wait for all compute ranks (Fig. 20.17, line 53)
    MPI_Barrier(MPI_COMM_WORLD);

    // Send interior (no halo) portion to data server
    MPI_Send(h_output + (size_t)left_bnd_z * dimx * dimy,
             dimz_per * dimx * dimy, MPI_FLOAT,
             server, 0, MPI_COMM_WORLD);

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaStreamDestroy(stream0); cudaStreamDestroy(stream1);
    cudaFree(d_input); cudaFree(d_output);
    cudaFreeHost(h_left_bnd);  cudaFreeHost(h_right_bnd);
    cudaFreeHost(h_left_halo); cudaFreeHost(h_right_halo);
    free(h_input); free(h_output);
}

// ─────────────────────────────────────────────────────────────────────────────
// main  (Fig. 20.6)
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int pid, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (np < 3) {
        if (pid == 0)
            printf("Need at least 3 MPI ranks "
                   "(2 compute + 1 data server).\n"
                   "Run: mpirun -np 5 ./mpi_cuda_stencil\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int num_comp = np - 1;
    if (DIMZ % num_comp != 0 && pid == 0)
        printf("Warning: DIMZ=%d not evenly divisible by %d compute ranks.\n",
               DIMZ, num_comp);

    if (pid == 0)
        printf("=== §20.3-20.7  MPI+CUDA Stencil ===\n"
               "Grid: %dx%dx%d  HALO=%d  NREPS=%d  %d compute ranks\n\n",
               DIMX, DIMY, DIMZ, HALO, NREPS, num_comp);

    if (pid < num_comp)
        compute_process(DIMX, DIMY, DIMZ / num_comp, NREPS);
    else
        data_server(DIMX, DIMY, DIMZ, NREPS);

    MPI_Finalize();
    return 0;
}
