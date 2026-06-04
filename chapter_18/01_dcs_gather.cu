// §18.2  Direct Coulomb Summation — gather kernel with constant memory
//        (Figs 18.3, 18.4, 18.6)
//
// Electrostatic potential at grid point j:
//   energy[j] = Σ_i  charge[i] / sqrt(dx²+dy²+dz²)   (direct Coulomb)
//
// Problem: sequential C code (Fig 18.3) has inner loop over atoms.
//   Parallelising directly maps to scatter (one thread per atom,
//   atomicAdd to grid points) — heavy contention, § 18.2.
//
// Gather strategy (Fig 18.6):
//   One thread per grid point.  Each thread accumulates contributions
//   from *all* atoms in its inner loop.  No write conflicts.
//
// Constant memory for atoms:
//   All threads in a warp iterate the atom loop at the same m,
//   so every warp reads the *same* atom — ideal broadcast pattern.
//   atoms[] is chunked to fit in 64 KB constant memory (CHUNK_SIZE*16 ≤ 64 KB).
//
// Array layout: atoms[4*i .. 4*i+3] = {x, y, z, charge} (Fig 18.3).
//
// Reference: Fig 18.3 (sequential), Fig 18.4 (optimised sequential),
//            Fig 18.5 (scatter — unused), Fig 18.6 (gather kernel).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_DIM   16     // 16×16 = 256 threads/block
#define CHUNK_SIZE  256    // atoms per constant-memory chunk; 256*16 = 4096 B

// Atom data in constant memory: 4 floats per atom {x, y, z, charge}
__constant__ float atoms_c[CHUNK_SIZE * 4];

// ── Gather kernel (Fig 18.6) ──────────────────────────────────────────────────
// Thread (i,j) computes the energy at grid point [j][i] from one chunk of atoms.
// k is the z-slice coordinate (pre-computed by host, passed as a float z value).
__global__ void cenergy_gather(float *energygrid, int grid_x, int grid_y,
                                float gridspacing, float z, int numatoms) {
    int i = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int j = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if (i >= grid_x || j >= grid_y) return;

    int atomarrdim = numatoms * 4;
    float x = gridspacing * (float)i;
    float y = gridspacing * (float)j;
    float energy = 0.0f;

    for (int n = 0; n < atomarrdim; n += 4) {
        float dx = x - atoms_c[n  ];
        float dy = y - atoms_c[n+1];
        float dz = z - atoms_c[n+2];
        energy += atoms_c[n+3] / sqrtf(dx*dx + dy*dy + dz*dz);
    }
    energygrid[grid_x*j + i] += energy;
}

// ── CPU reference (Fig 18.4 — optimised sequential) ──────────────────────────
// Loop interchange: outer atom (n), inner grid (j, then i).
// Hoists dz and dy computations out of inner loops.
static void cenergy_cpu(float *energygrid, int grid_x, int grid_y,
                         float gridspacing, float z,
                         const float *atoms, int numatoms) {
    for (int n = 0; n < numatoms * 4; n += 4) {
        float dz = z - atoms[n+2];
        float dz2 = dz*dz;
        float charge = atoms[n+3];
        for (int j = 0; j < grid_y; j++) {
            float dy = gridspacing*(float)j - atoms[n+1];
            float dy2 = dy*dy;
            for (int i = 0; i < grid_x; i++) {
                float dx = gridspacing*(float)i - atoms[n];
                energygrid[grid_x*j + i] += charge / sqrtf(dx*dx + dy2 + dz2);
            }
        }
    }
}

static float max_rel_err(const float *a, const float *b, int n) {
    float mx=0.f;
    for (int i=0;i<n;i++){float e=fabsf(a[i]-b[i])/(1.f+fabsf(b[i]));if(e>mx)mx=e;}
    return mx;
}

int main(void) {
    printf("=== Direct Coulomb Summation: Gather Kernel (§18.2, Fig 18.6) ===\n\n");

    int grid_x = 64, grid_y = 64;
    int numatoms = 1024;
    float gridspacing = 0.5f;   // Angstroms
    float z = 0.5f;             // z-coordinate of this 2-D slice

    int nGrid = grid_x * grid_y;
    size_t szGrid = nGrid * sizeof(float);
    size_t szAtoms = numatoms * 4 * sizeof(float);

    float *h_atoms  = (float *)malloc(szAtoms);
    float *h_energy_cpu = (float *)calloc(nGrid, sizeof(float));
    float *h_energy_gpu = (float *)calloc(nGrid, sizeof(float));

    srand(42);
    for (int i = 0; i < numatoms * 4; i += 4) {
        h_atoms[i  ] = (rand()/(float)RAND_MAX) * grid_x * gridspacing;  // x
        h_atoms[i+1] = (rand()/(float)RAND_MAX) * grid_y * gridspacing;  // y
        h_atoms[i+2] = (rand()/(float)RAND_MAX) * 2.f;                   // z
        h_atoms[i+3] = (rand()/(float)RAND_MAX) * 2.f - 1.f;             // charge
    }

    cenergy_cpu(h_energy_cpu, grid_x, grid_y, gridspacing, z, h_atoms, numatoms);

    // GPU setup
    float *d_energy;
    cudaMalloc(&d_energy, szGrid);
    cudaMemset(d_energy, 0, szGrid);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((grid_x+BLOCK_DIM-1)/BLOCK_DIM, (grid_y+BLOCK_DIM-1)/BLOCK_DIM);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    // Chunked constant-memory loop (analogous to Fig 17.12 for k-space)
    for (int chunk = 0; chunk * CHUNK_SIZE < numatoms; chunk++) {
        int offset   = chunk * CHUNK_SIZE;
        int actual   = (numatoms - offset < CHUNK_SIZE) ? numatoms - offset : CHUNK_SIZE;
        cudaMemcpyToSymbol(atoms_c, &h_atoms[offset * 4],
                           actual * 4 * sizeof(float), 0, cudaMemcpyHostToDevice);
        cenergy_gather<<<grid, block>>>(d_energy, grid_x, grid_y,
                                        gridspacing, z, actual);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(h_energy_gpu, d_energy, szGrid, cudaMemcpyDeviceToHost);

    float err = max_rel_err(h_energy_gpu, h_energy_cpu, nGrid);
    printf("Grid %d×%d, %d atoms, z=%.1f Å, gridspacing=%.2f Å\n",
           grid_x, grid_y, numatoms, z, gridspacing);
    printf("GPU time: %.3f ms\n", ms);
    printf("Max rel error: %.2e  %s\n", err, err < 1e-4f ? "PASS" : "FAIL");

    printf("\nGather DCS design (§18.2):\n");
    printf("  + One thread per grid point — no atomics\n");
    printf("  + atoms_c[] in constant memory: broadcast access (all threads share atom)\n");
    printf("  + 9 FLOP per atom: dx,dy,dz,dx²,dy²,dz²,sum,rsqrt,*charge\n");
    printf("  + Constant cache nearly eliminates DRAM traffic for atoms[]\n");
    printf("  − 4 constant-memory accesses per iteration vs 4 FLOP per grid op\n");
    printf("  → Thread coarsening (file 02) reduces memory pressure\n");

    free(h_atoms); free(h_energy_cpu); free(h_energy_gpu);
    cudaFree(d_energy);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
