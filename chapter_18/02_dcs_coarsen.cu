// §18.3  Direct Coulomb Summation — thread coarsening (Fig 18.8)
//
// Problem with the basic gather kernel (Fig 18.6):
//   Each thread processes one grid point and reads 4 constant-memory
//   values per atom (x, y, z, charge).  For COARSEN_FACTOR=1 this is
//   4 constant-memory accesses per atom regardless of how the grid
//   points are arranged.
//
// Thread coarsening idea (§18.3, Fig 18.7):
//   All grid points along the same row (same j) have *identical* y and z
//   distances to every atom.  Instead of computing dy and dz redundantly
//   for each grid point in a row, each thread handles COARSEN_FACTOR
//   adjacent grid points in the x direction and amortises the dy+dz²
//   computation across them.
//
//   Additionally, COARSEN_FACTOR=4 means each thread reads the atom
//   information once but produces 4 output energy values, reducing the
//   number of constant-memory accesses from 4 per grid point to 1 per
//   grid point (4 grid points / 4 constant-memory accesses = 1:1 instead
//   of 4:1).
//
// Grid layout change (Fig 18.8, lines 05–06):
//   i = blockIdx.x * blockDim.x * COARSEN_FACTOR + threadIdx.x
//   Each adjacent thread handles blockDim.x-spaced grid points
//   (set up for coalescing in the next optimisation).
//   However, the dx values for the 4 grid points simply add gridspacing
//   (lines 17–20 of Fig 18.8) — simple arithmetic offsets.
//
// Reference: Fig 18.7 (illustration), Fig 18.8 (coarsening kernel).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_DIM       16
#define COARSEN_FACTOR   4
#define CHUNK_SIZE      256

__constant__ float atoms_c[CHUNK_SIZE * 4];

// ── Gather kernel with thread coarsening (Fig 18.8) ───────────────────────────
// Each thread computes energy for COARSEN_FACTOR *consecutive* grid points.
// Non-overlapping assignment: thread t handles points [t*CF, t*CF+CF-1].
// dy and dz are calculated once per atom and shared across all 4 x-positions.
//
// Write pattern: adjacent threads write to locations CF apart (stride-CF),
// which is UNCOALESCED — the problem fixed by file 03 (Fig 18.10).
__global__ void cenergy_coarsen(float *energygrid, int grid_x, int grid_y,
                                 float gridspacing, float z, int numatoms) {
    // Each x-thread handles COARSEN_FACTOR consecutive x-points.
    // Multiply by COARSEN_FACTOR so adjacent threads don't overlap.
    int i = (blockIdx.x * BLOCK_DIM + threadIdx.x) * COARSEN_FACTOR;
    int j =  blockIdx.y * BLOCK_DIM + threadIdx.y;
    if (j >= grid_y) return;

    int atomarrdim = numatoms * 4;
    float x = gridspacing * (float)i;   // x of first grid point for this thread

    // Four accumulators — one per coarsened grid point (Fig 18.8, lines 10–13)
    float energy0 = 0.0f, energy1 = 0.0f, energy2 = 0.0f, energy3 = 0.0f;

    for (int n = 0; n < atomarrdim; n += 4) {
        float dx0 = x                - atoms_c[n];
        float dx1 = x +   gridspacing - atoms_c[n];
        float dx2 = x + 2*gridspacing - atoms_c[n];
        float dx3 = x + 3*gridspacing - atoms_c[n];
        float dy  = gridspacing*(float)j - atoms_c[n+1];
        float dz  = z                    - atoms_c[n+2];
        float dysqdzsq = dy*dy + dz*dz;          // shared across all 4 x-positions
        float charge   = atoms_c[n+3];

        energy0 += charge / sqrtf(dx0*dx0 + dysqdzsq);
        energy1 += charge / sqrtf(dx1*dx1 + dysqdzsq);
        energy2 += charge / sqrtf(dx2*dx2 + dysqdzsq);
        energy3 += charge / sqrtf(dx3*dx3 + dysqdzsq);
    }

    // Write four results — stride COARSEN_FACTOR between adjacent threads → uncoalesced
    if (i   < grid_x) energygrid[grid_x*j + i  ] += energy0;
    if (i+1 < grid_x) energygrid[grid_x*j + i+1] += energy1;
    if (i+2 < grid_x) energygrid[grid_x*j + i+2] += energy2;
    if (i+3 < grid_x) energygrid[grid_x*j + i+3] += energy3;
}

// ── Basic gather kernel (Fig 18.6) for timing comparison ─────────────────────
__global__ void cenergy_gather(float *energygrid, int grid_x, int grid_y,
                                float gridspacing, float z, int numatoms) {
    int i = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int j = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if (i >= grid_x || j >= grid_y) return;
    int atomarrdim = numatoms * 4;
    float x = gridspacing*(float)i, y = gridspacing*(float)j;
    float energy = 0.f;
    for (int n=0;n<atomarrdim;n+=4){
        float dx=x-atoms_c[n], dy=y-atoms_c[n+1], dz=z-atoms_c[n+2];
        energy+=atoms_c[n+3]/sqrtf(dx*dx+dy*dy+dz*dz);
    }
    energygrid[grid_x*j+i]+=energy;
}

static void cenergy_cpu(float *eg, int gx, int gy, float gs, float z,
                         const float *atoms, int na) {
    for (int n=0;n<na*4;n+=4){
        float dz=z-atoms[n+2],dz2=dz*dz,ch=atoms[n+3];
        for (int j=0;j<gy;j++){float dy=gs*(float)j-atoms[n+1],dy2=dy*dy;
            for(int i=0;i<gx;i++){float dx=gs*(float)i-atoms[n];eg[gx*j+i]+=ch/sqrtf(dx*dx+dy2+dz2);}}
    }
}
static float max_rel_err(const float *a,const float *b,int n){float mx=0.f;for(int i=0;i<n;i++){float e=fabsf(a[i]-b[i])/(1.f+fabsf(b[i]));if(e>mx)mx=e;}return mx;}

int main(void) {
    printf("=== Direct Coulomb Summation: Thread Coarsening (§18.3, Fig 18.8) ===\n\n");

    int grid_x=64, grid_y=64;
    int numatoms=1024;
    float gridspacing=0.5f, z=0.5f;
    int nGrid=grid_x*grid_y;
    size_t szGrid=nGrid*sizeof(float), szAtoms=numatoms*4*sizeof(float);

    float *h_atoms=(float*)malloc(szAtoms);
    float *h_cpu=(float*)calloc(nGrid,sizeof(float));
    float *h_gpu=(float*)calloc(nGrid,sizeof(float));

    srand(42);
    for(int i=0;i<numatoms*4;i+=4){
        h_atoms[i  ]=(rand()/(float)RAND_MAX)*grid_x*gridspacing;
        h_atoms[i+1]=(rand()/(float)RAND_MAX)*grid_y*gridspacing;
        h_atoms[i+2]=(rand()/(float)RAND_MAX)*2.f;
        h_atoms[i+3]=(rand()/(float)RAND_MAX)*2.f-1.f;
    }
    cenergy_cpu(h_cpu,grid_x,grid_y,gridspacing,z,h_atoms,numatoms);

    float *d_energy;
    cudaMalloc(&d_energy,szGrid);

    dim3 block(BLOCK_DIM,BLOCK_DIM);
    // For gather: grid covers all (i,j)
    dim3 grid_g((grid_x+BLOCK_DIM-1)/BLOCK_DIM,(grid_y+BLOCK_DIM-1)/BLOCK_DIM);
    // For coarsened: each thread handles CF x-points → need grid_x/CF x-threads
    dim3 grid_c((grid_x/(BLOCK_DIM*COARSEN_FACTOR)),
                (grid_y+BLOCK_DIM-1)/BLOCK_DIM);

    cudaEvent_t t0,t1;
    cudaEventCreate(&t0);cudaEventCreate(&t1);

    // ── Timing: basic gather ──────────────────────────────────────────────────
    cudaMemset(d_energy,0,szGrid);
    cudaEventRecord(t0);
    for(int r=0;r<10;r++){
        cudaMemset(d_energy,0,szGrid);
        for(int chunk=0;chunk*CHUNK_SIZE<numatoms;chunk++){
            int off=chunk*CHUNK_SIZE,act=((numatoms-off)<CHUNK_SIZE)?(numatoms-off):CHUNK_SIZE;
            cudaMemcpyToSymbol(atoms_c,&h_atoms[off*4],act*4*sizeof(float),0,cudaMemcpyHostToDevice);
            cenergy_gather<<<grid_g,block>>>(d_energy,grid_x,grid_y,gridspacing,z,act);
        }
    }
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    float ms_g; cudaEventElapsedTime(&ms_g,t0,t1);

    // ── Timing: coarsened ─────────────────────────────────────────────────────
    cudaMemset(d_energy,0,szGrid);
    cudaEventRecord(t0);
    for(int r=0;r<10;r++){
        cudaMemset(d_energy,0,szGrid);
        for(int chunk=0;chunk*CHUNK_SIZE<numatoms;chunk++){
            int off=chunk*CHUNK_SIZE,act=((numatoms-off)<CHUNK_SIZE)?(numatoms-off):CHUNK_SIZE;
            cudaMemcpyToSymbol(atoms_c,&h_atoms[off*4],act*4*sizeof(float),0,cudaMemcpyHostToDevice);
            cenergy_coarsen<<<grid_c,block>>>(d_energy,grid_x,grid_y,gridspacing,z,act);
        }
    }
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    float ms_c; cudaEventElapsedTime(&ms_c,t0,t1);

    cudaMemcpy(h_gpu,d_energy,szGrid,cudaMemcpyDeviceToHost);
    float err=max_rel_err(h_gpu,h_cpu,nGrid);
    printf("Grid %d×%d, %d atoms, COARSEN_FACTOR=%d (10-run avg)\n",
           grid_x,grid_y,numatoms,COARSEN_FACTOR);
    printf("Gather (Fig 18.6):    %.3f ms/run\n", ms_g/10.f);
    printf("Coarsen (Fig 18.8):   %.3f ms/run  speedup=%.2fx\n",
           ms_c/10.f, ms_g/ms_c);
    printf("Max rel error: %.2e  %s\n", err, err<1e-4f?"PASS":"FAIL");
    printf("\nCoarsening trade-offs (§18.3):\n");
    printf("  + dy²+dz² computed once per atom, shared across %d x-positions\n",COARSEN_FACTOR);
    printf("  + atom data (x,y,z,charge) fetched once per %d outputs\n",COARSEN_FACTOR);
    printf("  + Constant-memory accesses: 16 → 4 per %d grid points\n",COARSEN_FACTOR);
    printf("  − Writes to energygrid spaced 1 apart → uncoalesced (fix in file 03)\n");
    printf("  → File 03 spaces writes blockDim.x apart for coalescing\n");

    free(h_atoms);free(h_cpu);free(h_gpu);
    cudaFree(d_energy);
    cudaEventDestroy(t0);cudaEventDestroy(t1);
    return 0;
}
