// §18.4  Direct Coulomb Summation — coarsening + memory coalescing (Fig 18.10)
//
// Problem with the coarsened kernel (Fig 18.8):
//   In lines 30–33 of Fig 18.8, thread i writes to grid points i, i+1,
//   i+2, i+3 — four *consecutive* indices.  In a warp of 32 threads,
//   adjacent threads (thread 0, 1, 2, …) write to locations
//   (0,1,2,3), (1,2,3,4), (2,3,4,5), … which are *four-element strides*
//   apart.  The 32 warp threads collectively access 32×4=128 different
//   array positions — spread across multiple cache lines, uncoalesced.
//
// Fix (§18.4, Fig 18.9 → Fig 18.10):
//   Assign the four grid points that each thread computes so that they
//   are spaced blockDim.x elements apart rather than 1:
//     dx0 = x                          (original base for thread i)
//     dx1 = x +   blockDim.x*gridspacing
//     dx2 = x + 2*blockDim.x*gridspacing
//     dx3 = x + 3*blockDim.x*gridspacing
//
//   Then, within a warp, adjacent threads write to consecutive locations
//   for each of the four output statements.  All four write patterns are
//   now coalesced (each writes one blockDim.x-wide stripe).
//
// Reference: §18.4, Fig 18.10 (coarsened + coalesced kernel).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_DIM       16
#define COARSEN_FACTOR   4
#define CHUNK_SIZE      256

__constant__ float atoms_c[CHUNK_SIZE * 4];

// ── Coarsened + coalesced kernel (Fig 18.10) ─────────────────────────────────
// Grid-point assignments are spaced blockDim.x apart in x so that warp-wide
// writes are contiguous in memory.
__global__ void cenergy_coalesce(float *energygrid, int grid_x, int grid_y,
                                  float gridspacing, float z, int numatoms) {
    // i is the base x-index for this thread's *first* grid point (Fig 18.10 line 05)
    int i = blockIdx.x * BLOCK_DIM * COARSEN_FACTOR + threadIdx.x;
    int j = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if (j >= grid_y) return;

    int atomarrdim = numatoms * 4;
    float x = gridspacing * (float)i;   // x-coord of first assigned grid point

    float energy0=0.f, energy1=0.f, energy2=0.f, energy3=0.f;

    // dx offsets: spaced blockDim.x * gridspacing apart (coalescing fix)
    // Compared with file 02 where spacing was just gridspacing, here we
    // jump BLOCK_DIM grid points at a time so adjacent warp threads write
    // to adjacent memory locations (Fig 18.10 lines 17–20).
    float xstep = blockDim.x * gridspacing;   // = BLOCK_DIM * gridspacing

    for (int n = 0; n < atomarrdim; n += 4) {
        float dx0 = x            - atoms_c[n];
        float dx1 = x + 1*xstep - atoms_c[n];
        float dx2 = x + 2*xstep - atoms_c[n];
        float dx3 = x + 3*xstep - atoms_c[n];
        float dy  = gridspacing*(float)j - atoms_c[n+1];
        float dz  = z                    - atoms_c[n+2];
        float dysqdzsq = dy*dy + dz*dz;
        float charge   = atoms_c[n+3];

        energy0 += charge / sqrtf(dx0*dx0 + dysqdzsq);
        energy1 += charge / sqrtf(dx1*dx1 + dysqdzsq);
        energy2 += charge / sqrtf(dx2*dx2 + dysqdzsq);
        energy3 += charge / sqrtf(dx3*dx3 + dysqdzsq);
    }

    // Coalesced writes: adjacent threads write to adjacent i (Fig 18.10 lines 30–33)
    if (i              < grid_x) energygrid[grid_x*j + i            ] += energy0;
    if (i +   blockDim.x < grid_x) energygrid[grid_x*j + i+  blockDim.x] += energy1;
    if (i + 2*blockDim.x < grid_x) energygrid[grid_x*j + i+2*blockDim.x] += energy2;
    if (i + 3*blockDim.x < grid_x) energygrid[grid_x*j + i+3*blockDim.x] += energy3;
}

// ── Non-coalesced coarsen (Fig 18.8) for timing comparison ───────────────────
__global__ void cenergy_coarsen(float *energygrid, int grid_x, int grid_y,
                                 float gridspacing, float z, int numatoms) {
    int i=(blockIdx.x*BLOCK_DIM+threadIdx.x)*COARSEN_FACTOR;  // no overlap
    int j=blockIdx.y*BLOCK_DIM+threadIdx.y;
    if(j>=grid_y)return;
    int ad=numatoms*4;
    float x=gridspacing*(float)i;
    float e0=0.f,e1=0.f,e2=0.f,e3=0.f;
    for(int n=0;n<ad;n+=4){
        float dx0=x-atoms_c[n],dx1=x+gridspacing-atoms_c[n],dx2=x+2*gridspacing-atoms_c[n],dx3=x+3*gridspacing-atoms_c[n];
        float dy=gridspacing*(float)j-atoms_c[n+1],dz=z-atoms_c[n+2],dsq=dy*dy+dz*dz,ch=atoms_c[n+3];
        e0+=ch/sqrtf(dx0*dx0+dsq);e1+=ch/sqrtf(dx1*dx1+dsq);e2+=ch/sqrtf(dx2*dx2+dsq);e3+=ch/sqrtf(dx3*dx3+dsq);
    }
    if(i  <grid_x)energygrid[grid_x*j+i  ]+=e0;
    if(i+1<grid_x)energygrid[grid_x*j+i+1]+=e1;
    if(i+2<grid_x)energygrid[grid_x*j+i+2]+=e2;
    if(i+3<grid_x)energygrid[grid_x*j+i+3]+=e3;
}

static void cenergy_cpu(float *eg,int gx,int gy,float gs,float z,const float *atoms,int na){
    for(int n=0;n<na*4;n+=4){float dz=z-atoms[n+2],dz2=dz*dz,ch=atoms[n+3];
        for(int j=0;j<gy;j++){float dy=gs*(float)j-atoms[n+1],dy2=dy*dy;
            for(int i=0;i<gx;i++){float dx=gs*(float)i-atoms[n];eg[gx*j+i]+=ch/sqrtf(dx*dx+dy2+dz2);}}}
}
static float max_rel_err(const float *a,const float *b,int n){float mx=0.f;for(int i=0;i<n;i++){float e=fabsf(a[i]-b[i])/(1.f+fabsf(b[i]));if(e>mx)mx=e;}return mx;}

int main(void) {
    printf("=== Direct Coulomb Summation: Coarsening + Coalescing (§18.4, Fig 18.10) ===\n\n");

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
    dim3 grid_c(grid_x/(BLOCK_DIM*COARSEN_FACTOR),
                (grid_y+BLOCK_DIM-1)/BLOCK_DIM);

    cudaEvent_t t0,t1;
    cudaEventCreate(&t0);cudaEventCreate(&t1);

    auto run = [&](bool coalesce) {
        cudaMemset(d_energy,0,szGrid);
        cudaEventRecord(t0);
        for(int r=0;r<10;r++){
            cudaMemset(d_energy,0,szGrid);
            for(int chunk=0;chunk*CHUNK_SIZE<numatoms;chunk++){
                int off=chunk*CHUNK_SIZE,act=((numatoms-off)<CHUNK_SIZE)?(numatoms-off):CHUNK_SIZE;
                cudaMemcpyToSymbol(atoms_c,&h_atoms[off*4],act*4*sizeof(float),0,cudaMemcpyHostToDevice);
                if (coalesce)
                    cenergy_coalesce<<<grid_c,block>>>(d_energy,grid_x,grid_y,gridspacing,z,act);
                else
                    cenergy_coarsen <<<grid_c,block>>>(d_energy,grid_x,grid_y,gridspacing,z,act);
            }
        }
        cudaEventRecord(t1);cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms,t0,t1);
        return ms;
    };

    float ms_c = run(false);
    float ms_co = run(true);

    cudaMemcpy(h_gpu,d_energy,szGrid,cudaMemcpyDeviceToHost);
    float err=max_rel_err(h_gpu,h_cpu,nGrid);
    printf("Grid %d×%d, %d atoms, COARSEN_FACTOR=%d, BLOCK_DIM=%d (10-run avg)\n",
           grid_x,grid_y,numatoms,COARSEN_FACTOR,BLOCK_DIM);
    printf("Coarsen only (Fig 18.8):        %.3f ms/run\n", ms_c/10.f);
    printf("Coarsen+coalesce (Fig 18.10):   %.3f ms/run  speedup=%.2fx\n",
           ms_co/10.f, ms_c/ms_co);
    printf("Max rel error: %.2e  %s\n", err, err<1e-4f?"PASS":"FAIL");
    printf("\nCoalescing fix (§18.4):\n");
    printf("  Before: thread i writes to i, i+1, i+2, i+3 — 4-stride in warp\n");
    printf("  After:  thread i writes to i, i+%-2d, i+%-2d, i+%-2d — 1-stride\n",
           BLOCK_DIM,2*BLOCK_DIM,3*BLOCK_DIM);
    printf("  Adjacent threads in warp write consecutive memory → coalesced\n");
    printf("  Book reports combined gains of ~10× from gather→register→constmem→coarsen→coalesce\n");

    free(h_atoms);free(h_cpu);free(h_gpu);
    cudaFree(d_energy);
    cudaEventDestroy(t0);cudaEventDestroy(t1);
    return 0;
}
