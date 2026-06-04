// §15.6  Frontier-based push BFS with privatized frontiers — Figures 15.14–15.15
//
// Problem with §15.5: all threads atomicAdd to a single global numCurrFrontier
// counter.  On large frontiers this creates a high-contention bottleneck.
//
// Fix (privatization, same idea as §9.4 histogram privatization):
//   Each thread block maintains its own LOCAL frontier in shared memory.
//   Threads atomicAdd to the block-local counter (fast, shared memory).
//   Only when flushing the local frontier to global memory does the block
//   do ONE atomicAdd on the global counter to reserve a contiguous range.
//   All threads then copy from local to global in consecutive thread-index
//   order → consecutive global addresses → COALESCED writes (Fig 15.15).
//
// Algorithm (Fig 15.14):
//   1. Initialize: one thread per block sets numCurrFrontier_s = 0.
//   2. For each vertex v in prevFrontier (one thread per frontier element):
//      a. atomicCAS(&level[nb], UINT_MAX, currLevel) — claim unvisited nb.
//      b. If claimed: atomicAdd(&numCurrFrontier_s, 1) to get local slot.
//         If slot < LOCAL_FRONTIER_CAPACITY: store in currFrontier_s[slot].
//         Else (overflow): fall back to global atomicAdd on numCurrFrontier.
//   3. __syncthreads() — wait for all threads to finish local updates.
//   4. Thread 0: atomicAdd(numCurrFrontier, numCurrFrontier_s) — reserve
//      contiguous global range for this block's local frontier entries.
//   5. __syncthreads().
//   6. All threads cooperatively copy local frontier to the reserved global
//      range.  Thread t copies currFrontier_s[t] (for t < numCurrFrontier_s).
//      Consecutive threads write to consecutive global locations → coalesced.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE              256
#define LOCAL_FRONTIER_CAPACITY BLOCK_SIZE
#define UINT_INF                UINT_MAX

typedef struct {
    unsigned int *srcPtrs;
    unsigned int *dst;
    unsigned int  numVertices;
    unsigned int  numEdges;
} CSRGraph;

// ── Kernel (Fig 15.14): privatized local frontiers ────────────────────────────
__global__ void bfs_privatized_kernel(CSRGraph g, unsigned int *level,
                                       unsigned int *prevFrontier,
                                       unsigned int *currFrontier,
                                       unsigned int  numPrevFrontier,
                                       unsigned int *numCurrFrontier,
                                       unsigned int  currLevel) {
    // Shared-memory local frontier for this block (Fig 15.14 lines 06-08)
    __shared__ unsigned int currFrontier_s[LOCAL_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;

    if (threadIdx.x == 0) numCurrFrontier_s = 0;
    __syncthreads();

    // Each thread processes one element of the previous frontier
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int v = prevFrontier[i];
        for (unsigned int e = g.srcPtrs[v]; e < g.srcPtrs[v + 1]; e++) {
            unsigned int nb = g.dst[e];
            // Atomically claim unvisited neighbor
            if (atomicCAS(&level[nb], UINT_INF, currLevel) == UINT_INF) {
                // Try to add to local frontier
                unsigned int localIdx = atomicAdd(&numCurrFrontier_s, 1u);
                if (localIdx < LOCAL_FRONTIER_CAPACITY) {
                    currFrontier_s[localIdx] = nb;
                } else {
                    // Local frontier full: fall back to global frontier
                    numCurrFrontier_s = LOCAL_FRONTIER_CAPACITY;  // cap (don't overflow)
                    unsigned int globalIdx = atomicAdd(numCurrFrontier, 1u);
                    currFrontier[globalIdx] = nb;
                }
            }
        }
    }
    __syncthreads();

    // Reserve a contiguous range in the global frontier for this block's
    // local entries (one atomicAdd per block instead of one per thread).
    __shared__ unsigned int currFrontierStartIdx;
    if (threadIdx.x == 0)
        currFrontierStartIdx = atomicAdd(numCurrFrontier, numCurrFrontier_s);
    __syncthreads();

    // Flush local frontier to global frontier in coalesced order
    for (unsigned int t = threadIdx.x; t < numCurrFrontier_s; t += blockDim.x)
        currFrontier[currFrontierStartIdx + t] = currFrontier_s[t];
}

// ── CPU reference ─────────────────────────────────────────────────────────────
static void bfs_cpu(const unsigned int *sp, const unsigned int *dst,
                    unsigned int NV, unsigned int root, unsigned int *level) {
    for(unsigned int i=0;i<NV;i++) level[i]=UINT_INF;
    level[root]=0;
    unsigned int *q=(unsigned int *)malloc(NV*sizeof(unsigned int));
    unsigned int h=0,t=0; q[t++]=root;
    while(h<t){
        unsigned int v=q[h++];
        for(unsigned int e=sp[v];e<sp[v+1];e++) if(level[dst[e]]==UINT_INF){level[dst[e]]=level[v]+1;q[t++]=dst[e];}
    }
    free(q);
}

static int verify(const unsigned int *ref, const unsigned int *gpu, unsigned int n) {
    for(unsigned int i=0;i<n;i++) if(ref[i]!=gpu[i]){printf("  MISMATCH v=%u ref=%u gpu=%u\n",i,ref[i],gpu[i]);return 0;}
    return 1;
}

static void gen_csr(unsigned int N, unsigned int D,
                    unsigned int **pp, unsigned int **pd, unsigned int *pE) {
    srand(42);
    *pp=(unsigned int*)malloc((N+1)*sizeof(unsigned int));
    *pd=(unsigned int*)malloc(N*D*sizeof(unsigned int));
    unsigned int k=0;(*pp)[0]=0;
    for(unsigned int v=0;v<N;v++){
        for(unsigned int j=0;j<D;j++){unsigned int d;do{d=rand()%N;}while(d==v);(*pd)[k++]=d;}
        (*pp)[v+1]=k;
    }
    *pE=k;
}

int main(void) {
    printf("=== Frontier BFS with Privatized Frontiers (§15.6, Fig 15.14) ===\n\n");

    // ── Small test: Fig 15.1 graph ────────────────────────────────────────────
    {
        unsigned int NV=9,NE=15,root=0;
        unsigned int h_sp[]={0,2,4,7,9,11,12,13,15,15};
        unsigned int h_d[] ={1,2,3,4,5,6,7,4,8,5,8,8,8,0,6};
        unsigned int exp[] ={0,1,1,2,2,2,2,2,3};

        CSRGraph g_d;
        cudaMalloc(&g_d.srcPtrs,(NV+1)*sizeof(unsigned int));
        cudaMalloc(&g_d.dst,NE*sizeof(unsigned int));
        g_d.numVertices=NV; g_d.numEdges=NE;
        cudaMemcpy(g_d.srcPtrs,h_sp,(NV+1)*sizeof(unsigned int),cudaMemcpyHostToDevice);
        cudaMemcpy(g_d.dst,h_d,NE*sizeof(unsigned int),cudaMemcpyHostToDevice);

        unsigned int *d_level,*d_prev,*d_curr,*d_nCurr;
        cudaMalloc(&d_level,NV*sizeof(unsigned int));
        cudaMalloc(&d_prev,NV*sizeof(unsigned int));
        cudaMalloc(&d_curr,NV*sizeof(unsigned int));
        cudaMalloc(&d_nCurr,sizeof(unsigned int));

        unsigned int init[9]; for(int i=0;i<9;i++) init[i]=UINT_INF; init[root]=0;
        cudaMemcpy(d_level,init,NV*sizeof(unsigned int),cudaMemcpyHostToDevice);
        unsigned int r0=root; cudaMemcpy(d_prev,&r0,sizeof(unsigned int),cudaMemcpyHostToDevice);
        unsigned int nPrev=1;

        for(unsigned int lvl=1;lvl<NV&&nPrev>0;lvl++){
            cudaMemset(d_nCurr,0,sizeof(unsigned int));
            int nb=(nPrev+BLOCK_SIZE-1)/BLOCK_SIZE;
            bfs_privatized_kernel<<<nb,BLOCK_SIZE>>>(g_d,d_level,d_prev,d_curr,nPrev,d_nCurr,lvl);
            cudaDeviceSynchronize();
            cudaMemcpy(&nPrev,d_nCurr,sizeof(unsigned int),cudaMemcpyDeviceToHost);
            unsigned int *tmp=d_prev; d_prev=d_curr; d_curr=tmp;
        }

        unsigned int h_level[9];
        cudaMemcpy(h_level,d_level,NV*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        printf("Fig 15.1 BFS from root 0:\n  GPU:      ");
        for(int i=0;i<9;i++) printf("%u ",h_level[i]);
        printf("\n  Expected: "); for(int i=0;i<9;i++) printf("%u ",exp[i]);
        printf("\n  %s\n\n", verify(exp,h_level,NV)?"PASS":"FAIL");

        cudaFree(g_d.srcPtrs);cudaFree(g_d.dst);
        cudaFree(d_level);cudaFree(d_prev);cudaFree(d_curr);cudaFree(d_nCurr);
    }

    // ── Large test ────────────────────────────────────────────────────────────
    {
        unsigned int N=4096, D=8;
        unsigned int *h_sp,*h_dst,NE;
        gen_csr(N,D,&h_sp,&h_dst,&NE);
        printf("Large test: %u vertices, %u edges\n", N, NE);

        unsigned int *ref=(unsigned int *)malloc(N*sizeof(unsigned int));
        bfs_cpu(h_sp,h_dst,N,0,ref);

        CSRGraph g_d;
        cudaMalloc(&g_d.srcPtrs,(N+1)*sizeof(unsigned int));
        cudaMalloc(&g_d.dst,NE*sizeof(unsigned int));
        g_d.numVertices=N; g_d.numEdges=NE;
        cudaMemcpy(g_d.srcPtrs,h_sp,(N+1)*sizeof(unsigned int),cudaMemcpyHostToDevice);
        cudaMemcpy(g_d.dst,h_dst,NE*sizeof(unsigned int),cudaMemcpyHostToDevice);

        unsigned int *d_level,*d_prev,*d_curr,*d_nCurr;
        cudaMalloc(&d_level,N*sizeof(unsigned int));
        cudaMalloc(&d_prev,N*sizeof(unsigned int));
        cudaMalloc(&d_curr,N*sizeof(unsigned int));
        cudaMalloc(&d_nCurr,sizeof(unsigned int));

        unsigned int *init=(unsigned int *)malloc(N*sizeof(unsigned int));
        for(unsigned int i=0;i<N;i++) init[i]=UINT_INF; init[0]=0;
        cudaMemcpy(d_level,init,N*sizeof(unsigned int),cudaMemcpyHostToDevice);
        unsigned int r0=0; cudaMemcpy(d_prev,&r0,sizeof(unsigned int),cudaMemcpyHostToDevice);
        unsigned int nPrev=1;

        cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        unsigned int lev=0;
        cudaEventRecord(t0);
        for(unsigned int lvl=1;lvl<N&&nPrev>0;lvl++){
            cudaMemset(d_nCurr,0,sizeof(unsigned int));
            int nb=(nPrev+BLOCK_SIZE-1)/BLOCK_SIZE;
            bfs_privatized_kernel<<<nb,BLOCK_SIZE>>>(g_d,d_level,d_prev,d_curr,nPrev,d_nCurr,lvl);
            cudaDeviceSynchronize(); lev++;
            cudaMemcpy(&nPrev,d_nCurr,sizeof(unsigned int),cudaMemcpyDeviceToHost);
            unsigned int *tmp=d_prev; d_prev=d_curr; d_curr=tmp;
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms,t0,t1);

        unsigned int *gpu=(unsigned int *)malloc(N*sizeof(unsigned int));
        cudaMemcpy(gpu,d_level,N*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        printf("  GPU time: %.3f ms  %u level iterations  %s\n\n", ms, lev,
               verify(ref,gpu,N)?"PASS":"FAIL");

        printf("Privatization benefits (§15.6, Fig 15.15):\n");
        printf("  §15.5 frontier: every thread calls atomicAdd(numCurrFrontier)\n");
        printf("    → all threads contend on one counter (high latency)\n");
        printf("  §15.6 privatized: each block uses shared-memory local frontier\n");
        printf("    → threads contend only within their block (fast shared mem)\n");
        printf("    → only 1 global atomicAdd per BLOCK to reserve a range\n");
        printf("    → local-to-global flush is coalesced (consecutive thread indices\n");
        printf("       write to consecutive global addresses, as in Fig 15.15)\n");
        printf("  Block size %d × LOCAL_FRONTIER_CAPACITY %d\n",
               BLOCK_SIZE, LOCAL_FRONTIER_CAPACITY);

        free(h_sp);free(h_dst);free(ref);free(init);free(gpu);
        cudaFree(g_d.srcPtrs);cudaFree(g_d.dst);
        cudaFree(d_level);cudaFree(d_prev);cudaFree(d_curr);cudaFree(d_nCurr);
        cudaEventDestroy(t0);cudaEventDestroy(t1);
    }
    return 0;
}
