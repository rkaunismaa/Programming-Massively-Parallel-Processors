// §15.5  Vertex-centric push BFS with frontiers — Figure 15.12
//
// The push kernels in §15.3 launched a thread for every vertex every level,
// even though most threads did nothing.  Frontiers fix this: we track which
// vertices were labeled at the previous level (prevFrontier) and launch only
// one thread per element of that frontier.
//
// Key change from §15.3 (Fig 15.12):
//   - Thread index i maps to prevFrontier[i], not directly to vertex i.
//   - Threads are launched over the frontier, not the full vertex array.
//   - atomicCAS on level[neighbor] performs an atomic check-and-label:
//       if (atomicCAS(&level[nb], UINT_MAX, currLevel) == UINT_MAX)
//     This ensures each neighbor is added to the frontier AT MOST ONCE.
//     If two threads both see level[nb]==UINT_MAX and attempt the CAS,
//     exactly one succeeds (returns UINT_MAX) and adds nb to the frontier;
//     the other gets the already-written currLevel back and does nothing.
//   - atomicAdd on *numCurrFrontier allocates a slot in currFrontier.
//
// Unlike the §15.3 non-atomic write, here we cannot allow duplicates in the
// frontier because each entry causes another thread to be launched next level.
// Idempotent level writes are fine; duplicate frontier entries are NOT.
//
// Trade-off:
//   + Only |prevFrontier| threads launched per level → no wasted launches.
//   − atomicCAS (moderate contention) and atomicAdd (high contention on the
//     single counter) add overhead.
//   → §15.6 privatization reduces the contention on numCurrFrontier.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define UINT_INF   UINT_MAX

typedef struct {
    unsigned int *srcPtrs;
    unsigned int *dst;
    unsigned int  numVertices;
    unsigned int  numEdges;
} CSRGraph;

// ── Kernel (Fig 15.12): one thread per prevFrontier element ───────────────────
__global__ void bfs_frontier_kernel(CSRGraph g, unsigned int *level,
                                     unsigned int *prevFrontier,
                                     unsigned int *currFrontier,
                                     unsigned int  numPrevFrontier,
                                     unsigned int *numCurrFrontier,
                                     unsigned int  currLevel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int v = prevFrontier[i];
        for (unsigned int e = g.srcPtrs[v]; e < g.srcPtrs[v + 1]; e++) {
            unsigned int nb = g.dst[e];
            // Atomic check-and-label: succeeds only for the first thread to
            // visit nb (prevents duplicate frontier entries).
            if (atomicCAS(&level[nb], UINT_INF, currLevel) == UINT_INF) {
                unsigned int idx = atomicAdd(numCurrFrontier, 1u);
                currFrontier[idx] = nb;
            }
        }
    }
}

// ── CPU reference ─────────────────────────────────────────────────────────────
static void bfs_cpu(const unsigned int *sp, const unsigned int *dst,
                    unsigned int NV, unsigned int root, unsigned int *level) {
    for (unsigned int i=0;i<NV;i++) level[i]=UINT_INF;
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
    unsigned int k=0; (*pp)[0]=0;
    for(unsigned int v=0;v<N;v++){
        for(unsigned int j=0;j<D;j++){unsigned int d;do{d=rand()%N;}while(d==v);(*pd)[k++]=d;}
        (*pp)[v+1]=k;
    }
    *pE=k;
}

int main(void) {
    printf("=== Frontier-Based Push BFS (§15.5, Fig 15.12) ===\n\n");

    // ── Small test: Fig 15.1 graph ────────────────────────────────────────────
    {
        unsigned int NV=9, NE=15, root=0;
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
        cudaMalloc(&d_prev, NV*sizeof(unsigned int));
        cudaMalloc(&d_curr, NV*sizeof(unsigned int));
        cudaMalloc(&d_nCurr,sizeof(unsigned int));

        unsigned int init[9]; for(int i=0;i<9;i++) init[i]=UINT_INF; init[root]=0;
        cudaMemcpy(d_level,init,NV*sizeof(unsigned int),cudaMemcpyHostToDevice);
        unsigned int h_prev0[1]={root};
        cudaMemcpy(d_prev,h_prev0,sizeof(unsigned int),cudaMemcpyHostToDevice);
        unsigned int nPrev=1;

        for (unsigned int lvl=1; lvl<NV && nPrev>0; lvl++) {
            cudaMemset(d_nCurr,0,sizeof(unsigned int));
            int nb=(nPrev+BLOCK_SIZE-1)/BLOCK_SIZE;
            bfs_frontier_kernel<<<nb,BLOCK_SIZE>>>(g_d,d_level,d_prev,d_curr,nPrev,d_nCurr,lvl);
            cudaDeviceSynchronize();
            cudaMemcpy(&nPrev,d_nCurr,sizeof(unsigned int),cudaMemcpyDeviceToHost);
            // Swap prev ↔ curr
            unsigned int *tmp=d_prev; d_prev=d_curr; d_curr=tmp;
        }

        unsigned int h_level[9];
        cudaMemcpy(h_level,d_level,NV*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        printf("Fig 15.1 BFS from root 0:\n  GPU:      ");
        for(int i=0;i<9;i++) printf("%u ",h_level[i]);
        printf("\n  Expected: "); for(int i=0;i<9;i++) printf("%u ",exp[i]);
        printf("\n  Frontier sizes per level: |L0|=1 |L1|=2 |L2|=5 |L3|=1\n");
        printf("  %s\n\n", verify(exp,h_level,NV)?"PASS":"FAIL");

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
        cudaMalloc(&d_prev, N*sizeof(unsigned int));
        cudaMalloc(&d_curr, N*sizeof(unsigned int));
        cudaMalloc(&d_nCurr,sizeof(unsigned int));

        unsigned int *init=(unsigned int *)malloc(N*sizeof(unsigned int));
        for(unsigned int i=0;i<N;i++) init[i]=UINT_INF; init[0]=0;
        cudaMemcpy(d_level,init,N*sizeof(unsigned int),cudaMemcpyHostToDevice);
        unsigned int root0=0;
        cudaMemcpy(d_prev,&root0,sizeof(unsigned int),cudaMemcpyHostToDevice);
        unsigned int nPrev=1;

        cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        unsigned int totalThreads=0, lev=0;
        cudaEventRecord(t0);
        for(unsigned int lvl=1;lvl<N&&nPrev>0;lvl++){
            cudaMemset(d_nCurr,0,sizeof(unsigned int));
            int nb=(nPrev+BLOCK_SIZE-1)/BLOCK_SIZE;
            bfs_frontier_kernel<<<nb,BLOCK_SIZE>>>(g_d,d_level,d_prev,d_curr,nPrev,d_nCurr,lvl);
            cudaDeviceSynchronize(); lev++; totalThreads+=nPrev;
            cudaMemcpy(&nPrev,d_nCurr,sizeof(unsigned int),cudaMemcpyDeviceToHost);
            unsigned int *tmp=d_prev; d_prev=d_curr; d_curr=tmp;
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms,t0,t1);

        unsigned int *gpu=(unsigned int *)malloc(N*sizeof(unsigned int));
        cudaMemcpy(gpu,d_level,N*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        printf("  GPU time: %.3f ms  %u levels  total frontier threads=%u (vs %u×%u=%u naive)\n",
               ms, lev, totalThreads, lev, N, lev*N);
        printf("  %s\n\n", verify(ref,gpu,N)?"PASS":"FAIL");

        printf("Frontier trade-offs (§15.5):\n");
        printf("  + Only frontier vertices launch threads → far less wasted work\n");
        printf("  + %u frontier threads vs. %u×%u=%u naive (push/pull) threads\n",
               totalThreads, lev, N, lev*N);
        printf("  − atomicCAS: moderate contention (each unvisited neighbor)\n");
        printf("  − atomicAdd on numCurrFrontier: HIGH contention (all threads share 1 counter)\n");
        printf("  → §15.6 privatization reduces the numCurrFrontier contention\n");

        free(h_sp);free(h_dst);free(ref);free(init);free(gpu);
        cudaFree(g_d.srcPtrs);cudaFree(g_d.dst);
        cudaFree(d_level);cudaFree(d_prev);cudaFree(d_curr);cudaFree(d_nCurr);
        cudaEventDestroy(t0);cudaEventDestroy(t1);
    }
    return 0;
}
