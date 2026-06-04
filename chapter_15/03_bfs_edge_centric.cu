// §15.4  Edge-centric BFS — Figure 15.10
//
// Each thread is assigned to an EDGE, not a vertex.  Per level, a thread
// checks whether the source vertex of its edge belongs to the previous level.
// If so and if the destination vertex is unvisited, the thread labels the
// destination as belonging to the current level.
//
// Graph representation: COO (coordinate list)
//   src[numEdges]  — source vertex of each edge
//   dst[numEdges]  — destination vertex of each edge
//
// Advantages over vertex-centric (§15.4):
//   + More parallelism: one thread per edge vs. one thread per vertex.
//     For sparse graphs |E| >> |V|, more threads are available.
//   + Less load imbalance: each thread traverses exactly one edge regardless
//     of vertex degree (vertex-centric threads loop over all their edges).
//   + Reduced control divergence: all threads do the same amount of work.
//
// Disadvantages:
//   − Must check every edge every level, even if the source is irrelevant.
//     Vertex-centric can skip a vertex's entire edge list in O(1).
//   − Requires COO storage (extra src[] array vs. CSR's compact srcPtrs[]).
//
// The level write is idempotent (multiple threads may write the same currLevel
// to the same destination; no atomics needed for correctness).

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define UINT_INF   UINT_MAX

typedef struct {
    unsigned int *src;        // [numEdges]
    unsigned int *dst;        // [numEdges]
    unsigned int  numVertices;
    unsigned int  numEdges;
} COOGraph;

// ── Kernel (Fig 15.10): one thread per edge ───────────────────────────────────
__global__ void bfs_edge_kernel(COOGraph g, unsigned int *level,
                                 unsigned int *newVertexVisited,
                                 unsigned int currLevel) {
    unsigned int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < g.numEdges) {
        unsigned int v  = g.src[e];
        unsigned int nb = g.dst[e];
        if (level[v] == currLevel - 1) {     // source in previous level
            if (level[nb] == UINT_INF) {     // destination unvisited
                level[nb]         = currLevel;
                *newVertexVisited = 1;
            }
        }
    }
}

// ── CPU reference ─────────────────────────────────────────────────────────────
static void bfs_cpu(const unsigned int *srcPtrs, const unsigned int *dst,
                    unsigned int numV, unsigned int root, unsigned int *level) {
    for (unsigned int i = 0; i < numV; i++) level[i] = UINT_INF;
    level[root] = 0;
    unsigned int *q = (unsigned int *)malloc(numV * sizeof(unsigned int));
    unsigned int h = 0, t = 0; q[t++] = root;
    while (h < t) {
        unsigned int v = q[h++];
        for (unsigned int e = srcPtrs[v]; e < srcPtrs[v+1]; e++)
            if (level[dst[e]] == UINT_INF) { level[dst[e]] = level[v]+1; q[t++] = dst[e]; }
    }
    free(q);
}

static int verify(const unsigned int *ref, const unsigned int *gpu, unsigned int n) {
    for (unsigned int i = 0; i < n; i++)
        if (ref[i] != gpu[i]) { printf("  MISMATCH v=%u ref=%u gpu=%u\n",i,ref[i],gpu[i]); return 0; }
    return 1;
}

// Build COO from CSR
static void csr_to_coo(const unsigned int *srcPtrs, const unsigned int *dst,
                        unsigned int NV, unsigned int NE,
                        unsigned int **coo_src, unsigned int **coo_dst) {
    *coo_src = (unsigned int *)malloc(NE * sizeof(unsigned int));
    *coo_dst = (unsigned int *)malloc(NE * sizeof(unsigned int));
    for (unsigned int v = 0; v < NV; v++)
        for (unsigned int e = srcPtrs[v]; e < srcPtrs[v+1]; e++) {
            (*coo_src)[e] = v;
            (*coo_dst)[e] = dst[e];
        }
}

static void gen_csr(unsigned int N, unsigned int D,
                    unsigned int **pp, unsigned int **pd, unsigned int *pE) {
    srand(42);
    *pp = (unsigned int *)malloc((N+1)*sizeof(unsigned int));
    *pd = (unsigned int *)malloc(N*D*sizeof(unsigned int));
    unsigned int k=0; (*pp)[0]=0;
    for (unsigned int v=0;v<N;v++) {
        for (unsigned int j=0;j<D;j++) { unsigned int d; do{d=rand()%N;}while(d==v); (*pd)[k++]=d; }
        (*pp)[v+1]=k;
    }
    *pE=k;
}

int main(void) {
    printf("=== Edge-Centric BFS (§15.4, Fig 15.10) ===\n\n");

    // ── Small test: Fig 15.1 ──────────────────────────────────────────────────
    {
        unsigned int NV=9, NE=15;
        unsigned int h_sp[]={0,2,4,7,9,11,12,13,15,15};
        unsigned int h_d[] ={1,2,3,4,5,6,7,4,8,5,8,8,8,0,6};
        unsigned int root=0, exp[]={0,1,1,2,2,2,2,2,3};

        unsigned int *h_cs, *h_cd;
        csr_to_coo(h_sp, h_d, NV, NE, &h_cs, &h_cd);

        COOGraph g_d;
        cudaMalloc(&g_d.src, NE*sizeof(unsigned int));
        cudaMalloc(&g_d.dst, NE*sizeof(unsigned int));
        g_d.numVertices=NV; g_d.numEdges=NE;
        cudaMemcpy(g_d.src, h_cs, NE*sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(g_d.dst, h_cd, NE*sizeof(unsigned int), cudaMemcpyHostToDevice);

        unsigned int init[9]; for(int i=0;i<9;i++) init[i]=UINT_INF; init[root]=0;
        unsigned int *d_level, *d_flag;
        cudaMalloc(&d_level,NV*sizeof(unsigned int)); cudaMalloc(&d_flag,sizeof(unsigned int));
        cudaMemcpy(d_level,init,NV*sizeof(unsigned int),cudaMemcpyHostToDevice);

        int nb=(NE+BLOCK_SIZE-1)/BLOCK_SIZE;
        for (unsigned int lvl=1;lvl<NV;lvl++) {
            cudaMemset(d_flag,0,sizeof(unsigned int));
            bfs_edge_kernel<<<nb,BLOCK_SIZE>>>(g_d,d_level,d_flag,lvl);
            cudaDeviceSynchronize();
            unsigned int flag=0; cudaMemcpy(&flag,d_flag,sizeof(unsigned int),cudaMemcpyDeviceToHost);
            if (!flag) break;
        }

        unsigned int h_level[9];
        cudaMemcpy(h_level,d_level,NV*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        printf("Fig 15.1 BFS from root 0:\n  GPU:      ");
        for(int i=0;i<9;i++) printf("%u ",h_level[i]);
        printf("\n  Expected: "); for(int i=0;i<9;i++) printf("%u ",exp[i]);
        printf("\n  %s\n\n", verify(exp,h_level,NV)?"PASS":"FAIL");

        free(h_cs); free(h_cd);
        cudaFree(g_d.src); cudaFree(g_d.dst); cudaFree(d_level); cudaFree(d_flag);
    }

    // ── Large test ────────────────────────────────────────────────────────────
    {
        unsigned int N=4096, D=8;
        unsigned int *h_sp, *h_csr_dst, NE;
        gen_csr(N, D, &h_sp, &h_csr_dst, &NE);

        unsigned int *h_cs, *h_cd;
        csr_to_coo(h_sp, h_csr_dst, N, NE, &h_cs, &h_cd);
        printf("Large test: %u vertices, %u edges\n", N, NE);

        unsigned int *ref=(unsigned int *)malloc(N*sizeof(unsigned int));
        bfs_cpu(h_sp, h_csr_dst, N, 0, ref);

        COOGraph g_d;
        cudaMalloc(&g_d.src,NE*sizeof(unsigned int)); cudaMalloc(&g_d.dst,NE*sizeof(unsigned int));
        g_d.numVertices=N; g_d.numEdges=NE;
        cudaMemcpy(g_d.src,h_cs,NE*sizeof(unsigned int),cudaMemcpyHostToDevice);
        cudaMemcpy(g_d.dst,h_cd,NE*sizeof(unsigned int),cudaMemcpyHostToDevice);

        unsigned int *init=(unsigned int *)malloc(N*sizeof(unsigned int));
        for(unsigned int i=0;i<N;i++) init[i]=UINT_INF; init[0]=0;
        unsigned int *d_level,*d_flag;
        cudaMalloc(&d_level,N*sizeof(unsigned int)); cudaMalloc(&d_flag,sizeof(unsigned int));
        cudaMemcpy(d_level,init,N*sizeof(unsigned int),cudaMemcpyHostToDevice);

        cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        int nb=(NE+BLOCK_SIZE-1)/BLOCK_SIZE; unsigned int lev=0;
        cudaEventRecord(t0);
        for (unsigned int lvl=1;lvl<N;lvl++) {
            cudaMemset(d_flag,0,sizeof(unsigned int));
            bfs_edge_kernel<<<nb,BLOCK_SIZE>>>(g_d,d_level,d_flag,lvl);
            cudaDeviceSynchronize(); lev++;
            unsigned int flag=0; cudaMemcpy(&flag,d_flag,sizeof(unsigned int),cudaMemcpyDeviceToHost);
            if (!flag) break;
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms,t0,t1);

        unsigned int *gpu=(unsigned int *)malloc(N*sizeof(unsigned int));
        cudaMemcpy(gpu,d_level,N*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        printf("  %u threads/level (%u edges vs. %u vertices)\n", NE, NE, N);
        printf("  GPU time: %.3f ms  %u level iters  %s\n\n", ms, lev,
               verify(ref,gpu,N)?"PASS":"FAIL");

        printf("Edge-centric trade-offs (§15.4):\n");
        printf("  + More parallelism: %u edge threads vs. %u vertex threads\n", NE, N);
        printf("  + Uniform work per thread (exactly 1 edge traversal)\n");
        printf("  − Checks all %u edges per level (even irrelevant ones)\n", NE);
        printf("  − Uses COO storage (src[] adds %u ints vs. CSR srcPtrs[] = %u ints)\n",
               NE, N+1);

        free(h_sp);free(h_csr_dst);free(h_cs);free(h_cd);free(ref);free(init);free(gpu);
        cudaFree(g_d.src);cudaFree(g_d.dst);cudaFree(d_level);cudaFree(d_flag);
        cudaEventDestroy(t0);cudaEventDestroy(t1);
    }
    return 0;
}
