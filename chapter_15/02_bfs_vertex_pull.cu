// §15.3  Vertex-centric pull (bottom-up) BFS — Figure 15.8
//
// Each thread is assigned to a vertex.  Per level, a thread checks whether
// its vertex has NOT yet been visited (level[v] == UINT_MAX).  If so, it
// iterates over the vertex's INCOMING edges (CSC dstPtrs / src) looking for
// any neighbor that belongs to the previous level.  Once such a neighbor is
// found, the thread labels its vertex as belonging to the current level and
// BREAKS out of the loop early.
//
// This is called the "pull" or "bottom-up" approach because each unvisited
// vertex reaches back (pulls) to its predecessors to discover whether it
// should be labeled in the current level.
//
// Key difference from push (§15.3 comparison):
//   Push: loops over ALL outgoing edges of active (previously labeled) vertices.
//   Pull: loops over incoming edges of ALL unvisited vertices, but may break
//         early once a single predecessor in the previous level is found.
//
//   For graphs with high degree and variance (e.g. social networks), the early
//   break in pull can substantially reduce load imbalance and control divergence
//   at levels where many vertices have already been visited.
//   At early levels (few labeled vertices), push is usually faster because
//   most threads in pull loop over all incoming edges without finding a hit.
//
// Graph representation: CSC (compressed sparse column)
//   dstPtrs[numVertices+1] — column pointer (incoming edges per vertex)
//   src[numEdges]          — source vertex of each edge

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define UINT_INF   UINT_MAX

typedef struct {
    unsigned int *dstPtrs;   // [numVertices+1]
    unsigned int *src;        // [numEdges]
    unsigned int  numVertices;
    unsigned int  numEdges;
} CSCGraph;

// ── Kernel (Fig 15.8): one thread per vertex ──────────────────────────────────
__global__ void bfs_pull_kernel(CSCGraph g, unsigned int *level,
                                 unsigned int *newVertexVisited,
                                 unsigned int currLevel) {
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < g.numVertices) {
        if (level[v] == UINT_INF) {          // unvisited vertex
            for (unsigned int e = g.dstPtrs[v]; e < g.dstPtrs[v + 1]; e++) {
                unsigned int nb = g.src[e];
                if (level[nb] == currLevel - 1) {   // predecessor in prev level
                    level[v]          = currLevel;
                    *newVertexVisited = 1;
                    break;                    // early exit — one predecessor is enough
                }
            }
        }
    }
}

// ── Build CSC by transposing CSR ──────────────────────────────────────────────
static void csr_to_csc(const unsigned int *srcPtrs, const unsigned int *dst,
                        unsigned int NV, unsigned int NE,
                        unsigned int **dstPtrs, unsigned int **src) {
    *dstPtrs = (unsigned int *)calloc(NV + 1, sizeof(unsigned int));
    *src     = (unsigned int *)malloc(NE * sizeof(unsigned int));
    // Count in-degree
    for (unsigned int e = 0; e < NE; e++) (*dstPtrs)[dst[e] + 1]++;
    // Prefix sum
    for (unsigned int v = 1; v <= NV; v++) (*dstPtrs)[v] += (*dstPtrs)[v - 1];
    // Fill src array
    unsigned int *fill = (unsigned int *)calloc(NV, sizeof(unsigned int));
    for (unsigned int v = 0; v < NV; v++) {
        for (unsigned int e = srcPtrs[v]; e < srcPtrs[v + 1]; e++) {
            unsigned int d   = dst[e];
            unsigned int pos = (*dstPtrs)[d] + fill[d]++;
            (*src)[pos] = v;
        }
    }
    free(fill);
}

// ── CPU reference BFS (using CSR) ────────────────────────────────────────────
static void bfs_cpu(const unsigned int *srcPtrs, const unsigned int *dst,
                    unsigned int numV, unsigned int root, unsigned int *level) {
    for (unsigned int i = 0; i < numV; i++) level[i] = UINT_INF;
    level[root] = 0;
    unsigned int *queue = (unsigned int *)malloc(numV * sizeof(unsigned int));
    unsigned int head = 0, tail = 0;
    queue[tail++] = root;
    while (head < tail) {
        unsigned int v = queue[head++];
        for (unsigned int e = srcPtrs[v]; e < srcPtrs[v + 1]; e++) {
            unsigned int nb = dst[e];
            if (level[nb] == UINT_INF) { level[nb] = level[v] + 1; queue[tail++] = nb; }
        }
    }
    free(queue);
}

static int verify(const unsigned int *ref, const unsigned int *gpu, unsigned int n) {
    for (unsigned int i = 0; i < n; i++)
        if (ref[i] != gpu[i]) { printf("  MISMATCH v=%u ref=%u gpu=%u\n", i, ref[i], gpu[i]); return 0; }
    return 1;
}

static void gen_csr(unsigned int N, unsigned int D,
                    unsigned int **pp, unsigned int **pd, unsigned int *pE) {
    srand(42);
    *pp = (unsigned int *)malloc((N + 1) * sizeof(unsigned int));
    *pd = (unsigned int *)malloc(N * D * sizeof(unsigned int));
    unsigned int k = 0; (*pp)[0] = 0;
    for (unsigned int v = 0; v < N; v++) {
        for (unsigned int j = 0; j < D; j++) {
            unsigned int d; do { d = rand() % N; } while (d == v);
            (*pd)[k++] = d;
        }
        (*pp)[v + 1] = k;
    }
    *pE = k;
}

int main(void) {
    printf("=== Vertex-Centric Pull (Bottom-Up) BFS (§15.3, Fig 15.8) ===\n\n");

    // ── Small test: Fig 15.1 graph ────────────────────────────────────────────
    {
        unsigned int NV = 9, NE = 15;
        unsigned int h_srcPtrs[] = {0, 2, 4, 7, 9, 11, 12, 13, 15, 15};
        unsigned int h_dst[]     = {1, 2, 3, 4, 5, 6, 7, 4, 8, 5, 8, 8, 8, 0, 6};
        unsigned int root = 0;
        unsigned int exp[]= {0, 1, 1, 2, 2, 2, 2, 2, 3};

        unsigned int *h_dstPtrs, *h_src;
        csr_to_csc(h_srcPtrs, h_dst, NV, NE, &h_dstPtrs, &h_src);

        printf("CSC dstPtrs: "); for (int i=0;i<=9;i++) printf("%u ",h_dstPtrs[i]); printf("\n");
        printf("CSC src:     "); for (int i=0;i<15;i++) printf("%u ",h_src[i]); printf("\n\n");

        CSCGraph g_d;
        cudaMalloc(&g_d.dstPtrs, (NV+1)*sizeof(unsigned int));
        cudaMalloc(&g_d.src,     NE*sizeof(unsigned int));
        g_d.numVertices = NV; g_d.numEdges = NE;
        cudaMemcpy(g_d.dstPtrs, h_dstPtrs, (NV+1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(g_d.src,     h_src,     NE*sizeof(unsigned int),     cudaMemcpyHostToDevice);

        unsigned int init[9]; for (int i=0;i<9;i++) init[i]=UINT_INF; init[root]=0;
        unsigned int *d_level, *d_flag;
        cudaMalloc(&d_level, NV*sizeof(unsigned int)); cudaMalloc(&d_flag, sizeof(unsigned int));
        cudaMemcpy(d_level, init, NV*sizeof(unsigned int), cudaMemcpyHostToDevice);

        int nb = (NV + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (unsigned int lvl = 1; lvl < NV; lvl++) {
            cudaMemset(d_flag, 0, sizeof(unsigned int));
            bfs_pull_kernel<<<nb, BLOCK_SIZE>>>(g_d, d_level, d_flag, lvl);
            cudaDeviceSynchronize();
            unsigned int flag = 0;
            cudaMemcpy(&flag, d_flag, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            if (!flag) break;
        }

        unsigned int h_level[9];
        cudaMemcpy(h_level, d_level, NV*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        printf("Fig 15.1 BFS from root 0:\n  GPU:      ");
        for (int i=0;i<9;i++) printf("%u ",h_level[i]);
        printf("\n  Expected: "); for (int i=0;i<9;i++) printf("%u ",exp[i]);
        printf("\n  %s\n\n", verify(exp,h_level,NV)?"PASS":"FAIL");

        free(h_dstPtrs); free(h_src);
        cudaFree(g_d.dstPtrs); cudaFree(g_d.src); cudaFree(d_level); cudaFree(d_flag);
    }

    // ── Large test ────────────────────────────────────────────────────────────
    {
        unsigned int N = 4096, D = 8;
        unsigned int *h_sp, *h_dst, NE;
        gen_csr(N, D, &h_sp, &h_dst, &NE);
        unsigned int *h_dp, *h_src;
        csr_to_csc(h_sp, h_dst, N, NE, &h_dp, &h_src);
        printf("Large test: %u vertices, %u edges\n", N, NE);

        unsigned int *ref = (unsigned int *)malloc(N * sizeof(unsigned int));
        bfs_cpu(h_sp, h_dst, N, 0, ref);

        CSCGraph g_d;
        cudaMalloc(&g_d.dstPtrs, (N+1)*sizeof(unsigned int));
        cudaMalloc(&g_d.src,     NE*sizeof(unsigned int));
        g_d.numVertices = N; g_d.numEdges = NE;
        cudaMemcpy(g_d.dstPtrs, h_dp,  (N+1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(g_d.src,     h_src, NE*sizeof(unsigned int),    cudaMemcpyHostToDevice);

        unsigned int *init = (unsigned int *)malloc(N*sizeof(unsigned int));
        for (unsigned int i=0;i<N;i++) init[i]=UINT_INF; init[0]=0;
        unsigned int *d_level, *d_flag;
        cudaMalloc(&d_level,N*sizeof(unsigned int)); cudaMalloc(&d_flag,sizeof(unsigned int));
        cudaMemcpy(d_level,init,N*sizeof(unsigned int),cudaMemcpyHostToDevice);

        cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        int nb=(N+BLOCK_SIZE-1)/BLOCK_SIZE; unsigned int lev=0;
        cudaEventRecord(t0);
        for (unsigned int lvl=1;lvl<N;lvl++) {
            cudaMemset(d_flag,0,sizeof(unsigned int));
            bfs_pull_kernel<<<nb,BLOCK_SIZE>>>(g_d,d_level,d_flag,lvl);
            cudaDeviceSynchronize(); lev++;
            unsigned int flag=0; cudaMemcpy(&flag,d_flag,sizeof(unsigned int),cudaMemcpyDeviceToHost);
            if (!flag) break;
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms,t0,t1);

        unsigned int *gpu=(unsigned int *)malloc(N*sizeof(unsigned int));
        cudaMemcpy(gpu,d_level,N*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        printf("  GPU time: %.3f ms  %u level iters  %s\n\n", ms, lev,
               verify(ref,gpu,N)?"PASS":"FAIL");

        printf("Pull vs. Push (§15.3):\n");
        printf("  Pull breaks early when one predecessor found → less work per thread\n");
        printf("  Pull uses CSC (incoming edges); push uses CSR (outgoing edges)\n");
        printf("  Push: better for early levels (few labeled vertices)\n");
        printf("  Pull: better for later levels (many vertices already visited)\n");
        printf("  Direction-optimized BFS switches between the two adaptively\n");

        free(h_sp);free(h_dst);free(h_dp);free(h_src);free(ref);free(init);free(gpu);
        cudaFree(g_d.dstPtrs);cudaFree(g_d.src);cudaFree(d_level);cudaFree(d_flag);
        cudaEventDestroy(t0);cudaEventDestroy(t1);
    }
    return 0;
}
