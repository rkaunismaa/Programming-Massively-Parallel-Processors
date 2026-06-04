// §15.3  Vertex-centric push (top-down) BFS — Figure 15.6
//
// Each thread is assigned to a vertex.  Per level, a thread checks whether
// its vertex belongs to the previous level (level[v] == currLevel-1).
// If so, it iterates over the vertex's outgoing edges (CSR srcPtrs / dst)
// and labels every unvisited neighbor as belonging to the current level.
//
// This is called the "push" or "top-down" approach because each active
// vertex pushes its depth label forward to its neighbors.
//
// Idempotence: multiple threads may concurrently write the same currLevel
// to the same level[neighbor].  Since they all write the same value, no
// atomicAdd is needed for the level assignment.  The newVertexVisited flag
// is also idempotent (all writers set it to 1).
//
// Trade-offs (§15.3):
//   + Simple kernel; no atomics for level write.
//   − Launches a thread for every vertex every level, even vertices that are
//     irrelevant for that level (wasteful for early / late levels).
//   − Control divergence: threads iterate different numbers of edges.
//   → §15.5 frontier approach eliminates the redundant thread launches.
//
// Graph representation (Fig 15.3 A): CSR
//   srcPtrs[numVertices+1]  — row pointer (start of each vertex's edge list)
//   dst[numEdges]           — destination vertex of each edge

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define UINT_INF   UINT_MAX

typedef struct {
    unsigned int *srcPtrs;   // [numVertices+1]
    unsigned int *dst;        // [numEdges]
    unsigned int  numVertices;
    unsigned int  numEdges;
} CSRGraph;

// ── Kernel (Fig 15.6): one thread per vertex ──────────────────────────────────
__global__ void bfs_push_kernel(CSRGraph g, unsigned int *level,
                                 unsigned int *newVertexVisited,
                                 unsigned int currLevel) {
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < g.numVertices) {
        if (level[v] == currLevel - 1) {   // vertex in previous level
            for (unsigned int e = g.srcPtrs[v]; e < g.srcPtrs[v + 1]; e++) {
                unsigned int nb = g.dst[e];
                if (level[nb] == UINT_INF) {
                    level[nb]          = currLevel;
                    *newVertexVisited  = 1;
                }
            }
        }
    }
}

// ── CPU reference BFS ─────────────────────────────────────────────────────────
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
            if (level[nb] == UINT_INF) {
                level[nb] = level[v] + 1;
                queue[tail++] = nb;
            }
        }
    }
    free(queue);
}

static int verify(const unsigned int *ref, const unsigned int *gpu, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        if (ref[i] != gpu[i]) {
            printf("  MISMATCH v=%u ref=%u gpu=%u\n", i, ref[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

// Random graph: N vertices, each with avgDeg outgoing edges (no self-loops).
static void gen_csr(unsigned int N, unsigned int avgDeg,
                    unsigned int **pp, unsigned int **pd, unsigned int *pE) {
    srand(42);
    *pp = (unsigned int *)malloc((N + 1) * sizeof(unsigned int));
    *pd = (unsigned int *)malloc(N * avgDeg * sizeof(unsigned int));
    unsigned int k = 0;
    (*pp)[0] = 0;
    for (unsigned int v = 0; v < N; v++) {
        for (unsigned int j = 0; j < avgDeg; j++) {
            unsigned int d;
            do { d = rand() % N; } while (d == v);
            (*pd)[k++] = d;
        }
        (*pp)[v + 1] = k;
    }
    *pE = k;
}

int main(void) {
    printf("=== Vertex-Centric Push (Top-Down) BFS (§15.3, Fig 15.6) ===\n\n");

    // ── Small test: Fig 15.1 graph ────────────────────────────────────────────
    {
        unsigned int NV = 9, NE = 15;
        unsigned int h_srcPtrs[] = {0, 2, 4, 7, 9, 11, 12, 13, 15, 15};
        unsigned int h_dst[]     = {1, 2, 3, 4, 5, 6, 7, 4, 8, 5, 8, 8, 8, 0, 6};
        unsigned int root = 0;
        unsigned int exp[]= {0, 1, 1, 2, 2, 2, 2, 2, 3};

        CSRGraph g_d;
        cudaMalloc(&g_d.srcPtrs, (NV+1)*sizeof(unsigned int));
        cudaMalloc(&g_d.dst,     NE*sizeof(unsigned int));
        g_d.numVertices = NV; g_d.numEdges = NE;
        cudaMemcpy(g_d.srcPtrs, h_srcPtrs, (NV+1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(g_d.dst,     h_dst,     NE*sizeof(unsigned int),     cudaMemcpyHostToDevice);

        unsigned int *d_level, *d_flag;
        cudaMalloc(&d_level, NV*sizeof(unsigned int));
        cudaMalloc(&d_flag,  sizeof(unsigned int));

        // Init: all UINT_MAX except root
        unsigned int init_levels[9];
        for (int i = 0; i < 9; i++) init_levels[i] = UINT_INF;
        init_levels[root] = 0;
        cudaMemcpy(d_level, init_levels, NV*sizeof(unsigned int), cudaMemcpyHostToDevice);

        int nb = (NV + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (unsigned int lvl = 1; lvl < NV; lvl++) {
            cudaMemset(d_flag, 0, sizeof(unsigned int));
            bfs_push_kernel<<<nb, BLOCK_SIZE>>>(g_d, d_level, d_flag, lvl);
            cudaDeviceSynchronize();
            unsigned int flag = 0;
            cudaMemcpy(&flag, d_flag, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            if (!flag) break;
        }

        unsigned int h_level[9];
        cudaMemcpy(h_level, d_level, NV*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        printf("Fig 15.1 BFS from root 0:\n  GPU:      ");
        for (int i = 0; i < 9; i++) printf("%u ", h_level[i]);
        printf("\n  Expected: ");
        for (int i = 0; i < 9; i++) printf("%u ", exp[i]);
        printf("\n  %s\n\n", verify(exp, h_level, NV) ? "PASS" : "FAIL");

        cudaFree(g_d.srcPtrs); cudaFree(g_d.dst); cudaFree(d_level); cudaFree(d_flag);
    }

    // ── Large test ────────────────────────────────────────────────────────────
    {
        unsigned int N = 4096, D = 8;
        unsigned int *h_sp, *h_dst, NE;
        gen_csr(N, D, &h_sp, &h_dst, &NE);
        printf("Large test: %u vertices, %u edges (avg %.0f out-degree)\n",
               N, NE, (float)NE/N);

        unsigned int *ref_level = (unsigned int *)malloc(N * sizeof(unsigned int));
        bfs_cpu(h_sp, h_dst, N, 0, ref_level);
        unsigned int reached = 0;
        for (unsigned int i = 0; i < N; i++) if (ref_level[i] != UINT_INF) reached++;
        printf("  CPU BFS: %u/%u vertices reachable from root 0\n", reached, N);

        CSRGraph g_d;
        cudaMalloc(&g_d.srcPtrs, (N+1)*sizeof(unsigned int));
        cudaMalloc(&g_d.dst,     NE*sizeof(unsigned int));
        g_d.numVertices = N; g_d.numEdges = NE;
        cudaMemcpy(g_d.srcPtrs, h_sp,  (N+1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(g_d.dst,     h_dst, NE*sizeof(unsigned int),    cudaMemcpyHostToDevice);

        unsigned int *d_level, *d_flag;
        cudaMalloc(&d_level, N*sizeof(unsigned int));
        cudaMalloc(&d_flag,  sizeof(unsigned int));

        unsigned int *init = (unsigned int *)malloc(N * sizeof(unsigned int));
        for (unsigned int i = 0; i < N; i++) init[i] = UINT_INF;
        init[0] = 0;
        cudaMemcpy(d_level, init, N*sizeof(unsigned int), cudaMemcpyHostToDevice);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        int nb = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int levels_run = 0;
        cudaEventRecord(t0);
        for (unsigned int lvl = 1; lvl < N; lvl++) {
            cudaMemset(d_flag, 0, sizeof(unsigned int));
            bfs_push_kernel<<<nb, BLOCK_SIZE>>>(g_d, d_level, d_flag, lvl);
            cudaDeviceSynchronize();
            levels_run++;
            unsigned int flag = 0;
            cudaMemcpy(&flag, d_flag, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            if (!flag) break;
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);

        unsigned int *gpu_level = (unsigned int *)malloc(N * sizeof(unsigned int));
        cudaMemcpy(gpu_level, d_level, N*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        printf("  GPU time: %.3f ms  %u level iterations  %s\n\n", ms, levels_run,
               verify(ref_level, gpu_level, N) ? "PASS" : "FAIL");

        printf("Push BFS trade-offs (§15.3):\n");
        printf("  + Simple: no atomics (idempotent level write)\n");
        printf("  − Launches %u threads/level regardless of frontier size\n", N);
        printf("  − Most threads do nothing in early/late levels\n");
        printf("  → Frontier approach (file 04) eliminates wasted launches\n");

        free(h_sp); free(h_dst); free(ref_level); free(init); free(gpu_level);
        cudaFree(g_d.srcPtrs); cudaFree(g_d.dst); cudaFree(d_level); cudaFree(d_flag);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    return 0;
}
