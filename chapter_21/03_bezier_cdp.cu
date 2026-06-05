// §21.3  Bezier curve tessellation — with CUDA Dynamic Parallelism
//        Figs 21.7 and 21.12
//
// The original flat kernel (file 02) assigns one block per curve.  Blocks
// whose curve requires few vertices leave threads idle; blocks with many
// vertices serialise work across the block.  The workload imbalance across
// SMs limits occupancy.
//
// CDP redesign (Fig 21.7):
//   computeBezierLines_parent  — one thread per curve (not one block).
//     • determines nVertices from curvature
//     • calls cudaMalloc to allocate exactly the needed vertex storage
//     • launches computeBezierLines_child<<<ceil(nVerts/32), 32>>>
//
//   computeBezierLines_child   — tessellates the points for one curve.
//     Child grid size matches the work exactly → no wasted threads.
//
//   freeVertexMem              — reclaims device-side cudaMalloc allocations
//     (device-allocated memory must be freed by a device kernel, Fig 21.7
//      lines 42-47).
//
// Stream optimization (Fig 21.12):
//   By default all child grids launched by threads in the *same* parent
//   block share the NULL stream of that block and are serialised.
//   Creating a per-thread non-blocking stream places each child in its
//   own stream, enabling concurrent execution of child grids.
//
// Pending launch pool (§21.5):
//   The runtime maintains a fixed pool of 2048 pending launches.  When
//   N_LINES > 2048 we must raise the pool size with cudaDeviceSetLimit().

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N_LINES         512
#define MAX_TESS_POINTS 256   // upper bound; actual size is dynamic
#define CHILD_BLOCK      32

struct BezierLine {
    float2  CP[3];           // control points
    float2 *vertexPos;       // device pointer (cudaMalloc'd by parent kernel)
    int     nVertices;
};

__device__ float computeCurvature(const BezierLine &line) {
    float dx = line.CP[2].x - line.CP[0].x;
    float dy = line.CP[2].y - line.CP[0].y;
    float chord = sqrtf(dx*dx + dy*dy);
    if (chord < 1e-6f) return 0.f;
    float mx = 0.5f*(line.CP[0].x + line.CP[2].x);
    float my = 0.5f*(line.CP[0].y + line.CP[2].y);
    float dvx = line.CP[1].x - mx, dvy = line.CP[1].y - my;
    return sqrtf(dvx*dvx + dvy*dvy) / chord;
}

// ── Fig 21.7 lines 23-41: child kernel ───────────────────────────────────────
__global__ void computeBezierLines_child(int lineIdx, BezierLine *bLines,
                                         int nTessPoints) {
    int vi = threadIdx.x + blockDim.x * blockIdx.x;  // Fig 21.7 line 25
    if (vi >= nTessPoints) return;
    float u   = (nTessPoints == 1) ? 0.f
                                   : (float)vi / (float)(nTessPoints - 1);
    float omu = 1.0f - u;
    float B3u[3] = { omu*omu, 2.0f*u*omu, u*u };

    float2 pos = make_float2(0.f, 0.f);
    for (int k = 0; k < 3; k++)
        pos = make_float2(pos.x + B3u[k] * bLines[lineIdx].CP[k].x,
                          pos.y + B3u[k] * bLines[lineIdx].CP[k].y);
    bLines[lineIdx].vertexPos[vi] = pos;             // Fig 21.7 line 39
}

// ── Fig 21.7 lines 6-22: parent kernel ───────────────────────────────────────
// One thread per curve.  Work per thread is tiny; the child handles the bulk.
__global__ void computeBezierLines_parent(BezierLine *bLines, int nLines,
                                          bool use_streams) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;  // Fig 21.7 line 08
    if (idx >= nLines) return;

    float curvature = computeCurvature(bLines[idx]);   // Fig 21.7 line 12

    int nv = min(max((int)(curvature * 16.0f), 4), MAX_TESS_POINTS);
    bLines[idx].nVertices = nv;                        // Fig 21.7 line 14 (analog)

    // Allocate exactly the memory needed (Fig 21.7 line 15)
    cudaMalloc((void **)&bLines[idx].vertexPos, nv * sizeof(float2));

    int nblocks = (int)ceilf((float)nv / CHILD_BLOCK);

    if (use_streams) {
        // Fig 21.12: per-thread stream so sibling child grids can run concurrently
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        computeBezierLines_child<<<nblocks, CHILD_BLOCK, 0, stream>>>(
            idx, bLines, nv);                          // Fig 21.12 launch
        cudaStreamDestroy(stream);
    } else {
        // Fig 21.7 line 19: default NULL stream → all children in same block
        // are serialised
        computeBezierLines_child<<<nblocks, CHILD_BLOCK>>>(idx, bLines, nv);
    }
}

// ── Device-side verification ──────────────────────────────────────────────────
// Verify Bezier positions directly on the GPU.  Device-side cudaMalloc memory
// is readable by GPU kernels but NOT reliably by host cudaMemcpy, so we
// verify in-place and write a pass/fail flag to a regular device allocation.
__global__ void verifyBezierLines(const BezierLine *bLines, int nLines,
                                   int *d_fail_count) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nLines) return;

    int nv = bLines[b].nVertices;
    if (nv <= 0 || bLines[b].vertexPos == NULL) { atomicAdd(d_fail_count, 1); return; }

    for (int vi = 0; vi < nv; vi++) {
        float u   = (nv == 1) ? 0.f : (float)vi / (float)(nv - 1);
        float omu = 1.f - u;
        float B[3] = { omu*omu, 2.f*u*omu, u*u };
        float2 exp = make_float2(0.f, 0.f);
        for (int k = 0; k < 3; k++)
            exp = make_float2(exp.x + B[k]*bLines[b].CP[k].x,
                              exp.y + B[k]*bLines[b].CP[k].y);
        float2 got = bLines[b].vertexPos[vi];
        if (fabsf(got.x - exp.x) > 1e-5f || fabsf(got.y - exp.y) > 1e-5f) {
            atomicAdd(d_fail_count, 1);
            return;
        }
    }
}

// ── Fig 21.7 lines 42-47: free device-allocated vertex memory ─────────────────
// Memory allocated by cudaMalloc inside a kernel must be freed by a kernel.
__global__ void freeVertexMem(BezierLine *bLines, int nLines) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nLines)
        cudaFree(bLines[idx].vertexPos);  // Fig 21.7 line 46
}


static void run_and_verify(BezierLine *d_bLines, BezierLine *h_input,
                            bool use_streams, const char *label) {
    cudaMemcpy(d_bLines, h_input, N_LINES * sizeof(BezierLine), cudaMemcpyHostToDevice);

    // §21.5 Pending launch pool: increase pool to at least N_LINES
    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, N_LINES);

    int parent_threads = 128;
    int parent_blocks  = (N_LINES + parent_threads - 1) / parent_threads;
    computeBezierLines_parent<<<parent_blocks, parent_threads>>>(
        d_bLines, N_LINES, use_streams);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("  CUDA error after parent: %s\n", cudaGetErrorString(err));

    // Verify on device: device-side cudaMalloc memory is not reliably
    // accessible via host cudaMemcpy; read-back the results in a GPU kernel.
    int *d_fail_count;
    cudaMalloc(&d_fail_count, sizeof(int));
    cudaMemset(d_fail_count, 0, sizeof(int));
    verifyBezierLines<<<parent_blocks, parent_threads>>>(d_bLines, N_LINES, d_fail_count);
    cudaDeviceSynchronize();

    int h_fail_count = 0;
    cudaMemcpy(&h_fail_count, d_fail_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_fail_count);

    // Copy nVertices stats (but NOT vertexPos data) for reporting
    int total_verts = 0;
    BezierLine *h_result = (BezierLine *)malloc(N_LINES * sizeof(BezierLine));
    cudaMemcpy(h_result, d_bLines, N_LINES * sizeof(BezierLine), cudaMemcpyDeviceToHost);
    for (int b = 0; b < N_LINES; b++) total_verts += h_result[b].nVertices;
    free(h_result);

    printf("%-30s %s  (%d lines, %d total verts)\n",
           label, h_fail_count == 0 ? "PASS" : "FAIL", N_LINES, total_verts);

    // Free device-allocated vertex memory via a device kernel
    freeVertexMem<<<parent_blocks, parent_threads>>>(d_bLines, N_LINES);
    cudaDeviceSynchronize();
}

int main(void) {
    printf("=== Bezier Tessellation With CDP (§21.3, Figs 21.7/21.12) ===\n\n");

    srand(42);
    BezierLine *h_input = (BezierLine *)malloc(N_LINES * sizeof(BezierLine));
    for (int i = 0; i < N_LINES; i++) {
        for (int k = 0; k < 3; k++) {
            h_input[i].CP[k].x = (rand()/(float)RAND_MAX)*2.f - 1.f;
            h_input[i].CP[k].y = (rand()/(float)RAND_MAX)*2.f - 1.f;
        }
        h_input[i].vertexPos = NULL;
        h_input[i].nVertices = 0;
    }

    BezierLine *d_bLines;
    cudaMalloc(&d_bLines, N_LINES * sizeof(BezierLine));

    run_and_verify(d_bLines, h_input, false, "CDP (NULL stream, Fig 21.7):");
    run_and_verify(d_bLines, h_input, true,  "CDP (per-thread stream, Fig 21.12):");

    printf("\nCDP advantages (§21.3):\n");
    printf("  • vertexPos allocated dynamically — no MAX_TESS_POINTS waste\n");
    printf("  • One parent thread per curve — balanced parent workload\n");
    printf("  • Child grid sized to actual vertex count — no idle threads\n");
    printf("  • Per-thread streams (Fig 21.12) allow child grids to overlap\n");
    printf("  • freeVertexMem kernel frees device-malloc'd memory (§21.3)\n");

    free(h_input);
    cudaFree(d_bLines);
    return 0;
}
