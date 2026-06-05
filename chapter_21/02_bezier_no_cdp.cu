// §21.3  Bezier curve tessellation — without dynamic parallelism
//        Fig 21.6
//
// Computes sample points on N_LINES quadratic Bezier curves.
// Each curve is defined by three control points (P0, P1, P2).
//
// Quadratic Bezier formula:
//   B(t) = (1-t)²·P0 + 2(1-t)t·P1 + t²·P2,   t ∈ [0,1]
//
// Kernel design (Fig 21.6 — without CDP):
//   One block per curve.  Threads within the block collaborate to fill
//   the tessellation array, stepping by blockDim.x each iteration.
//   The curvature determines nVertices; blocks with high curvature do
//   more iterations.  Variable block workloads → control divergence
//   across blocks when scheduled on the same SM.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N_LINES          512
#define MAX_TESS_POINTS  32    // Fig 21.6 line 04: MAX_TESS_POINTS 32

struct BezierLine {
    float2 CP[3];                          // control points (Fig 21.6 line 08)
    float2 vertexPos[MAX_TESS_POINTS];     // tessellated positions (Fig 21.6 line 09)
    int    nVertices;                      // actual count (Fig 21.6 line 10)
};

// Returns a curvature proxy: deviation of P1 from the chord mid-point,
// divided by chord length.  Higher value → more tessellation points needed.
__device__ float computeCurvature(const BezierLine &line) {
    float dx = line.CP[2].x - line.CP[0].x;
    float dy = line.CP[2].y - line.CP[0].y;
    float chord = sqrtf(dx*dx + dy*dy);
    if (chord < 1e-6f) return 0.f;
    float mx = 0.5f * (line.CP[0].x + line.CP[2].x);
    float my = 0.5f * (line.CP[0].y + line.CP[2].y);
    float devx = line.CP[1].x - mx;
    float devy = line.CP[1].y - my;
    return sqrtf(devx*devx + devy*devy) / chord;
}

// ── Fig 21.6: Tessellation kernel (without CDP) ───────────────────────────────
// One block per curve.  The block's threads collaboratively write all
// nTessPoints sample positions.
__global__ void computeBezierLines(BezierLine *bLines, int nLines) {
    int bidx = blockIdx.x;                          // Fig 21.6 line 14
    if (bidx >= nLines) return;

    float curvature = computeCurvature(bLines[bidx]);   // Fig 21.6 line 16

    // Number of tessellation points proportional to curvature (Fig 21.6 lines 20-21)
    int nTessPoints = min(max((int)(curvature * 16.0f), 4), MAX_TESS_POINTS);
    bLines[bidx].nVertices = nTessPoints;

    // Each thread handles a strided subset of vertices (Fig 21.6 lines 23-40)
    for (int inc = 0; inc < nTessPoints; inc += blockDim.x) {
        int idx = inc + threadIdx.x;
        if (idx < nTessPoints) {
            float u   = (float)idx / (float)(nTessPoints - 1);  // Fig 21.6 line 27
            float omu = 1.0f - u;                                // Fig 21.6 line 28
            float B3u[3];                                        // Fig 21.6 line 29
            B3u[0] = omu * omu;                                  // Fig 21.6 line 30
            B3u[1] = 2.0f * u * omu;                             // Fig 21.6 line 31
            B3u[2] = u * u;                                      // Fig 21.6 line 32
            float2 pos = make_float2(0.f, 0.f);                  // Fig 21.6 line 33
            for (int k = 0; k < 3; k++)                         // Fig 21.6 line 34
                pos = make_float2(pos.x + B3u[k] * bLines[bidx].CP[k].x,
                                  pos.y + B3u[k] * bLines[bidx].CP[k].y);
            bLines[bidx].vertexPos[idx] = pos;                  // Fig 21.6 line 39
        }
    }
}

// CPU reference for verification
static void computeBezierLines_cpu(const BezierLine *bLines_h, int nLines,
                                   float2 *ref_pos, int *ref_nv) {
    for (int b = 0; b < nLines; b++) {
        float dx = bLines_h[b].CP[2].x - bLines_h[b].CP[0].x;
        float dy = bLines_h[b].CP[2].y - bLines_h[b].CP[0].y;
        float chord = sqrtf(dx*dx + dy*dy);
        float curv = 0.f;
        if (chord > 1e-6f) {
            float mx = 0.5f*(bLines_h[b].CP[0].x + bLines_h[b].CP[2].x);
            float my = 0.5f*(bLines_h[b].CP[0].y + bLines_h[b].CP[2].y);
            float dvx = bLines_h[b].CP[1].x - mx;
            float dvy = bLines_h[b].CP[1].y - my;
            curv = sqrtf(dvx*dvx + dvy*dvy) / chord;
        }
        int nv = min(max((int)(curv * 16.0f), 4), MAX_TESS_POINTS);
        ref_nv[b] = nv;
        for (int idx = 0; idx < nv; idx++) {
            float u   = (nv == 1) ? 0.f : (float)idx / (float)(nv - 1);
            float omu = 1.f - u;
            float B[3] = { omu*omu, 2.f*u*omu, u*u };
            float2 p = make_float2(0.f, 0.f);
            for (int k = 0; k < 3; k++)
                p = make_float2(p.x + B[k]*bLines_h[b].CP[k].x,
                                p.y + B[k]*bLines_h[b].CP[k].y);
            ref_pos[b * MAX_TESS_POINTS + idx] = p;
        }
    }
}

int main(void) {
    printf("=== Bezier Tessellation Without CDP (§21.3, Fig 21.6) ===\n\n");

    srand(42);

    BezierLine *h_bLines = (BezierLine *)malloc(N_LINES * sizeof(BezierLine));
    for (int i = 0; i < N_LINES; i++) {
        for (int k = 0; k < 3; k++) {
            h_bLines[i].CP[k].x = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            h_bLines[i].CP[k].y = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        }
        h_bLines[i].nVertices = 0;
    }

    // CPU reference
    float2 *ref_pos = (float2 *)malloc(N_LINES * MAX_TESS_POINTS * sizeof(float2));
    int    *ref_nv  = (int    *)malloc(N_LINES * sizeof(int));
    computeBezierLines_cpu(h_bLines, N_LINES, ref_pos, ref_nv);

    BezierLine *d_bLines;
    cudaMalloc(&d_bLines, N_LINES * sizeof(BezierLine));
    cudaMemcpy(d_bLines, h_bLines, N_LINES * sizeof(BezierLine), cudaMemcpyHostToDevice);

    // One block per curve, 32 threads per block
    computeBezierLines<<<N_LINES, 32>>>(d_bLines, N_LINES);
    cudaDeviceSynchronize();

    BezierLine *h_result = (BezierLine *)malloc(N_LINES * sizeof(BezierLine));
    cudaMemcpy(h_result, d_bLines, N_LINES * sizeof(BezierLine), cudaMemcpyDeviceToHost);

    // Verify
    int pass = 1;
    for (int b = 0; b < N_LINES && pass; b++) {
        if (h_result[b].nVertices != ref_nv[b]) { pass = 0; break; }
        for (int v = 0; v < h_result[b].nVertices; v++) {
            float ex = fabsf(h_result[b].vertexPos[v].x - ref_pos[b*MAX_TESS_POINTS+v].x);
            float ey = fabsf(h_result[b].vertexPos[v].y - ref_pos[b*MAX_TESS_POINTS+v].y);
            if (ex > 1e-5f || ey > 1e-5f) { pass = 0; break; }
        }
    }
    printf("%d Bezier curves tessellated, block_dim=32: %s\n",
           N_LINES, pass ? "PASS" : "FAIL");

    int total_verts = 0;
    for (int b = 0; b < N_LINES; b++) total_verts += h_result[b].nVertices;
    printf("Total vertices: %d  (avg %.1f per curve)\n",
           total_verts, (float)total_verts / N_LINES);

    printf("\nDesign note (§21.3): one block per curve; variable nVertices per\n");
    printf("  block causes workload imbalance across SMs. CDP (file 03) fixes\n");
    printf("  this: parent assigns exactly one thread per curve, launches child\n");
    printf("  grid sized to that curve's vertex count.\n");

    free(h_bLines); free(h_result); free(ref_pos); free(ref_nv);
    cudaFree(d_bLines);
    return 0;
}
