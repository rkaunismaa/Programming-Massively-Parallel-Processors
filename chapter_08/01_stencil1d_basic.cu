// §8.1 Background — 1D stencil: the conceptual foundation
//
// A stencil computes each output grid point as a weighted combination of
// neighbouring input grid points.  This is identical to convolution but the
// weights come from discretisation of a differential equation rather than
// from an image-processing filter.
//
// 1D finite-difference first derivative (§8.1 / Fig 8.2A — order-1 stencil):
//   FD[i] = (F[i+1] - F[i-1]) / (2*h)
//   Stencil weights: c_left = -1/(2h), c_center = 0, c_right = +1/(2h)
//
// 1D order-2 stencil (Fig 8.2B — five-point): also shown for the second
// derivative approximation using the three-point Laplacian:
//   FD2[i] = (F[i-1] - 2*F[i] + F[i+1]) / h^2
//
// Boundary conditions (§8.2 / Fig 8.5):
//   Boundary points (first and last) hold boundary conditions and are NOT
//   updated during a sweep.  Only interior points [1, N-2] are written.
//
// Arithmetic intensity (basic 1D kernel):
//   3-point stencil: 3 FLOP per output, 3×4 bytes loaded → 0.25 OP/B
//   5-point stencil: 5 FLOP per output, 5×4 bytes loaded → 0.25 OP/B

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ── 1D order-1 (three-point) stencil kernel ───────────────────────────────────
// FD[i] = c_l * in[i-1] + c_c * in[i] + c_r * in[i+1]
// Boundary cells (i == 0 or i == N-1) are not updated.
__global__ void stencil1d_3pt_kernel(const float *in, float *out,
                                      float c_l, float c_c, float c_r,
                                      unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Interior points only — boundary cells hold boundary conditions
    if (i >= 1 && i < N - 1)
        out[i] = c_l * in[i-1] + c_c * in[i] + c_r * in[i+1];
    else if (i < N)
        out[i] = in[i];   // copy boundary unchanged
}

// ── 1D order-2 (five-point) stencil kernel ────────────────────────────────────
// FD2[i] = c0*in[i-2] + c1*in[i-1] + c2*in[i] + c3*in[i+1] + c4*in[i+2]
__global__ void stencil1d_5pt_kernel(const float *in, float *out,
                                      float c0, float c1, float c2,
                                      float c3, float c4,
                                      unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 && i < N - 2)
        out[i] = c0*in[i-2] + c1*in[i-1] + c2*in[i] + c3*in[i+1] + c4*in[i+2];
    else if (i < N)
        out[i] = in[i];
}

// ── CPU reference ─────────────────────────────────────────────────────────────
static void cpu_stencil1d_3pt(const float *in, float *out,
                               float c_l, float c_c, float c_r,
                               unsigned int N) {
    out[0] = in[0];
    for (unsigned int i = 1; i < N-1; i++)
        out[i] = c_l*in[i-1] + c_c*in[i] + c_r*in[i+1];
    out[N-1] = in[N-1];
}

static int verify(const float *ref, const float *gpu, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        float err = fabsf(ref[i] - gpu[i]);
        if (err > 1e-4f * (fabsf(ref[i]) + 1.0f)) {
            printf("  MISMATCH i=%u  ref=%.6f  gpu=%.6f\n", i, ref[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    const unsigned int N = 1 << 20;   // 1 M grid points
    const float H = 1.0f / N;         // grid spacing

    float *in_h  = (float *)malloc(N * sizeof(float));
    float *out_h = (float *)malloc(N * sizeof(float));
    float *ref_h = (float *)malloc(N * sizeof(float));

    // Smooth input: f(x) = sin(2πx) for x ∈ [0,1)
    for (unsigned int i = 0; i < N; i++)
        in_h[i] = sinf(2.0f * (float)M_PI * i * H);

    float *in_d, *out_d;
    cudaMalloc(&in_d,  N * sizeof(float));
    cudaMalloc(&out_d, N * sizeof(float));
    cudaMemcpy(in_d, in_h, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + 255) / 256);

    // ── First-derivative stencil: c_left = -1/(2h), c_center = 0, c_right = +1/(2h)
    float c_l = -1.0f / (2.0f * H);
    float c_c =  0.0f;
    float c_r =  1.0f / (2.0f * H);

    stencil1d_3pt_kernel<<<grid, block>>>(in_d, out_d, c_l, c_c, c_r, N);
    cudaDeviceSynchronize();
    cudaMemcpy(out_h, out_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    cpu_stencil1d_3pt(in_h, ref_h, c_l, c_c, c_r, N);

    printf("3-point 1D stencil (first derivative):  %s\n",
           verify(ref_h, out_h, N) ? "PASS" : "FAIL");

    // Spot-check: df/dx = 2π*cos(2πx); check at midpoint
    unsigned int mid = N / 2;
    float exact = 2.0f * (float)M_PI * cosf(2.0f * (float)M_PI * mid * H);
    printf("  At i=%u: GPU=%.4f  exact=%.4f  (h=%.2e)\n",
           mid, out_h[mid], exact, (double)H);

    // ── Second-derivative stencil: (f[i-1] - 2*f[i] + f[i+1]) / h^2
    // This is the 1D Laplacian (order-1 stencil, three-point)
    float cd_l = 1.0f / (H*H);
    float cd_c = -2.0f / (H*H);
    float cd_r = 1.0f / (H*H);

    stencil1d_3pt_kernel<<<grid, block>>>(in_d, out_d, cd_l, cd_c, cd_r, N);
    cudaDeviceSynchronize();
    cudaMemcpy(out_h, out_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    // d²/dx²[sin(2πx)] = -(2π)² sin(2πx)
    float exact2 = -(2.0f*(float)M_PI)*(2.0f*(float)M_PI) * sinf(2.0f*(float)M_PI*mid*H);
    printf("3-point 1D stencil (Laplacian, 2nd deriv):\n");
    printf("  At i=%u: GPU=%.4f  exact=%.4f\n", mid, out_h[mid], exact2);

    printf("\nStencil patterns (Fig 8.2):\n");
    printf("  3-point (order 1): [-1, 0, +1]   (first derivative, scaled)\n");
    printf("  3-point Laplacian: [+1, -2, +1]  (second derivative)\n");
    printf("  5-point (order 2): uses i±2 neighbours\n");
    printf("Arithmetic intensity: 3 FLOP / (3×4 B) = 0.25 OP/B\n");
    printf("Boundary points: NOT updated (hold boundary conditions, §8.2)\n");

    free(in_h); free(out_h); free(ref_h);
    cudaFree(in_d); cudaFree(out_d);
    return 0;
}
