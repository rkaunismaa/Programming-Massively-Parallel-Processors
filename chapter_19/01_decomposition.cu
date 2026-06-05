// §19.3  Problem Decomposition — Output-centric (gather) vs Input-centric (scatter)
//
// Chapter 19 identifies two universal decomposition strategies (Fig. 19.3):
//
//   Output-centric (Fig. 19.3A): each thread owns one output element and
//     gathers contributions from whatever inputs it needs.  No atomics
//     required — only one thread ever writes a given output location.
//
//   Input-centric (Fig. 19.3B): each thread owns one input element and
//     scatters its value to all output elements it contributes to.
//     Multiple threads may update the same output → atomicAdd needed.
//
// This program benchmarks both strategies for a 1-D "windowed sum":
//
//   out[i] = sum of in[j] for all j with |j - i| <= RADIUS
//
// Both kernels produce identical results.  The output-centric kernel is
// expected to run significantly faster because:
//   • No atomics — no serialisation of writes to the same address.
//   • Only one write per thread (register accumulation → single store).
//   • Input reads are sequential (coalesced access).
//
// §19.3 notes that output-centric decomposition is preferred for most
// CUDA workloads for exactly these reasons (stencil, convolution, matrix
// multiply, DCS/gather all use output-centric decomposition).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N      (1 << 20)   // 1 M elements
#define RADIUS 32          // window half-width; each source touches 2*R+1 outputs
#define BLOCK  256

// ── Output-centric kernel (gather, Fig. 19.3A) ────────────────────────────────
// Thread i computes: out[i] = sum_{j=i-R}^{i+R} in[j]
// No atomics — thread i is the sole writer of out[i].
__global__ void gather_kernel(const float * __restrict__ in,
                                     float * __restrict__ out,
                              int n, int R)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sum = 0.0f;
    int jlo = max(0, i - R);
    int jhi = min(n - 1, i + R);
    for (int j = jlo; j <= jhi; j++)
        sum += in[j];
    out[i] = sum;   // single write, no contention
}

// ── Input-centric kernel (scatter, Fig. 19.3B) ────────────────────────────────
// Thread j reads in[j] and distributes it to every out[i] with |i-j| <= R.
// Multiple threads write to the same out[i] → atomicAdd required.
__global__ void scatter_kernel(const float * __restrict__ in,
                                      float * __restrict__ out,
                               int n, int R)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float val = in[j];
    int ilo = max(0, j - R);
    int ihi = min(n - 1, j + R);
    for (int i = ilo; i <= ihi; i++)
        atomicAdd(&out[i], val);   // serialised when multiple threads target same i
}

static double elapsed_ms(cudaEvent_t a, cudaEvent_t b)
{
    float ms; cudaEventElapsedTime(&ms, a, b); return (double)ms;
}

int main(void)
{
    printf("=== §19.3  Output-centric (gather) vs Input-centric (scatter) ===\n\n");
    printf("Problem : windowed sum  out[i] = sum_{|j-i|<=R} in[j]\n");
    printf("N=%d  RADIUS=%d  window_width=%d\n\n", N, RADIUS, 2*RADIUS+1);

    size_t bytes = (size_t)N * sizeof(float);

    // All-ones input: every interior out[i] should equal 2*RADIUS+1.
    float *h_in  = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);

    // ── Warm-up ────────────────────────────────────────────────────────────────
    cudaMemset(d_out, 0, bytes);
    gather_kernel<<<grid, block>>>(d_in, d_out, N, RADIUS);
    cudaMemset(d_out, 0, bytes);
    scatter_kernel<<<grid, block>>>(d_in, d_out, N, RADIUS);
    cudaDeviceSynchronize();

    // ── Output-centric (gather) ────────────────────────────────────────────────
    cudaMemset(d_out, 0, bytes);
    cudaEventRecord(ev0);
    gather_kernel<<<grid, block>>>(d_in, d_out, N, RADIUS);
    cudaEventRecord(ev1);
    cudaDeviceSynchronize();
    double t_gather = elapsed_ms(ev0, ev1);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    float mid_gather = h_out[N / 2];

    // ── Input-centric (scatter) ────────────────────────────────────────────────
    cudaMemset(d_out, 0, bytes);
    cudaEventRecord(ev0);
    scatter_kernel<<<grid, block>>>(d_in, d_out, N, RADIUS);
    cudaEventRecord(ev1);
    cudaDeviceSynchronize();
    double t_scatter = elapsed_ms(ev0, ev1);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    float mid_scatter = h_out[N / 2];

    // ── Results ───────────────────────────────────────────────────────────────
    int expected = 2 * RADIUS + 1;

    printf("Output-centric (gather, no atomics): %7.2f ms  "
           "out[N/2] = %.0f\n", t_gather, mid_gather);
    printf("Input-centric  (scatter, atomicAdd): %7.2f ms  "
           "out[N/2] = %.0f  (%.2fx slower)\n\n",
           t_scatter, mid_scatter, t_scatter / t_gather);

    printf("Expected out[N/2] = %d  (%s)\n\n", expected,
           (fabsf(mid_gather - expected) < 0.5f &&
            fabsf(mid_scatter - expected) < 0.5f) ? "PASS" : "FAIL");

    printf("Key observations (§19.3):\n");
    printf("  • Gather : thread i accumulates %d reads then does 1 write.\n",
           2*RADIUS+1);
    printf("    No race condition — output-centric decomposition is safe.\n");
    printf("  • Scatter: thread j does %d atomicAdd operations.\n",
           2*RADIUS+1);
    printf("    With RADIUS=%d, each output bin receives writes from %d threads.\n",
           RADIUS, 2*RADIUS+1);
    printf("    Contention serialises those writes, multiplying latency.\n");
    printf("  • The output-centric (gather) decomposition is preferred for\n");
    printf("    stencil, convolution, matrix multiply, and DCS (§19.3).\n");
    printf("  • Input-centric decomposition is preferred only when the\n");
    printf("    number of outputs is small, load is imbalanced, or the\n");
    printf("    mapping from inputs to outputs is hard to invert (histogram).\n");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return 0;
}
