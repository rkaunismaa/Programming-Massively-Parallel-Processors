// §19.1  Amdahl's Law — how the sequential fraction limits application speedup
//
// §19.1 shows (molecular dynamics example, Fig. 19.1) that a large-scale
// application consists of multiple modules.  A parallel programmer must
// decide which modules are worth accelerating on the GPU.
//
// Amdahl's Law (§19.1):
//
//   Application speedup = 1 / (f_serial + f_parallel / kernel_speedup)
//
//   where  f_serial   = fraction of total work that stays on the CPU
//          f_parallel = 1 - f_serial
//          kernel_speedup = speedup of the GPU kernel over sequential CPU
//
// Book example (§19.1): nonbonded force = 95% of time, accelerated 100×.
//   Without overlap: speedup = 1/(5% + 95%/100) ≈ 17×
//   With    overlap: speedup = 1/5%             = 20×
//
// This program:
//   1. Measures the raw GPU kernel time for a large SAXPY (the parallel part).
//   2. Sweeps over serial fractions from 0% to 20% and prints the Amdahl
//      speedup table — both theoretical and empirically measured by adding
//      real CPU busy-work that simulates the serial module.
//   3. Prints the achieved speedup, showing how even small serial fractions
//      crush the benefit of a powerful GPU.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define N       (1 << 23)   // 8 M floats — substantial parallel work
#define BLOCK   256
#define NWARM   3
#define NITER   10

// ── Parallel kernel: SAXPY (the GPU-accelerated module) ──────────────────────
__global__ void saxpy(float a,
                      const float * __restrict__ x,
                            float * __restrict__ y,
                      int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// ── CPU serial busy-work: sequential reduction.
// volatile float s prevents the compiler from eliminating the loop.
__attribute__((noinline))
static float cpu_serial_work(const float *h, int n)
{
    volatile float s = 0.0f;
    for (int i = 0; i < n; i++) s += h[i];
    return (float)s;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

int main(void)
{
    printf("=== §19.1  Amdahl's Law — Sequential Fraction vs Application Speedup ===\n\n");

    // ── Allocate device + host buffers ────────────────────────────────────────
    size_t bytes = (size_t)N * sizeof(float);
    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    float *h_buf = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_buf[i] = (float)(i & 0xFF) * 0.01f;
    cudaMemcpy(d_x, h_buf, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_buf, bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK);

    // ── Warm-up ───────────────────────────────────────────────────────────────
    for (int i = 0; i < NWARM; i++)
        saxpy<<<grid, block>>>(2.0f, d_x, d_y, N);
    cudaDeviceSynchronize();

    // ── Measure pure GPU kernel time T_gpu ───────────────────────────────────
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);

    cudaEventRecord(ev0);
    for (int i = 0; i < NITER; i++)
        saxpy<<<grid, block>>>(2.0f, d_x, d_y, N);
    cudaEventRecord(ev1);
    cudaDeviceSynchronize();
    float gpu_ms_f;
    cudaEventElapsedTime(&gpu_ms_f, ev0, ev1);
    double T_gpu = gpu_ms_f / NITER;

    printf("Parallel kernel (SAXPY, N=%d): T_gpu = %.3f ms/iter\n\n", N, T_gpu);

    // ── Measure CPU serial work time for varying problem sizes ────────────────
    // We calibrate: for the host reduction over k floats, how long does it take?
    // Then pick k_serial so that cpu_time ≈ f * T_total, where
    // T_total ≡ T_gpu * gpu_speedup (the hypothetical sequential baseline).

    const double gpu_speedup = 100.0;
    const double T_seq_total = T_gpu * gpu_speedup;   // hypothetical baseline

    printf("Assumed GPU kernel speedup over sequential: %.0f×\n", gpu_speedup);
    printf("Equivalent sequential baseline time:        %.2f ms\n\n", T_seq_total);

    // Calibrate CPU throughput: how many floats can we reduce per ms?
    int cal_n = 1 << 20;
    float *h_cal = (float *)malloc((size_t)cal_n * sizeof(float));
    for (int i = 0; i < cal_n; i++) h_cal[i] = 1.0f;
    double t0 = now_ms();
    float dummy = cpu_serial_work(h_cal, cal_n);
    double t1 = now_ms();
    (void)dummy;
    double cpu_floats_per_ms = cal_n / (t1 - t0 + 1e-9);
    free(h_cal);

    // ── Sweep serial fractions ────────────────────────────────────────────────
    double fracs[] = {0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20};
    int nf = (int)(sizeof(fracs) / sizeof(fracs[0]));

    printf("%-10s  %-12s  %-14s  %-14s  %-12s  %-12s\n",
           "f_serial", "T_serial", "T_parallel", "T_total",
           "Amdahl(theory)", "Measured");
    printf("%-10s  %-12s  %-14s  %-14s  %-12s  %-12s\n",
           "--------", "--------", "----------", "-------",
           "--------------", "--------");

    for (int fi = 0; fi < nf; fi++) {
        double f = fracs[fi];

        // Theoretical Amdahl
        double t_serial_theory   = f * T_seq_total;
        double t_parallel_theory = (1.0 - f) * T_seq_total / gpu_speedup;
        double t_app_theory      = t_serial_theory + t_parallel_theory;
        double speedup_theory    = T_seq_total / t_app_theory;

        // Empirical: run GPU kernel + CPU serial busy-work in series
        int n_serial = (int)(t_serial_theory * cpu_floats_per_ms);
        float *h_serial = (float *)malloc(((size_t)n_serial + 1) * sizeof(float));
        for (int i = 0; i < n_serial; i++) h_serial[i] = 1.0f;

        // Time the combined application (GPU kernel + serial CPU work).
        // The GPU kernel runs asynchronously; CPU serial work runs on the host
        // while the GPU is busy.  cudaDeviceSynchronize() waits for both.
        volatile float serial_sink = 0.0f;
        double t_measured = 0.0;
        for (int it = 0; it < NITER; it++) {
            double wall0 = now_ms();
            saxpy<<<grid, block>>>(2.0f, d_x, d_y, N);
            if (n_serial > 0)
                serial_sink += cpu_serial_work(h_serial, n_serial);
            cudaDeviceSynchronize();
            t_measured += (now_ms() - wall0);
        }
        t_measured /= NITER;
        (void)serial_sink;

        double speedup_measured = (T_gpu + (f > 0 ? t_serial_theory : 0.0))
                                  / t_measured;
        // Normalise measured speedup relative to T_seq_total
        double app_speedup_meas = T_seq_total / t_measured;

        free(h_serial);

        printf("  %5.1f%%    %8.2f ms    %8.2f ms    %8.2f ms    %8.2f×    %8.2f×\n",
               f * 100.0,
               t_serial_theory,
               t_parallel_theory,
               t_app_theory,
               speedup_theory,
               app_speedup_meas);
    }

    printf("\nFormula: Speedup = 1 / (f_serial + (1 - f_serial) / %.0f)\n\n",
           gpu_speedup);

    printf("Key insights (§19.1):\n");
    printf("  • At 0%% serial: speedup = %.0f× (full GPU benefit)\n", gpu_speedup);
    printf("  • At 5%% serial: speedup = %.1f× — the §19.1 molecular dynamics example\n",
           1.0 / (0.05 + 0.95 / gpu_speedup));
    printf("  • At 10%% serial: speedup = %.1f× (10× ceiling even with 100× GPU)\n",
           1.0 / (0.10 + 0.90 / gpu_speedup));
    printf("  • At 20%% serial: speedup = %.1f× (only 5× ceiling)\n",
           1.0 / (0.20 + 0.80 / gpu_speedup));
    printf("  • Amdahl's law motivates task-level parallelism (§19.1) and\n");
    printf("    overlapping host serial work with device execution.\n");
    printf("  • With full host–device overlap (CPU serial runs while GPU runs)\n");
    printf("    the overhead drops to zero and speedup → %.0f× (the GPU speedup).\n",
           gpu_speedup);

    cudaFree(d_x); cudaFree(d_y);
    free(h_buf);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return 0;
}
