// §9.1 Background  §9.2 Atomic operations and a basic histogram kernel
// §9.3 Latency and throughput of atomic operations
//
// A histogram counts the number of occurrences of data values in each interval.
// For a text histogram of the alphabet (§9.1 / Fig 9.2) each value interval
// spans four consecutive letters: a-d, e-h, …, y-z → 7 bins total.
//   bin = (data[i] - 'a') / 4
//
// The race condition (§9.2 / Figs 9.4–9.5):
//   Sequential code: histo[bin]++  (safe — single thread)
//   Naïve parallel:  histo[bin]++  (WRONG — read-modify-write race)
//   Fixed parallel:  atomicAdd(&histo[bin], 1)   (Fig 9.6)
//
// The contention bottleneck (§9.3 / Fig 9.7):
//   atomicAdd serialises all updates to the same location.
//   If M threads all update the same bin, throughput = 1 op / (2 × latency).
//   Example: 200-cycle DRAM latency → max throughput = 1/(400 cycles).
//   For 7 uniform bins across 1 M threads: effective throughput ≈ 7× better.
//
// This file demonstrates all three cases so the race condition is visible.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define NUM_BINS  7     // 7 four-letter intervals across the 26-letter alphabet
#define BLOCK_SIZE 256

// ── Figure 9.2: sequential CPU histogram ─────────────────────────────────────
void histogram_sequential(const char *data, unsigned int length,
                          unsigned int *histo) {
    for (unsigned int i = 0; i < length; i++) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26)
            histo[alphabet_position / 4]++;
    }
}

// ── Naïve GPU kernel — NO atomics (demonstrates race condition) ───────────────
// WARNING: produces incorrect results because histo[bin]++ is not atomic.
__global__ void histo_naive_kernel(const char *data, unsigned int length,
                                    unsigned int *histo) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26)
            histo[alphabet_position / 4]++;    // RACE CONDITION
    }
}

// ── Figure 9.6: atomic histogram kernel ───────────────────────────────────────
// atomicAdd serialises concurrent read-modify-write on the same bin.
// Sequential code: histo[bin]++  →  atomicAdd(&histo[bin], 1)
__global__ void histo_kernel(const char *data, unsigned int length,
                               unsigned int *histo) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26)
            atomicAdd(&histo[alphabet_position / 4], 1);
    }
}

static bool histos_equal(const unsigned int *a, const unsigned int *b, int n) {
    for (int i = 0; i < n; i++)
        if (a[i] != b[i]) return false;
    return true;
}

int main(void) {
    // Build a text string — "programming massively parallel processors" repeated
    const char *phrase = "programmingmassiveleparallelprocessors";
    const unsigned int REPS = 1 << 14;     // repeat ~262K times → ~10 M chars
    unsigned int length = (unsigned int)(strlen(phrase) * REPS);

    char *data_h = (char *)malloc(length);
    for (unsigned int r = 0; r < REPS; r++)
        memcpy(data_h + r * strlen(phrase), phrase, strlen(phrase));

    // CPU reference
    unsigned int cpu_histo[NUM_BINS] = {0};
    histogram_sequential(data_h, length, cpu_histo);

    printf("CPU reference histogram:\n");
    const char *labels[] = {"a-d","e-h","i-l","m-p","q-t","u-x","y-z"};
    for (int b = 0; b < NUM_BINS; b++)
        printf("  [%s]: %u\n", labels[b], cpu_histo[b]);
    printf("  Total: %u  (non-alpha chars ignored)\n\n",
           cpu_histo[0]+cpu_histo[1]+cpu_histo[2]+cpu_histo[3]+
           cpu_histo[4]+cpu_histo[5]+cpu_histo[6]);

    // Device data
    char     *data_d;
    unsigned int *histo_d;
    cudaMalloc(&data_d, length * sizeof(char));
    cudaMalloc(&histo_d, NUM_BINS * sizeof(unsigned int));
    cudaMemcpy(data_d, data_h, length * sizeof(char), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((length + BLOCK_SIZE - 1) / BLOCK_SIZE);

    unsigned int gpu_histo[NUM_BINS];

    // ── Naïve (no atomics) ────────────────────────────────────────────────────
    cudaMemset(histo_d, 0, NUM_BINS * sizeof(unsigned int));
    histo_naive_kernel<<<grid, block>>>(data_d, length, histo_d);
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_histo, histo_d, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    bool naive_correct = histos_equal(cpu_histo, gpu_histo, NUM_BINS);
    unsigned int naive_total = 0;
    for (int b = 0; b < NUM_BINS; b++) naive_total += gpu_histo[b];
    printf("Naive GPU (no atomics): %s  total=%u  (expected %u)\n",
           naive_correct ? "CORRECT (unlikely!)" : "WRONG — race condition",
           naive_total, length > 0 ? cpu_histo[0]+cpu_histo[1]+cpu_histo[2]+
           cpu_histo[3]+cpu_histo[4]+cpu_histo[5]+cpu_histo[6] : 0);

    // ── Atomic kernel (Fig 9.6) ────────────────────────────────────────────────
    cudaMemset(histo_d, 0, NUM_BINS * sizeof(unsigned int));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    histo_kernel<<<grid, block>>>(data_d, length, histo_d);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(gpu_histo, histo_d, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Atomic GPU (Fig 9.6):   %s  time=%.3f ms\n\n",
           histos_equal(cpu_histo, gpu_histo, NUM_BINS) ? "PASS" : "FAIL", ms);

    // ── §9.3 Contention analysis ───────────────────────────────────────────────
    unsigned int total_alpha = 0;
    for (int b = 0; b < NUM_BINS; b++) total_alpha += cpu_histo[b];
    printf("Contention analysis (§9.3):\n");
    printf("  Total alpha chars: %u  across %d bins\n", total_alpha, NUM_BINS);
    for (int b = 0; b < NUM_BINS; b++) {
        float pct = cpu_histo[b] * 100.0f / total_alpha;
        printf("  bin [%s]: %5.1f%%  (avg updates/atomicAdd serialisation)\n",
               labels[b], pct);
    }
    printf("\nWith uniform distribution across 7 bins:\n");
    printf("  Throughput boost vs single-bin: ~7x\n");
    printf("  With heavy m-p bias: throughput limited by m-p contention.\n");

    free(data_h);
    cudaFree(data_d); cudaFree(histo_d);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
