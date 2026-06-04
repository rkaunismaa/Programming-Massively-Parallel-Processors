// §13.2  Sequential LSD radix sort — Figure 13.1
//
// Least-significant-digit (LSD) radix sort for unsigned 32-bit integers.
// One bit per iteration (1-bit radix), so 32 passes cover all bits.
//
// Each pass is a stable two-way partition on the current bit:
//   keys with bit=0 land in the 0-bucket (left half of output)
//   keys with bit=1 land in the 1-bucket (right half of output)
//
// Stability preserves relative order within each bucket, which is what
// allows the multi-pass LSD approach to produce a fully sorted result.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ── Sequential LSD radix sort (1-bit radix, 32 iterations) ───────────────────
static void radix_sort(unsigned int *keys, int n) {
    unsigned int *buf = (unsigned int *)malloc(n * sizeof(unsigned int));

    for (int iter = 0; iter < 32; iter++) {
        // Count 0-bucket size
        int zeros = 0;
        for (int i = 0; i < n; i++)
            if (((keys[i] >> iter) & 1u) == 0) zeros++;

        // Stable scatter: preserve original order within each bucket
        int start0 = 0, start1 = zeros;
        for (int i = 0; i < n; i++) {
            int bit = (keys[i] >> iter) & 1;
            buf[(bit == 0) ? start0++ : start1++] = keys[i];
        }
        memcpy(keys, buf, n * sizeof(unsigned int));
    }
    free(buf);
}

// Stable key-value variant used for the stability test
static void radix_sort_kv(unsigned int *keys, unsigned int *vals, int n) {
    unsigned int *kbuf = (unsigned int *)malloc(n * sizeof(unsigned int));
    unsigned int *vbuf = (unsigned int *)malloc(n * sizeof(unsigned int));
    for (int iter = 0; iter < 32; iter++) {
        int zeros = 0;
        for (int i = 0; i < n; i++)
            if (((keys[i] >> iter) & 1u) == 0) zeros++;
        int s0 = 0, s1 = zeros;
        for (int i = 0; i < n; i++) {
            int dst = ((keys[i] >> iter) & 1) ? s1++ : s0++;
            kbuf[dst] = keys[i]; vbuf[dst] = vals[i];
        }
        memcpy(keys, kbuf, n * sizeof(unsigned int));
        memcpy(vals, vbuf, n * sizeof(unsigned int));
    }
    free(kbuf); free(vbuf);
}

static int cmp_uint(const void *a, const void *b) {
    unsigned int x = *(unsigned int *)a, y = *(unsigned int *)b;
    return (x > y) - (x < y);
}

int main(void) {
    printf("=== Sequential LSD Radix Sort (§13.2, Fig 13.1) ===\n\n");

    // ── Small example from Fig 13.1 (16 four-bit values) ─────────────────────
    {
        unsigned int keys[] = {0xC, 0x3, 0x6, 0x9, 0xF, 0x8, 0x5, 0xA,
                               0xA, 0x6, 0xB, 0xD, 0x4, 0xA, 0x7, 0x0};
        int n = 16;
        unsigned int ref[16];
        memcpy(ref, keys, n * sizeof(unsigned int));
        qsort(ref, n, sizeof(unsigned int), cmp_uint);

        printf("Input:  ");
        for (int i = 0; i < n; i++) printf("%X ", keys[i]);
        printf("\n");

        radix_sort(keys, n);

        printf("Sorted: ");
        for (int i = 0; i < n; i++) printf("%X ", keys[i]);
        printf("\n");
        printf("Small test: %s\n\n",
               memcmp(keys, ref, n * sizeof(unsigned int)) == 0 ? "PASS" : "FAIL");
    }

    // ── Stability test ────────────────────────────────────────────────────────
    // Pairs (key, value): equal keys must appear in original order in the output.
    {
        unsigned int keys[] = {3, 1, 2, 1, 3, 2};
        unsigned int vals[] = {0, 1, 2, 3, 4, 5};  // values track original positions
        int n = 6;
        radix_sort_kv(keys, vals, n);

        printf("Stability test — sort (key,value) pairs by key:\n");
        printf("  Result: ");
        for (int i = 0; i < n; i++) printf("(%u,%u) ", keys[i], vals[i]);
        printf("\n");
        // Expected: (1,1),(1,3),(2,2),(2,5),(3,0),(3,4)
        int ok = keys[0]==1 && vals[0]==1 && keys[1]==1 && vals[1]==3 &&
                 keys[2]==2 && vals[2]==2 && keys[3]==2 && vals[3]==5 &&
                 keys[4]==3 && vals[4]==0 && keys[5]==3 && vals[5]==4;
        printf("  Stability: %s\n\n",
               ok ? "PASS (relative order preserved within equal keys)" : "FAIL");
    }

    // ── Performance test ──────────────────────────────────────────────────────
    {
        int n = 1 << 20;
        unsigned int *keys = (unsigned int *)malloc(n * sizeof(unsigned int));
        unsigned int *ref  = (unsigned int *)malloc(n * sizeof(unsigned int));
        srand(42);
        for (int i = 0; i < n; i++) keys[i] = ref[i] = (unsigned int)rand();
        qsort(ref, n, sizeof(unsigned int), cmp_uint);

        clock_t t0 = clock();
        radix_sort(keys, n);
        double ms = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC;

        int ok = (memcmp(keys, ref, n * sizeof(unsigned int)) == 0);
        printf("Performance test: N=1M  CPU time=%.1f ms  %s\n", ms,
               ok ? "PASS" : "FAIL");
        printf("  32 iterations × one stable scatter pass per iteration\n");

        free(keys); free(ref);
    }
    return 0;
}
