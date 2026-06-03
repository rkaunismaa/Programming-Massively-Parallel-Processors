/*
 * Chapter 2 — Section 2.3: A Vector Addition Kernel
 *
 * This is the sequential CPU baseline shown in Figure 2.4 of the book.
 * Vector addition is the "Hello World" of data-parallel programming.
 * Variable names are suffixed with "_h" to indicate host (CPU) data,
 * following the book's convention of using "_h" for host and "_d" for device.
 *
 * Build:
 *   gcc -O2 -o vec_add_sequential 01_vec_add_sequential.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Compute vector sum C_h = A_h + B_h  (Figure 2.4) */
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; ++i) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

int main() {
    int n = 1 << 20;   /* 1 048 576 elements */
    size_t size = n * sizeof(float);

    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    vecAdd(A, B, C, n);

    /* Every element of C should be 3.0 */
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(C[i] - 3.0f);
        if (err > max_err) max_err = err;
    }
    printf("Sequential vecAdd — max error: %e  [%s]\n",
           max_err, max_err < 1e-5f ? "PASSED" : "FAILED");

    free(A);
    free(B);
    free(C);
    return 0;
}
