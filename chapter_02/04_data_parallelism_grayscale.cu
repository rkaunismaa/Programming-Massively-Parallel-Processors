/*
 * Chapter 2 — Section 2.1: Data Parallelism
 *
 * The chapter introduces data parallelism using color-to-grayscale
 * conversion as the motivating example (Figure 2.1, Figure 2.2).
 *
 * Key idea (Section 2.1):
 *   Each output pixel depends only on the corresponding input pixel,
 *   so all N pixels can be computed independently — data parallelism.
 *   The computation of O[0], O[1], … O[N-1] forms the independent units
 *   shown in Figure 2.2.
 *
 * Luminance formula from the book (Section 2.1):
 *   L = r * 0.21 + g * 0.72 + b * 0.07
 *
 * Thread mapping (consistent with Section 2.5):
 *   One thread per pixel, using the same 1-D index calculation as vecAdd.
 *   The 2-D thread organisation for images is covered in Chapter 3.
 *
 * Input layout: packed RGB bytes — [R0, G0, B0, R1, G1, B1, …]
 *               3 unsigned chars per pixel, values 0–255
 * Output layout: one unsigned char per pixel (grayscale luminance)
 *
 * Build:
 *   nvcc -O2 -arch=sm_70 -o grayscale 04_data_parallelism_grayscale.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHANNELS 3   /* RGB */
#define BLOCK_SIZE 256

/* -----------------------------------------------------------------------
 * Kernel: one thread → one output pixel
 *
 * Each thread:
 *   1. Computes its global pixel index  (same formula as vecAddKernel)
 *   2. Guards against out-of-bounds access
 *   3. Reads R, G, B from the packed input array
 *   4. Applies the luminance formula
 *   5. Writes one byte to the output array
 *
 * This demonstrates the core data-parallelism idea from Section 2.1:
 * every output element is independent, so all N threads can run in parallel.
 * ----------------------------------------------------------------------- */
__global__
void colorToGrayscaleKernel(unsigned char* Pin,
                             unsigned char* Pout,
                             int width, int height) {
    /* 1-D global thread index — covers all pixels when grid is large enough */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        /* Each color pixel occupies CHANNELS consecutive bytes in Pin */
        int rgb_offset = idx * CHANNELS;

        unsigned char r = Pin[rgb_offset];       /* red   */
        unsigned char g = Pin[rgb_offset + 1];   /* green */
        unsigned char b = Pin[rgb_offset + 2];   /* blue  */

        /* Luminance formula (Section 2.1) — cast back to unsigned char */
        Pout[idx] = (unsigned char)(0.21f * r + 0.72f * g + 0.07f * b);
    }
}

/* -----------------------------------------------------------------------
 * Host wrapper: mirrors the structure of vecAdd from Section 2.3–2.6
 * ----------------------------------------------------------------------- */
void colorToGrayscale(unsigned char* Pin_h,
                      unsigned char* Pout_h,
                      int width, int height) {
    int total_pixels = width * height;
    int in_size  = total_pixels * CHANNELS * sizeof(unsigned char);
    int out_size = total_pixels             * sizeof(unsigned char);

    unsigned char *Pin_d, *Pout_d;

    /* Part 1: allocate device memory, copy input host → device */
    cudaMalloc((void**)&Pin_d,  in_size);
    cudaMalloc((void**)&Pout_d, out_size);
    cudaMemcpy(Pin_d, Pin_h, in_size, cudaMemcpyHostToDevice);

    /* Part 2: launch — one thread per pixel (Section 2.2, 2.6) */
    int blocks = (int)ceil(total_pixels / (float)BLOCK_SIZE);
    colorToGrayscaleKernel<<<blocks, BLOCK_SIZE>>>(Pin_d, Pout_d, width, height);

    /* Part 3: retrieve result, free device memory */
    cudaMemcpy(Pout_h, Pout_d, out_size, cudaMemcpyDeviceToHost);
    cudaFree(Pin_d);
    cudaFree(Pout_d);
}

/* -----------------------------------------------------------------------
 * Verification: compute expected grayscale values on the CPU
 * ----------------------------------------------------------------------- */
static void reference_grayscale(unsigned char* Pin,
                                 unsigned char* Pout,
                                 int width, int height) {
    int n = width * height;
    for (int i = 0; i < n; i++) {
        int off = i * CHANNELS;
        Pout[i] = (unsigned char)(0.21f * Pin[off]
                                + 0.72f * Pin[off + 1]
                                + 0.07f * Pin[off + 2]);
    }
}

int main() {
    /* Simulate a small image */
    int width  = 1920;
    int height = 1080;
    int n      = width * height;

    unsigned char* rgb    = (unsigned char*)malloc(n * CHANNELS);
    unsigned char* gray   = (unsigned char*)malloc(n);
    unsigned char* ref    = (unsigned char*)malloc(n);

    /* Fill with a simple pattern: pixel i has (i%256, (i*2)%256, (i*3)%256) */
    for (int i = 0; i < n; i++) {
        rgb[i * CHANNELS]     = (unsigned char)(i % 256);
        rgb[i * CHANNELS + 1] = (unsigned char)((i * 2) % 256);
        rgb[i * CHANNELS + 2] = (unsigned char)((i * 3) % 256);
    }

    /* GPU result */
    colorToGrayscale(rgb, gray, width, height);

    /* CPU reference */
    reference_grayscale(rgb, ref, width, height);

    /* Compare */
    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        /* Allow ±1 due to floating-point rounding in the byte cast */
        int diff = (int)gray[i] - (int)ref[i];
        if (diff < -1 || diff > 1) mismatches++;
    }
    printf("Grayscale conversion (%dx%d = %d pixels): %d mismatch(es) — [%s]\n",
           width, height, n,
           mismatches, mismatches == 0 ? "PASSED" : "FAILED");

    free(rgb);
    free(gray);
    free(ref);
    return 0;
}
