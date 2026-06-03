/*
 * Chapter 3 — Section 3.2: Mapping Threads to Multidimensional Data
 *             Figure 3.4: colorToGrayscaleConversion kernel
 *
 * This is the first complete example of a 2-D thread grid working on a
 * 2-D dataset.  Two key ideas from Section 3.2 are demonstrated:
 *
 *  1. Row-major linearisation (Figure 3.3):
 *       1D index of element at (row, col) = row * width + col
 *     The j*4+i formula from Figure 3.3 generalises to row*width+col.
 *
 *  2. Thread-to-pixel mapping (Section 3.2):
 *       col = blockIdx.x * blockDim.x + threadIdx.x   (horizontal)
 *       row = blockIdx.y * blockDim.y + threadIdx.y   (vertical)
 *     The if-guard (col < width && row < height) is needed because
 *     the grid may generate extra threads when the image dimensions
 *     are not multiples of the block size (same reason as the i < n
 *     guard in vecAddKernel from Chapter 2).
 *
 *  3. RGB → grayscale formula (Section 3.2, Figure 3.4 line 19):
 *       L = 0.21*r + 0.71*g + 0.07*b
 *     Each color pixel occupies 3 consecutive bytes: [R, G, B].
 *     The "CHANNELS" multiplier converts a pixel index into a byte offset.
 *
 * Build:
 *   nvcc -O2 -arch=sm_70 -o color_to_grayscale 02_color_to_grayscale.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHANNELS 3   /* 3 bytes per pixel: R, G, B */
#define BLOCK_DIM 16 /* 16×16 = 256 threads per block — multiple of 32 */

/* -----------------------------------------------------------------------
 * Kernel — Figure 3.4 (reproduced verbatim with added inline comments)
 *
 * Input  Pin : RGB image, width*height*3 unsigned chars
 * Output Pout: grayscale image, width*height unsigned chars
 * ----------------------------------------------------------------------- */
__global__
void colorToGrayscaleConversion(unsigned char* Pout,
                                 unsigned char* Pin,
                                 int width, int height) {
    /* Map this thread to a pixel (col, row) in the image */
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    /* Guard: only threads whose (col, row) fall within the image process pixels.
     * Extra threads at the right/bottom edges of the grid do nothing. */
    if (col < width && row < height) {
        /* 1-D offset into the grayscale output (one byte per pixel) */
        int grayOffset = row * width + col;

        /* The RGB input has CHANNELS times as many bytes as the grayscale output.
         * Multiply by CHANNELS to find the start of this pixel's [R, G, B] triplet. */
        int rgbOffset = grayOffset * CHANNELS;

        unsigned char r = Pin[rgbOffset];       /* Red   channel */
        unsigned char g = Pin[rgbOffset + 1];   /* Green channel */
        unsigned char b = Pin[rgbOffset + 2];   /* Blue  channel */

        /* Luminance formula — Figure 3.4 line 19 */
        Pout[grayOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

/* -----------------------------------------------------------------------
 * Host wrapper
 *
 * Grid launch (Section 3.2):
 *   dimGrid.x = ceil(width  / 16.0)   — blocks needed in x direction
 *   dimGrid.y = ceil(height / 16.0)   — blocks needed in y direction
 *
 * For a 1500×2000 image:  ceil(1500/16)=94 and ceil(2000/16)=125
 *   → 94×125 = 11 750 blocks of 16×16 = 256 threads → 3 008 000 threads
 * ----------------------------------------------------------------------- */
void colorToGrayscale(unsigned char* Pin_h,
                      unsigned char* Pout_h,
                      int width, int height) {
    int in_bytes  = width * height * CHANNELS * sizeof(unsigned char);
    int out_bytes = width * height             * sizeof(unsigned char);

    unsigned char *Pin_d, *Pout_d;
    cudaMalloc((void**)&Pin_d,  in_bytes);
    cudaMalloc((void**)&Pout_d, out_bytes);
    cudaMemcpy(Pin_d, Pin_h, in_bytes, cudaMemcpyHostToDevice);

    dim3 dimGrid((int)ceil(width  / (float)BLOCK_DIM),
                 (int)ceil(height / (float)BLOCK_DIM), 1);
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, 1);
    colorToGrayscaleConversion<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, width, height);

    cudaMemcpy(Pout_h, Pout_d, out_bytes, cudaMemcpyDeviceToHost);
    cudaFree(Pin_d);
    cudaFree(Pout_d);
}

/* CPU reference for verification */
static void cpu_grayscale(unsigned char* Pin, unsigned char* Pout,
                           int width, int height) {
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int off  = (r * width + c) * CHANNELS;
            Pout[r * width + c] = (unsigned char)(
                0.21f * Pin[off] + 0.71f * Pin[off+1] + 0.07f * Pin[off+2]);
        }
    }
}

int main() {
    /* Use the 62×76 running example from Section 3.2 */
    const int width = 76, height = 62;
    int n = width * height;

    unsigned char* rgb  = (unsigned char*)malloc(n * CHANNELS);
    unsigned char* gray = (unsigned char*)malloc(n);
    unsigned char* ref  = (unsigned char*)malloc(n);

    /* Synthetic pixel data */
    for (int i = 0; i < n; i++) {
        rgb[i * CHANNELS]     = (unsigned char)(i % 256);
        rgb[i * CHANNELS + 1] = (unsigned char)((i * 2) % 256);
        rgb[i * CHANNELS + 2] = (unsigned char)((i * 3) % 256);
    }

    colorToGrayscale(rgb, gray, width, height);
    cpu_grayscale(rgb, ref, width, height);

    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        int diff = (int)gray[i] - (int)ref[i];
        if (diff < -1 || diff > 1) mismatches++;
    }
    printf("colorToGrayscale (%d×%d): %d mismatch(es) — [%s]\n",
           width, height, mismatches,
           mismatches == 0 ? "PASSED" : "FAILED");

    /* Show grid dimensions for a 1500×2000 image (Section 3.2 example) */
    int big_w = 1500, big_h = 2000;
    printf("\nFor a %d×%d image:\n", big_w, big_h);
    printf("  dimGrid  = (%d, %d)\n",
           (int)ceil(big_w / (float)BLOCK_DIM),
           (int)ceil(big_h / (float)BLOCK_DIM));
    printf("  dimBlock = (%d, %d)\n", BLOCK_DIM, BLOCK_DIM);
    printf("  Total threads = %d\n",
           (int)ceil(big_w / (float)BLOCK_DIM) * BLOCK_DIM *
           (int)ceil(big_h / (float)BLOCK_DIM) * BLOCK_DIM);

    free(rgb); free(gray); free(ref);
    return 0;
}
