/*
 * Chapter 3 — Section 3.3: Image Blur — A More Complex Kernel
 *             Figure 3.8: blurKernel
 *
 * Image blurring computes each output pixel as the average of a
 * (2*BLUR_SIZE+1) × (2*BLUR_SIZE+1) patch of input pixels centred on that
 * pixel (Section 3.3).  This is a simplified box blur; Chapter 7 will
 * present the more general weighted-sum (convolution) approach.
 *
 * New concepts relative to the grayscale kernel:
 *
 *  1. Each thread reads MULTIPLE input pixels (the neighbourhood),
 *     not just one.  This breaks the simple 1-thread-1-input mapping.
 *
 *  2. Boundary handling (Figure 3.9):
 *     The patch of a pixel near an edge extends outside the image.
 *     The kernel must check (curRow, curCol) bounds and skip
 *     out-of-range pixels.  The running sum of valid pixels is kept
 *     in `pixels` so the correct average is computed even at corners
 *     and edges (where fewer than the full patch is available).
 *
 *  3. BLUR_SIZE convention:
 *     BLUR_SIZE = 1  → 3×3   patch  (9 pixels max)
 *     BLUR_SIZE = 3  → 7×7   patch  (49 pixels max)
 *     The loop bounds are:  blurRow/Col ∈ [-BLUR_SIZE, +BLUR_SIZE]
 *
 * Build:
 *   nvcc -O2 -arch=sm_70 -o image_blur 03_image_blur.cu
 *   To change patch size: nvcc -DBLUR_SIZE=3 ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef BLUR_SIZE
#define BLUR_SIZE 1   /* radius — default 3×3 patch */
#endif

#define BLOCK_DIM 16

/* -----------------------------------------------------------------------
 * blurKernel — Figure 3.8 (grayscale, single-channel image)
 *
 *  in  : input  image  width*height unsigned chars
 *  out : output image  width*height unsigned chars
 *  w   : image width  (number of columns)
 *  h   : image height (number of rows)
 *
 * Each thread calculates ONE output pixel by averaging its patch.
 * The thread's (row, col) is the centre of the patch.
 * ----------------------------------------------------------------------- */
__global__
void blurKernel(unsigned char* in, unsigned char* out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;   /* counts valid (in-bounds) neighbours */

        /* Iterate over the (2*BLUR_SIZE+1)^2 patch */
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                /* Boundary check — skip pixels that lie outside the image.
                 * This handles the 5 edge/corner cases shown in Figure 3.9. */
                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow * w + curCol];
                    ++pixels;
                }
            }
        }

        /* Write the average — integer division truncates, matching the book */
        out[row * w + col] = (unsigned char)(pixVal / pixels);
    }
}

/* -----------------------------------------------------------------------
 * Host wrapper
 * ----------------------------------------------------------------------- */
void imageBlur(unsigned char* in_h, unsigned char* out_h, int w, int h) {
    int bytes = w * h * sizeof(unsigned char);
    unsigned char *in_d, *out_d;

    cudaMalloc((void**)&in_d,  bytes);
    cudaMalloc((void**)&out_d, bytes);
    cudaMemcpy(in_d, in_h, bytes, cudaMemcpyHostToDevice);

    dim3 dimGrid((int)ceil(w / (float)BLOCK_DIM),
                 (int)ceil(h / (float)BLOCK_DIM), 1);
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, 1);
    blurKernel<<<dimGrid, dimBlock>>>(in_d, out_d, w, h);
    cudaDeviceSynchronize();

    cudaMemcpy(out_h, out_d, bytes, cudaMemcpyDeviceToHost);
    cudaFree(in_d);
    cudaFree(out_d);
}

/* -----------------------------------------------------------------------
 * CPU reference — box blur with identical boundary handling
 * ----------------------------------------------------------------------- */
static void cpu_blur(unsigned char* in, unsigned char* out, int w, int h) {
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            int pixVal = 0, pixels = 0;
            for (int dr = -BLUR_SIZE; dr <= BLUR_SIZE; dr++) {
                for (int dc = -BLUR_SIZE; dc <= BLUR_SIZE; dc++) {
                    int r = row + dr, c = col + dc;
                    if (r >= 0 && r < h && c >= 0 && c < w) {
                        pixVal += in[r * w + c];
                        pixels++;
                    }
                }
            }
            out[row * w + col] = (unsigned char)(pixVal / pixels);
        }
    }
}

int main() {
    const int width = 640, height = 480;
    int n = width * height;

    unsigned char* in   = (unsigned char*)malloc(n);
    unsigned char* out  = (unsigned char*)malloc(n);
    unsigned char* ref  = (unsigned char*)malloc(n);

    /* Synthetic gradient image */
    for (int i = 0; i < n; i++) in[i] = (unsigned char)(i % 256);

    imageBlur(in, out, width, height);
    cpu_blur(in, ref, width, height);

    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        int diff = (int)out[i] - (int)ref[i];
        if (diff < -1 || diff > 1) mismatches++;
    }
    printf("blurKernel BLUR_SIZE=%d, image %dx%d: %d mismatch(es) — [%s]\n",
           BLUR_SIZE, width, height, mismatches,
           mismatches == 0 ? "PASSED" : "FAILED");

    /* ---------------------------------------------------------------
     * Demonstrate the boundary-condition cases from Figure 3.9.
     * For a pixel at (0,0) with BLUR_SIZE=1 only the top-left 2×2
     * sub-patch is valid — 4 pixels, not 9.
     * --------------------------------------------------------------- */
    printf("\nBoundary-condition spot-checks (BLUR_SIZE=%d, %dx%d image):\n",
           BLUR_SIZE, width, height);

    /* Corner pixel (0,0) */
    {
        int pixVal = 0, pixels = 0;
        for (int dr = -BLUR_SIZE; dr <= BLUR_SIZE; dr++)
            for (int dc = -BLUR_SIZE; dc <= BLUR_SIZE; dc++) {
                int r = dr, c = dc;
                if (r >= 0 && r < height && c >= 0 && c < width) {
                    pixVal += in[r * width + c];
                    pixels++;
                }
            }
        int patch = (2*BLUR_SIZE+1);
        printf("  Corner (0,0)   : valid pixels = %d / %d\n",
               pixels, patch * patch);
    }

    /* Edge pixel (0, width/2) */
    {
        int row = 0, col = width / 2;
        int pixVal = 0, pixels = 0;
        for (int dr = -BLUR_SIZE; dr <= BLUR_SIZE; dr++)
            for (int dc = -BLUR_SIZE; dc <= BLUR_SIZE; dc++) {
                int r = row + dr, c = col + dc;
                if (r >= 0 && r < height && c >= 0 && c < width) {
                    pixVal += in[r * width + c];
                    pixels++;
                }
            }
        int patch = (2*BLUR_SIZE+1);
        printf("  Top-edge (%d,%d): valid pixels = %d / %d\n",
               row, col, pixels, patch * patch);
    }

    /* Interior pixel (height/2, width/2) — should use full patch */
    {
        int row = height / 2, col = width / 2;
        int pixVal = 0, pixels = 0;
        for (int dr = -BLUR_SIZE; dr <= BLUR_SIZE; dr++)
            for (int dc = -BLUR_SIZE; dc <= BLUR_SIZE; dc++) {
                int r = row + dr, c = col + dc;
                if (r >= 0 && r < height && c >= 0 && c < width) {
                    pixVal += in[r * width + c];
                    pixels++;
                }
            }
        int patch = (2*BLUR_SIZE+1);
        printf("  Interior (%d,%d): valid pixels = %d / %d\n",
               row, col, pixels, patch * patch);
    }

    free(in); free(out); free(ref);
    return 0;
}
