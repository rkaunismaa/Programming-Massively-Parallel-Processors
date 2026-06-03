/*
 * Chapter 3 — Section 3.1: Multidimensional Grid Organization
 *
 * CUDA grids and blocks can be 1D, 2D, or 3D.  Grid and block dimensions
 * are specified with the dim3 type — an integer vector with fields x, y, z.
 * Unused dimensions default to 1.
 *
 * Built-in kernel variables (always read-only inside a kernel):
 *   gridDim  — dimensions of the grid   (type dim3)
 *   blockDim — dimensions of each block (type dim3)
 *   blockIdx — this block's coordinates (type uint3)
 *   threadIdx— this thread's coordinates within its block (type uint3)
 *
 * Key limits (Section 3.1):
 *   • gridDim.x  : 1 … 2^31-1   (gridDim.y / .z : 1 … 65535)
 *   • Total threads per block ≤ 1024  (e.g. 32×32, 16×16×4, 1024×1×1)
 *   • Block size should be a multiple of 32 for hardware efficiency
 *
 * Build:
 *   nvcc -O2 -arch=sm_70 -o multidim_grid 01_multidim_grid_organization.cu
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* -----------------------------------------------------------------------
 * 1-D kernel — identical to Chapter 2's vecAddKernel.
 * The shorthand vecAddKernel<<<N, 256>>> is equivalent to
 *   dim3 dimGrid(N, 1, 1);  dim3 dimBlock(256, 1, 1);
 * ----------------------------------------------------------------------- */
__global__
void kernel1D(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;
}

/* -----------------------------------------------------------------------
 * 2-D kernel — maps (row, col) thread indices to a 2-D array.
 * Row-major linearisation: index = row * width + col  (Section 3.2)
 * ----------------------------------------------------------------------- */
__global__
void kernel2D(int* out, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        out[row * width + col] = row * 100 + col;   /* encode (row,col) */
    }
}

/* -----------------------------------------------------------------------
 * 3-D kernel — maps (plane, row, col) to a 3-D array.
 * Linearisation: index = plane * height * width + row * width + col
 * ----------------------------------------------------------------------- */
__global__
void kernel3D(int* out, int width, int height, int depth) {
    int col   = blockIdx.x * blockDim.x + threadIdx.x;
    int row   = blockIdx.y * blockDim.y + threadIdx.y;
    int plane = blockIdx.z * blockDim.z + threadIdx.z;
    if (col < width && row < height && plane < depth) {
        out[plane * height * width + row * width + col] =
            plane * 10000 + row * 100 + col;
    }
}

int main() {
    /* ---------------------------------------------------------------
     * 1-D example
     * Equivalent shorthand: kernel1D<<<ceil(n/256.0), 256>>>(d, n);
     * --------------------------------------------------------------- */
    {
        const int n = 512;
        int h[n];
        int* d;
        cudaMalloc((void**)&d, n * sizeof(int));

        dim3 dimGrid(2, 1, 1);    /* 2 blocks */
        dim3 dimBlock(256, 1, 1); /* 256 threads/block → 512 threads total */
        kernel1D<<<dimGrid, dimBlock>>>(d, n);
        cudaDeviceSynchronize();
        cudaMemcpy(h, d, n * sizeof(int), cudaMemcpyDeviceToHost);

        int ok = 1;
        for (int i = 0; i < n; i++) ok &= (h[i] == i);
        printf("1-D grid (%d blocks × %d threads): %s\n",
               dimGrid.x, dimBlock.x, ok ? "PASSED" : "FAILED");
        cudaFree(d);
    }

    /* ---------------------------------------------------------------
     * 2-D example — process a 62×76 image with 16×16 blocks
     * (the running example from Section 3.2 / Figure 3.2)
     * gridDim = (ceil(76/16), ceil(62/16)) = (5, 4)
     * Total threads = 80×64, covering the 76×62 valid pixels
     * --------------------------------------------------------------- */
    {
        const int width = 76, height = 62;
        int n = width * height;
        int* h = (int*)malloc(n * sizeof(int));
        int* d;
        cudaMalloc((void**)&d, n * sizeof(int));

        dim3 dimGrid((int)ceil(width  / 16.0),
                     (int)ceil(height / 16.0), 1);
        dim3 dimBlock(16, 16, 1);
        kernel2D<<<dimGrid, dimBlock>>>(d, width, height);
        cudaDeviceSynchronize();
        cudaMemcpy(h, d, n * sizeof(int), cudaMemcpyDeviceToHost);

        int ok = 1;
        for (int r = 0; r < height; r++)
            for (int c = 0; c < width; c++)
                ok &= (h[r * width + c] == r * 100 + c);
        printf("2-D grid (%d×%d blocks, 16×16 threads) for %d×%d image: %s\n",
               dimGrid.x, dimGrid.y, width, height, ok ? "PASSED" : "FAILED");
        free(h);
        cudaFree(d);
    }

    /* ---------------------------------------------------------------
     * 3-D example — a small 8×8×4 volume, 4×4×2 threads per block
     * --------------------------------------------------------------- */
    {
        const int W = 8, H = 8, D = 4;
        int n = W * H * D;
        int* h = (int*)malloc(n * sizeof(int));
        int* d;
        cudaMalloc((void**)&d, n * sizeof(int));

        dim3 dimGrid((int)ceil(W / 4.0),
                     (int)ceil(H / 4.0),
                     (int)ceil(D / 2.0));
        dim3 dimBlock(4, 4, 2);
        kernel3D<<<dimGrid, dimBlock>>>(d, W, H, D);
        cudaDeviceSynchronize();
        cudaMemcpy(h, d, n * sizeof(int), cudaMemcpyDeviceToHost);

        int ok = 1;
        for (int p = 0; p < D; p++)
            for (int r = 0; r < H; r++)
                for (int c = 0; c < W; c++)
                    ok &= (h[p*H*W + r*W + c] == p*10000 + r*100 + c);
        printf("3-D grid for %d×%d×%d volume: %s\n", W, H, D,
               ok ? "PASSED" : "FAILED");
        free(h);
        cudaFree(d);
    }

    return 0;
}
