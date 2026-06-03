# Chapter 3 — Multidimensional Grids and Data

Chapter 2 worked exclusively with 1-D data and 1-D grids.  This chapter
generalises both: CUDA grids and blocks can be **1-D, 2-D, or 3-D**, and the
chapter shows how to choose the right organisation for different data shapes.

Three worked examples of increasing complexity are developed:

1. **Color-to-grayscale** — each thread maps to one 2-D pixel (Section 3.2)
2. **Image blur** — each thread reads a *neighbourhood* of pixels (Section 3.3)
3. **Matrix multiplication** — each thread computes one dot product (Section 3.4)

A key technical skill introduced here is **row-major linearisation**: how to
correctly convert multi-dimensional array indices into the 1-D byte offset
that CUDA device pointers require.

---

## Programs

### `01_multidim_grid_organization.cu` — dim3 and multi-dimensional grids (Section 3.1)

Shows how to declare and use `dim3` launch configurations for 1-D, 2-D, and
3-D grids, and verifies that `blockIdx`, `threadIdx`, and `blockDim` report
the expected values inside each kernel.

**Key concepts:**

- `dim3` is a struct with fields `x`, `y`, `z`; unused dimensions default to 1.
- The two execution configuration parameters in `<<<...>>>` are both `dim3`.
- Inside a kernel, `gridDim` and `blockDim` are built-in read-only variables
  that reflect those configuration values.
- The total number of threads in a block must not exceed **1024**.
- Block sizes should be **multiples of 32** for hardware efficiency (warp alignment).

```bash
nvcc -O2 -arch=sm_89 -o multidim_grid 01_multidim_grid_organization.cu
./multidim_grid
```

Expected output:
```
1-D grid (2 blocks × 256 threads): PASSED
2-D grid (5×4 blocks, 16×16 threads) for 76×62 image: PASSED
3-D grid for 8×8×4 volume: PASSED
```

---

### `02_color_to_grayscale.cu` — 2-D thread-to-pixel mapping (Section 3.2, Figure 3.4)

The `colorToGrayscaleConversion` kernel from Figure 3.4 — the chapter's first
complete 2-D kernel.  Demonstrates how a 2-D thread grid maps onto a 2-D image.

**Thread-to-pixel mapping:**
```
col = blockIdx.x * blockDim.x + threadIdx.x   (horizontal / column)
row = blockIdx.y * blockDim.y + threadIdx.y   (vertical   / row)
```

**Row-major linearisation (Figure 3.3):**
```
grayOffset = row * width + col          (1 byte per pixel in Pout)
rgbOffset  = grayOffset * CHANNELS      (3 bytes per pixel in Pin)
```

**Luminance formula (Figure 3.4, line 19):**
```
L = 0.21*r + 0.71*g + 0.07*b
```

**Grid launch for an image of width × height pixels:**
```c
dim3 dimGrid(ceil(width/16.0), ceil(height/16.0), 1);
dim3 dimBlock(16, 16, 1);
```
For a 1500×2000 image this produces 94×125 = 11,750 blocks.
Extra threads at the right and bottom edges are suppressed by the
`if (col < width && row < height)` guard — the same pattern as
`if (i < n)` in Chapter 2's `vecAddKernel`.

```bash
nvcc -O2 -arch=sm_89 -o color_to_grayscale 02_color_to_grayscale.cu
./color_to_grayscale
```

---

### `03_image_blur.cu` — Box blur with boundary handling (Section 3.3, Figures 3.8–3.9)

The `blurKernel` from Figure 3.8.  Each thread computes *one output pixel* by
averaging a `(2*BLUR_SIZE+1)²` patch of input pixels centred on that pixel.

**New concepts relative to the grayscale kernel:**

| Concept | Details |
|---------|---------|
| Multi-input per thread | Each thread reads up to `(2*BLUR_SIZE+1)²` input pixels |
| Boundary handling | Pixels near edges have smaller valid patches; the kernel counts only in-bounds neighbours in `pixels` and divides by that count, not by the full patch size (Figure 3.9) |
| BLUR_SIZE convention | BLUR_SIZE=1 → 3×3 patch; BLUR_SIZE=3 → 7×7 patch |

**Boundary cases (Figure 3.9):**
```
Corner pixel (0,0) with BLUR_SIZE=1: only 4 of 9 patch pixels are valid
Top-edge pixel with BLUR_SIZE=1:     only 6 of 9 patch pixels are valid
Interior pixel:                      all 9 patch pixels are valid
```

The `BLUR_SIZE` constant is set at compile time with `-DBLUR_SIZE=N`:

```bash
# Default 3×3 patch (BLUR_SIZE=1)
nvcc -O2 -arch=sm_89 -o image_blur 03_image_blur.cu
./image_blur

# 7×7 patch (BLUR_SIZE=3)
nvcc -O2 -arch=sm_89 -DBLUR_SIZE=3 -o image_blur_7x7 03_image_blur.cu
./image_blur_7x7

# Or with the Makefile
make SM_ARCH=sm_89 image_blur_7x7
```

---

### `04_matrix_multiply.cu` — Naïve square matrix multiplication (Section 3.4, Figures 3.11–3.13)

The `MatrixMulKernel` from Figure 3.11 — one thread per output element `P[row][col]`.

**Thread-to-output mapping (same as color-to-grayscale):**
```
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
```

**Inner product loop (Figure 3.11, lines 07–09):**
```c
for (int k = 0; k < Width; ++k)
    Pvalue += M[row * Width + k]    // k-th element of row `row` of M
             * N[k   * Width + col]; // k-th element of column `col` of N
```

The file includes:
- The **4×4 trace-through** from Figure 3.12: M×I = M (multiplication by the identity matrix), showing how thread (0,0) of block (0,0) computes `P[0][0]` and thread (0,0) of block (1,0) computes `P[2][0]`.
- A **512×512 random matrix** test verified against a CPU reference.

> **Limitation noted in Section 3.4:** This naïve kernel reads every element
> of M and N from global memory repeatedly — once per output element that
> needs it.  Chapter 5 addresses this with **tiled matrix multiplication**
> using shared memory.

```bash
nvcc -O2 -arch=sm_89 -o matrix_multiply 04_matrix_multiply.cu -lm
./matrix_multiply
```

---

### `05_row_major_linearization.cu` — Row-major layout (Section 3.2, Figure 3.3)

A focused demonstration of the linearisation mechanics that underpin all 2-D
and 3-D kernels in this chapter.

**2-D row-major (Figure 3.3):**
```
M[row][col]  →  M[row * Width + col]
```
The file reproduces the book's worked example: `M_{2,1}` in a 4×4 matrix has
1-D index `2*4+1 = 9`.

**3-D row-major (Section 3.2):**
```
P[plane][row][col]  →  P[plane * height * width + row * width + col]
```

Kernels demonstrated:
- **Transpose** (2-D): writes `M[row][col]` to `T[col][row]`
- **Sum planes** (3-D): reduces a depth-D tensor to a 2-D matrix by summing across planes

```bash
nvcc -O2 -arch=sm_89 -o row_major 05_row_major_linearization.cu
./row_major
```

---

## Building all programs

```bash
# Release build
make SM_ARCH=sm_89

# Debug build (adds -g -G for cuda-gdb)
make SM_ARCH=sm_89 DEBUG=1

# Build blur with 7×7 patch
make SM_ARCH=sm_89 image_blur_7x7

# Clean
make clean
```

---

## Debugging in VS Code

Good kernel breakpoint targets:

| File | Line to break on | What you observe |
|------|-----------------|-----------------|
| `02_color_to_grayscale.cu` | Inside `colorToGrayscaleConversion`, after `int col = ...` | `col`, `row`, `grayOffset`, `rgbOffset` for the focused thread |
| `03_image_blur.cu` | Inside the nested loop, line `pixVal += in[...]` | `curRow`, `curCol`, `pixVal`, `pixels` accumulating |
| `04_matrix_multiply.cu` | Inside the k-loop, line `Pvalue += M[...]` | `k`, `Pvalue` accumulating, `row`, `col` |

Set `"breakOnLaunch": true` in `launch.json` to pause at the very first kernel
invocation — useful when you want to examine thread 0 of block 0 before
any computation has occurred.

See the [root README](../README.md) for full `.vscode/launch.json` setup.

---

## Concepts covered

| Concept | Where |
|---------|-------|
| `dim3` type and 1-D/2-D/3-D grid/block configurations | Section 3.1 |
| `gridDim` and `blockDim` built-in variables | Section 3.1 |
| Row-major linearisation of 2-D and 3-D arrays | Section 3.2, Figure 3.3 |
| 2-D thread-to-pixel mapping | Section 3.2, Figure 3.4 |
| Boundary guards for non-multiple-of-block-size data | Section 3.2 |
| Multi-input-per-thread kernel (blur) | Section 3.3, Figure 3.8 |
| Edge/corner boundary handling | Section 3.3, Figure 3.9 |
| Matrix multiplication — one thread per output element | Section 3.4, Figures 3.11–3.13 |
| Row-major access pattern for M and N | Section 3.4 |
