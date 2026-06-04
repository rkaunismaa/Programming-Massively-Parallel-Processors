# Chapter 16 — Deep Learning

Code samples for **Chapter 16: Deep learning** from *Programming Massively Parallel Processors* (4th ed.).

The chapter implements the **forward pass of a convolutional layer** (CNN inference)
in two progressively more sophisticated ways.

| File | Section | Strategy | Key idea |
|------|---------|----------|----------|
| `01_conv_forward_basic.cu` | §16.3 | Basic CUDA inference kernel | One thread per output element; 3-D grid (M, T, N) with 2-D tiles |
| `02_conv_forward_gemm.cu` | §16.4 | Unroll + GEMM | `unroll_Kernel` expands input patches; tiled GEMM computes Y = W·X_unroll |

---

## Array layouts (§16.2)

| Array | Shape | Description |
|-------|-------|-------------|
| X | [N, C, H, W] | Input feature maps (N samples in minibatch) |
| W | [M, C, K, K] | Filter banks (M output maps, each C×K×K) |
| Y | [N, M, H_out, W_out] | Output feature maps, H_out = H−K+1 |

Ghost cells are handled by the "no-padding" convention from §16.2 (LeNet-5 style):
the output is strictly smaller than the input.

---

## §16.3 — Basic CUDA inference kernel (Fig 16.15)

```
Grid:  (M, T, N)     — M output feature maps × T tiles × N samples
Block: (TILE_WIDTH, TILE_WIDTH, 1)   — 2-D tile of output pixels
```

Thread `(m, tile_y*TILE_WIDTH+ty, tile_x*TILE_WIDTH+tx, n)` computes one
output element by iterating the innermost c-, p-, q-loops serially to avoid
write-after-write atomics on Y.

**Limitation:** filter weights W and input patches X are re-read from global
memory for every output element.  A tiled version using constant or shared
memory (like Chapter 7) is left as a book exercise.

---

## §16.4 — Unroll + GEMM (Figs 16.17–16.18)

The convolution is recast as a single matrix multiplication:

```
Y [M, H_out·W_out]  =  W_filter [M, C·K·K]  ×  X_unroll [C·K·K, H_out·W_out]
```

**Unroll step** (`unroll_Kernel`): each CUDA thread gathers one column of
`X_unroll` — all `C·K·K` input values needed to produce one output pixel.
Adjacent threads write adjacent columns → coalesced writes.

**GEMM step**: the sample uses a self-contained tiled GEMM kernel.
In production, `cublasSgemm` achieves near-peak FLOP/s on this shape.

**Expansion ratio**: `C·K²·H_out·W_out / (C·H·W)` ≈ K² for large feature
maps, which is the main disadvantage of materialising `X_unroll` in DRAM.
CUDNN avoids this by performing lazy on-chip unrolling (§16.5).

---

## Test cases

| Test | N | M | C | H×W | K | Notes |
|------|---|---|---|-----|---|-------|
| Tiny (file 1) | 1 | 1 | 1 | 5×5 | 3 | Hand-checkable; all-ones filter |
| Multi-channel (file 1) | 4 | 6 | 3 | 10×10 | 3 | Random weights |
| Larger (file 1) | 8 | 16 | 3 | 32×32 | 5 | Timed; LeNet C1-like |
| Tiny GEMM (file 2) | 1 | 2 | 3 | 3×3 | 2 | Prints X_unroll matrix |
| Mid (file 2) | 4 | 8 | 3 | 10×10 | 3 | GPU unroll+GEMM vs CPU |
| LeNet-scale (file 2) | 8 | 16 | 6 | 14×14 | 5 | ~C3 layer of LeNet-5 |

---

## Building

```bash
make SM_ARCH=sm_89
make SM_ARCH=sm_89 DEBUG=1
```
