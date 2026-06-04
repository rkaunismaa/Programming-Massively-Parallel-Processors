# Chapter 14 — Sparse Matrix Computation

Code samples for **Chapter 14: Sparse matrix computation** from *Programming Massively Parallel Processors* (4th ed.).

The chapter uses **SpMV** (sparse matrix–vector multiply, `y = A·x + y`) to compare five
sparse storage formats, each representing a different trade-off between space efficiency,
memory coalescing, and load balance.

| File | Section | Format | Key characteristic |
|------|---------|--------|--------------------|
| `01_spmv_coo.cu` | §14.2 | COO | One thread/NZ; coalesced reads; `atomicAdd` required |
| `02_spmv_csr.cu` | §14.3 | CSR | One thread/row; no atomics; non-coalesced reads |
| `03_spmv_ell.cu` | §14.4 | ELL | Column-major layout → coalesced; padding overhead |
| `04_spmv_ell_coo.cu` | §14.5 | ELL-COO hybrid | Threshold T caps ELL width; overflow → COO |
| `05_spmv_jds.cu` | §14.6 | JDS | Sort rows by nnz; coalesced + reduced divergence; no padding |

---

## Test matrix (Fig 14.1)

```
A (4×4) = [ 1  7  0  0 ]     x = [1, 2, 3, 4]
           [ 5  0  3  9 ]
           [ 0  0  2  8 ]     y_expected = [15, 50, 38, 24]
           [ 0  0  0  6 ]
```

Every kernel produces `y = [15, 50, 38, 24]` for this input.

---

## Format comparison

| Format | Coalesced? | Atomics? | Divergence | Padding |
|--------|-----------|---------|-----------|---------|
| COO    | ✓ (all 3 arrays) | required | low | none |
| CSR    | ✗ | none | high | none |
| ELL    | ✓ (column-major) | none | low | can be large |
| ELL-COO | ✓ (ELL part) | COO overflow only | low | reduced |
| JDS    | ✓ (column-major) | none | reduced | none |

---

## Building

```bash
make SM_ARCH=sm_89          # build all five targets
make SM_ARCH=sm_89 DEBUG=1  # add -g -G for cuda-gdb
```

| GPU family | `SM_ARCH` |
|------------|-----------|
| RTX 40xx   | `sm_89`   |
| RTX 30xx   | `sm_86`   |
| A100       | `sm_80`   |
| V100       | `sm_70`   |
