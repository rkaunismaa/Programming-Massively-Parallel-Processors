# Chapter 17 — Iterative MRI Reconstruction

Code samples for **Chapter 17: Iterative magnetic resonance imaging reconstruction** from *Programming Massively Parallel Processors* (4th ed.).

The chapter implements the **F^H D computation** — the dominant kernel in iterative MRI reconstruction — through four progressive optimisation steps.

| File | Section | Step | Key idea |
|------|---------|------|----------|
| `01_fhd_scatter.cu` | §17.3 Step 1 | Scatter + atomics (Fig 17.5) | One thread per k-space sample; atomicAdd to all N voxels — heavy contention |
| `02_fhd_gather.cu` | §17.3 Step 1 | Gather, no atomics (Figs 17.6/17.9/17.10) | Loop fission + loop interchange; one thread per voxel, no write conflicts |
| `03_fhd_register.cu` | §17.3 Step 2 | Register promotion (Fig 17.11) | x,y,z and rFhD,iFhD promoted to registers; 14→5 memory accesses/iteration |
| `04_fhd_constmem.cu` | §17.3 Steps 2–3 | Constant memory + AoS + fast trig (Figs 17.16/17.17) | k-space coords in constant cache; AoS struct; `__sinf`/`__cosf` |

---

## The computation (§17.2)

F^H D reconstructs the image from k-space scanner data:

```
Mu[m]   = Phi[m] * D[m]          (complex weight per k-space sample)
FhD[n]  = Σ_m Mu[m] * exp(j·2π·(kx[m]·x[n] + ky[m]·y[n] + kz[m]·z[n]))
```

- **M** k-space samples (kx, ky, kz coordinates; complex data D; weight Phi)
- **N** image voxels at spatial positions x, y, z
- Each F^H D[n] element depends on all M samples → O(M·N) work

---

## Optimisation progression

### Step 1 — Scatter vs Gather

**Scatter (Fig 17.5)**: parallelise the M-loop → each thread scatters to all N
voxels via `atomicAdd`.  With M threads all writing to the same N locations,
atomic contention dominates.

**Gather (Fig 17.10)**: loop fission splits the m-loop so the inner n-loop
body has no dependencies on preceding outer-loop iterations.  Loop interchange
then makes n the outer (parallelised) loop.  Each thread owns one rFhD[n] —
no conflicts, no atomics.

### Step 2 — Register optimisation (Fig 17.11)

The gather kernel reads x[n], y[n], z[n] from global memory on every inner
iteration even though they never change.  Assigning them to automatic variables
(registers) before the loop reduces global memory accesses from 14 to 5 per
inner iteration.

### Steps 2–3 — Constant memory + AoS + hardware trig (Figs 17.15–17.17)

kx[m], ky[m], kz[m] are accessed by all threads in a warp at the *same* index
m in every inner-loop iteration — ideal broadcast access for the constant cache.

- **Chunking** (Fig 17.12): constant memory is 64 KB; k-space data is split into
  chunks of CHUNK_SIZE samples, each loaded via `cudaMemcpyToSymbol` before the
  kernel processes that chunk.
- **AoS layout** (Fig 17.15/17.16): storing {x,y,z} as a struct rather than three
  separate arrays packs all three coordinates for sample m into one cache line,
  served by a single constant-cache fetch.
- **Hardware trig** (Fig 17.17): `__cosf`/`__sinf` use GPU SFU units at higher
  throughput than the software `cosf`/`sinf` library functions.

---

## Building

```bash
make SM_ARCH=sm_89
make SM_ARCH=sm_89 DEBUG=1
```
