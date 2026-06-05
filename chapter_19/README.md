# Chapter 19 — Parallel Programming and Computational Thinking

Code samples for **Chapter 19: Parallel programming and computational thinking** from *Programming Massively Parallel Processors* (4th ed.).

Chapter 19 is the methodology chapter of the book. It generalises the practical patterns seen in Chapters 2–18 into three repeatable steps that a parallel programmer takes when approaching any problem: **algorithm selection** (§19.2), **problem decomposition** (§19.3), and **performance optimisation and tuning** (§19.1, §19.4). No new CUDA APIs are introduced; the chapter illustrates *when and why* to reach for the APIs already learned.

| File | Sections | Key idea |
|------|---------|----------|
| `01_decomposition.cu` | §19.3, Fig. 19.3 | Output-centric (gather) vs input-centric (scatter) decomposition |
| `02_amdahl.cu` | §19.1, Amdahl's Law | How the sequential fraction limits application-level GPU speedup |

---

## §19.1 — Goals of parallel computing

Three reasons to parallelize (p. 433–436):

| Goal | Example |
|------|---------|
| **Solve faster** | Risk-analysis portfolio run in 4 h instead of 200 h |
| **Solve bigger** | Expand portfolio beyond what sequential time allows |
| **Solve better** | Use a more accurate model within the same time budget |

### Amdahl's Law

The application speedup is bounded by the serial fraction that cannot be parallelised:

```
Application speedup = 1 / (f_serial + f_parallel / kernel_speedup)
```

Book example — molecular dynamics (Fig. 19.1):

| Scenario | Calculation | Result |
|----------|-------------|--------|
| Nonbonded force = 95% of time, 100× GPU speedup, no overlap | 1/(5% + 95%/100) | **17×** |
| Same, host serial work hidden inside GPU execution | 1/5% | **20×** |

The law motivates two responses:
1. **Task-level parallelism**: run small serial modules concurrently using multicore host or multiple CUDA streams.
2. **Host–device overlap**: design kernels so the host can run its serial portion while the GPU is busy (§20.5, `chapter_20/01_stencil_streams.cu`).

---

## §19.2 — Algorithm selection

An algorithm is selected by trading off four properties:

| Property | Example tradeoff |
|----------|-----------------|
| **Algorithmic complexity** | DCS O(N·M) vs cutoff-sum O(N·k) — Ch. 18 |
| **Degree of parallelism** | Kogge-Stone more parallel than Brent-Kung — Ch. 11 |
| **Generality** | Radix sort keys-only vs merge sort any comparison — Ch. 13 |
| **Accuracy** | Cutoff summation sacrifices small accuracy for O(N) complexity |

Fig. 19.2 (p. 439) shows empirically that all three cutoff-binning variants (SmallBin, LargeBin, SmallBin-Overlap) maintain the same scalability for large grid volumes, while direct summation scales quadratically and eventually falls behind the CPU.

Key rule: **there is rarely a single best algorithm** — the best choice depends on the hardware, the input distribution, and the acceptable accuracy.

---

## §19.3 — Problem decomposition

After selecting an algorithm, the problem must be decomposed into parallel subproblems. Two fundamental strategies (Fig. 19.3):

### Output-centric decomposition (gather)

Each thread is assigned one **output** element and reads from whatever inputs contribute to it.

```
Thread i: out[i] = f(in[j0], in[j1], ..., in[jk])
```

- **Access pattern**: gather — multiple reads, one write.
- **No atomics**: thread `i` is the only writer of `out[i]`.
- **Used by**: stencil, convolution, matrix multiply, DCS, merge, BFS-pull.

### Input-centric decomposition (scatter)

Each thread is assigned one **input** element and writes its contribution to every output it affects.

```
Thread j: for each output i that in[j] contributes to:
    atomicAdd(&out[i], f(in[j]))
```

- **Access pattern**: scatter — one read, multiple writes.
- **Atomics required** when multiple threads may write the same output.
- **Used by**: histogram (Ch. 9), SpMV-COO (Ch. 14), BFS-push (Ch. 15).
- **Preferred when** the number of outputs is much smaller than inputs, or load balancing is otherwise hard to achieve with output-centric decomposition.

### Benchmark (`01_decomposition.cu`)

Windowed sum over N=1M elements with RADIUS=32 (window width = 65):

| Kernel | Strategy | Atomics | Time | Speedup |
|--------|----------|---------|------|---------|
| `gather_kernel` | output-centric | none | 0.02 ms | 1.0× (baseline) |
| `scatter_kernel` | input-centric | `atomicAdd` | 0.23 ms | **12.5× slower** |

Each of the 65 neighbour threads that write to the same `out[i]` must serialise through the atomic — exactly the contention penalty §19.3 warns about.

---

## §19.4 — Computational thinking

§19.4 frames three levels of parallelisation effort ("good, better, best"):

| Level | Approach | Example |
|-------|---------|---------|
| **Good** | Re-compile with parallel libraries or pragmas | Replace `std::sort` with `thrust::sort` |
| **Better** | Rewrite hot loops as CUDA kernels | Port DCS gather loop to `cenergy` kernel |
| **Best** | Holistic redesign — algorithm + decomposition + optimisation | Change algorithm to cutoff binning, adopt output-centric decomposition, add thread coarsening and memory coalescing |

The "best" level combines all three steps from §19.1–19.3 and is what the application chapters (Ch. 17–18) demonstrate in practice.

---

## §19.1 Demo — Amdahl's Law (`02_amdahl.cu`)

The program:
1. Measures `T_gpu` — the raw GPU kernel time (large SAXPY).
2. Defines a hypothetical sequential baseline `T_seq = T_gpu × 100` (assumes GPU is 100× faster than CPU for the parallel portion).
3. Sweeps serial fractions from 0% to 20%, computes **theoretical** Amdahl speedup, and **empirically** measures it by running the GPU kernel plus a calibrated CPU serial workload.

Sample output (RTX 4090):

```
Parallel kernel (SAXPY, N=8388608): T_gpu = 0.021 ms/iter

f_serial    T_serial   T_parallel   T_total    Amdahl(theory)  Measured
--------    --------   ----------   -------    --------------  --------
    0.0%      0.00 ms     0.02 ms    0.02 ms      100.00×       82.57×
    5.0%      0.11 ms     0.02 ms    0.13 ms       16.81×       19.60×
   10.0%      0.21 ms     0.02 ms    0.23 ms        9.17×        9.89×
   20.0%      0.42 ms     0.02 ms    0.44 ms        4.81×        4.96×
```

Measured tracks theory closely for `f >= 5%`. At `f < 2%`, the serial work completes while the GPU kernel is still running (host–device overlap), so the overhead is free — this is the "with overlap" case from §19.1.

---

## Building

```bash
make SM_ARCH=sm_89          # builds both programs
./decomposition             # §19.3 gather vs scatter
./amdahl                    # §19.1 Amdahl's law
make clean
```
