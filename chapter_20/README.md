# Chapter 20 — Programming a Heterogeneous Computing Cluster

Code samples for **Chapter 20: Programming a heterogeneous computing cluster — An introduction to CUDA streams** from *Programming Massively Parallel Processors* (4th ed.).

Chapter 20 introduces joint MPI+CUDA programming for HPC clusters where each node has one or more GPUs. The running example is a 3-D heat-transfer Jacobi iteration (25-point stencil) partitioned along the z-dimension across MPI ranks.

| File | Sections | Key idea |
|------|---------|----------|
| `01_stencil_streams.cu` | §20.5 | Single-GPU stream-overlap demo — no MPI required; runs on this machine |
| `02_mpi_cuda_stencil.cu` | §20.3–20.7 | Full MPI+CUDA stencil; requires `mpirun -np N` |

---

## §20.1–20.2 — Background and running example

HPC clusters today use both CPUs and GPUs per node. The dominant inter-node programming interface is **MPI** (Message Passing Interface), a distributed-memory model where processes communicate by sending and receiving messages.

The running example is a **3-D structured grid** (heat flow in a rectangular duct, Fig. 20.3) partitioned along the z-dimension into **domain slices**. Each MPI rank owns one slice. The Jacobi method updates each grid point as a weighted average of its 24 nearest neighbours (four in each of ±x, ±y, ±z).

Process roles (Fig. 20.6):

| Rank | Role |
|------|------|
| 0 .. np−2 | Compute processes — each owns a GPU and one z-partition |
| np−1 | Data server — distributes input data and collects output |

---

## §20.3 — MPI basics

Five essential MPI functions (Fig. 20.5):

```c
MPI_Init(&argc, &argv)                  // initialise MPI runtime
MPI_Comm_rank(MPI_COMM_WORLD, &pid)     // unique process id (0 .. np-1)
MPI_Comm_size(MPI_COMM_WORLD, &np)      // total number of processes
MPI_Abort(MPI_COMM_WORLD, error_code)   // terminate all processes on error
MPI_Finalize()                          // shut down, free all resources
```

All MPI processes run the same program (SPMD). Each uses `pid` to decide its role — identical to how CUDA threads use `threadIdx` / `blockIdx`.

---

## §20.4 — Point-to-point communication

```c
MPI_Send(buf, count, MPI_FLOAT, dest, tag, MPI_COMM_WORLD)
MPI_Recv(buf, count, MPI_FLOAT, src,  tag, MPI_COMM_WORLD, &status)

// Combined send-to-right + receive-from-left in one call (Fig. 20.16):
MPI_Sendrecv(send_buf, count, MPI_FLOAT, dest, tag,
             recv_buf, count, MPI_FLOAT, src,  tag,
             MPI_COMM_WORLD, &status)
```

The data server uses `MPI_Send` to distribute z-partitions (plus halo cells) to every compute rank. Edge processes (rank 0 and rank np−2) receive only one set of halos; internal processes receive two.

**Halo cells** (ghost cells): each compute rank must receive the boundary slices from its left and right neighbours to compute its own boundary points in the next iteration. For the 25-point stencil with four neighbours in each direction, four halo slices are needed on each side.

---

## §20.5 — Overlapping computation and communication

The naive approach serialises: compute all → exchange halos → repeat. This leaves the GPU idle during the MPI transfer and the network idle during GPU compute.

**Two-stage strategy** (Fig. 20.12):

```
Iteration i:
  Stage 1 — stream 0:  compute boundary slices (the slices neighbours need)
  Stage 2 — concurrent:
    stream 1: compute interior slices (the bulk of the work)
    stream 0: async D→H of boundary data  (PCIe transfer)
  After stream0 sync:
    host:     MPI_Sendrecv (network halo exchange)
    stream 0: async H→D of received halo data
  cudaDeviceSynchronize()
  Swap d_input ↔ d_output
```

The interior kernel in stream 1 overlaps with the PCIe transfer in stream 0, and on a real cluster also overlaps with the MPI network transfer.

Two CUDA APIs make this possible:

| API | Purpose |
|-----|---------|
| `cudaHostAlloc(cudaHostAllocDefault)` | Pinned (page-locked) host memory — required for `cudaMemcpyAsync` to be truly non-blocking |
| `cudaMemcpyAsync(dst, src, bytes, dir, stream)` | Enqueue a PCIe transfer into a stream; returns immediately |

**Measured speedup on RTX 4090** (single-GPU simulation, 64×64×512 partition):

| Approach | Time/iter | Speedup |
|----------|-----------|---------|
| Serial (sync D↔H) | 0.045 ms | 1.0× |
| Overlapped (§20.5) | 0.026 ms | **1.7×** |

On a real cluster the speedup is larger because the MPI network latency (milliseconds) is much longer than the PCIe transfer time (microseconds), giving the interior kernel plenty of time to run.

---

## §20.6 — Collective communication

MPI also provides optimised collective operations:

| Function | Effect |
|----------|--------|
| `MPI_Barrier(comm)` | Synchronise all processes — none continues until all arrive |
| `MPI_Bcast(buf, count, type, root, comm)` | Root broadcasts to all |
| `MPI_Reduce(sbuf, rbuf, count, type, op, root, comm)` | Reduce across all → root |
| `MPI_Gather(sbuf, sc, type, rbuf, rc, type, root, comm)` | Collect from all → root |
| `MPI_Scatter(sbuf, sc, type, rbuf, rc, type, root, comm)` | Distribute root → all |

`MPI_Barrier` is used before the iteration loop to ensure all compute nodes have received their input data before any begin computing.

---

## §20.7 — CUDA-aware MPI

Modern MPI libraries (MVAPICH2, IBM Platform MPI, Open MPI 4+) can read and write **device memory directly**, removing the need for host bounce buffers and the two async H↔D copies per halo exchange. The revised `MPI_Sendrecv` calls pass device pointers (Fig. 20.19):

```c
// Without CUDA-aware MPI: D→H + MPI_Sendrecv + H→D (three steps, two copies)

// With CUDA-aware MPI: just MPI_Sendrecv with device addresses (one step)
MPI_Sendrecv(d_output + right_bnd_offset, halo_pts, MPI_FLOAT, right_nbr, ...
             d_output + left_halo_offset,  halo_pts, MPI_FLOAT, left_nbr,  ...
             MPI_COMM_WORLD, &status);
```

Build `02_mpi_cuda_stencil` with `-DCUDA_AWARE_MPI` to enable this path:

```bash
make SM_ARCH=sm_89 CUDA_AWARE=1 mpi_cuda_stencil
```

---

## Building

```bash
# Standalone stream-overlap demo (no MPI needed)
make SM_ARCH=sm_89
./stencil_streams

# Full MPI+CUDA stencil (requires Open MPI)
sudo apt install libopenmpi-dev openmpi-bin
make SM_ARCH=sm_89 mpi_cuda_stencil
mpirun -np 5 ./mpi_cuda_stencil    # 4 compute + 1 data server

# CUDA-aware MPI variant (§20.7)
make SM_ARCH=sm_89 CUDA_AWARE=1 mpi_cuda_stencil
mpirun -np 5 ./mpi_cuda_stencil
```

> `02_mpi_cuda_stencil` requires a CUDA-aware MPI library.  
> Each compute rank uses one GPU (`cudaSetDevice(pid)` should be added in  
> a real multi-GPU deployment; omitted here for clarity).
