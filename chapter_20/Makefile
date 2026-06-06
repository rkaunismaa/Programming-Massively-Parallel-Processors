SM_ARCH  ?= sm_61
NVCC     := nvcc
NVCCFLAGS := -arch=$(SM_ARCH) -O2 -std=c++14

ifdef DEBUG
  NVCCFLAGS += -g -G
endif

# ── Target 1: standalone CUDA stream-overlap demo (no MPI) ───────────────────
.PHONY: all clean stencil_streams mpi_cuda_stencil

all: stencil_streams

stencil_streams: 01_stencil_streams.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# ── Target 2: full MPI+CUDA stencil (requires Open MPI / MVAPICH2) ───────────
# Install MPI:  sudo apt install libopenmpi-dev openmpi-bin
# Then build:   make SM_ARCH=sm_61 mpi_cuda_stencil
# Run:          mpirun -np 5 ./mpi_cuda_stencil
#
# For CUDA-aware MPI (§20.7):
#   make SM_ARCH=sm_61 CUDA_AWARE=1 mpi_cuda_stencil

MPICC    := mpicxx
CUDA_INC := $(shell nvcc --dryrun 2>&1 | grep -o '\-I[^ ]*' | head -1 || echo -I/usr/local/cuda/include)
CUDA_LIB := -L/usr/local/cuda/lib64 -lcudart

ifdef CUDA_AWARE
  MPI_EXTRA := -DCUDA_AWARE_MPI
endif

mpi_cuda_stencil: 02_mpi_cuda_stencil.cu
	@command -v $(MPICC) >/dev/null 2>&1 || \
	  { echo "ERROR: $(MPICC) not found.  Install MPI first:"; \
	    echo "  sudo apt install libopenmpi-dev openmpi-bin"; exit 1; }
	$(MPICC) -x cu $(NVCCFLAGS) $(MPI_EXTRA) -o $@ $< $(CUDA_LIB)

clean:
	rm -f stencil_streams mpi_cuda_stencil
