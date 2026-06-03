# Chapter 5 — Memory Architecture and Data Locality
#
# Adjust SM_ARCH to match your GPU (sm_70=V100, sm_80=A100, sm_86=RTX30xx, sm_89=RTX40xx)
SM_ARCH  ?= sm_70
NVCC      = nvcc
NVCCFLAGS = -O2 -arch=$(SM_ARCH)

ifdef DEBUG
  NVCCFLAGS += -g -G
endif

TARGETS = memory_types        \
          tiled_matmul         \
          tiled_matmul_bc      \
          dynamic_shmem        \
          roofline

.PHONY: all clean

all: $(TARGETS)

memory_types: 01_memory_types_demo.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

tiled_matmul: 02_tiled_matmul.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lm

# Build with alternative tile width: make SM_ARCH=sm_89 TILE=32 tiled_matmul
tiled_matmul: 02_tiled_matmul.cu
	$(NVCC) $(NVCCFLAGS) $(if $(TILE),-DTILE_WIDTH=$(TILE)) -o $@ $< -lm

tiled_matmul_bc: 03_tiled_matmul_boundary.cu
	$(NVCC) $(NVCCFLAGS) $(if $(TILE),-DTILE_WIDTH=$(TILE)) -o $@ $< -lm

dynamic_shmem: 04_dynamic_shared_memory.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lm

roofline: 05_roofline_analysis.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lm

clean:
	rm -f $(TARGETS)
