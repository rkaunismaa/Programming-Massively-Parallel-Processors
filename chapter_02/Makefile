# Chapter 2 — Heterogeneous Data Parallel Computing
#
# Adjust SM_ARCH to match your GPU's compute capability:
#   RTX 20xx / Titan V   → sm_70
#   RTX 30xx             → sm_86
#   RTX 40xx             → sm_89
#   A100                 → sm_80
#   H100                 → sm_90
#
SM_ARCH  ?= sm_70
NVCC      = nvcc
CC        = gcc
NVCCFLAGS = -O2 -arch=$(SM_ARCH)
CFLAGS    = -O2 -Wall

# Pass DEBUG=1 to add host (-g) and device (-G) debug symbols for cuda-gdb.
# Example: make SM_ARCH=sm_61 DEBUG=1
# Note: -G disables GPU optimisations — only use for debugging, not benchmarking.
ifdef DEBUG
  NVCCFLAGS += -g -G
  CFLAGS    += -g
endif

TARGETS = vec_add_sequential \
          vec_add_cuda        \
          error_checking      \
          grayscale           \
          function_qualifiers

.PHONY: all clean

all: $(TARGETS)

vec_add_sequential: 01_vec_add_sequential.c
	$(CC) $(CFLAGS) -o $@ $< -lm

vec_add_cuda: 02_vec_add_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lm

error_checking: 03_error_checking.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

grayscale: 04_data_parallelism_grayscale.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

function_qualifiers: 05_function_qualifiers.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
