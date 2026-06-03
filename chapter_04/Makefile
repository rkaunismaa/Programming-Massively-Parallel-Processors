# Chapter 4 — Compute Architecture and Scheduling
#
# Adjust SM_ARCH to match your GPU (sm_70=V100, sm_80=A100, sm_86=RTX30xx, sm_89=RTX40xx)
SM_ARCH  ?= sm_70
NVCC      = nvcc
NVCCFLAGS = -O2 -arch=$(SM_ARCH)

ifdef DEBUG
  NVCCFLAGS += -g -G
endif

TARGETS = barrier_sync      \
          warp_partitioning  \
          control_divergence \
          query_device       \
          occupancy

.PHONY: all clean

all: $(TARGETS)

barrier_sync: 01_barrier_synchronization.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

warp_partitioning: 02_warp_partitioning.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

control_divergence: 03_control_divergence.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

query_device: 04_query_device_properties.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

occupancy: 05_occupancy.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
