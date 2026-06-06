SM_ARCH  ?= sm_61
NVCC     := nvcc
NVCCFLAGS := -arch=$(SM_ARCH) -O2 -std=c++14

ifdef DEBUG
  NVCCFLAGS += -g -G
endif

TARGETS := unified_memory zero_copy concurrent_streams cooperative_kernel

.PHONY: all clean

all: $(TARGETS)

unified_memory: 01_unified_memory.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

zero_copy: 02_zero_copy.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

concurrent_streams: 03_concurrent_streams.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

cooperative_kernel: 04_cooperative_kernel.cu
	$(NVCC) $(NVCCFLAGS) -rdc=true -o $@ $<

clean:
	rm -f $(TARGETS)
