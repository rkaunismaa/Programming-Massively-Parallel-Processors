SM_ARCH      ?= sm_89
COARSE_FACTOR ?= 4
NVCC         := nvcc
NVCCFLAGS    := -arch=$(SM_ARCH) -O2 -std=c++14 -DCOARSE_FACTOR=$(COARSE_FACTOR)

ifdef DEBUG
    NVCCFLAGS += -g -G
endif

TARGETS := reduction_simple reduction_convergent reduction_shared \
           reduction_multiblock reduction_coarsened

.PHONY: all clean

all: $(TARGETS)

reduction_simple:     01_reduction_simple.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

reduction_convergent: 02_reduction_convergent.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

reduction_shared:     03_reduction_shared_mem.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

reduction_multiblock: 04_reduction_multiblock.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

reduction_coarsened:  05_reduction_coarsened.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o *.ptx *.cubin *.fatbin

# Example: coarsen by 8
#   make SM_ARCH=sm_89 COARSE_FACTOR=8
