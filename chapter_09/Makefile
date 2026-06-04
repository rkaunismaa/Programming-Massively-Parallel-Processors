SM_ARCH  ?= sm_89
CFACTOR  ?= 4
NVCC     := nvcc
NVCCFLAGS := -arch=$(SM_ARCH) -O2 -std=c++14 -DCFACTOR=$(CFACTOR)

ifdef DEBUG
    NVCCFLAGS += -g -G
endif

TARGETS := histogram_basic histogram_priv_global histogram_priv_shared \
           histogram_coarsened histogram_aggregated

.PHONY: all clean

all: $(TARGETS)

histogram_basic:        01_histogram_atomic_basic.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

histogram_priv_global:  02_histogram_privatized_global.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

histogram_priv_shared:  03_histogram_privatized_shared.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

histogram_coarsened:    04_histogram_coarsened.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

histogram_aggregated:   05_histogram_aggregated.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o *.ptx *.cubin *.fatbin

# Example: build with coarsening factor 8
#   make SM_ARCH=sm_89 CFACTOR=8
