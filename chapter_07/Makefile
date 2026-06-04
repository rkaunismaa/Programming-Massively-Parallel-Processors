SM_ARCH       ?= sm_89
FILTER_RADIUS ?= 2
NVCC          := nvcc
NVCCFLAGS     := -arch=$(SM_ARCH) -O2 -std=c++14 \
                  -DFILTER_RADIUS=$(FILTER_RADIUS)

ifdef DEBUG
    NVCCFLAGS += -g -G
endif

TARGETS := conv1d conv2d_basic conv2d_const conv2d_tiled conv2d_cached

.PHONY: all clean

all: $(TARGETS)

conv1d:       01_conv1d_basic.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

conv2d_basic: 02_conv2d_basic.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

conv2d_const: 03_conv2d_constant_mem.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

conv2d_tiled: 04_conv2d_tiled_halo.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

conv2d_cached: 05_conv2d_cached_halo.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o *.ptx *.cubin *.fatbin

# Example: build with a 9×9 filter (FILTER_RADIUS=4)
#   make SM_ARCH=sm_89 FILTER_RADIUS=4
