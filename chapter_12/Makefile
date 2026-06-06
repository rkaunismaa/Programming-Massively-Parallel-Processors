SM_ARCH   ?= sm_61
TILE_SIZE ?= 512
NVCC      := nvcc
NVCCFLAGS := -arch=$(SM_ARCH) -O2 -std=c++14 \
             -DTILE_SIZE=$(TILE_SIZE)

ifdef DEBUG
    NVCCFLAGS += -g -G
endif

TARGETS := merge_sequential merge_basic merge_tiled merge_circular_buffer

.PHONY: all clean

all: $(TARGETS)

merge_sequential:      01_merge_sequential.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

merge_basic:           02_merge_basic.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

merge_tiled:           03_merge_tiled.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

merge_circular_buffer: 04_merge_circular_buffer.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o *.ptx *.cubin *.fatbin

# Examples:
#   make SM_ARCH=sm_61 TILE_SIZE=1024
#   make SM_ARCH=sm_61 DEBUG=1
