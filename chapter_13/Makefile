SM_ARCH   ?= sm_89
SEG_SIZE  ?= 256
TILE_SIZE ?= 256
NVCC      := nvcc
CC        := gcc
NVCCFLAGS := -arch=$(SM_ARCH) -O2 -std=c++14 \
             -DSEG_SIZE=$(SEG_SIZE) -DTILE_SIZE=$(TILE_SIZE)
CFLAGS    := -O2 -std=c11

ifdef DEBUG
    NVCCFLAGS += -g -G
    CFLAGS    += -g
endif

TARGETS := radix_sort_sequential radix_sort_parallel \
           radix_sort_coalesced  parallel_merge_sort

.PHONY: all clean

all: $(TARGETS)

radix_sort_sequential: 01_radix_sort_sequential.c
	$(CC) $(CFLAGS) -o $@ $<

radix_sort_parallel: 02_radix_sort_parallel.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

radix_sort_coalesced: 03_radix_sort_coalesced.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

parallel_merge_sort: 04_parallel_merge_sort.cu
	$(NVCC) $(NVCCFLAGS) -lm -o $@ $<

clean:
	rm -f $(TARGETS) *.o *.ptx *.cubin *.fatbin

# Examples:
#   make SM_ARCH=sm_89
#   make SM_ARCH=sm_89 SEG_SIZE=512 TILE_SIZE=512
#   make SM_ARCH=sm_89 DEBUG=1 radix_sort_parallel
