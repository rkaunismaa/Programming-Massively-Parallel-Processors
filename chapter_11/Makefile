SM_ARCH      ?= sm_89
SECTION_SIZE ?= 1024
BLOCK_DIM    ?= 256
CFACTOR      ?= 4
NVCC         := nvcc
NVCCFLAGS    := -arch=$(SM_ARCH) -O2 -std=c++14 \
                -DSECTION_SIZE=$(SECTION_SIZE) \
                -DBLOCK_DIM=$(BLOCK_DIM) \
                -DCFACTOR=$(CFACTOR)

ifdef DEBUG
    NVCCFLAGS += -g -G
endif

TARGETS := scan_kogge_stone scan_brent_kung scan_coarsened \
           scan_hierarchical scan_single_pass

.PHONY: all clean

all: $(TARGETS)

scan_kogge_stone:  01_scan_kogge_stone.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

scan_brent_kung:   02_scan_brent_kung.cu
	$(NVCC) $(NVCCFLAGS) -DSECTION_SIZE=2048 -o $@ $<

scan_coarsened:    03_scan_coarsened.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

scan_hierarchical: 04_scan_hierarchical.cu
	$(NVCC) $(NVCCFLAGS) -DSECTION_SIZE=2048 -o $@ $<

scan_single_pass:  05_scan_single_pass.cu
	$(NVCC) $(NVCCFLAGS) -DSECTION_SIZE=2048 -o $@ $<

clean:
	rm -f $(TARGETS) *.o *.ptx *.cubin *.fatbin

# Examples:
#   make SM_ARCH=sm_89 SECTION_SIZE=512    # smaller sections
#   make SM_ARCH=sm_89 CFACTOR=8           # more coarsening
