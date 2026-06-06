SM_ARCH  ?= sm_61
NVCC     := nvcc
NVCCFLAGS := -arch=$(SM_ARCH) -O2 -std=c++14

# Dynamic parallelism requires relocatable device code (-rdc=true).
# Files 01, 03, 04 launch kernels from device code; 02 does not.
CDP_FLAGS := -rdc=true

ifdef DEBUG
    NVCCFLAGS += -g -G
endif

TARGETS := cdp_overview bezier_no_cdp bezier_cdp quadtree_cdp

.PHONY: all clean

all: $(TARGETS)

cdp_overview:  01_cdp_overview.cu
	$(NVCC) $(NVCCFLAGS) $(CDP_FLAGS) -o $@ $<

bezier_no_cdp: 02_bezier_no_cdp.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

bezier_cdp:    03_bezier_cdp.cu
	$(NVCC) $(NVCCFLAGS) $(CDP_FLAGS) -o $@ $<

quadtree_cdp:  04_quadtree_cdp.cu
	$(NVCC) $(NVCCFLAGS) $(CDP_FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o *.ptx *.cubin *.fatbin

# Examples:
#   make SM_ARCH=sm_61
#   make SM_ARCH=sm_61 DEBUG=1 bezier_cdp
#   make SM_ARCH=sm_61 quadtree_cdp
