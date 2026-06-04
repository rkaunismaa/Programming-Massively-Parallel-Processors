SM_ARCH  ?= sm_89
NVCC     := nvcc
NVCCFLAGS := -arch=$(SM_ARCH) -O2 -std=c++14

ifdef DEBUG
    NVCCFLAGS += -g -G
endif

TARGETS := stencil1d stencil3d_basic stencil3d_shared stencil3d_coarsened stencil3d_regtile

.PHONY: all clean

all: $(TARGETS)

stencil1d:           01_stencil1d_basic.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

stencil3d_basic:     02_stencil3d_basic.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

stencil3d_shared:    03_stencil3d_shared_mem.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

stencil3d_coarsened: 04_stencil3d_coarsened.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

stencil3d_regtile:   05_stencil3d_register_tiling.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o *.ptx *.cubin *.fatbin
