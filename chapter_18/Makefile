SM_ARCH  ?= sm_61
NVCC     := nvcc
NVCCFLAGS := -arch=$(SM_ARCH) -O2 -std=c++14

ifdef DEBUG
    NVCCFLAGS += -g -G
endif

TARGETS := dcs_gather dcs_coarsen dcs_coalesce

.PHONY: all clean

all: $(TARGETS)

dcs_gather:   01_dcs_gather.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

dcs_coarsen:  02_dcs_coarsen.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

dcs_coalesce: 03_dcs_coalesce.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o *.ptx *.cubin *.fatbin

# Examples:
#   make SM_ARCH=sm_61
#   make SM_ARCH=sm_61 DEBUG=1 dcs_coalesce
