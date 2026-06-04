SM_ARCH  ?= sm_89
NVCC     := nvcc
NVCCFLAGS := -arch=$(SM_ARCH) -O2 -std=c++14

ifdef DEBUG
    NVCCFLAGS += -g -G
endif

TARGETS := fhd_scatter fhd_gather fhd_register fhd_constmem

.PHONY: all clean

all: $(TARGETS)

fhd_scatter:  01_fhd_scatter.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

fhd_gather:   02_fhd_gather.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

fhd_register: 03_fhd_register.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

fhd_constmem: 04_fhd_constmem.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o *.ptx *.cubin *.fatbin

# Examples:
#   make SM_ARCH=sm_89
#   make SM_ARCH=sm_89 DEBUG=1 fhd_constmem
