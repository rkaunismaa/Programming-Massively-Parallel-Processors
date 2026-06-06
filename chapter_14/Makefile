SM_ARCH  ?= sm_61
NVCC     := nvcc
NVCCFLAGS := -arch=$(SM_ARCH) -O2 -std=c++14

ifdef DEBUG
    NVCCFLAGS += -g -G
endif

TARGETS := spmv_coo spmv_csr spmv_ell spmv_ell_coo spmv_jds

.PHONY: all clean

all: $(TARGETS)

spmv_coo:     01_spmv_coo.cu
	$(NVCC) $(NVCCFLAGS) -lm -o $@ $<

spmv_csr:     02_spmv_csr.cu
	$(NVCC) $(NVCCFLAGS) -lm -o $@ $<

spmv_ell:     03_spmv_ell.cu
	$(NVCC) $(NVCCFLAGS) -lm -o $@ $<

spmv_ell_coo: 04_spmv_ell_coo.cu
	$(NVCC) $(NVCCFLAGS) -lm -o $@ $<

spmv_jds:     05_spmv_jds.cu
	$(NVCC) $(NVCCFLAGS) -lm -o $@ $<

clean:
	rm -f $(TARGETS) *.o *.ptx *.cubin *.fatbin

# Examples:
#   make SM_ARCH=sm_61
#   make SM_ARCH=sm_61 DEBUG=1 spmv_jds
