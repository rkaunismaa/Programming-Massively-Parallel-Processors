SM_ARCH  ?= sm_89
NVCC     := nvcc
NVCCFLAGS := -arch=$(SM_ARCH) -O2 -std=c++14

ifdef DEBUG
    NVCCFLAGS += -g -G
endif

TARGETS := conv_forward_basic conv_forward_gemm

.PHONY: all clean

all: $(TARGETS)

conv_forward_basic: 01_conv_forward_basic.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

conv_forward_gemm: 02_conv_forward_gemm.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o *.ptx *.cubin *.fatbin

# Examples:
#   make SM_ARCH=sm_89
#   make SM_ARCH=sm_89 DEBUG=1 conv_forward_gemm
