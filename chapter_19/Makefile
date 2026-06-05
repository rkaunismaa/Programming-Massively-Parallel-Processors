SM_ARCH   ?= sm_89
NVCC      := nvcc
NVCCFLAGS := -arch=$(SM_ARCH) -O2 -std=c++14

ifdef DEBUG
  NVCCFLAGS += -g -G
endif

.PHONY: all clean decomposition amdahl

all: decomposition amdahl

decomposition: 01_decomposition.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

amdahl: 02_amdahl.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f decomposition amdahl
