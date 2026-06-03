# Chapter 6 — Performance Considerations
SM_ARCH  ?= sm_70
NVCC      = nvcc
NVCCFLAGS = -O2 -arch=$(SM_ARCH)

ifdef DEBUG
  NVCCFLAGS += -g -G
endif

TARGETS = coalescing_demo    \
          corner_turning      \
          thread_coarsening   \
          opt_progression

.PHONY: all clean

all: $(TARGETS)

coalescing_demo: 01_coalescing_demo.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

corner_turning: 02_corner_turning.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lm

thread_coarsening: 03_thread_coarsening_matmul.cu
	$(NVCC) $(NVCCFLAGS) $(if $(CF),-DCOARSE_FACTOR=$(CF)) -o $@ $< -lm

opt_progression: 04_optimization_progression.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lm

clean:
	rm -f $(TARGETS)
