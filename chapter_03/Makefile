# Chapter 3 — Multidimensional Grids and Data
#
# Adjust SM_ARCH to match your GPU (sm_70=V100, sm_80=A100, sm_86=RTX30xx, sm_89=RTX40xx)
SM_ARCH  ?= sm_70
NVCC      = nvcc
NVCCFLAGS = -O2 -arch=$(SM_ARCH)

# Pass DEBUG=1 to add host (-g) and device (-G) debug symbols for cuda-gdb.
# Example: make SM_ARCH=sm_61 DEBUG=1
ifdef DEBUG
  NVCCFLAGS += -g -G
endif

TARGETS = multidim_grid      \
          color_to_grayscale  \
          image_blur          \
          matrix_multiply     \
          row_major

.PHONY: all clean

all: $(TARGETS)

multidim_grid: 01_multidim_grid_organization.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

color_to_grayscale: 02_color_to_grayscale.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

image_blur: 03_image_blur.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Build blur with a larger 7×7 patch (BLUR_SIZE=3):
#   make image_blur_7x7
image_blur_7x7: 03_image_blur.cu
	$(NVCC) $(NVCCFLAGS) -DBLUR_SIZE=3 -o $@ $<

matrix_multiply: 04_matrix_multiply.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lm

row_major: 05_row_major_linearization.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) image_blur_7x7
