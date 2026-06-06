SM_ARCH  ?= sm_61
NVCC     := nvcc
NVCCFLAGS := -arch=$(SM_ARCH) -O2 -std=c++14

ifdef DEBUG
    NVCCFLAGS += -g -G
endif

TARGETS := bfs_vertex_push bfs_vertex_pull bfs_edge_centric \
           bfs_frontier_push bfs_frontier_privatized

.PHONY: all clean

all: $(TARGETS)

bfs_vertex_push:       01_bfs_vertex_push.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

bfs_vertex_pull:       02_bfs_vertex_pull.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

bfs_edge_centric:      03_bfs_edge_centric.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

bfs_frontier_push:     04_bfs_frontier_push.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

bfs_frontier_privatized: 05_bfs_frontier_privatized.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o *.ptx *.cubin *.fatbin

# Examples:
#   make SM_ARCH=sm_61
#   make SM_ARCH=sm_61 DEBUG=1 bfs_frontier_privatized
