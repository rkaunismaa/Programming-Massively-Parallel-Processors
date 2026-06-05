// §21.4  Quadtree construction — recursive CUDA Dynamic Parallelism
//        Figs 21.8–21.11, Appendix A21.1
//
// A quadtree partitions a 2-D plane by recursively splitting each node
// into four equal quadrants until the node contains ≤ MIN_POINTS_PER_NODE
// points or the maximum recursion depth is reached (§21.4).
//
// CDP kernel design (Fig 21.8 / 21.10):
//   One block per quadtree node.  The block:
//     1. Checks termination (too few points or at max depth).
//     2. Computes the centre of the node's bounding box.
//     3. Counts points in each of the 4 child quadrants (shared memory).
//     4. Scans those counts to get per-quadrant placement offsets.
//     5. Reorders the points into the output buffer.
//     6. Thread 0 atomically allocates 4 child nodes, prepares them,
//        and launches build_quadtree_kernel<<<4, BLOCK_DIM>>> recursively.
//
// Two point buffers ping-pong between levels (Fig 21.9).  The last
// thread (check_num_points_and_depth) copies buffer 1 → buffer 0 when
// the recursion stops so the caller always reads from buffer 0.
//
// Node allocation: the book's Fig 21.10 uses a level-based scheme that
// only works when the tree is full.  This implementation uses a global
// atomic counter (g_node_count) so that arbitrary partial trees are
// supported correctly.
//
// Support code (Appendix A21.1): Points, Bounding_box, Quadtree_node,
// Parameters classes are all defined here.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ── Tuneable constants ────────────────────────────────────────────────────────
#define N_POINTS            4096
#define MAX_DEPTH              6    // max recursion depth (hardware limit is 24)
#define MIN_POINTS_PER_NODE    4
#define BLOCK_DIM             64    // threads per quadtree block
#define MAX_NODES           8192    // pre-allocated node-pool size

// ─────────────────────────────────────────────────────────────────────────────
// Appendix A21.1 — Support types
// ─────────────────────────────────────────────────────────────────────────────

// SoA storage for 2-D points (A21.1 lines 1-29)
class Points {
    float *m_x;
    float *m_y;
public:
    __host__ __device__ Points() : m_x(NULL), m_y(NULL) {}
    __host__ __device__ Points(float *x, float *y) : m_x(x), m_y(y) {}

    __host__ __device__ __forceinline__ float2 get_point(int i) const {
        return make_float2(m_x[i], m_y[i]);
    }
    __host__ __device__ __forceinline__ void set_point(int i, const float2 &p) {
        m_x[i] = p.x;  m_y[i] = p.y;
    }
    __host__ __device__ __forceinline__ void set(float *x, float *y) {
        m_x = x;  m_y = y;
    }
};

// Axis-aligned bounding box (A21.1 lines 31-71)
class Bounding_box {
    float2 m_p_min, m_p_max;
public:
    __host__ __device__ Bounding_box() {
        m_p_min = make_float2(0.f, 0.f);
        m_p_max = make_float2(1.f, 1.f);
    }
    __host__ __device__ void set(float min_x, float min_y,
                                  float max_x, float max_y) {
        m_p_min = make_float2(min_x, min_y);
        m_p_max = make_float2(max_x, max_y);
    }
    __host__ __device__ __forceinline__ void compute_center(float2 &c) const {
        c.x = 0.5f * (m_p_min.x + m_p_max.x);
        c.y = 0.5f * (m_p_min.y + m_p_max.y);
    }
    __host__ __device__ __forceinline__ const float2 &get_max() const { return m_p_max; }
    __host__ __device__ __forceinline__ const float2 &get_min() const { return m_p_min; }
    __host__ __device__ __forceinline__ bool contains(const float2 &p) const {
        return p.x >= m_p_min.x && p.x < m_p_max.x &&
               p.y >= m_p_min.y && p.y < m_p_max.y;
    }
};

// A single node in the quadtree (A21.1 lines 73-126)
class Quadtree_node {
    int          m_id;
    Bounding_box m_bbox;
    int          m_begin, m_end;
    int          m_is_leaf;      // 1 when recursion stops here, 0 otherwise
public:
    __host__ __device__ Quadtree_node()
        : m_id(0), m_begin(0), m_end(0), m_is_leaf(0) {}

    __host__ __device__ int  id()              const { return m_id; }
    __host__ __device__ void set_id(int id)          { m_id = id; }
    __host__ __device__ int  is_leaf()         const { return m_is_leaf; }
    __host__ __device__ void set_is_leaf(int v)      { m_is_leaf = v; }

    __host__ __device__ const Bounding_box &bounding_box() const { return m_bbox; }
    __host__ __device__ void set_bounding_box(float min_x, float min_y,
                                               float max_x, float max_y) {
        m_bbox.set(min_x, min_y, max_x, max_y);
    }

    __host__ __device__ int  num_points()   const { return m_end - m_begin; }
    __host__ __device__ int  points_begin() const { return m_begin; }
    __host__ __device__ int  points_end()   const { return m_end; }
    __host__ __device__ void set_range(int begin, int end) {
        m_begin = begin;  m_end = end;
    }
};

// Algorithm parameters (A21.1 lines 128-156)
struct Parameters {
    int point_selector;      // which buffer is the input (0 or 1)
    int depth;               // current recursion depth
    int max_depth;
    int min_points_per_node;

    __host__ __device__ Parameters(int max_d, int min_pts)
        : point_selector(0), depth(0),
          max_depth(max_d), min_points_per_node(min_pts) {}

    // Copy constructor for child launch: flips selector, increments depth
    __host__ __device__ Parameters(const Parameters &p, bool /*next_level*/)
        : point_selector((p.point_selector + 1) % 2),
          depth(p.depth + 1),
          max_depth(p.max_depth),
          min_points_per_node(p.min_points_per_node) {}
};

// ── Global node counter (replaces the level-offset scheme in Fig 21.10) ───────
// Root occupies slot 0; every subsequent allocation increments this counter.
__device__ int g_node_count;

// ─────────────────────────────────────────────────────────────────────────────
// Fig 21.11 — device helper functions
// ─────────────────────────────────────────────────────────────────────────────

// Checks whether this block should stop recursing (Fig 21.11 lines 1-15).
// If stopping: if buffer 1 is the current input, copy points → buffer 0.
__device__ bool check_num_points_and_depth(Quadtree_node *node, Points *points,
                                            int num_points,
                                            const Parameters &params) {
    if (params.depth >= params.max_depth ||
        num_points  <= params.min_points_per_node) {
        // Mark as leaf (thread 0 does the write; all threads do the copy)
        if (threadIdx.x == 0) node->set_is_leaf(1);
        // Ensure output is always in buffer 0
        if (params.point_selector == 1) {
            int begin = node->points_begin(), end = node->points_end();
            for (int it = begin + threadIdx.x; it < end; it += blockDim.x)
                points[0].set_point(it, points[1].get_point(it));
        }
        __syncthreads();
        return true;
    }
    return false;
}

// Counts how many points fall in each of the 4 child quadrants
// (Fig 21.11 lines 17-36).
// Quadrant encoding: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
__device__ void count_points_in_children(const Points &in_pts, int *smem,
                                          int range_begin, int range_end,
                                          float2 center) {
    if (threadIdx.x < 4) smem[threadIdx.x] = 0;
    __syncthreads();

    for (int it = range_begin + threadIdx.x; it < range_end; it += blockDim.x) {
        float2 p = in_pts.get_point(it);
        int q = (p.x >= center.x ? 1 : 0) + (p.y >= center.y ? 2 : 0);
        atomicAdd(&smem[q], 1);
    }
    __syncthreads();
}

// Exclusive-scan of the 4 counts → write per-quadrant starting offsets
// into smem[4..7] (Fig 21.11 lines 38-48).
__device__ void scan_for_offsets(int node_pts_begin, int *smem) {
    int *smem2 = &smem[4];
    if (threadIdx.x == 0) {
        int sum = node_pts_begin;
        for (int q = 0; q < 4; q++) {
            smem2[q] = sum;
            sum += smem[q];
        }
    }
    __syncthreads();
}

// Reorders points into child-quadrant groups in the output buffer
// (Fig 21.11 lines 50-71).
__device__ void reorder_points(Points &out_pts, const Points &in_pts,
                                int *smem,
                                int range_begin, int range_end,
                                float2 center) {
    int *smem2 = &smem[4];
    for (int it = range_begin + threadIdx.x; it < range_end; it += blockDim.x) {
        float2 p = in_pts.get_point(it);
        int q = (p.x >= center.x ? 1 : 0) + (p.y >= center.y ? 2 : 0);
        int dest = atomicAdd(&smem2[q], 1);
        out_pts.set_point(dest, p);
    }
    __syncthreads();
}

// Initialise the 4 child nodes with their IDs, bounding boxes, and point
// ranges (Fig 21.11 lines 73-103).
__device__ void prepare_children(Quadtree_node *children,
                                  const Quadtree_node *parent,
                                  const Bounding_box &bbox,
                                  const int *smem) {
    const int *smem2 = &smem[4];
    const float2 &pmin = bbox.get_min();
    const float2 &pmax = bbox.get_max();
    float2 center;
    bbox.compute_center(center);

    // Quadrant layout matches count_points_in_children:
    //   q=0: x<cx, y<cy  (bottom-left)   q=1: x≥cx, y<cy  (bottom-right)
    //   q=2: x<cx, y≥cy  (top-left)      q=3: x≥cx, y≥cy  (top-right)
    children[0].set_id(4 * parent->id() + 0);
    children[1].set_id(4 * parent->id() + 1);
    children[2].set_id(4 * parent->id() + 2);
    children[3].set_id(4 * parent->id() + 3);

    children[0].set_bounding_box(pmin.x,    pmin.y,    center.x, center.y);
    children[1].set_bounding_box(center.x,  pmin.y,    pmax.x,   center.y);
    children[2].set_bounding_box(pmin.x,    center.y,  center.x, pmax.y);
    children[3].set_bounding_box(center.x,  center.y,  pmax.x,   pmax.y);

    // After reorder_points, smem2[q] = initial_offset[q] + count[q] = END of q's range.
    // smem[q] = count[q] (unchanged by reorder_points).
    // So start = smem2[q] - smem[q], end = smem2[q].
    children[0].set_range(smem2[0] - smem[0], smem2[0]);
    children[1].set_range(smem2[1] - smem[1], smem2[1]);
    children[2].set_range(smem2[2] - smem[2], smem2[2]);
    children[3].set_range(smem2[3] - smem[3], smem2[3]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fig 21.10 — recursive quadtree kernel
//
// `nodes` is ALWAYS the full node pool (d_nodes).  `node_offset` is the
// index of the first node in this batch within that pool.  This is necessary
// because `child_base` from atomicAdd(&g_node_count,4) is an absolute index
// into the pool; if we passed &nodes[child_base] as the new `nodes` pointer
// instead, subsequent levels would compute &nodes[child_base] as
// (pool_base + level_offset) + child_base rather than pool_base + child_base,
// accumulating an ever-growing offset that pushes nodes beyond g_node_count.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void build_quadtree_kernel(Quadtree_node *nodes, int node_offset,
                                       Points *points, Parameters params) {
    __shared__ int smem[8];   // smem[0..3]: counts; smem[4..7]: offsets

    Quadtree_node *node = &nodes[node_offset + blockIdx.x]; // one block → one node
    int num_points = node->num_points();

    bool exit = check_num_points_and_depth(node, points, num_points, params);
    if (exit) return;

    const Bounding_box &bbox = node->bounding_box();
    float2 center;
    bbox.compute_center(center);

    int range_begin = node->points_begin();
    int range_end   = node->points_end();

    const Points &in_pts  = points[params.point_selector];
    Points       &out_pts = points[(params.point_selector + 1) % 2];

    count_points_in_children(in_pts,  smem, range_begin, range_end, center);
    scan_for_offsets(range_begin, smem);
    reorder_points(out_pts, in_pts, smem, range_begin, range_end, center);

    // Thread 0 allocates child slots and launches the recursive grid
    if (threadIdx.x == 0) {
        int child_base = atomicAdd(&g_node_count, 4);
        if (child_base + 4 <= MAX_NODES) {
            Quadtree_node *children = &nodes[child_base];
            prepare_children(children, node, bbox, smem);
            // Fig 21.10 line 43: launch 4 child blocks — one per quadrant.
            // Pass the full pool pointer and child_base as the absolute offset
            // so every recursive level uses the same base address.
            build_quadtree_kernel<<<4, BLOCK_DIM, 8*sizeof(int)>>>(
                nodes, child_base, points, Parameters(params, true));
        }
    }
}

// ── Device-side verification (avoids CDP write-visibility issues) ─────────────
// CDP child-kernel writes to device memory are not reliably visible via host
// cudaMemcpy until after an additional host-launched kernel syncs.  Run
// verification in a post-kernel GPU pass to ensure we read the correct state.
__global__ void verify_quadtree_kernel(const Quadtree_node *nodes, int node_count,
                                        const Points *points,    // points[0] = final buffer
                                        int n_pts, int *d_fail) {
    int ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= node_count) return;
    if (!nodes[ni].is_leaf()) return;

    int begin = nodes[ni].points_begin();
    int end   = nodes[ni].points_end();
    if (begin >= end) return;

    const Bounding_box &bb = nodes[ni].bounding_box();
    for (int pi = begin; pi < end; pi++) {
        if (pi < 0 || pi >= n_pts) { atomicAdd(d_fail, 1); return; }
        float2 pt = points[0].get_point(pi);
        if (!bb.contains(pt)) { atomicAdd(d_fail, 1); return; }
    }
}

__global__ void count_leaf_points_kernel(const Quadtree_node *nodes, int node_count,
                                          int *d_count) {
    int ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= node_count) return;
    if (nodes[ni].is_leaf())
        atomicAdd(d_count, nodes[ni].num_points());
}

int main(void) {
    printf("=== Quadtree with CUDA Dynamic Parallelism (§21.4, Figs 21.10/21.11) ===\n\n");
    printf("N_POINTS=%d  MAX_DEPTH=%d  MIN_PTS_PER_NODE=%d\n\n",
           N_POINTS, MAX_DEPTH, MIN_POINTS_PER_NODE);

    // ── Host data ────────────────────────────────────────────────────────────
    srand(42);
    float *h_px = (float *)malloc(N_POINTS * sizeof(float));
    float *h_py = (float *)malloc(N_POINTS * sizeof(float));
    // Divide by RAND_MAX+1 (as double) to stay strictly in [0, 1).
    // rand()/(float)RAND_MAX can produce exactly 1.0 on some platforms
    // because (float)RAND_MAX rounds up to 2^31, and the division yields 1.0.
    for (int i = 0; i < N_POINTS; i++) {
        h_px[i] = (float)(rand() / ((double)RAND_MAX + 1.0));
        h_py[i] = (float)(rand() / ((double)RAND_MAX + 1.0));
    }

    // ── Device allocations ───────────────────────────────────────────────────
    // Two point buffers for ping-pong (Fig 21.9)
    float *d_px0, *d_py0, *d_px1, *d_py1;
    cudaMalloc(&d_px0, N_POINTS * sizeof(float));
    cudaMalloc(&d_py0, N_POINTS * sizeof(float));
    cudaMalloc(&d_px1, N_POINTS * sizeof(float));
    cudaMalloc(&d_py1, N_POINTS * sizeof(float));
    cudaMemcpy(d_px0, h_px, N_POINTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py0, h_py, N_POINTS * sizeof(float), cudaMemcpyHostToDevice);

    // Points array on device: two Points objects (buffer 0 and buffer 1)
    Points h_pts_arr[2];
    h_pts_arr[0].set(d_px0, d_py0);
    h_pts_arr[1].set(d_px1, d_py1);
    Points *d_pts_arr;
    cudaMalloc(&d_pts_arr, 2 * sizeof(Points));
    cudaMemcpy(d_pts_arr, h_pts_arr, 2 * sizeof(Points), cudaMemcpyHostToDevice);

    // Node pool
    Quadtree_node *d_nodes;
    cudaMalloc(&d_nodes, MAX_NODES * sizeof(Quadtree_node));
    cudaMemset(d_nodes, 0, MAX_NODES * sizeof(Quadtree_node));

    // Initialise root node (Fig 21.10: host sets up root before first launch)
    Quadtree_node h_root;
    h_root.set_id(0);
    h_root.set_bounding_box(0.f, 0.f, 1.f, 1.f);
    h_root.set_range(0, N_POINTS);
    cudaMemcpy(&d_nodes[0], &h_root, sizeof(Quadtree_node), cudaMemcpyHostToDevice);

    // Initialise global counter to 1 (root is at slot 0)
    int init_count = 1;
    cudaMemcpyToSymbol(g_node_count, &init_count, sizeof(int));

    // §21.5: raise pending launch pool to handle deep trees
    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 4096);

    // ── Launch root kernel (1 block) ─────────────────────────────────────────
    Parameters params(MAX_DEPTH, MIN_POINTS_PER_NODE);
    build_quadtree_kernel<<<1, BLOCK_DIM, 8*sizeof(int)>>>(d_nodes, 0, d_pts_arr, params);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // ── Copy results back ─────────────────────────────────────────────────────
    int h_node_count;
    cudaMemcpyFromSymbol(&h_node_count, g_node_count, sizeof(int));

    Quadtree_node *h_nodes = (Quadtree_node *)malloc(h_node_count * sizeof(Quadtree_node));
    cudaMemcpy(h_nodes, d_nodes, h_node_count * sizeof(Quadtree_node), cudaMemcpyDeviceToHost);

    // ── Stats (device-side to avoid CDP write-visibility issues) ─────────────
    printf("Nodes allocated: %d / %d\n", h_node_count, MAX_NODES);

    int vblocks = (h_node_count + 255) / 256;
    int *d_fail, *d_leaf_pts;
    cudaMalloc(&d_fail,     sizeof(int));
    cudaMalloc(&d_leaf_pts, sizeof(int));
    cudaMemset(d_fail,     0, sizeof(int));
    cudaMemset(d_leaf_pts, 0, sizeof(int));

    verify_quadtree_kernel<<<vblocks, 256>>>(
        d_nodes, h_node_count, d_pts_arr, N_POINTS, d_fail);
    count_leaf_points_kernel<<<vblocks, 256>>>(
        d_nodes, h_node_count, d_leaf_pts);
    cudaDeviceSynchronize();

    int h_fail = 0, h_leaf_pts = 0;
    cudaMemcpy(&h_fail,     d_fail,     sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_leaf_pts, d_leaf_pts, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_fail); cudaFree(d_leaf_pts);

    // Host-side leaf count from node metadata
    int leaf_count = 0;
    for (int i = 0; i < h_node_count; i++)
        if (h_nodes[i].is_leaf()) leaf_count++;
    printf("Leaf nodes (host-visible): %d  |  leaf pts (device count): %d / %d\n",
           leaf_count, h_leaf_pts, N_POINTS);

    // ── Verify ────────────────────────────────────────────────────────────────
    printf("Spatial correctness (every point inside its node): %s\n",
           h_fail == 0 ? "PASS" : "FAIL");

    printf("\nRecursion structure (§21.4, Fig 21.8):\n");
    printf("  • Each block owns one quadrant; depth limit = %d\n", MAX_DEPTH);
    printf("  • Blocks with ≤ %d points exit without launching children\n",
           MIN_POINTS_PER_NODE);
    printf("  • Thread 0 allocates 4 child slots atomically and launches\n");
    printf("    build_quadtree_kernel<<<4, %d>>> recursively (Fig 21.10)\n", BLOCK_DIM);
    printf("  • Max nesting depth supported by hardware: 24 levels (§21.5)\n");

    free(h_nodes); free(h_px);
    cudaFree(d_px0); cudaFree(d_py0); cudaFree(d_px1); cudaFree(d_py1);
    cudaFree(d_pts_arr); cudaFree(d_nodes);
    return 0;
}
