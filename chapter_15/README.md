# Chapter 15 — Graph Traversal

Code samples for **Chapter 15: Graph traversal** from *Programming Massively Parallel Processors* (4th ed.).

The chapter implements **BFS (breadth-first search)** on a directed graph, progressively
optimizing across five strategies.

| File | Section | Strategy | Key idea |
|------|---------|----------|----------|
| `01_bfs_vertex_push.cu` | §15.3 | Vertex-centric push (top-down) | One thread/vertex; push level to unvisited neighbors via CSR |
| `02_bfs_vertex_pull.cu` | §15.3 | Vertex-centric pull (bottom-up) | One thread/vertex; pull from incoming edges via CSC; early exit |
| `03_bfs_edge_centric.cu` | §15.4 | Edge-centric | One thread/edge; uniform work; COO graph |
| `04_bfs_frontier_push.cu` | §15.5 | Frontier push with atomicCAS | Only frontier vertices get threads; atomicCAS prevents duplicates |
| `05_bfs_frontier_privatized.cu` | §15.6 | Privatized frontiers | Per-block local frontier in shared memory; coalesced global flush |

---

## Test graph (Fig 15.1)

9 vertices, 15 directed edges. BFS from root 0:

```
Expected levels: [0, 1, 1, 2, 2, 2, 2, 2, 3]

Level 0: {0}
Level 1: {1, 2}           (neighbors of 0)
Level 2: {3, 4, 5, 6, 7}  (neighbors of 1, 2)
Level 3: {8}              (neighbors of 3, 4, 5, 6)
```

CSR (Fig 15.3 A): `srcPtrs = [0,2,4,7,9,11,12,13,15,15]`  
CSC (Fig 15.3 B): `dstPtrs = [0,1,2,3,4,6,8,10,11,15]`

---

## Strategy comparison

| Strategy | Threads/level | Atomics | Graph format | Best for |
|----------|--------------|---------|-------------|----------|
| Push (top-down) | `numVertices` | none | CSR | Early levels |
| Pull (bottom-up) | `numVertices` | none | CSC | Late levels |
| Edge-centric | `numEdges` | none | COO | High-degree graphs |
| Frontier push | `|prevFrontier|` | CAS + Add | CSR | Sparse frontiers |
| Frontier privatized | `|prevFrontier|` | CAS + block Add | CSR | Large frontiers |

A **direction-optimized** BFS (§15.3) uses push for early levels and switches to pull for
later levels, combining both advantages.

---

## Building

```bash
make SM_ARCH=sm_89
make SM_ARCH=sm_89 DEBUG=1
```
