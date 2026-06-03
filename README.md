# Programming Massively Parallel Processors — Code Samples

Runnable CUDA C code samples for every chapter of  
**"Programming Massively Parallel Processors: A Hands-on Approach" (4th edition, 2023)**  
by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj.

Each chapter subdirectory contains standalone `.cu` / `.c` files, one per major concept, with comments that cross-reference the relevant figures and sections in the book.

---

## Repository layout

```
.
├── chapter_02/    Heterogeneous Data Parallel Computing
├── chapter_03/    Multidimensional Grids and Data
└── ...
```

---

## Prerequisites

| Tool | Version used | Notes |
|------|-------------|-------|
| NVIDIA GPU | RTX 4090 (sm_89) | Any CUDA-capable GPU works; adjust `SM_ARCH` |
| CUDA Toolkit | 12.4 | `nvcc` and `cuda-gdb` at `/usr/local/cuda/bin/` |
| GCC | system default | For the sequential `.c` examples |
| GNU Make | system default | Each chapter has its own `Makefile` |
| VS Code | latest | See debugger setup below |

---

## Building

Every chapter directory has a `Makefile`.  
Set `SM_ARCH` to match your GPU's compute capability:

| GPU family | `SM_ARCH` |
|------------|-----------|
| RTX 40xx   | `sm_89`   |
| RTX 30xx   | `sm_86`   |
| A100       | `sm_80`   |
| V100       | `sm_70`   |

```bash
# Build all programs in a chapter
cd chapter_02
make SM_ARCH=sm_89

# Build a single program
make SM_ARCH=sm_89 vec_add_cuda

# Clean
make clean
```

---

## Debugging CUDA kernels in VS Code

### 1. Install the required extension

Open VS Code, press `Ctrl+Shift+X`, and install:

> **NVIDIA Nsight Visual Studio Code Edition**  
> Publisher: NVIDIA  
> Extension ID: `nvidia.nsight-vscode-edition`

This extension adds a `cuda-gdb` debug type that lets you set breakpoints *inside device kernel functions*, switch between GPU threads and warps, and inspect GPU registers and memory.

---

### 2. Create a debug build

`cuda-gdb` needs host debug symbols (`-g`) **and** device debug symbols (`-G`).  
The `-G` flag disables GPU optimisations so line numbers map cleanly to device code.

```bash
# Build a single file with debug symbols
nvcc -g -G -arch=sm_89 -o vec_add_cuda_dbg chapter_02/02_vec_add_cuda.cu

# Or use the per-chapter Makefile debug target
cd chapter_02
make SM_ARCH=sm_89 DEBUG=1 vec_add_cuda
```

Each chapter `Makefile` supports `DEBUG=1` to inject `-g -G` automatically:

```makefile
ifdef DEBUG
  NVCCFLAGS += -g -G
endif
```

> **Note:** `-G` can make kernels run 10–100× slower.  Always remove it for performance testing.

---

### 3. Configure `.vscode/launch.json`

Create (or open) `.vscode/launch.json` at the **root of this repository** and add a configuration for each program you want to debug.  
The template below debugs `vec_add_cuda_dbg` from Chapter 2:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Ch02: CUDA vecAdd",
            "type": "cuda-gdb",
            "request": "launch",
            "program": "${workspaceFolder}/chapter_02/vec_add_cuda_dbg",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}/chapter_02",
            "environment": [],
            "externalConsole": false,
            "cuda": {
                "runtime": "cuda",
                "breakOnLaunch": false,
                "breakOnExit": false
            }
        },
        {
            "name": "Ch02: error checking",
            "type": "cuda-gdb",
            "request": "launch",
            "program": "${workspaceFolder}/chapter_02/error_checking_dbg",
            "args": [],
            "cwd": "${workspaceFolder}/chapter_02"
        },
        {
            "name": "Ch03: matrix multiply",
            "type": "cuda-gdb",
            "request": "launch",
            "program": "${workspaceFolder}/chapter_03/matrix_multiply_dbg",
            "args": [],
            "cwd": "${workspaceFolder}/chapter_03"
        }
    ]
}
```

> **Tip:** Set `"breakOnLaunch": true` to automatically pause execution the moment the first CUDA kernel is launched.

---

### 4. Configure `.vscode/tasks.json`

Add a build task so you can press `Ctrl+Shift+B` to recompile before debugging:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build Ch02 debug",
            "type": "shell",
            "command": "make -C ${workspaceFolder}/chapter_02 SM_ARCH=sm_89 DEBUG=1",
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": ["$nvcc"]
        },
        {
            "label": "Build Ch03 debug",
            "type": "shell",
            "command": "make -C ${workspaceFolder}/chapter_03 SM_ARCH=sm_89 DEBUG=1"
        }
    ]
}
```

Link a build task to a debug configuration by adding `"preLaunchTask"` to any launch entry:

```json
"preLaunchTask": "Build Ch02 debug"
```

---

### 5. Set breakpoints and run

1. Open a `.cu` source file (e.g. `chapter_02/02_vec_add_cuda.cu`).
2. Click the gutter to the left of a line number **inside a kernel function** to set a device breakpoint.
3. Select the desired configuration from the **Run and Debug** panel (`Ctrl+Shift+D`).
4. Press **F5** to launch.

When execution hits the breakpoint inside the kernel, VS Code pauses and the **Variables** panel shows:

| Variable | What you see |
|----------|-------------|
| `threadIdx` | `{x: 0, y: 0, z: 0}` (the focused thread) |
| `blockIdx`  | `{x: 0, y: 0, z: 0}` |
| `blockDim`  | `{x: 256, y: 1, z: 1}` |
| local vars  | e.g. `i = 0`, `C[0]` |

---

### 6. Switch between GPU threads

The **CUDA** panel (left sidebar, after the Nsight extension is installed) lets you:

- Browse all active **warps** and **lanes** on each SM.
- Click any warp/lane to move the debugger focus there and inspect its local variables.
- Use the **CUDA Focus** command to jump to a specific `(device, SM, warp, lane)`.

You can also use the **Debug Console** to run `cuda-gdb` commands directly:

```
-exec info cuda threads
-exec cuda thread (0,0,0)
-exec cuda block (1,0,0)
-exec info locals
```

---

### 7. Common `cuda-gdb` debug console commands

| Command | Effect |
|---------|--------|
| `info cuda threads` | List all active CUDA threads |
| `info cuda blocks` | List all active thread blocks |
| `cuda thread (tx,ty,tz)` | Switch focus to a specific thread |
| `cuda block (bx,by,bz)` | Switch focus to a specific block |
| `print threadIdx` | Print current thread's index |
| `print *A_d@16` | Print first 16 elements of device array `A_d` |
| `watch C_h[0]` | Watchpoint on a host variable |

---

## Chapters

| Chapter | Title | Directory |
|---------|-------|-----------|
| 2 | Heterogeneous Data Parallel Computing | [chapter_02/](chapter_02/) |
| 3 | Multidimensional Grids and Data | [chapter_03/](chapter_03/) |
| 4 | Compute Architecture and Scheduling | [chapter_04/](chapter_04/) |
| 5 | Memory Architecture and Data Locality | [chapter_05/](chapter_05/) |

---

## Quick reference: CUDA C keywords

| Keyword | Meaning |
|---------|---------|
| `__global__` | Kernel — called from host, runs on device, launches a new thread grid |
| `__device__` | Device function — called from device only, no new grid |
| `__host__`   | Host function (default if no qualifier) |
| `threadIdx`  | Thread's position within its block |
| `blockIdx`   | Block's position within the grid |
| `blockDim`   | Block dimensions (threads per block) |
| `gridDim`    | Grid dimensions (blocks per grid) |
| `__syncthreads()` | Barrier — all threads in a block must reach this point before any continue |
