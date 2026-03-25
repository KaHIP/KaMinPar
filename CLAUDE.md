# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

After completing any task, update this file if you learned anything that should be recorded here.

## Build Commands

```bash
# Shared-memory partitioner (default)
cmake -B build --preset=default && cmake --build build --parallel

# Distributed partitioner (requires MPI)
cmake -B build --preset=distributed && cmake --build build --parallel

# Large graphs (64-bit edge IDs/weights)
cmake -B build --preset=memory && cmake --build build --parallel
```

Key CMake options:
- `-DKAMINPAR_BUILD_TESTS=On` — enable unit tests
- `-DKAMINPAR_BUILD_DISTRIBUTED=On` — enable distributed partitioner
- `-DKAMINPAR_ASSERTION_LEVEL=light|normal|heavy` — assertion verbosity
- `-DKAMINPAR_BUILD_WITH_ASAN=On -DKAMINPAR_BUILD_WITH_UBSAN=On` — sanitizers

## Running Tests

```bash
cmake -B build --preset=default \
  -DKAMINPAR_BUILD_TESTS=On \
  -DKAMINPAR_BUILD_APPS=On
cmake --build build --parallel
cd build && ctest --output-on-failure

# Run a specific test
ctest -R <test_name> --output-on-failure
```

Tests also cover distributed code when `-DKAMINPAR_BUILD_DISTRIBUTED=On` is set.

## Code Formatting

```bash
./scripts/run_clang_format.sh
```

Applies clang-format (LLVM-based style, 2-space indents) to all `.cc` and `.h` files in `apps/`, `tests/`, `kaminpar-*/`, `kaminpar-cli/`, `kaminpar-mpi/`, and `external/growt/`.

## Architecture

KaMinPar is a multilevel graph partitioning library with shared-memory and distributed-memory variants.

### Module Layout

- **`kaminpar-common/`** — Shared utilities (OBJECT library): data structures (binary heap, marker arrays, hash maps), graph compression (StreamVByte, interval/high-degree encoding), parallel algorithms, logging, and `KAssert` assertions.
- **`kaminpar-shm/`** — Shared-memory partitioner (TBB-parallelized). Implements the full multilevel pipeline: coarsening (matching/clustering), initial partitioning (greedy graph growing, LP), refinement (FM, label propagation, flow-based), and uncoarsening. Main class: `include/kaminpar-shm/kaminpar.h`.
- **`kaminpar-dist/`** — Distributed partitioner (MPI). Mirrors the structure of `kaminpar-shm/` but operates on distributed graphs. Main class: `include/kaminpar-dist/dkaminpar.h`.
- **`kaminpar-mpi/`** — MPI primitives: sparse alltoall/allreduce/allgather, grid topology. Used only by `kaminpar-dist/`.
- **`kaminpar-io/`** — Graph I/O for METIS (`.metis`) and ParHIP (`.parhip`) formats, plus a compressed binary format. Separate dist-variants for distributed I/O.
- **`kaminpar-cli/`** — CLI argument parsing (CLI11). Wires configuration structs for both `KaMinPar` and `dKaMinPar`.
- **`apps/`** — Executable entry points (`KaMinPar.cc`, `dKaMinPar.cc`) plus benchmarking and analysis tools.
- **`bindings/`** — Python and NetworKit language bindings.
- **`external/growt/`** — Bundled concurrent hash table library used by `kaminpar-dist/`.

### Multilevel Partitioning Pipeline

```
Graph Input → Coarsening (contract) → Initial Partitioning → Refinement → Uncoarsening → Partition Output
```

Partitioning configuration presets (`-P` flag): `default` (fast, Metis-quality), `eco` (more FM), `strong` (flow refinement), `terapart` (memory-efficient), `largek` (k > 1024).

### Data Type Variants

The library can be compiled with different data type widths controlled by `KAMINPAR_64BIT_*` CMake flags. The `default` preset uses 32-bit node/edge IDs and weights; `memory` uses 64-bit edge IDs and weights for graphs exceeding 2³² edges.

### Public API

- **C++**: `#include <kaminpar-shm/kaminpar.h>` → `KaMinPar` class; `#include <kaminpar-dist/dkaminpar.h>` → `dKaMinPar` class.
- **C**: `#include <kaminpar-shm/ckaminpar.h>` (C wrapper around the C++ API).
- **Python**: installable wheel via pip (`pykaminpar`).
