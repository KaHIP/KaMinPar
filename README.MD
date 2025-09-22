<h1 align="center">KaMinPar</h1>

<h3 align="center">A Parallel Heuristic Solver for the Balanced k-Way Graph Partitioning Problem</h3>

<p align="center">
  <a href="#quick-start-python-and-networkit">Python Quick Start</a> •
  <a href="#installing-pre-built-binaries">Installing Pre-built Binaries</a> •
  <a href="#building-from-source">Building from Source</a> •
  <a href="#using-the-command-line-binaries">Using the CLI Application</a> •
  <a href="#using-the-c-library-interface">Using the C/C++ Library</a> •
  <a href="#licensing">Licensing</a>
</p>

Given a graph, KaMinPar aims to divide its vertices into k disjoint blocks of approximately equal weight, minimizing the number of edges crossing between blocks.
KaMinPar offers high efficiency and low memory overheads while achieving partitions of similar quality as the widely used Metis algorithm.
For example, it can partition the massive [hyperlink-2012](https://webdatacommons.org/hyperlinkgraph/index.html) graph (approx. 3.5 billion vertices and 112 billion edges) into 30,000 blocks in under 6 minutes on 96 cores, using around 300 GiB RAM.
Notably, KaMinPar is also optimized for large k, where it often achieves an order-of-magnitude speedup over competing partitioners.
For unweighted input graphs, it guarantees strict adherence to the balance constraint.

## Quick-Start (Python and NetworKit)

KaMinPar offers bindings for Python (only available on Linux and macOS), which can be installed via `pip`:

```shell
pip install kaminpar
```

Check out our [documentation](bindings/python/README.md) and [examples](examples/bindings-python/) to get started.
Additionally, we offer bindings for [NetworKit](https://networkit.github.io/), which can also be installed via `pip`:

```shell
pip install kaminpar-networkit
```

For instance, these allow you to generate, partition and visualize a graph in just a few lines of code:

```python
import networkit as nk
import kaminpar_networkit as kaminpar
from networkit import vizbridges

# Generate a random hyperbolic graph with 100 vertices, average degree 16 and power-law exponent 2.7
graph = nk.generators.HyperbolicGenerator(100, k = 16, gamma = 2.7, T = 0).generate()

# Partition the graph into 4 blocks with a maximum imbalance of 3%
partition = kaminpar.KaMinPar(graph).computePartitionWithEpsilon(4, 0.03)

# Draw the graph and the partition
fig = nk.vizbridges.widgetFromGraph(
    graph, 
    dimension = nk.vizbridges.Dimension.Three, 
    nodePartition = partition, 
    nodePalette = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
)
fig.write_image("partitioned_hyperbolic.png")
```

Refer to our [documentation](bindings/networkit/README.md) and [examples](examples/bindings-networkit/) to get started.

## Installing Pre-built Binaries

### Debian / Ubuntu

KaMinPar is not available in the official Debian/Ubuntu repositories, but you can install it manually using the prebuilt `.deb` packages provided on the [GitHub Releases page](https://github.com/KaHIP/KaMinPar/releases/latest), as follows:

```shell
# For AMD64
wget https://github.com/KaHIP/KaMinPar/releases/download/v3.6.0/kaminpar_3.6.0-1_amd64.deb
sudo apt install ./kaminpar_3.6.0-1_amd64.deb

# For ARM64
wget https://github.com/KaHIP/KaMinPar/releases/download/v3.6.0/kaminpar_3.6.0-1_arm64.deb
sudo apt install ./kaminpar_3.6.0-1_arm64.deb
```

> [!TIP]
> Avoid using `dpkg -i` directly, as it doesn’t resolve dependencies. APT handles them automatically.

### RHEL (AlmaLinux / CentOS / RockyLinux)

An `.rpm` package is also available on the [GitHub Releases page](https://github.com/KaHIP/KaMinPar/releases/latest) for RHEL-compatible distributions:

```shell
# For AMD64
sudo dnf install https://github.com/KaHIP/KaMinPar/releases/download/v3.6.0/kaminpar-3.6.0-1.x86_64.rpm

# For ARM64
sudo dnf install https://github.com/KaHIP/KaMinPar/releases/download/v3.6.0/kaminpar-3.6.0-1.aarch64.rpm
```

### ArchLinux (AUR)

You can install the `kaminpar` package via the [Arch User Repository](https://aur.archlinux.org/packages/kaminpar). You can install it manually:

```
git clone https://aur.archlinux.org/kaminpar.git
cd kaminpar
makepkg -si
```

Or install it using your preferred [AUR helper](https://wiki.archlinux.org/title/AUR_helpers), for example:

```
paru -S kaminpar
```

### NixOS

The package `kaminpar` is available via the Nix [unstable](https://search.nixos.org/packages?channel=unstable&show=kaminpar) channel. If you're using [Nix flakes](https://nixos.wiki/wiki/flakes), you can install or run KaMinPar directly from its GitHub repository:

```shell
# Add KaMinPar to your user profile
nix profile install github:KaHIP/KaMinPar

# Run KaMinPar / dKaMinPar directly
nix run github:KaHIP/KaMinPar -- <graph filename> <number of blocks> ...
nix run github:KaHIP/KaMinPar#dKaMinPar -- <graph filename> <number of blocks> ...

# Enter a development shell with Python and NetworKit bindings
nix develop github:KaHIP/KaMinPar#python
```

KaMinPar is also available as a flake in the project root. To use KaMinPar as an input in your own flake-based project, add the following to your `flake.nix`:

```nix
inputs = {
  # Define your inputs ...

  kaminpar = {
    url = "github:KaHIP/KaMinPar";
  };
};

outputs = { self, nixpkgs, kaminpar, ... }@inputs: {
  # Define your outputs...

  # You can access the KaMinPar packages like:
  # kaminpar.packages.${system}.kaminpar
  # kaminpar.packages.${system}.kaminpar-python
  # kaminpar.packages.${system}.kaminpar-networkit
};
```

## Building from Source

### Requirements

* **Compiler:** C++20-ready GCC or Clang compiler
* **Dependencies:** CMake, oneAPI TBB, Google Sparsehash (optional), MPI (optional)
* **System:** Linux (x86, ARM) or macOS (ARM)

### Instructions

After cloning the repository, follow the standard CMake build procedure:

```shell
cmake -B build --preset=default && cmake --build build --parallel
```

> [!IMPORTANT]
> Per default, node and edge IDs and weights use 32 bit data types. 
> For larger graphs, use some of the following CMake flags:
> 
> * `-DKAMINPAR_64BIT_EDGE_IDS=On`: required for partitioning graphs with more than 2^31 undirected edges.
> * `-DKAMINPAR_64BIT_NODE_IDS=On`: required for partitioning graphs with more than 2^32 nodes.
> * `-DKAMINPAR_64BIT_WEIGHTS=On`: required for graphs with more than 2^31 undirected edges, or for graphs with non-unit edge weights whose sum cannot be represented in 32 bits.

To build the distributed components, use `--preset=distributed` instead of the default preset.

## Using the Command-Line Binaries

To partition a graph in [Metis](docs/graph_file_format.md) format, run:

```shell
# KaMinPar: shared-memory partitioning
./build/apps/KaMinPar <graph filename> <number of blocks> [-P default|terapart|strong|largek] [-t <nproc>] [-e <eps, e.g., 0.03 for 3%>] [-o <output partition>]

# dKaMinPar: distributed partitioning
mpirun -n <nproc> ./build/apps/dKaMinPar <graph filename> <number of blocks> [-P default|strong|xterapart] [-e <eps, e.g., 0.03 for 3%>] [-o <output partition>]
```

The computed partition is written to a text file (controlled via `-o <filename>`), where the n-th line contains the block ID (0-based) of the n-th node.

There are multiple configuration presets that tune the algorithm for different scenarios:

* `-P default`: fast partitioning with quality comparable to Metis.
* `-P terapart`: same partition quality as `default`, but with reduced memory consumption (slightly slower).
* `-P strong`: better quality than `default` through additional FM refinement at the cost of increased runtime.
* `-P largek`: faster for large values of k (e.g., k > 1024); reduces partition quality slightly for smaller k.

The `-k <k>` option directs (d)KaMinPar to partition the graph into k blocks of roughly equal weight (use `-e <eps>` to control the maximum allowed imbalance, e.g., `-e 0.03` for 3%).
Alternatively, one can specify the maximum weight for each block explicitly through one of the following options:

* `-B <W0> ... <Wk-1>`: Explicitly specifies the maximum block weights, i.e., the weight of the i-th block should be bounded by `Wi`.
* `-b <w0> ... <wk-1>`: Same as `-B`, but specifies the maximum weights as fractions of the total node weight, i.e., the weight of the i-th block should be bounded by `wi * total node weight`.

> [!TIP]
> KaMinPar recently added support for balanced minimum block weights. 
> These can be configured analogously to the maximum block weights via the following options:
>
> * `--min-epsilon=<e.g., 0.03 for 3%>`: mirrors `-e`
> * `--min-block-weights <W0> ... <Wk-1>`: mirrors `-B`
> * `--min-block-weight-factors <w0> ... <wk-1>`: mirrors `-b`
>
> Use `--no-empty-blocks` if you need all blocks to be non-empty.

Other common command line options include:

* `--help`: Prints all command line options.
* `--version`: Prints the build configuration and current version.
* `--validate`: Enables basic input graph validation.
* `-q`: Quiet mode, suppresses all console output.

## Using the C++ Library Interface

If you are using CMake, you can use the partitioners as libraries by adding this repository as a Git submodule to your project and including it in your CMake configuration:

```cmake
add_subdirectory(external/KaMinPar)

target_link_libraries(<your-target> PUBLIC KaMinPar::KaMinPar)  # Shared-memory partitioning
target_link_libraries(<your-target> PUBLIC KaMinPar::dKaMinPar) # Distributed partitioning
```

Or, if KaMinPar is installed system-wide (see section on [Installing Pre-built Binaries](#installing-pre-built-binaries)), you can simply use:

```cmake
find_package(KaMinPar REQUIRED)

target_link_libraries(<your-target> PUBLIC KaMinPar::KaMinPar)  # Shared-memory partitioning
target_link_libraries(<your-target> PUBLIC KaMinPar::dKaMinPar) # Distributed partitioning
```

Alternatively, you can use `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(KaMinPar
  GIT_REPOSITORY https://github.com/KaHIP/KaMinPar.git
  GIT_TAG main
  EXCLUDE_FROM_ALL)
set(KAMINPAR_BUILD_DISTRIBUTED ON CACHE BOOL "" FORCE) # optional, required for dKaMinPar
FetchContent_MakeAvailable(KaMinPar)

target_link_libraries(<your-target> PUBLIC KaMinPar::KaMinPar)  # Shared-memory partitioning
target_link_libraries(<your-target> PUBLIC KaMinPar::dKaMinPar) # Distributed partitioning
```

### Shared-Memory API

The shared-memory partitioner can be used as follows:

> [!CAUTION]
> Only one thread can call `KaMinPar::compute_partition()` at a time (even on different `KaMinPar` objects).
> This is because the performance timers are implemented as global singletons, which are inherently not thread-safe.
> To compute multiple partitions in parallel, you must disable the timers by passing `-DKAMINPAR_ENABLE_TIMERS=Off` to CMake.

```c++
#include <kaminpar.h>
using namespace kaminpar;

KaMinPar shm(int num_threads, shm::create_default_context());

// Pass a copy of the graph:
shm.copy_graph(
  std::span<const EdgeID> xadj, 
  std::span<const NodeID> adjncy, 
  std::span<const NodeWeight> vwgt = {}, 
  std::span<const EdgeWeight> adjwgt = {}
);

// Alternatively, let KaMinPar borrow the graph: this avoids the copy, but the
// spans must stay valid throughout partitioning and KaMinPar might modify the 
// data:
shm.borrow_and_mutate_graph(
  std::span<EdgeID> xadj, 
  std::span<NodeID> adjncy, 
  std::span<NodeWeight> vwgt = {}, 
  std::span<EdgeWeight> adjwgt = {}
);

// Set the number of blocks:
shm.set_k(BlockID k);

// Configure max block weights:
shm.set_uniform_max_block_weights(double epsilon); // -e <...>
// shm.set_absolute_max_block_weights(std::span<const BlockWeight> max_block_weights); // -B <...>
// shm.set_relative_max_block_weights(std::span<const double> max_block_weight_factors); // -b <...>

// Optionally, configure min block weights:
// shm.set_uniform_min_block_weights(double epsilon); // --min-epsilon <...>
// shm.set_absolute_min_block_weights(std::span<const BlockWeight> min_block_weights); // --min-block-weights <...>
// shm.set_relative_min_block_weights(std::span<const double> min_block_weight_factors); // --min-block-weight-factors <...>

// Compute the partition:
EdgeWeight cut = shm.compute_partition(std::span<BlockID> out_partition);
```

> [!TIP]
> If you want to partition a graph that does not fit into memory, you can use our [Builder interface](docs/graph_compression.md) to construct a graph while compressing it on-the-fly.

Check out our [examples](examples/) to see the library interface in action.

### Distributed-Memory API

The distributed-memory partitioner can be used as follows:

```c++
#include <dkaminpar.h>
using namespace kaminpar;

dKaMinPar dist(MPI_Comm comm, int num_threads, dist::create_default_context());

// Pass a copy of the graph:
dist.copy_graph(
  std::span<GlobalNodeID> vtxdist, 
  std::span<GlobalEdgeID> xadj, 
  std::span<GlobalNodeID> adjncy, 
  std::span<GlobalNodeWeight> vwvgt = {}, 
  std::span<GlobalEdgeWeight> adjwgt = {}
);

// Set the number of blocks:
dist.set_k(BlockID k);

// Configure max block weights:
dist.set_uniform_max_block_weights(double epsilon); // -e <...>
// dist.set_absolute_max_block_weights(std::span<const BlockWeight> max_block_weights); // -B <...>
// dist.set_relative_max_block_weights(std::span<const double> max_block_weight_factors); // -b <...>

// Compute the partition:
EdgeWeight cut = dist.compute_partition(std::span<BlockID> out_partition);
```

> [!NOTE]
> The `num_threads` parameter controls the number of threads *per MPI process*. 
> While the partitioner is mostly optimized for using one thread per MPI process (i.e., setting the parameter to `1`), applications that already use a hybrid MPI+X parallelization scheme might benefit from using multiple threads per MPI process.

## Licensing

KaMinPar is free software provided under the MIT license.
If you use KaMinPar in an academic setting, please cite the appropriate publication(s) listed below.

```
// KaMinPar
@InProceedings{DeepMultilevelGraphPartitioning,
  author    = {Lars Gottesb{\"{u}}ren and
               Tobias Heuer and
               Peter Sanders and
               Christian Schulz and
               Daniel Seemaier},
  title     = {Deep Multilevel Graph Partitioning},
  booktitle = {29th Annual European Symposium on Algorithms, {ESA} 2021},
  series    = {LIPIcs},
  volume    = {204},
  pages     = {48:1--48:17},
  publisher = {Schloss Dagstuhl - Leibniz-Zentrum f{\"{u}}r Informatik},
  year      = {2021},
  url       = {https://doi.org/10.4230/LIPIcs.ESA.2021.48},
  doi       = {10.4230/LIPIcs.ESA.2021.48}
}

// dKaMinPar (distributed KaMinPar)
@InProceedings{DistributedDeepMultilevelGraphPartitioning,
  author    = {Sanders, Peter and Seemaier, Daniel},
  title     = {Distributed Deep Multilevel Graph Partitioning},
  booktitle = {Euro-Par 2023: Parallel Processing},
  year      = {2023},
  publisher = {Springer Nature Switzerland},
  pages     = {443--457},
  isbn      = {978-3-031-39698-4}
}

// [x]TeraPart (memory-efficient [d]KaMinPar)
@InProceedings{TeraPart,
  author       = {Daniel Salwasser and
                  Daniel Seemaier and
                  Lars Gottesb{\"{u}}ren and
                  Peter Sanders},
  title        = {Tera-Scale Multilevel Graph Partitioning},
  booktitle    = {{IEEE} International Parallel and Distributed Processing Symposium,
                  {IPDPS} 2025, Milano, Italy, June 3-7, 2025},
  pages        = {285--296},
  publisher    = {{IEEE}},
  year         = {2025},
  url          = {https://doi.org/10.1109/IPDPS64566.2025.00033},
  doi          = {10.1109/IPDPS64566.2025.00033},
  timestamp    = {Mon, 28 Jul 2025 15:42:51 +0200},
  biburl       = {https://dblp.org/rec/conf/ipps/SalwasserSG025.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

// Worst-Case Linear-Time Multilevel Graph Partitioning
@misc{LinearTimeMGP,
      title={Linear-Time Multilevel Graph Partitioning via Edge Sparsification},
      author={Lars Gottesbüren and Nikolai Maas and Dominik Rosch and Peter Sanders and Daniel Seemaier},
      year={2025},
      eprint={2504.17615},
      archivePrefix={arXiv},
      primaryClass={cs.DS},
      url={https://arxiv.org/abs/2504.17615},
}
```

