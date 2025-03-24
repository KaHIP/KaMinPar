# KaMinPar

**KaMinPar** is a parallel heuristic solver for the balanced k-way graph partitioning problem.
Given a graph, it aims to divide its vertices into k disjoint blocks of approximately equal weight, minimizing the number of edges crossing between blocks.
KaMinPar offers high efficiency and low memory overheads while achieving partitions of similar quality as the widely used Metis algorithm.
For example, it can partition the massive [hyperlink-2012](https://webdatacommons.org/hyperlinkgraph/index.html) graph (approx. 3.5 billion vertices and 112 billion edges) into 30,000 blocks in under 6 minutes on 96 cores, using around 300 GiB RAM.
Notably, KaMinPar is also optimized for large k, where it often achieves an order-of-magnitude speedup over competing partitioners.
For unweighted input graphs, it guarantees strict adherence to the balance constraint.

## Installation Nodes (Python and NetworKit)

KaMinPar offers bindings for Python (only available for Linux and macOS), which can be installed via `pip`:

```shell
pip install kaminpar
```

To get started, refer to our (quick-start documentation)[bindings/python/README.md] and basic (examples)[examples/bindings-python].

Additionally, we offer bindings for (NetworKit)[https://networkit.github.io/], which can also be installed via `pip`:

```shell
pip install kaminpar-networkit
```

For instance, these allow you to generate, partition and visualize a graph in just a few lines of Python:

```python
import networkit as nk
import kaminpar_networkit as kaminpar
from networkit import vizbridges

graph = nk.generators.HyperbolicGenerator(100, k = 16, gamma = 2.7, T = 0).generate()
partition = kaminpar.KaMinPar(graph).computePartitionWithEpsilon(4, 0.03)

fig = nk.vizbridges.widgetFromGraph(graph, dimension = nk.vizbridges.Dimension.Three, nodePartition = partition, nodePalette = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)])
fig.write_image("partitioned_hyperbolic.png")
```

Refer to our (quick-start guide)[bindings/networkit/README.md) and basic (examples)[examples/bindings-networkit/) to get started.

## Installation Notes (C/C++)

### Requirements

* **Compiler:** C++20-ready GCC or Clang compiler
* **Dependencies:** CMake, oneAPI TBB, MPI (optional)
* **System:** Linux (x86, ARM) or macOS (ARM)

### Installation

After cloning the repository, follow the standard CMake build procedure:

```shell
cmake -B build --preset=default && cmake --build build --parallel
```

To partition a graph in [Metis](docs/graph_file_format.md) format, run:

```shell
# KaMinPar: shared-memory partitioning
./build/apps/KaMinPar [-P default|terapart|strong|largek] -G <graph filename> -k <number of blocks> -t <nproc> [-o <output partition>]

# dKaMinPar: distributed partitioning
mpirun -n <nproc> ./build/apps/dKaMinPar [-P default|strong|xterapart] -G <graph filename> -k <number of blocks> [-o <output partition>]
```

The computed partition is written to a text file (controlled via `-o <filename>`), where the n-th line contains the block ID (0-based) of the n-th node.

There are multiple configuration presets that tune the algorithm for different scenarios:

* `-P default`: fast partitioning with quality comparable to Metis.
* `-P terapart`: same partition quality as `default`, but with reduced memory consumption (slightly slower).
* `-P strong`: better quality than `default` through additional FM refinement at the cost of increased runtime.
* `-P largek`: faster for large values of k (e.g., k > 1024); reduces partition quality slightly for smaller k.

If all blocks should have roughly the same way, use the `-k` and `-e` options to parameterize the partition:

* `-k <k>`: Controls the number of blocks in the partition (if all blocks should be subject to the same maximum block weight controlled by `-e`)
* `-e <epsilon>`: Controls the maximum imbalance, e.g., `-e 0.03` to allow 3% imbalance, **defaults to 3%**

Alternatively, specify the maximum weight for each block explicitly:

* `-B <W0> <W2> ... <Wk-1>`: Explicitly specifies the maximum block weights, i.e., the weight of the i-th block should be bounded by `Wi`.
* `-b <w0> <w2> ... <wk-1>`: Same as `-B`, but specifies the maximum weights as fractions of the total node weight, i.e., the weight of the i-th block should be bounded by `wi * total node weight`.

Other command line options include:

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

Alternatively, you can use `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(KaMinPar
  GIT_REPOSITORY https://github.com/KaHIP/KaMinPar.git
  GIT_TAG main)
FetchContent_MakeAvailable(KaMinPar)
set_property(DIRECTORY "${KaMinPar_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL YES) # optional

target_link_libraries(<your-target> PUBLIC KaMinPar::KaMinPar)  # Shared-memory partitioning
target_link_libraries(<your-target> PUBLIC KaMinPar::dKaMinPar) # Distributed partitioning
```

The shared-memory partitioner can be used as follows:

```c++
#include <kaminpar-shm/kaminpar.h>
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

// Compute a `k`-way partition where each block weight is bounded by 
// `(1 + epsilon) * average`:
shm.compute_partition(BlockID k, double epsilon, std::span<BlockID> out_partition);

// Compute a `max_block_weights.size()`-way partition where the `i`-th block 
// weight is bounded by `max_block_weights[i]`:
shm.compute_partition(std::vector<BlockWeight> max_block_weights, std::span<BlockID> out_partition);

// Compute a `max_block_weight_factors.size()`-way partition where the `i`-th 
// block weight is bounded by `max_block_weight_factors[i] * total_weight`:
shm.compute_partition(std::vector<double> max_block_weight_factors, std::span<BlockID> out_partition);
```

The distributed-memory partitioner can be used as follows:

```c++
#include <kaminpar-dist/dkaminpar.h>
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

// Compute a `k`-way partition where each block weight is bounded by 
// `(1 + epsilon) * average`:
dist.compute_partition(BlockID k, double epsilon, std::span<BlockID> out_partition);

// Compute a `max_block_weights.size()`-way partition where the `i`-th block 
// weight is bounded by `max_block_weights[i]`:
dist.compute_partition(std::vector<BlockWeight> max_block_weights, std::span<BlockID> out_partition);

// Compute a `max_block_weight_factors.size()`-way partition where the `i`-th 
// block weight is bounded by `max_block_weight_factors[i] * total_weight`:
dist.compute_partition(std::vector<double> max_block_weight_factors, std::span<BlockID> out_partition);
```

More examples can be found in the (examples/)[examples/] directory. 

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
@misc{TeraPart,
      title={Tera-Scale Multilevel Graph Partitioning}, 
      author={Daniel Salwasser and Daniel Seemaier and Lars Gottesbüren and Peter Sanders},
      year={2024},
      eprint={2410.19119},
      archivePrefix={arXiv},
      primaryClass={cs.DS},
      url={https://arxiv.org/abs/2410.19119}, 
}
```

