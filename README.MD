# KaMinPar

KaMinPar is a shared-memory parallel tool to heuristically solve the graph partitioning problem: divide a graph into k disjoint blocks of roughly equal weight while
minimizing the number of edges between blocks.
Competing algorithms are mostly evaluated for small values of k. If k is large, they often compute highly imbalance solutions, solutions of low quality or suffer excessive running time.
KaMinPar substantially mitigates these problems.
It computes partitions of comparable quality to other high-quality graph partitioning tools while guaranteeing the balance constraint for unweighted input graphs.
Moreover, for large values of k, it is an order of magnitude faster than competing algorithms.

## Installation Notes

### Requirements

* **Compiler:** C++20-ready GCC or Clang compiler
* **Dependencies:** CMake, Intel TBB, MPI (optional)
* **System:** Linux (x86, ARM) or macOS (ARM)

### Quickstart

After cloning the repository, make sure to initialize the submodules:

```shell
git submodule update --init --recursive
```

Then, follow the standard CMake build procedure:

```shell
cmake -B build --preset=default
cmake --build build --parallel
```

To partition a graph in METIS format (see, e.g., the [KaHIP manual](https://github.com/KaHIP/KaHIP/raw/master/manual/kahip.pdf)), run:

```shell
# KaMinPar: shared-memory partitioning
./build/apps/KaMinPar [-P default|terapart|strong|largek] -G <graph filename> -k <number of blocks> -t <nproc> [-o <output partition>]

# dKaMinPar: distributed partitioning
mpirun -n <nproc> ./build/apps/dKaMinPar [-P default|strong|xterapart] -G <graph filename> -k <number of blocks> [-o <output partition>]
```

The computed partition is written to a text file, where the n-th line contains the block ID (0-based) of the n-th vertex.

There are multiple configuration presets that tune the algorithm for different scenarios:

* `-P default`: fast partitioning with quality comparable to Metis
* `-P terapart`: same partition quality as `default`, but with reduced memory consumption (slightly slower)
* `-P strong`: better quality than `default` at the cost of increased runtime
* `-P largek`: faster for large values of k (e.g., k > 1024); may reduce partition quality for smaller k

Configuration presets can be inspected using the `--dump-config` flag.
To build a custom configuration, dump one of the presets to a file, modify it and load it using `-C <filename>`:

```shell
./build/KaMinPar -P terapart --dump-config > custom.ini
# ... modify custom.ini ...
./build/KaMinPar -C custom.ini <...>
```

## Using the Library Interface

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

Then, call the libraries as follows:

```c++
#include <kaminpar-shm/kaminpar.h>
#include <kaminpar-dist/dkaminpar.h>

using namespace kaminpar;

// Call the shared-memory partitioner:
KaMinPar shm(int num_threads, shm::create_default_context());
// KaMinPar::reseed(int seed);
shm.borrow_and_mutate_graph(NodeID n, EdgeID *xadj, NodeID *adjncy, NodeWeight *vwgt = nullptr, EdgeWeight *adjwgt = nullptr);
// alternatively: shm.copy_graph(n, xadj, adjncy, vwgt, adjwgt); will work on a copy of the graph
shm.compute_partition(BlockID number_of_blocks, double epsilon, std::span<BlockID> out_partition);
// alternatively: shm.compute_partition(std::vector<BlockWeight> max_block_weights, std::span<BlockID> out_partition);
// Note: you must ensure that the total max block weight is larger than the total node weight of the graph

// Call the distributed partitioner:
dKaMinPar dist(MPI_Comm comm, int num_threads, dist::create_default_context());
// dKaMinPar::reseed(int seed); 
dist.copy_graph(GlobalNodeID *vtxdist, GlobalEdgeID *xadj, GlobalNodeID *adjncy, GlobalNodeWeight *vwvgt = nullptr, GlobalEdgeWeight *adjwgt = nullptr);
dist.compute_partition(BlockID number_of_blocks, double epsilon, std::span<BlockID> out_partition);
// alternatively: dist.compute_partition(std::vector<BlockWeight> max_block_weights, std::span<BlockID> out_partition);
// Note: you must ensure that the total max block weight is larger than the total node weight of the graph
```

More examples can be found in the `examples/` directory. 

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

