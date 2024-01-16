# KaMinPar

KaMinPar is a shared-memory parallel tool to heuristically solve the graph partitioning problem: divide a graph into k disjoint blocks of roughly equal weight while
minimizing the number of edges between blocks.
Competing algorithms are mostly evaluated for small values of k. If k is large, they often compute highly imbalance solutions, solutions of low quality or suffer excessive running time.
KaMinPar substantially mitigates these problems.
It computes partitions of comparable quality to other high-quality graph partitioning tools while guaranteeing the balance constraint for unweighted input graphs.
Moreover, for large values of k, it is an order of magnitude faster than competing algorithms.

## Installation Notes

### Requirements

* C++17 compiler (GCC or Clang)
* CMake 
* Intel Thread Building Blocks library (TBB)
* MPI (optional)

### Building KaMinPar

Build KaMinPar following the standard CMake steps:

```shell
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKAMINPAR_BUILD_DISTRIBUTED=On
cmake --build build --parallel
```

## Using the Binaries

To partition a graph in METIS format using (d)KaMinPar, run

```shell
# KaMinPar: shared-memory partitioning
./build/apps/KaMinPar [-P fast|default|strong|largek] -G <graph filename> -k <number of blocks> -t <nproc> [--epsilon=0.03] [--seed=0]

# dKaMinPar: distributed partitioning
mpirun -n <nproc> ./build/apps/dKaMinPar [-P default|strong] -G <graph filename> -k <number of blocks> [--epsilon=0.03] [--seed=0]
```

Use the `--help` flag to see a list of all command line options.
To setup algorithmic tuning parameters, (d)KaMinPar offers configuration presets that can be loaded using the `-P <preset>` option (view `--help` for a list of all presets).
Presets can be viewed by using the `--dump-config` flag; to use a custom preset, load a configuration file using the `-C <filename>` option, e.g.,

```shell
# Write the default preset to a file
./KaMinPar [-P default|strong|largek] --dump-config > my_preset.ini

# ... modify the configuration by editing my_preset.ini ...

# Use your modified preset
./KaMinPar -C my_preset.ini -G <...> -k <...> -t <...>
```

For a description of the graph format, please refer to the [KaHiP manual](https://github.com/KaHIP/KaHIP/raw/master/manual/kahip.pdf).

## Using the Libraries

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
shm.compute_partition(BlockID number_of_blocks, BlockID *out_partition);

// Call the distributed partitioner:
dKaMinPar dist(MPI_Comm comm, int num_threads, dist::create_default_context());
// dKaMinPar::reseed(int seed); 
dist.import_graph(GlobalNodeID *vtxdist, GlobalEdgeID *xadj, GlobalNodeID *adjncy, GlobalNodeWeight *vwvgt = nullptr, GlobalEdgeWeight *adjwgt = nullptr);
dist.compute_partition(BlockID number_of_blocks, BlockID *out_partition);
```

Please take a look at `apps/KaMinPar.cc` and `apps/dKaMinPar.cc` for a full example on how to call the libraries.

## Licensing

KaMinPar is free software provided under the MIT License.

If you publish results using our shared-memory partitioner, please acknowledge our work by citing the following [paper](https://doi.org/10.4230/LIPIcs.ESA.2021.48) ([preprint](https://arxiv.org/abs/2105.02022), [resources](https://algo2.iti.kit.edu/seemaier/deep_mgp/)):

```
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
```

If you publish results using out distributed-memory partitioner, please cite the following [paper](https://link.springer.com/chapter/10.1007/978-3-031-39698-4_30) ([preprint](https://arxiv.org/abs/2303.01417), [resources](https://algo2.iti.kit.edu/seemaier/ddeep_mgp/)):

```
@InProceedings{DistributedDeepMultilevelGraphPartitioning,
  author    = {Sanders, Peter and Seemaier, Daniel},
  title     = {Distributed Deep Multilevel Graph Partitioning},
  booktitle = {Euro-Par 2023: Parallel Processing},
  year      = {2023},
  publisher = {Springer Nature Switzerland},
  pages     = {443--457},
  isbn      = {978-3-031-39698-4}
}
```

