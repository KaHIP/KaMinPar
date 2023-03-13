# KaMinPar

KaMinPar is a shared-memory parallel tool to heuristically solve the graph partitioning problem: divide a graph into k disjoint blocks of roughly equal weight while
minimizing the number of edges between blocks.
Competing algorithms are mostly evaluated for small values of k. If k is large, they often compute highly imbalance solutions, solutions of low quality or suffer excessive running time.
KaMinPar substantially mitigates these problems.
It computes partitions of comparable quality to other high-quality graph partitioning tools while guaranteeing the balance constraint for unweighted input graphs.
Moreover, for large values of k, it is an order of magnitude faster than competing algorithms.

## Installation Notes

### Requirements

* C++17 compiler (GCC, Clang)
* CMake 
* Intel Thread Building Blocks library (TBB)
* MPI (optional)

### Building KaMinPar

Build KaMinPar following the standard CMake steps:

```shell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Running KaMinPar

To partition a graph in METIS format using (d)KaMinPar, run

```shell
# KaMinPar: shared-memory partitioning
./build/apps/KaMinPar [-P default|largek] -G <graph filename> -k <number of blocks> -t <nproc> [--epsilon=0.03] [--seed=0]

# dKaMinPar: distributed partitioning
mpirun -n <nproc> ./build/apps/dKaMinPar [-P default|strong] -G <graph filename> -k <number of blocks> [--epsilon=0.03] [--seed=0]
```

Use the `--help` flag to see a list of all command line options.
To setup algorithmic tuning parameters, (d)KaMinPar offers configuration presets that can be loaded using the `-P <preset>` option (view `--help` for a list of all presets).
Presets can be viewed by using the `--dump-config` flag; to use a custom preset, load a configuration file using the `-C <filename>` option, e.g.,

```shell
# Write the default preset to a file
./KaMinPar [-P default] --dump-config > my_preset.ini

# ... modify the configuration by editing my_preset.ini ...

# Use your modified preset
./KaMinPar -C my_preset.ini -G <...> -k <...> -t <...>
```

For a description of the graph format, please have a look at the [KaHiP manual](https://github.com/KaHIP/KaHIP/raw/master/manual/kahip.pdf).

### Using KaMinPar as a Library

To use KaMinPar as a library, build the `kaminpar` target and follow the example given below.

```c++
// graph from the manual 
std::vector<libkaminpar::EdgeID> nodes{0, 2, 5, 7, 9, 12};
std::vector<libkaminpar::NodeID> edges{1, 4, 0, 2, 4, 1, 3, 2, 4, 0, 1, 3};

libkaminpar::Partitioner partitioner = 
        libkaminpar::PartitionerBuilder
        ::from_adjacency_array(nodes.data(), edges.data())
        .create();

partitioner.set_option("--threads", "6"); // use 6 cores
partitioner.set_option("--epsilon", "0.04"); // allow 4% imbalance

// compute 16-way partition
std::unique_ptr<libkaminpar::BlockID[]> partition = partitioner.partition(16); 
```

If you are already using Metis and want to give KaMinPar a try, you can build an interface compatible to that of Metis 
by passing `-DKAMINPAR_BUILD_METIS_INTERFACE=On` to CMake. 
Then, simply link against our library (together with `libtbb` and `libtbbmalloc`) instead of Metis.

## Licensing

KaMinPar is free software provided under the MIT License.
If you publish results using our shared-memory partitioner, please acknowledge our work by citing the following paper ([resources](https://algo2.iti.kit.edu/seemaier/deep_mgp/)):

```
@inproceedings{DeepMultilevel,
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

If you publish results using out distributed-memory partitioner, please cite the following technical report ([resources](https://algo2.iti.kit.edu/seemaier/ddeep_mgp/)):

```
@misc{DistributedDeepMultilevel,
  doi = {10.48550/ARXIV.2303.01417},
  url = {https://arxiv.org/abs/2303.01417},
  author = {Sanders, Peter and Seemaier, Daniel},
  title = {Distributed Deep Multilevel Graph Partitioning},
  publisher = {arXiv},
  year = {2023}
}
```

