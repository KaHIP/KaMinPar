# KaMinPar

The graph partitioning software KaMinPar -- Karlsruhe Minimal Graph Partitioning. 

KaMinPar is a shared-memory parallel tool to heuristically solve the graph partitioning problem: divide a graph into k disjoint blocks of roughly equal weight while
minimizing the number of edges between blocks.
Competing algorithms are mostly evaluated for small values of k. If k is large, they often compute highly imbalance solutions, solutions of low quality or suffer excessive running time.
KaMinPar substantially mitigates these problems.
It computes partitions of comparable quality to other high-quality graph partitioning tools while guaranteeing the balance constraint for unweighted input graphs.
Moreover, for large values of k, it is an order of magnitude faster than competing algorithms.

## Installation Notes

### Requirements

* Modern C++-20 ready compiler such as `g++` version 10 or higher
* CMake 
* Intel Thread Building Blocks library (TBB)

To build the distributed graph partitioner, you also need:

* MPI 

### Building KaMinPar

To build the software, clone this repository and type
```shell
./build.sh [DISTRIBUTED]
```

Alternatively, you can use the standard CMake build process:

1. Clone the repository including submodules: `git clone --depth=1 --recursive git@github.com:KaHIP/KaMinPar.git`
2. Create a build: `mkdir build && cd build`
3. Run CMake: `cmake .. -DCMAKE_BUILD_TYPE=Release`
4. Build the software: `make KaMinPar -j`
5. (optional) Build the distributed software: `make dKaMinPar -j`

The resulting binary is located in `build/apps/`.

## Running KaMinPar

To partition a graph in METIS format using KaMinPar, run

```shell
./KaMinPar -G <graph filename> -k <number of blocks> 
```

Common command line options include:

* `--help` - print a complete list of command line options
* `--threads=<int>` - maximum number of threads to be used
* `--epsilon=<double>` - maximum allowed imbalance,e.g., `0.03` for 3%
* `--seed=<int>` - seed for random number generation
* `--save-partition` - write the partition to a file 
  (you also might want to specify the partition filename using `--partition-name=<str>`) 

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

KaMinPar is free software provided under the GNU General Public License (GPLv3).
If you publish results using our algorithms, please acknowledge our work by citing the following paper:

```
@article{KAMINPAR,
  title = {Deep Multilevel Graph Partitioning},
  author = {Lars Gottesbüren, Tobias Heuer, Peter Sanders, Christian Schulz, Daniel Seemaier},
  journal = {arXiv preprint arXiv:2105.02022},
  year = {2021}
}  
```


