# KaMinPar

KaMinPar is a shared-memory parallel tool to heuristically solve the graph partitioning problem: divide a graph into k disjoint blocks of roughly equal weight while
minimizing the number of edges between blocks.
Competing algorithms are mostly evaluated for small values of k. If k is large, they often compute highly imbalance solutions, solutions of low quality or suffer excessive running time.
KaMinPar substantially mitigates these problems.
It computes partitions of comparable quality to other high-quality graph partitioning tools while guaranteeing the balance constraint for unweighted input graphs.
Moreover, for large values of k, it is an order of magnitude faster than competing algorithms.

## Installation Notes

### Requirements

* Modern C++-20 ready compiler such as `g++` version 10 or higher
  * A C++17 port requiring `g++` version 7.2.0 or higher is available in branch `c++17`
* CMake 
* Intel Thread Building Blocks library (TBB)

### Building KaMinPar

To build the software, clone this repository and type
```shell
./build.sh
```

Alternatively, you can use the standard CMake build process:

1. Clone the repository including submodules: `git clone --depth=1 --recursive git@github.com:KaHIP/KaMinPar.git`
2. Create a build directory: `mkdir build && cd build`
3. Run CMake: `cmake .. -DCMAKE_BUILD_TYPE=Release`
4. Build the software: `make -j`

The resulting binary is located in `build/apps/KaMinPar`.

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

License: MIT

If you publish results using our algorithms, please acknowledge our work by citing the following paper ([pdf](http://algo2.iti.kit.edu/seemaier/deep_mgp/)):

```
@inproceedings{DBLP:conf/esa/GottesburenH00S21,
  author    = {Lars Gottesb{\"{u}}ren and
               Tobias Heuer and
               Peter Sanders and
               Christian Schulz and
               Daniel Seemaier},
  title     = {Deep Multilevel Graph Partitioning},
  booktitle = {29th Annual European Symposium on Algorithms, {ESA} 2021, September
               6-8, 2021, Lisbon, Portugal (Virtual Conference)},
  series    = {LIPIcs},
  volume    = {204},
  pages     = {48:1--48:17},
  publisher = {Schloss Dagstuhl - Leibniz-Zentrum f{\"{u}}r Informatik},
  year      = {2021},
  url       = {https://doi.org/10.4230/LIPIcs.ESA.2021.48},
  doi       = {10.4230/LIPIcs.ESA.2021.48}
}
```

Detailed experimental results published in our paper are available [here](http://algo2.iti.kit.edu/seemaier/deep_mgp/).
