# Installation Notes

## Requirements

* Modern C++-20 ready compiler such as `g++` version 10 or higher
* CMake 
* Intel Thread Building Blocks library (TBB)

## Building KaMinPar

To build the software, clone this repository and type
```shell
./build.sh
```

Alternatively, you can use the standard CMake build process:

1. Clone the repository including submodules: `git clone --depth=1 --recursive git@github.com:KaHIP/KaMinPar.git`
2. Create a build: `mkdir build && cd build`
3. Run CMake: `cmake .. -DCMAKE_BUILD_TYPE=Release`
4. Build the software: `make -j`

The resulting binary is located in `build/apps/KaMinPar`.

## Running KaMinPar

To partition a graph in METIS format using `KaMinPar`, run

```shell
./KaMinPar -G <graph filename> -k <number of blocks> -t <number of threads> -e <max. imbalance, e.g. 0.03 for 3%> 
```

For a description of the graph format, please have a look at the [KaHiP manual](https://github.com/KaHIP/KaHIP/raw/master/manual/kahip.pdf).
A list of all options is available using the `--help` option.

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


