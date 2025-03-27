# NetworKit Bindings for KaMinPar

[KaMinPar](https://github.com/KaHIP/KaMinPar) is a shared-memory parallel tool to heuristically solve the graph partitioning problem: divide a graph into k disjoint blocks of roughly equal weight while minimizing the number of edges between blocks. This Python package provides bindings to KaMinPar and integrates with the [NetworKit](https://networkit.github.io/) library to offer a fast and scalable graph partitioning algorithm.

## Installation

You can install the NetworKit bindings for KaMinPar via pip:

```sh
pip install kaminpar-networkit
```

Alternatively, you can build the bindings from source:

```sh
git clone https://github.com/KaHiP/KaMinPar.git
pip install KaMinPar/bindings/networkit
```

When building the NetworKit bindings from source, the dependencies of KaMinPar must be available on the system. These include CMake, Intel TBB and Sparsehash. Additionally, if you install the package via pip on a target system and Python version where no pre-built wheel is available, the package will fall back to building from the source distribution. In this case, you must also ensure that the required dependencies are installed on your system.

## Usage

The following is a basic example of how to use the KaMinPar bindings. For the full API documentation, refer to the [documentation file](https://github.com/KaHIP/KaMinPar/blob/main/bindings/networkit/src/kaminpar_networkit/__init__.pyi).

```python
import kaminpar_networkit as kaminpar
import networkit

# Use NetworKit to read a graph from disk stored in METIS format and output statistics for the graph.
graph = networkit.readGraph("hyperlink.metis", networkit.Format.METIS)
networkit.overview(graph)

# Create a KaMinPar instance and compute a partition of the graph into four blocks using imbalance factor 3%.
shm = kaminpar.KaMinPar(graph)
partition = shm.computePartitionWithEpsilon(4, 0.03)

edge_cut = networkit.community.EdgeCut().getQuality(partition, graph)
print("Computed a partition with an edge cut of", edge_cut)
```

## License

The NetworKit bindings for KaMinPar and KaMinPar are free software provided under the MIT license. If you use KaMinPar in an academic setting, please cite the appropriate publication(s) listed below.

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
