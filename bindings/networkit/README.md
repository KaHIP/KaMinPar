# NetworKit Bindings for KaMinPar

[KaMinPar](https://github.com/KaHIP/KaMinPar) is a shared-memory parallel tool to heuristically solve the graph partitioning problem: divide a graph into k disjoint blocks of roughly equal weight while minimizing the number of edges between blocks. This Python package provides bindings to KaMinPar and integrates with the [NetworKit](https://networkit.github.io/) library to offer a fast and scalable graph partitioning algorithm.

## Installation

You can install the NetworKit bindings for KaMinPar via pip:

```sh
pip install kaminpar_networkit
```

Alternatively, you can build the bindings from source:

```sh
git clone https://github.com/KaHiP/KaMinPar.git
pip install KaMinPar/bindings/networkit
```

## Usage

The following is a basic example of how to use the KaMinPar bindings.

```python
import networkit
import kaminpar_networkit as kaminpar


graph = networkit.readGraph("graph.metis", networkit.Format.METIS)
networkit.overview(graph)

shm = kaminpar.KaMinPar(graph)
partition = shm.computePartitionWithFactors([0.53, 0.253, 0.253])

edge_cut = networkit.community.EdgeCut().getQuality(partition, graph)
print("Computed a partition with an edge cut of ", edge_cut)
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
