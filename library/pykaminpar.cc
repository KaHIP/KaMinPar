/*******************************************************************************
 * @file:   pykaminpar.cc
 * @author: Daniel Seemaier
 * @date:   28.10.2022
 * @brief:  Python binding for the KaMinPar library.
 ******************************************************************************/
#include <iostream>
#include <string>

#include <pybind11/pybind11.h>

#include "library/kaminpar.h"

using namespace libkaminpar;

PYBIND11_MODULE(pykaminpar, m) {
  m.doc() = "KaMinPar: (Somewhat) Minimal Shared-Memory Graph Partitioning for "
            "Large k";
  m.def(
      "partition_file",
      [](const std::string &filename, const BlockID k,
         const std::string &preset = "default", const int num_threads = 0,
         const double epsilon = 0.03, const int seed = 0,
         const bool quiet = false) {
        auto partitioner =
            libkaminpar::PartitionerBuilder::from_graph_file(filename)
                .rearrange_and_create();
        partitioner.set_option("--epsilon", std::to_string(epsilon));
        partitioner.set_preset(preset);
        partitioner.set_seed(seed);
        partitioner.set_quiet(quiet);
        partitioner.set_num_threads(num_threads);

        const auto partition_size = partitioner.partition_size();
        auto partition = partitioner.partition(k);

        pybind11::list py_partition;
        for (std::size_t i = 0; i < partition_size; ++i) {
          py_partition.append(partition[i]);
        }

        return py_partition;
      },
      "A function that partitions a graph in a METIS file",
      pybind11::arg("filename"), pybind11::arg("k"),
      pybind11::arg("preset") = "default", pybind11::arg("num_threads") = 0,
      pybind11::arg("epsilon") = 0.03, pybind11::arg("seed") = 0,
      pybind11::arg("quiet") = true);
  m.def(
      "partition",
      [](const pybind11::object &xadj, const pybind11::object &adjncy,
         const pybind11::object &vwgt, const pybind11::object &adjwgt,
         const BlockID k, const std::string &preset = "default",
         const int num_threads = 0, const double epsilon = 0.03,
         const int seed = 0, const bool quiet = false) {
        std::vector<EdgeID> xadjv;
        for (auto it : xadj) {
          xadjv.push_back(pybind11::cast<EdgeID>(*it));
        }

        std::vector<NodeID> adjncyv;
        for (auto it : adjncy) {
          adjncyv.push_back(pybind11::cast<NodeID>(*it));
        }

        std::vector<NodeWeight> vwgtv;
        for (auto it : vwgt) {
          vwgtv.push_back(pybind11::cast<NodeWeight>(*it));
        }

        std::vector<EdgeWeight> adjwgtv;
        for (auto it : adjwgt) {
          adjwgtv.push_back(pybind11::cast<EdgeWeight>(*it));
        }

        auto builder = libkaminpar::PartitionerBuilder::from_adjacency_array(
            xadjv.size() - 1, xadjv.data(), adjncyv.data());
        if (!vwgtv.empty()) {
          builder.with_node_weights(vwgtv.data());
        }
        if (!adjwgtv.empty()) {
          builder.with_edge_weights(adjwgtv.data());
        }
        auto partitioner = builder.rearrange_and_create();

        partitioner.set_option("--epsilon", std::to_string(epsilon));
        partitioner.set_preset(preset);
        partitioner.set_seed(seed);
        partitioner.set_quiet(quiet);
        partitioner.set_num_threads(num_threads);

        const auto partition_size = partitioner.partition_size();
        auto partition = partitioner.partition(k);

        pybind11::list py_partition;
        for (std::size_t i = 0; i < partition_size; ++i) {
          py_partition.append(partition[i]);
        }

        return py_partition;
      },
      "A function that partitions a graph held in memory",
      pybind11::arg("xadj"), pybind11::arg("adjncy"), pybind11::arg("vwgt"),
      pybind11::arg("adjwgt"), pybind11::arg("k"),
      pybind11::arg("preset") = "default", pybind11::arg("num_threads") = 0,
      pybind11::arg("epsilon") = 0.03, pybind11::arg("seed") = 0,
      pybind11::arg("quiet") = true);
}
