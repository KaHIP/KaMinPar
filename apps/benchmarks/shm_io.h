/*******************************************************************************
 * @file:   shm_io.h
 * @author: Daniel Seemaier
 * @date:   20.04.2023
 * @brief:  Common KaGen-based IO code for benchmarks.
 ******************************************************************************/
#pragma once

#include <fstream>
#include <iostream>
#include <memory>

#include <kagen.h>
#include <kaminpar-shm/datastructures/graph.h>
#include <kaminpar-shm/datastructures/partitioned_graph.h>
#include <kaminpar-shm/kaminpar.h>
#include <kaminpar-shm/metrics.h>

namespace kaminpar {
inline auto invoke_kagen(const std::string &options) {
  kagen::KaGen generator(MPI_COMM_WORLD);
  generator.UseCSRRepresentation();
  generator.EnableOutput(false);

  const bool generate = std::find(options.begin(), options.end(), ';') != options.end();

  if (generate) {
    return generator.GenerateFromOptionString(options);
  } else {
    return generator.ReadFromFile(
        options, kagen::FileFormat::EXTENSION, kagen::GraphDistribution::BALANCE_VERTICES
    );
  }
}
} // namespace kaminpar

namespace kaminpar::shm {
struct GraphWrapper {
  std::vector<EdgeID> xadj;
  std::vector<NodeID> adjncy;
  std::vector<NodeWeight> vwgt;
  std::vector<EdgeWeight> adjvwgt;
  std::unique_ptr<Graph> graph;
};

struct PartitionedGraphWrapper : public GraphWrapper {
  std::unique_ptr<PartitionedGraph> p_graph;
};

inline GraphWrapper load_graph(const std::string &graph_name, const bool is_sorted = false) {
  kagen::Graph kagen_graph = invoke_kagen(graph_name);

  GraphWrapper wrapper;
  wrapper.xadj = kagen_graph.TakeXadj<EdgeID>();
  wrapper.adjncy = kagen_graph.TakeAdjncy<NodeID>();
  wrapper.vwgt = kagen_graph.TakeVertexWeights<NodeWeight>();
  wrapper.adjvwgt = kagen_graph.TakeEdgeWeights<EdgeWeight>();
  wrapper.graph = std::make_unique<Graph>(
      StaticArray<EdgeID>(wrapper.xadj.data(), wrapper.xadj.size()),
      StaticArray<NodeID>(wrapper.adjncy.data(), wrapper.adjncy.size()),
      StaticArray<NodeWeight>(wrapper.vwgt.data(), wrapper.vwgt.size()),
      StaticArray<EdgeWeight>(wrapper.adjvwgt.data(), wrapper.adjvwgt.size()),
      is_sorted
  );

  std::cout << "Loaded graph with n=" << wrapper.graph->n() << ", m=" << wrapper.graph->m()
            << std::endl;
  return wrapper;
}

inline PartitionedGraphWrapper load_partitioned_graph(
    const std::string &graph_name, const std::string &partition_name, const bool is_sorted = false
) {
  GraphWrapper graph_wrapper = load_graph(graph_name, is_sorted);
  PartitionedGraphWrapper wrapper;
  wrapper.xadj = std::move(graph_wrapper.xadj);
  wrapper.adjncy = std::move(graph_wrapper.adjncy);
  wrapper.vwgt = std::move(graph_wrapper.vwgt);
  wrapper.adjvwgt = std::move(graph_wrapper.adjvwgt);
  wrapper.graph = std::move(graph_wrapper.graph);

  const NodeID n = wrapper.graph->n();
  StaticArray<BlockID> partition(n);
  std::ifstream partition_file(partition_name);
  for (NodeID u = 0; u < n; ++u) {
    BlockID b = kInvalidBlockID;
    if (!(partition_file >> b)) {
      throw std::runtime_error("partition size does not match graph size");
    }
    partition[u] = b;
  }

  const BlockID k = *std::max_element(partition.begin(), partition.end()) + 1;
  wrapper.p_graph = std::make_unique<PartitionedGraph>(*wrapper.graph, k, std::move(partition));

  std::cout << "Loaded partitioned graph with cut=" << metrics::edge_cut(*wrapper.p_graph)
            << ", imbalance=" << metrics::imbalance(*wrapper.p_graph) << std::endl;

  return wrapper;
}
} // namespace kaminpar::shm
