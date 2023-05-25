/*******************************************************************************
 * @file:   shm_io.h
 * @author: Daniel Seemaier
 * @date:   20.04.2023
 * @brief:  Common KaGen-based IO code for benchmarks.
 ******************************************************************************/
#pragma once

#include <fstream>
#include <iostream>

#include <kagen.h>
#include <kaminpar/datastructures/graph.h>
#include <kaminpar/datastructures/partitioned_graph.h>
#include <kaminpar/kaminpar.h>
#include <kaminpar/metrics.h>

namespace kaminpar {
inline auto invoke_kagen(const std::string &options) {
  kagen::KaGen generator(MPI_COMM_WORLD);
  generator.UseCSRRepresentation();
  generator.EnableAdvancedStatistics();
  generator.EnableOutput(true);

  const bool generate = std::find(options.begin(), options.end(), ';') != options.end();

  if (generate) {
    return generator.GenerateFromOptionString(options);
  } else {
    return generator.ReadFromFile(
        options, kagen::FileFormat::EXTENSION, kagen::GraphDistribution::BALANCE_VERTICES
    );
  }
}

struct ShmGraphWrapper {
  std::vector<shm::EdgeID> xadj;
  std::vector<shm::NodeID> adjncy;
  std::vector<shm::NodeWeight> vwgt;
  std::vector<shm::EdgeWeight> adjvwgt;
  std::unique_ptr<shm::Graph> graph;
};

struct ShmPartitionedGraphWrapper : public ShmGraphWrapper {
  std::unique_ptr<shm::PartitionedGraph> p_graph;
};

inline ShmGraphWrapper load_shm_graph(const std::string &graph_name) {
  using namespace kaminpar::shm;

  kagen::Graph kagen_graph = invoke_kagen(graph_name);

  ShmGraphWrapper wrapper;
  wrapper.xadj = kagen_graph.TakeXadj<EdgeID>();
  wrapper.adjncy = kagen_graph.TakeAdjncy<NodeID>();
  wrapper.vwgt = kagen_graph.TakeVertexWeights<NodeWeight>();
  wrapper.adjvwgt = kagen_graph.TakeEdgeWeights<EdgeWeight>();
  wrapper.graph = std::make_unique<Graph>(
      StaticArray<EdgeID>(wrapper.xadj.data(), wrapper.xadj.size()),
      StaticArray<NodeID>(wrapper.adjncy.data(), wrapper.adjncy.size()),
      StaticArray<NodeWeight>(wrapper.vwgt.data(), wrapper.vwgt.size()),
      StaticArray<EdgeWeight>(wrapper.adjvwgt.data(), wrapper.adjvwgt.size())
  );

  std::cout << "Loaded graph with n=" << wrapper.graph->n() << ", m=" << wrapper.graph->m()
            << std::endl;
  return wrapper;
}

inline ShmPartitionedGraphWrapper
load_partitioned_shm_graph(const std::string &graph_name, const std::string &partition_name) {
  using namespace kaminpar::shm;

  ShmGraphWrapper graph_wrapper = load_shm_graph(graph_name);
  ShmPartitionedGraphWrapper wrapper;
  wrapper.xadj = std::move(graph_wrapper.xadj);
  wrapper.adjncy = std::move(graph_wrapper.adjncy);
  wrapper.vwgt = std::move(graph_wrapper.vwgt);
  wrapper.adjvwgt = std::move(graph_wrapper.adjvwgt);
  wrapper.graph = std::move(graph_wrapper.graph);

  const NodeID n = wrapper.graph->n();
  StaticArray<BlockID> partition(n);
  std::ifstream partition_file(partition_name);
  for (NodeID u = 0; u < n; ++u) {
    BlockID b;
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
} // namespace kaminpar
