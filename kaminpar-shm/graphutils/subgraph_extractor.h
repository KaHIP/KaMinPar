/*******************************************************************************
 * @file:   subgraph_extraction.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Extracts the subgraphs induced by each block of a partition.
 ******************************************************************************/
#pragma once

#include <array>
#include <utility>
#include <vector>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/definitions.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::graph {
struct SubgraphMemoryStartPosition {
  std::size_t nodes_start_pos{0};
  std::size_t edges_start_pos{0};

  // operator overloads for parallel::prefix_sum()
  SubgraphMemoryStartPosition operator+(const SubgraphMemoryStartPosition &other) const {
    return {nodes_start_pos + other.nodes_start_pos, edges_start_pos + other.edges_start_pos};
  }

  SubgraphMemoryStartPosition &operator+=(const SubgraphMemoryStartPosition &other) {
    nodes_start_pos += other.nodes_start_pos;
    edges_start_pos += other.edges_start_pos;
    return *this;
  }
};

struct SubgraphMemory {
  SubgraphMemory() = default;

  SubgraphMemory(
      const NodeID n,
      const BlockID k,
      const EdgeID m,
      const bool is_node_weighted = true,
      const bool is_edge_weighted = true
  ) {
    resize(n, k, m, is_node_weighted, is_edge_weighted);
  }

  explicit SubgraphMemory(const PartitionedGraph &p_graph) {
    resize(p_graph);
  }

  void resize(const PartitionedGraph &p_graph) {
    resize(
        p_graph.n(),
        p_graph.k(),
        p_graph.m(),
        p_graph.is_node_weighted(),
        p_graph.is_edge_weighted()
    );
  }

  void resize(
      const NodeID n,
      const BlockID k,
      const EdgeID m,
      const bool is_node_weighted = true,
      const bool is_edge_weighted = true
  ) {
    SCOPED_TIMER("Allocation");

    nodes.resize(n + k);
    edges.resize(m);
    node_weights.resize(is_node_weighted * (n + k));
    edge_weights.resize(is_edge_weighted * m);
  }

  [[nodiscard]] bool empty() const {
    return nodes.empty();
  }

  StaticArray<EdgeID> nodes;
  StaticArray<NodeID> edges;
  StaticArray<NodeWeight> node_weights;
  StaticArray<EdgeWeight> edge_weights;
};

struct SubgraphExtractionResult {
  scalable_vector<Graph> subgraphs;
  scalable_vector<NodeID> node_mapping;
  scalable_vector<SubgraphMemoryStartPosition> positions;
};

struct SequentialSubgraphExtractionResult {
  std::array<Graph, 2> subgraphs;
  std::array<SubgraphMemoryStartPosition, 2> positions;
};

struct TemporarySubgraphMemory {
  constexpr static double kOverallocationFactor = 1.05;

  void ensure_size_nodes(const NodeID n, const bool is_node_weighed) {
    if (nodes.size() < n + 1) {
      nodes.resize(n * kOverallocationFactor + 1);
      ++num_node_reallocs;
    }
    if (is_node_weighed && node_weights.size() < n) {
      node_weights.resize(n * kOverallocationFactor);
    }
    if (mapping.size() < n) {
      mapping.resize(n * kOverallocationFactor);
    }
  }

  void ensure_size_edges(const EdgeID m, const bool is_edge_weighted) {
    if (edges.size() < m) {
      edges.resize(m * kOverallocationFactor);
      ++num_edge_reallocs;
    }
    if (is_edge_weighted && edge_weights.size() < m) {
      edge_weights.resize(m * kOverallocationFactor);
    }
  }

  std::vector<EdgeID> nodes;
  std::vector<NodeID> edges;
  std::vector<NodeWeight> node_weights;
  std::vector<EdgeWeight> edge_weights;
  std::vector<NodeID> mapping;

  std::size_t num_node_reallocs = 0;
  std::size_t num_edge_reallocs = 0;

  [[nodiscard]] std::size_t memory_in_kb() const {
    return nodes.size() * sizeof(EdgeID) / 1000 +            //
           edges.size() * sizeof(NodeID) / 1000 +            //
           node_weights.size() * sizeof(NodeWeight) / 1000 + //
           edge_weights.size() * sizeof(EdgeWeight) / 1000 + //
           mapping.size() * sizeof(NodeID) / 1000;           //
  }
};

SubgraphExtractionResult extract_subgraphs(
    const PartitionedGraph &p_graph, const BlockID input_k, SubgraphMemory &subgraph_memory
);

SequentialSubgraphExtractionResult extract_subgraphs_sequential(
    const PartitionedGraph &p_graph,
    const std::array<BlockID, 2> &final_ks,
    SubgraphMemoryStartPosition memory_position,
    SubgraphMemory &subgraph_memory,
    TemporarySubgraphMemory &tmp_subgraph_memory
);

PartitionedGraph copy_subgraph_partitions(
    PartitionedGraph p_graph,
    const scalable_vector<StaticArray<BlockID>> &p_subgraph_partitions,
    BlockID k_prime,
    BlockID input_k,
    const scalable_vector<NodeID> &mapping
);
} // namespace kaminpar::shm::graph
