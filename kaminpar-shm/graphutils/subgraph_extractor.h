/*******************************************************************************
 * @file:   subgraph_extraction.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Extracts the subgraphs induced by each block of a partition.
 ******************************************************************************/
#pragma once

#include <array>
#include <vector>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::graph {
struct SubgraphMemoryStartPosition {
  std::size_t nodes_start_pos = 0;
  std::size_t edges_start_pos = 0;

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
  SubgraphMemory() {
    RECORD_DATA_STRUCT(0, _struct);
  }

  SubgraphMemory(
      const NodeID n,
      const BlockID k,
      const EdgeID m,
      const bool is_node_weighted = true,
      const bool is_edge_weighted = true
  ) {
    RECORD_DATA_STRUCT(0, _struct);
    resize(n, k, m, is_node_weighted, is_edge_weighted);
  }

  explicit SubgraphMemory(const PartitionedGraph &p_graph) {
    RECORD_DATA_STRUCT(0, _struct);
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
    resize(n, k, m, is_node_weighted ? n : 0, is_edge_weighted ? m : 0);
  }

  void resize(
      const NodeID n,
      const BlockID k,
      const EdgeID m,
      const NodeID n_weights,
      const EdgeID m_weights
  ) {
    SCOPED_HEAP_PROFILER("SubgraphMemory resize");
    SCOPED_TIMER("Allocation");

    nodes.resize(n + k);
    edges.resize(m);
    node_weights.resize((n_weights == 0) ? 0 : (n_weights + k));
    edge_weights.resize(m_weights);

    IF_HEAP_PROFILING(
        _struct->size = std::max(
            _struct->size,
            nodes.size() * sizeof(EdgeID) + edges.size() * sizeof(NodeID) +
                node_weights.size() * sizeof(NodeWeight) + edge_weights.size() * sizeof(EdgeWeight)
        )
    );
  }

  [[nodiscard]] bool empty() const {
    return nodes.empty();
  }

  StaticArray<EdgeID> nodes;
  StaticArray<NodeID> edges;
  StaticArray<NodeWeight> node_weights;
  StaticArray<EdgeWeight> edge_weights;

  IF_HEAP_PROFILING(heap_profiler::DataStructure *_struct);
};

struct SubgraphExtractionResult {
  ScalableVector<Graph> subgraphs;
  StaticArray<NodeID> node_mapping;
  StaticArray<SubgraphMemoryStartPosition> positions;
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

struct OCSubgraphMemoryPreprocessingResult {
  StaticArray<NodeID> mapping;
  StaticArray<NodeID> block_nodes_offset;
  StaticArray<NodeID> block_nodes;
};

class OCSubgraphMemory {
public:
  OCSubgraphMemory(const NodeID n, const EdgeID m)
      : _nodes(heap_profiler::overcommit_memory<EdgeID>(n + 1)),
        _edges(heap_profiler::overcommit_memory<NodeID>(m)),
        _node_weights(heap_profiler::overcommit_memory<NodeWeight>(n)),
        _edge_weights(heap_profiler::overcommit_memory<EdgeWeight>(m)),
        _max_n(0),
        _max_m(0),
        _max_n_weights(0),
        _max_m_weights(0) {}

  [[nodiscard]] EdgeID *nodes() {
    return _nodes.get();
  }

  [[nodiscard]] NodeID *edges() {
    return _edges.get();
  }

  [[nodiscard]] NodeWeight *node_weights() {
    return _node_weights.get();
  }

  [[nodiscard]] EdgeWeight *edge_weights() {
    return _edge_weights.get();
  }

  void
  record(const NodeID n, const EdgeID m, const bool has_node_weights, const bool has_edge_weights) {
    _max_n = std::max(_max_n, n);
    _max_m = std::max(_max_m, m);

    if (has_node_weights) {
      _max_n_weights = std::max(_max_n_weights, n);
    }

    if (has_edge_weights) {
      _max_m_weights = std::max(_max_m_weights, m);
    }
  }

  void record_allocations() {
    if constexpr (kHeapProfiling) {
      heap_profiler::HeapProfiler::global().record_alloc(
          _nodes.get(), (_max_n + 1) * sizeof(EdgeID)
      );
      heap_profiler::HeapProfiler::global().record_alloc(_edges.get(), _max_m * sizeof(NodeID));
      heap_profiler::HeapProfiler::global().record_alloc(
          _node_weights.get(), _max_n_weights * sizeof(NodeWeight)
      );
      heap_profiler::HeapProfiler::global().record_alloc(
          _edge_weights.get(), _max_m_weights * sizeof(EdgeWeight)
      );
    }
  }

private:
  heap_profiler::unique_ptr<EdgeID> _nodes;
  heap_profiler::unique_ptr<NodeID> _edges;
  heap_profiler::unique_ptr<NodeWeight> _node_weights;
  heap_profiler::unique_ptr<EdgeWeight> _edge_weights;

  NodeID _max_n;
  EdgeID _max_m;
  NodeID _max_n_weights;
  EdgeID _max_m_weights;
};

[[nodiscard]] OCSubgraphMemoryPreprocessingResult
extract_subgraphs_preprocessing(const PartitionedGraph &p_graph);

Graph extract_subgraph(
    const PartitionedGraph &p_graph,
    const BlockID block,
    const StaticArray<NodeID> &block_nodes,
    const StaticArray<NodeID> &mapping,
    graph::OCSubgraphMemory &subgraph_memory
);

SequentialSubgraphExtractionResult extract_subgraphs_sequential(
    const PartitionedGraph &p_graph,
    const std::array<BlockID, 2> &final_ks,
    const SubgraphMemoryStartPosition memory_position,
    OCSubgraphMemory &subgraph_memory,
    TemporarySubgraphMemory &tmp_subgraph_memory
);

SubgraphExtractionResult extract_subgraphs(
    const PartitionedGraph &p_graph, BlockID input_k, SubgraphMemory &subgraph_memory
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
    const ScalableVector<StaticArray<BlockID>> &p_subgraph_partitions,
    BlockID k_prime,
    BlockID input_k,
    const StaticArray<NodeID> &mapping
);
} // namespace kaminpar::shm::graph
