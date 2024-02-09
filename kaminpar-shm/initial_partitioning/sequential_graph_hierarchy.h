/*******************************************************************************
 * @file:   sequential_graph_hierarchy.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

namespace kaminpar::shm::ip {
template <typename Graph> class SequentialGraphHierarchy {
  using PartitionedGraph = GenericPartitionedGraph<Graph>;

public:
  explicit SequentialGraphHierarchy(const Graph *finest_graph) : _finest_graph(finest_graph) {}

  SequentialGraphHierarchy(const SequentialGraphHierarchy &) = delete;
  SequentialGraphHierarchy &operator=(const SequentialGraphHierarchy &) = delete;

  SequentialGraphHierarchy(SequentialGraphHierarchy &&) noexcept = default;
  SequentialGraphHierarchy &operator=(SequentialGraphHierarchy &&) noexcept = default;

  void take_coarse_graph(Graph &&c_graph, std::vector<NodeID> &&c_mapping) {
    KASSERT(coarsest_graph().n() == c_mapping.size());
    _coarse_mappings.push_back(std::move(c_mapping));
    _coarse_graphs.push_back(std::move(c_graph));
  }

  [[nodiscard]] const Graph &coarsest_graph() const {
    return _coarse_graphs.empty() ? *_finest_graph : _coarse_graphs.back();
  }

  PartitionedGraph pop_and_project(PartitionedGraph &&coarse_p_graph) {
    KASSERT(!_coarse_graphs.empty());
    KASSERT(&_coarse_graphs.back() == &coarse_p_graph.graph());

    // goal: project partition of p_graph == c_graph onto new_c_graph
    std::vector<NodeID> c_mapping{std::move(_coarse_mappings.back())};
    _coarse_mappings.pop_back();

    const Graph &graph{get_second_coarsest_graph()};
    KASSERT(graph.n() == c_mapping.size());

    StaticArray<BlockID> partition{graph.n()};
    for (const NodeID u : graph.nodes()) {
      partition[u] = coarse_p_graph.block(c_mapping[u]);
    }

    // This destroys underlying Graph wrapped in p_graph
    _coarse_graphs.pop_back();

    return {typename PartitionedGraph::seq{}, graph, coarse_p_graph.k(), std::move(partition)};
  }

  [[nodiscard]] inline std::size_t size() const {
    return _coarse_graphs.size();
  }

  [[nodiscard]] inline bool empty() const {
    return _coarse_graphs.empty();
  }

  [[nodiscard]] inline const auto &coarse_mappings() const {
    return _coarse_mappings;
  }

  [[nodiscard]] inline const auto &coarse_graphs() const {
    return _coarse_graphs;
  }

private:
  [[nodiscard]] const Graph &get_second_coarsest_graph() const {
    KASSERT(!_coarse_graphs.empty());
    return (_coarse_graphs.size() > 1) ? _coarse_graphs[_coarse_graphs.size() - 2] : *_finest_graph;
  }

  const Graph *_finest_graph;
  std::vector<std::vector<NodeID>> _coarse_mappings;
  std::vector<Graph> _coarse_graphs;
};
} // namespace kaminpar::shm::ip
