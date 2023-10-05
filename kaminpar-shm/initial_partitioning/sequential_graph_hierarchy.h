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
class SequentialGraphHierarchy {
public:
  explicit SequentialGraphHierarchy(const Graph *finest_graph);

  SequentialGraphHierarchy(const SequentialGraphHierarchy &) = delete;
  SequentialGraphHierarchy &operator=(const SequentialGraphHierarchy &) = delete;

  SequentialGraphHierarchy(SequentialGraphHierarchy &&) noexcept = default;
  SequentialGraphHierarchy &operator=(SequentialGraphHierarchy &&) noexcept = default;

  void take_coarse_graph(Graph &&c_graph, std::vector<NodeID> &&c_mapping);

  [[nodiscard]] const Graph &coarsest_graph() const;

  PartitionedGraph pop_and_project(PartitionedGraph &&coarse_p_graph);

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
  [[nodiscard]] const Graph &get_second_coarsest_graph() const;

  const Graph *_finest_graph;
  std::vector<std::vector<NodeID>> _coarse_mappings;
  std::vector<Graph> _coarse_graphs;
};
} // namespace kaminpar::shm::ip
