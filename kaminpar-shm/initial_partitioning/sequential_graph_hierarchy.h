/*******************************************************************************
 * Sequential graph hierarchy used for the multilevel cycle during initial
 * bipartitioning.
 *
 * @file:   sequential_graph_hierarchy.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

#include "kaminpar-common/datastructures/scalable_vector.h"

namespace kaminpar::shm::ip {
class SequentialGraphHierarchy {
public:
  SequentialGraphHierarchy() = default;

  SequentialGraphHierarchy(const SequentialGraphHierarchy &) = delete;
  SequentialGraphHierarchy &operator=(const SequentialGraphHierarchy &) = delete;

  SequentialGraphHierarchy(SequentialGraphHierarchy &&) noexcept = default;
  SequentialGraphHierarchy &operator=(SequentialGraphHierarchy &&) noexcept = default;

  void init(const CSRGraph &graph);

  void push(CSRGraph &&c_graph, ScalableVector<NodeID> &&c_mapping);

  [[nodiscard]] const CSRGraph &current() const;

  PartitionedCSRGraph pop(PartitionedCSRGraph &&coarse_p_graph);

  [[nodiscard]] inline std::size_t level() const {
    return _coarse_graphs.size();
  }

  [[nodiscard]] inline bool empty() const {
    return _coarse_graphs.empty();
  }

  //[[nodiscard]] inline const auto &coarse_mappings() const {
  //  return _coarse_mappings;
  //}

  //[[nodiscard]] inline const auto &coarse_graphs() const {
  //  return _coarse_graphs;
  //}

private:
  [[nodiscard]] const CSRGraph &get_second_coarsest_graph() const;

  const CSRGraph *_finest_graph;
  ScalableVector<ScalableVector<NodeID>> _coarse_mappings;
  ScalableVector<CSRGraph> _coarse_graphs;
};
} // namespace kaminpar::shm::ip
