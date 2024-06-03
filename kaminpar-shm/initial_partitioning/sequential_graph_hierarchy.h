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
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"

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

  StaticArray<BlockID> alloc_partition_memory();
  ScalableVector<NodeID> alloc_mapping_memory();
  CSRGraphMemory alloc_graph_memory();

private:
  [[nodiscard]] const CSRGraph &get_second_coarsest_graph() const;

  void recover_partition_memory(StaticArray<BlockID> partition);
  void recover_mapping_memory(ScalableVector<NodeID> mapping);
  void recover_graph_memory(CSRGraph graph);

  const CSRGraph *_finest_graph;

  ScalableVector<ScalableVector<NodeID>> _coarse_mappings;
  ScalableVector<CSRGraph> _coarse_graphs;

  ScalableVector<CSRGraphMemory> _graph_memory_cache;
  ScalableVector<ScalableVector<NodeID>> _mapping_memory_cache;
  ScalableVector<StaticArray<BlockID>> _partition_memory_cache;
};
} // namespace kaminpar::shm::ip
