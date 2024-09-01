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

namespace kaminpar::shm {
class SequentialGraphHierarchy {
public:
  SequentialGraphHierarchy() = default;

  SequentialGraphHierarchy(const SequentialGraphHierarchy &) = delete;
  SequentialGraphHierarchy &operator=(const SequentialGraphHierarchy &) = delete;

  SequentialGraphHierarchy(SequentialGraphHierarchy &&) noexcept = default;
  SequentialGraphHierarchy &operator=(SequentialGraphHierarchy &&) noexcept = default;

  void init(const CSRGraph &graph);

  void push(CSRGraph &&c_graph, StaticArray<NodeID> &&c_mapping);

  [[nodiscard]] const CSRGraph &current() const;

  PartitionedCSRGraph pop(PartitionedCSRGraph &&coarse_p_graph);

  [[nodiscard]] inline std::size_t level() const {
    return _coarse_graphs.size();
  }

  [[nodiscard]] inline bool empty() const {
    return _coarse_graphs.empty();
  }

  StaticArray<BlockWeight> alloc_block_weights_memory(std::size_t size = 0);
  StaticArray<BlockID> alloc_partition_memory(std::size_t size = 0);
  StaticArray<NodeID> alloc_mapping_memory();
  CSRGraphMemory alloc_graph_memory();

private:
  [[nodiscard]] const CSRGraph &get_second_coarsest_graph() const;

  void recover_block_weights_memory(StaticArray<BlockWeight> block_weights);
  void recover_partition_memory(StaticArray<BlockID> partition);
  void recover_mapping_memory(StaticArray<NodeID> mapping);
  void recover_graph_memory(CSRGraph graph);

  const CSRGraph *_finest_graph;

  ScalableVector<StaticArray<NodeID>> _coarse_mappings;
  ScalableVector<CSRGraph> _coarse_graphs;

  ScalableVector<CSRGraphMemory> _graph_memory_cache;
  ScalableVector<StaticArray<NodeID>> _mapping_memory_cache;
  ScalableVector<StaticArray<BlockID>> _partition_memory_cache;
  ScalableVector<StaticArray<BlockWeight>> _block_weights_memory_cache;
};
} // namespace kaminpar::shm
