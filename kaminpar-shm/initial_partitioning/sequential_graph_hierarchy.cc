/*******************************************************************************
 * Sequential graph hierarchy used for the multilevel cycle during initial
 * bipartitioning.
 *
 * @file:   sequential_graph_hierarchy.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/sequential_graph_hierarchy.h"

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

#include "kaminpar-common/datastructures/scalable_vector.h"

namespace kaminpar::shm::ip {
void SequentialGraphHierarchy::init(const CSRGraph &graph) {
  _finest_graph = &graph;

  _coarse_mappings.clear();
  _coarse_graphs.clear();
}

void SequentialGraphHierarchy::push(CSRGraph &&c_graph, ScalableVector<NodeID> &&c_mapping) {
  KASSERT(current().n() == c_mapping.size());

  _coarse_mappings.push_back(std::move(c_mapping));
  _coarse_graphs.push_back(std::move(c_graph));
}

[[nodiscard]] const CSRGraph &SequentialGraphHierarchy::current() const {
  return _coarse_graphs.empty() ? *_finest_graph : _coarse_graphs.back();
}

PartitionedCSRGraph SequentialGraphHierarchy::pop(PartitionedCSRGraph &&coarse_p_graph) {
  KASSERT(!_coarse_graphs.empty());
  KASSERT(&_coarse_graphs.back() == &coarse_p_graph.graph());

  // goal: project partition of p_graph == c_graph onto new_c_graph
  ScalableVector<NodeID> c_mapping = std::move(_coarse_mappings.back());
  _coarse_mappings.pop_back();

  const CSRGraph &graph = get_second_coarsest_graph();
  KASSERT(graph.n() == c_mapping.size());

  StaticArray<BlockID> partition(graph.n(), static_array::small, static_array::seq);
  for (const NodeID u : graph.nodes()) {
    partition[u] = coarse_p_graph.block(c_mapping[u]);
  }

  // This destroys underlying Graph wrapped in p_graph
  _coarse_graphs.pop_back();

  return {PartitionedCSRGraph::seq{}, graph, coarse_p_graph.k(), std::move(partition)};
}

const CSRGraph &SequentialGraphHierarchy::get_second_coarsest_graph() const {
  KASSERT(!_coarse_graphs.empty());
  return (_coarse_graphs.size() > 1) ? _coarse_graphs[_coarse_graphs.size() - 2] : *_finest_graph;
}
} // namespace kaminpar::shm::ip
