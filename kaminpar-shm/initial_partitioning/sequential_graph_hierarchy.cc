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

namespace kaminpar::shm {
void SequentialGraphHierarchy::init(const CSRGraph &graph) {
  _finest_graph = &graph;

  _coarse_mappings.clear();
  _coarse_graphs.clear();
}

void SequentialGraphHierarchy::push(CSRGraph &&c_graph, StaticArray<NodeID> &&c_mapping) {
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

  // Goal: project partition of p_graph == c_graph onto new_c_graph
  StaticArray<NodeID> c_mapping = std::move(_coarse_mappings.back());
  _coarse_mappings.pop_back();

  const CSRGraph &graph = get_second_coarsest_graph();
  KASSERT(graph.n() == c_mapping.size());

  StaticArray<BlockID> partition = alloc_partition_memory();
  if (partition.size() < graph.n()) {
    partition.resize(graph.n(), static_array::small, static_array::seq);
  }

  for (const NodeID u : graph.nodes()) {
    partition[u] = coarse_p_graph.block(c_mapping[u]);
  }

  // Recover the memory of the coarsest graph before free'ing the graph object:
  recover_mapping_memory(std::move(c_mapping));
  recover_graph_memory(std::move(_coarse_graphs.back()));

  // ... the partition array of the coarsest graph is managed by the PoolBipartitioner
  // instead of the PartitionedCSRGraph object: do not push it back to the memory cache
  if (!coarse_p_graph.raw_partition().is_span()) {
    recover_partition_memory(coarse_p_graph.take_raw_partition());
  }

  _coarse_graphs.pop_back();

  return {PartitionedCSRGraph::seq{}, graph, coarse_p_graph.k(), std::move(partition)};
}

const CSRGraph &SequentialGraphHierarchy::get_second_coarsest_graph() const {
  KASSERT(!_coarse_graphs.empty());

  return (_coarse_graphs.size() > 1) ? _coarse_graphs[_coarse_graphs.size() - 2] : *_finest_graph;
}

void SequentialGraphHierarchy::recover_partition_memory(StaticArray<BlockID> partition) {
  KASSERT(!partition.is_span(), "span should not be cached");

  _partition_memory_cache.push_back(std::move(partition));
}

void SequentialGraphHierarchy::recover_mapping_memory(StaticArray<NodeID> mapping) {
  _mapping_memory_cache.push_back(std::move(mapping));
}

void SequentialGraphHierarchy::recover_graph_memory(CSRGraph graph) {
  KASSERT(!graph.raw_nodes().is_span(), "span should not be cached");
  KASSERT(!graph.raw_edges().is_span(), "span should not be cached");
  KASSERT(!graph.raw_node_weights().is_span(), "span should not be cached");
  KASSERT(!graph.raw_edge_weights().is_span(), "span should not be cached");

  _graph_memory_cache.push_back(CSRGraphMemory{
      .nodes = graph.take_raw_nodes(),
      .edges = graph.take_raw_edges(),
      .node_weights = graph.take_raw_node_weights(),
      .edge_weights = graph.take_raw_edge_weights(),
  });
}

StaticArray<BlockID> SequentialGraphHierarchy::alloc_partition_memory() {
  if (_partition_memory_cache.empty()) {
    _partition_memory_cache.emplace_back(0, static_array::small, static_array::seq);
  }

  auto memory = std::move(_partition_memory_cache.back());
  _partition_memory_cache.pop_back();
  return memory;
}

StaticArray<NodeID> SequentialGraphHierarchy::alloc_mapping_memory() {
  if (_mapping_memory_cache.empty()) {
    _mapping_memory_cache.emplace_back(0, static_array::seq);
  }

  auto memory = std::move(_mapping_memory_cache.back());
  _mapping_memory_cache.pop_back();
  return memory;
}

CSRGraphMemory SequentialGraphHierarchy::alloc_graph_memory() {
  if (_graph_memory_cache.empty()) {
    _graph_memory_cache.push_back(CSRGraphMemory{
        StaticArray<EdgeID>{0, static_array::seq},
        StaticArray<NodeID>{0, static_array::seq},
        StaticArray<NodeWeight>{0, static_array::seq},
        StaticArray<EdgeWeight>{0, static_array::seq},
    });
  }

  auto memory = std::move(_graph_memory_cache.back());
  _graph_memory_cache.pop_back();

  KASSERT(!memory.nodes.is_span(), "span should not be cached");
  KASSERT(!memory.edges.is_span(), "span should not be cached");
  KASSERT(!memory.node_weights.is_span(), "span should not be cached");
  KASSERT(!memory.edge_weights.is_span(), "span should not be cached");

  return memory;
}
} // namespace kaminpar::shm
