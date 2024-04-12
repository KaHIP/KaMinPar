/*******************************************************************************
 * Coarsener that is optimized to contract clusterings.
 *
 * @file:   cluster_coarsener.cc
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/cluster_coarsener.h"

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
std::pair<const Graph *, bool> ClusteringCoarsener::compute_coarse_graph(
    const NodeWeight max_cluster_weight, const NodeID to_size, const bool free_memory_afterwards
) {
  SCOPED_HEAP_PROFILER("Level", std::to_string(_hierarchy.size()));
  SCOPED_TIMER("Level", std::to_string(_hierarchy.size()));

  _clustering_algorithm->set_max_cluster_weight(max_cluster_weight);
  _clustering_algorithm->set_desired_cluster_count(to_size);

  START_HEAP_PROFILER("Label Propagation");
  START_TIMER("Label Propagation");
  auto &clustering =
      _clustering_algorithm->compute_clustering(*_current_graph, free_memory_afterwards);
  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_HEAP_PROFILER("Contract graph");
  auto coarsened = TIMED_SCOPE("Contract graph") {
    return contract(*_current_graph, _c_ctx.contraction, clustering, _contraction_m_ctx);
  };
  STOP_HEAP_PROFILER();

  const bool converged =
      _c_ctx.coarsening_should_converge(_current_graph->n(), coarsened->get().n());

  _hierarchy.push_back(std::move(coarsened));
  _current_graph = &_hierarchy.back()->get();

  if (free_memory_afterwards) {
    _contraction_m_ctx.buckets.free();
    _contraction_m_ctx.buckets_index.free();
    _contraction_m_ctx.all_buffered_nodes.free();
  }

  return {_current_graph, !converged};
}

PartitionedGraph ClusteringCoarsener::uncoarsen(PartitionedGraph &&p_graph) {
  KASSERT(&p_graph.graph() == _current_graph);
  KASSERT(!empty(), V(size()));

  SCOPED_HEAP_PROFILER("Level", std::to_string(_hierarchy.size()));
  SCOPED_TIMER("Level", std::to_string(_hierarchy.size()));

  const BlockID p_graph_k = p_graph.k();
  const auto p_graph_partition = p_graph.take_raw_partition();

  auto coarsened = std::move(_hierarchy.back());
  _hierarchy.pop_back();
  _current_graph = empty() ? &_input_graph : &_hierarchy.back()->get();

  START_HEAP_PROFILER("Allocation");
  START_TIMER("Allocation");
  RECORD("partition") StaticArray<BlockID> partition(_current_graph->n());
  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_TIMER("Project partition");
  coarsened->project(p_graph_partition, partition);
  STOP_TIMER();

  SCOPED_HEAP_PROFILER("Create graph");
  SCOPED_TIMER("Create graph");
  return {*_current_graph, p_graph_k, std::move(partition)};
}
} // namespace kaminpar::shm
