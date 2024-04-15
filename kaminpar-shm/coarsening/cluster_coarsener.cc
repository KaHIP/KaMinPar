/*******************************************************************************
 * Coarsener that is optimized to contract clusterings.
 *
 * @file:   cluster_coarsener.cc
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/cluster_coarsener.h"

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
void ClusteringCoarsener::initialize(const Graph *graph) {
  _hierarchy.clear();
  _input_graph = graph;
}

bool ClusteringCoarsener::coarsen(
    const NodeWeight max_cluster_weight, const NodeID to_size, const bool free_memory_afterwards
) {
  SCOPED_HEAP_PROFILER("Level", std::to_string(_hierarchy.size()));
  SCOPED_TIMER("Level", std::to_string(_hierarchy.size()));

  if (_clustering.size() < current().n()) {
    SCOPED_TIMER("Allocation");
    _clustering.resize(current().n());
  }

  START_HEAP_PROFILER("Label Propagation");
  START_TIMER("Label Propagation");
  _clustering_algorithm->set_max_cluster_weight(max_cluster_weight);
  _clustering_algorithm->set_desired_cluster_count(to_size);
  _clustering_algorithm->compute_clustering(_clustering, current(), free_memory_afterwards);
  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_HEAP_PROFILER("Contract graph");
  auto coarsened = TIMED_SCOPE("Contract graph") {
    return contract_clustering(current(), _clustering, _c_ctx.contraction, _contraction_m_ctx);
  };
  STOP_HEAP_PROFILER();

  const bool converged = _c_ctx.coarsening_should_converge(current().n(), coarsened->get().n());

  _hierarchy.push_back(std::move(coarsened));

  if (free_memory_afterwards) {
    _contraction_m_ctx.buckets.free();
    _contraction_m_ctx.buckets_index.free();
    _contraction_m_ctx.all_buffered_nodes.free();
  }

  return !converged;
}

PartitionedGraph ClusteringCoarsener::uncoarsen(PartitionedGraph &&p_graph) {
  KASSERT(&p_graph.graph() == &current());
  KASSERT(!empty(), "cannot call uncoarsen() on an empty graph hierarchy");

  SCOPED_HEAP_PROFILER("Level", std::to_string(_hierarchy.size()));
  SCOPED_TIMER("Level", std::to_string(_hierarchy.size()));

  const BlockID p_graph_k = p_graph.k();
  const auto p_graph_partition = p_graph.take_raw_partition();

  auto coarsened = std::move(_hierarchy.back());
  _hierarchy.pop_back();

  START_HEAP_PROFILER("Allocation");
  START_TIMER("Allocation");
  RECORD("partition") StaticArray<BlockID> partition(current().n());
  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_TIMER("Project partition");
  coarsened->project(p_graph_partition, partition);
  STOP_TIMER();

  SCOPED_HEAP_PROFILER("Create graph");
  SCOPED_TIMER("Create graph");
  return {current(), p_graph_k, std::move(partition)};
}
} // namespace kaminpar::shm
