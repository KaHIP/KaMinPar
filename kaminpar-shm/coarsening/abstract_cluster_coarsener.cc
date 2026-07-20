/*******************************************************************************
 * Provides common functionality for coarseners optimized for cluster
 * contraction.
 *
 * @file:   abstract_cluster_coarsener.cc
 * @author: Daniel Seemaier
 * @date:   03.04.2025
 ******************************************************************************/
#include "kaminpar-shm/coarsening/abstract_cluster_coarsener.h"

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(false);

}

AbstractClusterCoarsener::AbstractClusterCoarsener(
    const Context &ctx, const PartitionContext &p_ctx
)
    : _ctx(ctx),
      _c_ctx(ctx.coarsening),
      _p_ctx(p_ctx),
      _clustering_algorithm(factory::create_clusterer(ctx)) {}

void AbstractClusterCoarsener::initialize(const Graph *graph) {
  _hierarchy.clear();
  _input_graph = graph;
}

const Graph &AbstractClusterCoarsener::current() const {
  return _hierarchy.empty() ? *_input_graph : _hierarchy.back()->get();
}

std::size_t AbstractClusterCoarsener::level() const {
  return _hierarchy.size();
}

void AbstractClusterCoarsener::use_communities(std::span<const NodeID> communities) {
  _input_communities = communities;
  _communities_hierarchy.clear();
}

[[nodiscard]] std::span<const NodeID> AbstractClusterCoarsener::current_communities() const {
  return _communities_hierarchy.empty() ? _input_communities : _communities_hierarchy.back();
}

void AbstractClusterCoarsener::release_allocated_memory() {
  SCOPED_HEAP_PROFILER("Deallocation");
  SCOPED_TIMER("Deallocation");

  _clustering_algorithm.reset();

  _contraction_m_ctx.buckets.free();
  _contraction_m_ctx.buckets_index.free();
  _contraction_m_ctx.leader_mapping.free();
  _contraction_m_ctx.all_buffered_nodes.free();
}

std::unique_ptr<CoarseGraph>
AbstractClusterCoarsener::pop_hierarchy([[maybe_unused]] PartitionedGraph &&p_graph) {
  KASSERT(!empty(), "cannot pop from an empty graph hierarchy", assert::light);

  auto coarsened = std::move(_hierarchy.back());
  _hierarchy.pop_back();

  KASSERT(
      &coarsened->get() == &p_graph.graph(),
      "p_graph wraps a different graph (ptr="
          << &p_graph.graph() << ") than the one that was coarsened (ptr=" << &coarsened->get()
          << ")",
      assert::light
  );

  if (!_communities_hierarchy.empty()) {
    _communities_hierarchy.pop_back();
  }

  return coarsened;
}

bool AbstractClusterCoarsener::keep_allocated_memory() const {
  return level() >= _c_ctx.clustering.max_mem_free_coarsening_level;
}

void AbstractClusterCoarsener::compute_clustering_for_current_graph(
    StaticArray<NodeID> &clustering
) {
  const bool free_allocated_memory = !keep_allocated_memory();
  const NodeID prev_n = current().n();

  DBG << "Coarsening graph with " << prev_n << " nodes";

  START_HEAP_PROFILER("Label Propagation");
  START_TIMER("Label Propagation");

  if (!_input_communities.empty()) {
    _clustering_algorithm->set_communities(current_communities());
  }

  configure_clusterer(*_clustering_algorithm, current(), _ctx, _p_ctx);

  _clustering_algorithm->compute_clustering(clustering, current(), free_allocated_memory);
  STOP_TIMER();
  STOP_HEAP_PROFILER();
}

PartitionedGraph AbstractClusterCoarsener::uncoarsen(PartitionedGraph &&p_graph) {
  const BlockID p_graph_k = p_graph.k();
  const auto p_graph_partition = p_graph.take_raw_partition();

  auto coarsened = pop_hierarchy(std::move(p_graph));
  const NodeID next_n = current().n();

  START_HEAP_PROFILER("Allocation");
  START_TIMER("Allocation");
  StaticArray<BlockID> partition(next_n);
  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_TIMER("Project partition");
  coarsened->project_up(p_graph_partition, partition);
  STOP_TIMER();

  SCOPED_HEAP_PROFILER("Create graph");
  SCOPED_TIMER("Create graph");
  return {current(), p_graph_k, std::move(partition)};
}

void AbstractClusterCoarsener::contract_current_graph_and_push(StaticArray<NodeID> &clustering) {
  START_HEAP_PROFILER("Contract graph");
  START_TIMER("Contract graph");

  _hierarchy.push_back([&] {
    auto c_graph = contract_clustering(
        current(), std::move(clustering), _c_ctx.contraction, _contraction_m_ctx
    );
    c_graph->get().set_level(level() + 1);
    return c_graph;
  }());

  if (!_communities_hierarchy.empty()) {
    _communities_hierarchy.emplace_back(current().n());
    project_communities(
        *_hierarchy.back(),
        _communities_hierarchy[_communities_hierarchy.size() - 2],
        _communities_hierarchy.back()
    );
  } else if (!_input_communities.empty()) {
    _communities_hierarchy.emplace_back(current().n());
    project_communities(*_hierarchy.back(), _input_communities, _communities_hierarchy.back());
  }

  STOP_TIMER();
  STOP_HEAP_PROFILER();
}

[[nodiscard]] bool AbstractClusterCoarsener::has_not_converged(const NodeID prev_n) const {
  const NodeID next_n = current().n();
  const bool converged = (1.0 - 1.0 * next_n / prev_n) <= _c_ctx.convergence_threshold;
  return !converged;
}

} // namespace kaminpar::shm
