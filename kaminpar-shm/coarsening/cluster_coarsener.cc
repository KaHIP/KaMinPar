/*******************************************************************************
 * Coarsener that is optimized to contract clusterings.
 *
 * @file:   cluster_coarsener.cc
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/cluster_coarsener.h"

#include <algorithm>

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/coarsening/max_cluster_weights.h"
#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(false);

}

ClusteringCoarsener::ClusteringCoarsener(const Context &ctx, const PartitionContext &p_ctx)
    : _ctx(ctx),
      _c_ctx(ctx.coarsening),
      _p_ctx(p_ctx),
      _clustering_algorithm(factory::create_clusterer(ctx)) {}

void ClusteringCoarsener::initialize(const Graph *graph) {
  _hierarchy.clear();
  _input_graph = graph;
}

void ClusteringCoarsener::use_communities(std::span<const NodeID> communities) {
  _input_communities = communities;
  _communities_hierarchy.clear();
}

[[nodiscard]] std::span<const NodeID> ClusteringCoarsener::current_communities() const {
  return _communities_hierarchy.empty() ? _input_communities : _communities_hierarchy.back();
}

bool ClusteringCoarsener::coarsen() {
  START_HEAP_PROFILER("Allocation");
  RECORD("clustering") StaticArray<NodeID> clustering(current().n(), static_array::noinit);
  STOP_HEAP_PROFILER();

  const bool free_allocated_memory = !keep_allocated_memory();
  const NodeWeight total_node_weight = current().total_node_weight();
  const NodeID prev_n = current().n();

  DBG << "Coarsening graph with " << prev_n << " nodes";

  START_HEAP_PROFILER("Label Propagation");
  START_TIMER("Label Propagation");

  if (!_input_communities.empty()) {
    _clustering_algorithm->set_communities(current_communities());
  }

  _clustering_algorithm->set_max_cluster_weight(
      compute_max_cluster_weight<NodeWeight>(_c_ctx, _p_ctx, prev_n, total_node_weight)
  );

  {
    NodeID desired_cluster_count = prev_n / _c_ctx.clustering.shrink_factor;

    const double U = _c_ctx.clustering.forced_level_upper_factor;
    const double L = _c_ctx.clustering.forced_level_lower_factor;
    const BlockID k = _p_ctx.k;
    const int p = _ctx.parallel.num_threads;
    const NodeID C = _c_ctx.contraction_limit;

    if (_c_ctx.clustering.forced_kc_level) {
      if (prev_n > U * C * k) {
        desired_cluster_count = std::max<NodeID>(desired_cluster_count, L * C * k);
      }
    }
    if (_c_ctx.clustering.forced_pc_level) {
      if (prev_n > U * C * p) {
        desired_cluster_count = std::max<NodeID>(desired_cluster_count, L * C * p);
      }
    }

    DBG << "Desired cluster count: " << desired_cluster_count;
    _clustering_algorithm->set_desired_cluster_count(desired_cluster_count);
  }

  _clustering_algorithm->compute_clustering(clustering, current(), free_allocated_memory);
  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_HEAP_PROFILER("Contract graph");
  START_TIMER("Contract graph");
  _hierarchy.push_back(
      contract_clustering(current(), std::move(clustering), _c_ctx.contraction, _contraction_m_ctx)
  );

  auto project_communities = [&](const std::size_t fine_n,
                                 const NodeID *fine_ptr,
                                 const std::size_t coarse_n,
                                 NodeID *coarse_ptr) {
    if constexpr (std::is_same_v<BlockID, NodeID>) {
      const BlockID *fine = reinterpret_cast<const BlockID *>(fine_ptr);
      BlockID *coarse = reinterpret_cast<BlockID *>(coarse_ptr);
      _hierarchy.back()->project_down({fine, fine_n}, {coarse, coarse_n});
    } else {
      StaticArray<BlockID> fine(fine_n);
      StaticArray<BlockID> coarse(coarse_n);

      tbb::parallel_for<std::size_t>(0, fine_n, [&](const std::size_t i) {
        fine[i] = static_cast<BlockID>(fine_ptr[i]);
      });
      _hierarchy.back()->project_down(fine, coarse);
      tbb::parallel_for<std::size_t>(0, coarse_n, [&](const std::size_t i) {
        coarse_ptr[i] = static_cast<NodeID>(coarse[i]);
      });
    }
  };

  if (!_communities_hierarchy.empty()) {
    _communities_hierarchy.emplace_back(current().n());

    const std::size_t fine_n = _communities_hierarchy[_communities_hierarchy.size() - 2].size();
    NodeID *fine_ptr = _communities_hierarchy[_communities_hierarchy.size() - 2].data();
    const std::size_t coarse_n = _communities_hierarchy.back().size();
    NodeID *coarse_ptr = _communities_hierarchy.back().data();

    project_communities(fine_n, fine_ptr, coarse_n, coarse_ptr);
  } else if (!_input_communities.empty()) {
    _communities_hierarchy.emplace_back(current().n());

    const std::size_t fine_n = _input_communities.size();
    const NodeID *fine_ptr = _input_communities.data();
    const std::size_t coarse_n = _communities_hierarchy.back().size();
    NodeID *coarse_ptr = _communities_hierarchy.back().data();

    project_communities(fine_n, fine_ptr, coarse_n, coarse_ptr);
  }

  STOP_TIMER();
  STOP_HEAP_PROFILER();

  const NodeID next_n = current().n();
  const bool converged = (1.0 - 1.0 * next_n / prev_n) <= _c_ctx.convergence_threshold;

  if (free_allocated_memory) {
    _contraction_m_ctx.buckets.free();
    _contraction_m_ctx.buckets_index.free();
    _contraction_m_ctx.all_buffered_nodes.free();
  }

  return !converged;
}

PartitionedGraph ClusteringCoarsener::uncoarsen(PartitionedGraph &&p_graph) {
  const BlockID p_graph_k = p_graph.k();
  const auto p_graph_partition = p_graph.take_raw_partition();

  auto coarsened = pop_hierarchy(std::move(p_graph));
  const NodeID next_n = current().n();

  START_HEAP_PROFILER("Allocation");
  START_TIMER("Allocation");
  RECORD("partition") StaticArray<BlockID> partition(next_n);
  STOP_TIMER();
  STOP_HEAP_PROFILER();

  START_TIMER("Project partition");
  coarsened->project_up(p_graph_partition, partition);
  STOP_TIMER();

  SCOPED_HEAP_PROFILER("Create graph");
  SCOPED_TIMER("Create graph");
  return {current(), p_graph_k, std::move(partition)};
}

void ClusteringCoarsener::release_allocated_memory() {
  SCOPED_HEAP_PROFILER("Deallocation");
  SCOPED_TIMER("Deallocation");

  _clustering_algorithm.reset();

  _contraction_m_ctx.buckets.free();
  _contraction_m_ctx.buckets_index.free();
  _contraction_m_ctx.leader_mapping.free();
  _contraction_m_ctx.all_buffered_nodes.free();
}

std::unique_ptr<CoarseGraph>
ClusteringCoarsener::pop_hierarchy([[maybe_unused]] PartitionedGraph &&p_graph) {
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

bool ClusteringCoarsener::keep_allocated_memory() const {
  return level() >= _c_ctx.clustering.max_mem_free_coarsening_level;
}

} // namespace kaminpar::shm
