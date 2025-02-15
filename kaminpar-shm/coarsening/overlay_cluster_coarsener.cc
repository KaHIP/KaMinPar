/*******************************************************************************
 * Coarsener that computes multiple clusterings, overlays and contracts them to
 * coarsen the graph.
 *
 * @file:   overlay_cluster_coarsener.cc
 * @author: Daniel Seemaier
 * @date:   13.12.2024
 ******************************************************************************/
#include "kaminpar-shm/coarsening/overlay_cluster_coarsener.h"

#include <algorithm>

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction_preprocessing.h"
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

OverlayClusteringCoarsener::OverlayClusteringCoarsener(
    const Context &ctx, const PartitionContext &p_ctx
)
    : _ctx(ctx),
      _c_ctx(ctx.coarsening),
      _p_ctx(p_ctx),
      _clustering_algorithm(factory::create_clusterer(ctx)) {}

void OverlayClusteringCoarsener::initialize(const Graph *graph) {
  _hierarchy.clear();
  _input_graph = graph;
}

void OverlayClusteringCoarsener::use_communities(std::span<const NodeID> communities) {
  _input_communities = communities;
  _communities_hierarchy.clear();
}

[[nodiscard]] std::span<const NodeID> OverlayClusteringCoarsener::current_communities() const {
  return _communities_hierarchy.empty() ? _input_communities : _communities_hierarchy.back();
}

bool OverlayClusteringCoarsener::coarsen() {
  SCOPED_HEAP_PROFILER("Level", std::to_string(_hierarchy.size()));
  SCOPED_TIMER("Level", std::to_string(_hierarchy.size()));

  START_HEAP_PROFILER("Allocation");
  const int num_overlays = 1 << _c_ctx.overlay_clustering.num_levels;
  std::vector<StaticArray<NodeID>> clusterings;
  for (int i = 0; i < num_overlays; ++i) {
    clusterings.emplace_back(current().n(), static_array::noinit);
  }
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

  const bool compute_overlays = level() <= _c_ctx.overlay_clustering.max_level;

  if (compute_overlays) {
    for (auto &clustering : clusterings) {
      _clustering_algorithm->compute_clustering(clustering, current(), free_allocated_memory);
    }
  } else {
    _clustering_algorithm->compute_clustering(
        clusterings.front(), current(), free_allocated_memory
    );
  }
  STOP_TIMER();
  STOP_HEAP_PROFILER();

  TIMED_SCOPE("Overlay clusters") {
    if (compute_overlays) {
      for (int level = _c_ctx.overlay_clustering.num_levels; level > 0; --level) {
        const int num_overlays_in_level = 1 << level;
        for (int pair = 0; pair < num_overlays_in_level / 2; ++pair) {
          clusterings[pair] =
              overlay(std::move(clusterings[pair]), clusterings[num_overlays_in_level / 2 + pair]);
        }
      }
    }
  };

  START_HEAP_PROFILER("Contract graph");
  START_TIMER("Contract graph");
  _hierarchy.push_back(contract_clustering(
      current(), std::move(clusterings.front()), _c_ctx.contraction, _contraction_m_ctx
  ));

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

StaticArray<NodeID>
OverlayClusteringCoarsener::overlay(StaticArray<NodeID> a, const StaticArray<NodeID> &b) {
#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
  StaticArray<NodeID> a_copy(a.begin(), a.end());
#endif

  const Graph &graph = current();
  const NodeID n = graph.n();

  StaticArray<NodeID> index, buckets, leader_mapping;
  contraction::fill_leader_mapping(graph, a, leader_mapping);
  const NodeID c_n = leader_mapping[n - 1];
  auto mapping = contraction::compute_mapping(graph, std::move(a), leader_mapping);
  contraction::fill_cluster_buckets(c_n, graph, mapping, index, buckets);

  tbb::parallel_for<NodeID>(0, c_n, [&](const NodeID c_u) {
    std::sort(
        buckets.begin() + index[c_u],
        buckets.begin() + index[c_u + 1],
        [&](const NodeID u, const NodeID v) { return b[u] < b[v]; }
    );

    NodeID prev_b = kInvalidNodeID;
    NodeID cur_id = index[c_u] - 1;
    for (std::size_t i = index[c_u]; i < index[c_u + 1]; ++i) {
      const NodeID u = buckets[i];
      const NodeID cur_b = b[u];
      if (cur_b != prev_b) {
        ++cur_id;
      }
      mapping[u] = cur_id;
      prev_b = cur_b;
    }
  });

#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
  KASSERT(
      [&] {
        for (NodeID u = 0; u < n; ++u) {
          bool failed = false;
          graph.adjacent_nodes(u, [&](const NodeID v) {
            if (a_copy[u] == a_copy[v] && b[u] == b[v]) {
              if (mapping[u] != mapping[v]) {
                failed = true;
              }
            }
            if (a_copy[u] != a_copy[v] || b[u] != b[v]) {
              if (mapping[u] == mapping[v]) {
                failed = true;
              }
            }
          });
          if (failed) {
            return false;
          }
        }
        return true;
      }(),
      "Overlaying clusters failed",
      assert::heavy
  );
#endif

  return mapping;
}

PartitionedGraph OverlayClusteringCoarsener::uncoarsen(PartitionedGraph &&p_graph) {
  SCOPED_TIMER("Level", std::to_string(_hierarchy.size()));

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

void OverlayClusteringCoarsener::release_allocated_memory() {
  SCOPED_HEAP_PROFILER("Deallocation");
  SCOPED_TIMER("Deallocation");

  _clustering_algorithm.reset();

  _contraction_m_ctx.buckets.free();
  _contraction_m_ctx.buckets_index.free();
  _contraction_m_ctx.leader_mapping.free();
  _contraction_m_ctx.all_buffered_nodes.free();
}

std::unique_ptr<CoarseGraph> OverlayClusteringCoarsener::pop_hierarchy(PartitionedGraph &&p_graph) {
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

bool OverlayClusteringCoarsener::keep_allocated_memory() const {
  return level() >= _c_ctx.clustering.max_mem_free_coarsening_level;
}

} // namespace kaminpar::shm
