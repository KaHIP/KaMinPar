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

#include "kaminpar-shm/coarsening/abstract_cluster_coarsener.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/coarsening/contraction/cluster_contraction_preprocessing.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(false);

}

OverlayClusterCoarsener::OverlayClusterCoarsener(const Context &ctx, const PartitionContext &p_ctx)
    : AbstractClusterCoarsener(ctx, p_ctx) {}

bool OverlayClusterCoarsener::coarsen() {
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
  const NodeID prev_n = current().n();
  const bool compute_overlays =
      level() <= static_cast<std::size_t>(_c_ctx.overlay_clustering.max_level);

  if (compute_overlays) {
    for (auto &clustering : clusterings) {
      compute_clustering_for_current_graph(clustering);
    }
  } else {
    compute_clustering_for_current_graph(clusterings.front());
  }

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

  contract_current_graph_and_push(clusterings.front());

  if (free_allocated_memory) {
    _contraction_m_ctx.buckets.free();
    _contraction_m_ctx.buckets_index.free();
    _contraction_m_ctx.all_buffered_nodes.free();
  }

  return has_not_converged(prev_n);
}

StaticArray<NodeID>
OverlayClusterCoarsener::overlay(StaticArray<NodeID> a, const StaticArray<NodeID> &b) {
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

} // namespace kaminpar::shm
