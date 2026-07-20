/*******************************************************************************
 * Coarsener that computes multiple clusterings, overlays and contracts them to
 * coarsen the graph.
 *
 * @file:   overlay_cluster_coarsener.cc
 * @author: Daniel Seemaier
 * @date:   13.12.2024
 ******************************************************************************/
#include "kaminpar-shm/coarsening/overlay_cluster_coarsener.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

OverlayClusterCoarsener::OverlayClusterCoarsener(const Context &ctx, const PartitionContext &p_ctx)
    : AbstractClusterCoarsener(ctx, p_ctx) {
  auto ensemble_clusterer =
      std::make_unique<EnsembleClusterer>(std::move(_clustering_algorithm), 1);
  _ensemble_clusterer = ensemble_clusterer.get();
  _clustering_algorithm = std::move(ensemble_clusterer);
}

bool OverlayClusterCoarsener::coarsen() {
  SCOPED_HEAP_PROFILER("Level", std::to_string(_hierarchy.size()));
  SCOPED_TIMER("Level", std::to_string(_hierarchy.size()));

  START_HEAP_PROFILER("Allocation");
  StaticArray<NodeID> clustering(current().n(), static_array::noinit);
  STOP_HEAP_PROFILER();

  const bool free_allocated_memory = !keep_allocated_memory();
  const NodeID prev_n = current().n();
  const bool compute_overlays =
      level() <= static_cast<std::size_t>(_c_ctx.overlay_clustering.max_level);

  const std::size_t num_clusterings =
      compute_overlays ? std::size_t{1} << _c_ctx.overlay_clustering.num_levels : 1;
  _ensemble_clusterer->set_num_clusterings(num_clusterings);
  compute_clustering_for_current_graph(clustering);

  contract_current_graph_and_push(clustering);

  if (free_allocated_memory) {
    _contraction_m_ctx.buckets.free();
    _contraction_m_ctx.buckets_index.free();
    _contraction_m_ctx.all_buffered_nodes.free();
  }

  return has_not_converged(prev_n);
}

} // namespace kaminpar::shm
