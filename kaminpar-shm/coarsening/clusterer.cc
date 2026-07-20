/*******************************************************************************
 * Shared utilities for configuring clustering algorithms.
 *
 * @file:   clusterer.cc
 ******************************************************************************/
#include "kaminpar-shm/coarsening/clusterer.h"

#include <algorithm>

#include "kaminpar-shm/coarsening/max_cluster_weights.h"

#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(false);

} // namespace

void configure_clusterer(
    Clusterer &clusterer, const Graph &graph, const Context &ctx, const PartitionContext &p_ctx
) {
  const CoarseningContext &c_ctx = ctx.coarsening;
  const NodeID n = graph.n();

  clusterer.set_max_cluster_weight(
      compute_max_cluster_weight<NodeWeight>(c_ctx, p_ctx, n, graph.total_node_weight())
  );

  NodeID desired_cluster_count = n / c_ctx.clustering.shrink_factor;

  const double upper_factor = c_ctx.clustering.forced_level_upper_factor;
  const double lower_factor = c_ctx.clustering.forced_level_lower_factor;
  const BlockID k = p_ctx.k;
  const int num_threads = ctx.parallel.num_threads;
  const NodeID contraction_limit = c_ctx.contraction_limit;

  if (c_ctx.clustering.forced_kc_level && n > upper_factor * contraction_limit * k) {
    desired_cluster_count =
        std::max<NodeID>(desired_cluster_count, lower_factor * contraction_limit * k);
  }
  if (c_ctx.clustering.forced_pc_level && n > upper_factor * contraction_limit * num_threads) {
    desired_cluster_count =
        std::max<NodeID>(desired_cluster_count, lower_factor * contraction_limit * num_threads);
  }

  DBG << "Desired cluster count: " << desired_cluster_count;
  clusterer.set_desired_cluster_count(desired_cluster_count);
}

} // namespace kaminpar::shm
