/*******************************************************************************
 * Utility functions to compute the maximum allowed cluster weight during
 * coarsening.
 *
 * @file:   max_cluster_weights.h
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#pragma once

#include <cstdint>

#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {
template <typename NodeWeight, typename PartitionContext>
NodeWeight compute_max_cluster_weight(
    const CoarseningContext &c_ctx,
    const PartitionContext &p_ctx,
    const std::uint64_t n,
    const std::int64_t total_node_weight
) {
  double max_cluster_weight = 0.0;

  switch (c_ctx.clustering.cluster_weight_limit) {
  case ClusterWeightLimit::EPSILON_BLOCK_WEIGHT:
    max_cluster_weight = (p_ctx.epsilon * total_node_weight) /
                         std::clamp<BlockID>(n / c_ctx.contraction_limit, 2, p_ctx.k);
    break;

  case ClusterWeightLimit::BLOCK_WEIGHT:
    max_cluster_weight = (1.0 + p_ctx.epsilon) * total_node_weight / p_ctx.k;
    break;

  case ClusterWeightLimit::ONE:
    max_cluster_weight = 1.0;
    break;

  case ClusterWeightLimit::ZERO:
    max_cluster_weight = 0.0;
    break;

  case ClusterWeightLimit::AVERAGE_NODE_WEIGHT:
    max_cluster_weight = static_cast<NodeWeight>(total_node_weight / n);
    break;
  }

  return static_cast<NodeWeight>(max_cluster_weight * c_ctx.clustering.cluster_weight_multiplier);
}

template <typename NodeWeight, typename CoarseningContext, typename PartitionContext>
NodeWeight compute_max_cluster_weight(
    const CoarseningContext &c_ctx,
    const PartitionContext &p_ctx,
    const std::uint64_t n,
    const std::int64_t total_node_weight
) {
  double max_cluster_weight = 0.0;

  switch (c_ctx.cluster_weight_limit) {
  case ClusterWeightLimit::EPSILON_BLOCK_WEIGHT:
    max_cluster_weight = (p_ctx.epsilon * total_node_weight) /
                         std::clamp<BlockID>(n / c_ctx.contraction_limit, 2, p_ctx.k);
    break;

  case ClusterWeightLimit::BLOCK_WEIGHT:
    max_cluster_weight = (1.0 + p_ctx.epsilon) * total_node_weight / p_ctx.k;
    break;

  case ClusterWeightLimit::ONE:
    max_cluster_weight = 1.0;
    break;

  case ClusterWeightLimit::ZERO:
    max_cluster_weight = 0.0;
    break;

  case ClusterWeightLimit::AVERAGE_NODE_WEIGHT:
    max_cluster_weight = static_cast<NodeWeight>(total_node_weight / n);
    break;
  }

  return static_cast<NodeWeight>(max_cluster_weight * c_ctx.cluster_weight_multiplier);
}
} // namespace kaminpar::shm
