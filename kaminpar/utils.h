/*******************************************************************************
 * @file:   utils.h
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 * @brief:  Unsorted utility functions (@todo)
 ******************************************************************************/
#pragma once

#include "kaminpar/datastructures/graph.h"
#include "kaminpar/kaminpar.h"

namespace kaminpar::shm {
PartitionContext create_bipartition_context(const PartitionContext &k_p_ctx,
                                            const Graph &subgraph,
                                            BlockID final_k1, BlockID final_k2);

double compute_2way_adaptive_epsilon(const PartitionContext &p_ctx,
                                     NodeWeight subgraph_total_node_weight,
                                     BlockID subgraph_final_k);

template <typename NodeID_ = NodeID, typename NodeWeight_ = NodeWeight>
NodeWeight_ compute_max_cluster_weight(const NodeID_ n,
                                       const NodeWeight_ total_node_weight,
                                       const PartitionContext &input_p_ctx,
                                       const CoarseningContext &c_ctx) {
  double max_cluster_weight = 0.0;

  switch (c_ctx.cluster_weight_limit) {
  case ClusterWeightLimit::EPSILON_BLOCK_WEIGHT:
    max_cluster_weight =
        (input_p_ctx.epsilon * total_node_weight) /
        std::clamp<BlockID>(n / c_ctx.contraction_limit, 2, input_p_ctx.k);
    break;

  case ClusterWeightLimit::BLOCK_WEIGHT:
    max_cluster_weight =
        (1.0 + input_p_ctx.epsilon) * total_node_weight / input_p_ctx.k;
    break;

  case ClusterWeightLimit::ONE:
    max_cluster_weight = 1.0;
    break;

  case ClusterWeightLimit::ZERO:
    max_cluster_weight = 0.0;
    break;
  }

  return static_cast<NodeWeight_>(max_cluster_weight *
                                  c_ctx.cluster_weight_multiplier);
}

template <typename NodeWeight_ = NodeWeight, typename Graph_ = Graph>
NodeWeight_ compute_max_cluster_weight(const Graph_ &c_graph,
                                       const PartitionContext &input_p_ctx,
                                       const CoarseningContext &c_ctx) {
  return compute_max_cluster_weight(c_graph.n(), c_graph.total_node_weight(),
                                    input_p_ctx, c_ctx);
}
} // namespace kaminpar::shm
