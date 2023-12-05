/*******************************************************************************
 * Utility functions for partitioning.
 *
 * @file:   partition_utils.h
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/math.h"

namespace kaminpar::shm {

template <
    typename CoarseningContext,
    typename PartitionContext,
    typename NodeID,
    typename NodeWeight>
NodeWeight compute_max_cluster_weight(
    const CoarseningContext &c_ctx,
    const NodeID n,
    const NodeWeight total_node_weight,
    const PartitionContext &p_ctx
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
  }

  return static_cast<NodeWeight>(max_cluster_weight * c_ctx.cluster_weight_multiplier);
}

template <typename CoarseningContext>
NodeWeight compute_max_cluster_weight(
    const CoarseningContext &c_ctx, const Graph &graph, const PartitionContext &p_ctx
) {
  return compute_max_cluster_weight(c_ctx, graph.n(), graph.total_node_weight(), p_ctx);
}

double compute_2way_adaptive_epsilon(
    NodeWeight total_node_weight, BlockID k, const PartitionContext &p_ctx
);

PartitionContext create_bipartition_context(
    const Graph &subgraph, BlockID k1, BlockID k2, const PartitionContext &kway_p_ctx
);

/**
 * Given a block $0 <= B < k'$ of an intermediate partition with $k' < k$ blocks, this function
 * computes the number of blocks into which $B$ will be split for the final partition.
 *
 * More precisely, consider a binary tree with labels linked to each node constructed as follows:
 *
 * - The root node has label $k$.
 * - A node with label $\ell > 0$ has two children with labels $\lceil \ell / 2 \rceil$ and $\lfloor
 *   \ell / 2 \rfloor$.
 * - A node with label $\ell = 1$ has one child labelled $1$.
 * - The construction stops as soon as all nodes of a level have label $1$ / the level has size $k$.
 *
 * This function computes the label of any node in this tree, given the size of the nodes level
 * (i.e., $k'$) and its position within the level (i.e., $B$). Note that all levels have distinct
 * sizes, and thus, these two parameters uniquely identify a node of the tree).
 *
 * @param block The block $B$ / the position of a node within its level.
 * @param current_k The number of blocks $k'$ in the intermediate partition / the size of the node's
 * level.
 * @param inpuot_k The number of blocks $k$ in the final partition / the label of the root node.
 *
 * @return The number of blocks into which $B$ will be split for the final partition.
 */
BlockID compute_final_k(BlockID block, BlockID current_k, BlockID input_k);
} // namespace kaminpar::shm
