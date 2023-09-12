/*******************************************************************************
 * Unsorted utility functions (@todo).
 *
 * @file:   utils.cc
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#include "kaminpar/partition_utils.h"

#include "kaminpar/context.h"
#include "kaminpar/definitions.h"

#include "common/math.h"

namespace kaminpar::shm {
double compute_2way_adaptive_epsilon(
    const NodeWeight total_node_weight, const BlockID k, const PartitionContext &p_ctx
) {
  const double base =
      (1.0 + p_ctx.epsilon) * k * p_ctx.total_node_weight / p_ctx.k / total_node_weight;
  const double exponent = 1.0 / math::ceil_log2(k);
  const double epsilon_prime = std::pow(base, exponent) - 1.0;
  const double adaptive_epsilon = std::max(epsilon_prime, 0.0001);
  return adaptive_epsilon;
}

PartitionContext create_bipartition_context(
    const Graph &subgraph, const BlockID k1, const BlockID k2, const PartitionContext &kway_p_ctx
) {
  PartitionContext twoway_p_ctx;
  twoway_p_ctx.k = 2;
  twoway_p_ctx.setup(subgraph);
  twoway_p_ctx.epsilon =
      compute_2way_adaptive_epsilon(subgraph.total_node_weight(), k1 + k2, kway_p_ctx);
  twoway_p_ctx.block_weights.setup(twoway_p_ctx, {k1, k2});
  return twoway_p_ctx;
}

BlockID compute_final_k(const BlockID block, const BlockID current_k, const BlockID input_k) {
  const BlockID height = math::floor_log2(input_k);
  const BlockID level = math::floor_log2(current_k);
  const BlockID num_heavy_blocks = input_k - (1 << height);
  const BlockID num_leaves = 1 << (height - level);
  const BlockID first_leaf = block * num_leaves;
  const BlockID first_invalid_leaf = first_leaf + num_leaves;

  if (first_leaf > num_heavy_blocks) {
    return num_leaves;
  } else if (first_invalid_leaf <= num_heavy_blocks) {
    return 2 * num_leaves;
  } else {
    return num_heavy_blocks - (block - 1) * num_leaves;
  }
}
} // namespace kaminpar::shm
