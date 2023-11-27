/*******************************************************************************
 * Unsorted utility functions (@todo).
 *
 * @file:   utils.cc
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#include "kaminpar-shm/partition_utils.h"

#include <array>

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/definitions.h"

#include "kaminpar-common/math.h"

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
  twoway_p_ctx.block_weights.setup(twoway_p_ctx, k1 + k2);
  return twoway_p_ctx;
}

BlockID compute_final_k(const BlockID block, const BlockID current_k, const BlockID input_k) {
  if (current_k == 1) {
    return input_k;
  }
  if (current_k == input_k) {
    return 1;
  }

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

BlockID
compute_final_k_legacy(const BlockID block, const BlockID current_k, const BlockID input_k) {
  if (current_k == 1) {
    return input_k;
  }
  if (current_k == input_k) {
    return 1;
  }

  const BlockID level = math::floor_log2(current_k);                // == log2(current_k)
  const BlockID base = input_k >> level;                            // == input_k / current_k
  const BlockID num_plus_one_blocks = input_k & ((1 << level) - 1); // == input_k % current_k

  static_assert(sizeof(BlockID) == 4);
  std::array<BlockID, 16> lut = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};

  const BlockID height = math::ceil_log2(input_k);
  const BlockID position =
      (lut[(block & 0xF0'00'00'00) >> 28] | (lut[(block & 0x0F'00'00'00) >> 24] << 4) |
       (lut[(block & 0x00'F0'00'00) >> 20] << 8) | (lut[(block & 0x00'0F'00'00) >> 16] << 12) |
       (lut[(block & 0x00'00'F0'00) >> 12] << 16) | (lut[(block & 0x00'00'0F'00) >> 8] << 20) |
       (lut[(block & 0x00'00'00'F0) >> 4] << 24) | (lut[block & 0x00'00'00'0F] << 28)) >>
      (32 - level);
  return base + (position < num_plus_one_blocks);
}
} // namespace kaminpar::shm
