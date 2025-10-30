/*******************************************************************************
 * Unsorted utility functions (@todo).
 *
 * @file:   utils.cc
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#include "kaminpar-shm/partitioning/partition_utils.h"

#include <algorithm>
#include <array>
#include <cmath>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/math.h"

namespace kaminpar::shm::partitioning {

BlockID compute_final_k(const BlockID block, const BlockID current_k, const BlockID input_k) {
  if (current_k == input_k) {
    return 1;
  }

  // The level of the current block in the binary tree == log2(current_k)
  const BlockID level = math::floor_log2(current_k);

  // Within a level, each pair of labels l1, l2 satisfy |l1 - l2| <= 1, i.e., they differ by at most
  // one.
  // This is the smaller label of the level, i.e., the label is either base or base + 1.
  const BlockID base = input_k >> level;

  // This is the number of base + 1 labels of the level, all other have value base:
  const BlockID num_plus_one_blocks = input_k & ((1 << level) - 1);

  // Reverse the bits of the block label, i.e., 0b0000'1010 -> 0b0000'0101 (leading zeroes are
  // discarded). This gives the order in which nodes have their labels "flipped" from base to
  // base + 1.
  static_assert(sizeof(BlockID) == 4);
  std::array<BlockID, 16> lut = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
  const int64_t reversed_block = static_cast<int64_t>(
          lut[(block & 0xF0'00'00'00) >> 28] | (lut[(block & 0x0F'00'00'00) >> 24] << 4) |
          (lut[(block & 0x00'F0'00'00) >> 20] << 8) | (lut[(block & 0x00'0F'00'00) >> 16] << 12) |
          (lut[(block & 0x00'00'F0'00) >> 12] << 16) | (lut[(block & 0x00'00'0F'00) >> 8] << 20) |
          (lut[(block & 0x00'00'00'F0) >> 4] << 24) | (lut[block & 0x00'00'00'0F] << 28)
      ) >> (32 - level);

  return base + (reversed_block < num_plus_one_blocks);
}

// @todo optimize
BlockID
compute_first_sub_block(const BlockID block, const BlockID current_k, const BlockID input_k) {
  if (current_k == 1) {
    return 0;
  }

  int level = math::ceil_log2(current_k);
  int mask = 1 << (level - 1);

  BlockID width = 1;
  BlockID current_value = input_k;
  BlockID ans = 0;
  while (width <= current_k) {
    width *= 2;
    auto [lhs, rhs] = math::split_integral(current_value);
    if (block & mask) {
      current_value = rhs;
      ans += lhs;
    } else {
      current_value = lhs;
    }
    mask >>= 1;
  }
  return ans;

  // BlockID first_sub_block = 0;
  // for (BlockID b = 0; b < block; ++b) {
  // first_sub_block += compute_final_k(b, current_k, input_k);
  //}
  // return first_sub_block;
}

BlockID compute_first_invalid_sub_block(
    const BlockID block, const BlockID current_k, const BlockID input_k
) {
  return compute_first_sub_block(block, current_k, input_k) +
         compute_final_k(block, current_k, input_k);
}

BlockID compute_k_for_n(const NodeID n, const Context &input_ctx) {
  // Catch special case where log is negative:
  if (n < 2 * input_ctx.coarsening.contraction_limit) {
    return 2;
  }

  const BlockID k_prime = 1 << math::ceil_log2(n / input_ctx.coarsening.contraction_limit);
  return std::clamp<BlockID>(k_prime, 2, input_ctx.partition.k);
}

std::size_t compute_num_copies(
    const Context &input_ctx, const NodeID n, const bool converged, const std::size_t num_threads
) {
  KASSERT(num_threads > 0u);

  // Sequential base case;
  const NodeID C = input_ctx.coarsening.contraction_limit;
  if (converged || n <= 2 * C) {
    return num_threads;
  }

  // Parallel case:
  const std::size_t f = 1 << static_cast<std::size_t>(std::ceil(std::log2(1.0 * n / C)));

  // Continue with coarsening if the graph is still too large ...
  if (f > num_threads) {
    return 1;
  }

  // ... otherwise, split into groups:
  return num_threads / f;
}

int compute_num_threads_for_parallel_ip(const Context &input_ctx) {
  return math::floor2(
      static_cast<unsigned int>(
          1.0 * input_ctx.parallel.num_threads *
          input_ctx.partitioning.deep_initial_partitioning_load
      )
  );
}

BlockID compute_next_k(const BlockID current_k, const Context &input_ctx) {
  return std::min<BlockID>(current_k * 2, input_ctx.partition.k);
}

} // namespace kaminpar::shm::partitioning
