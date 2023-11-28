#include <gtest/gtest.h>

#include "kaminpar-shm/partition_utils.h"

using namespace kaminpar::shm;

namespace {
TEST(PartitionUtilsTest, TwoBlocks) {
  EXPECT_EQ(compute_final_k(0, 1, 2), 2);
}

TEST(PartitionUtilsTest, FourBlocks) {
  EXPECT_EQ(compute_final_k(0, 1, 4), 4);

  EXPECT_EQ(compute_final_k(0, 2, 4), 2);
  EXPECT_EQ(compute_final_k(1, 2, 4), 2);
}

TEST(PartitionUtilsTest, ElevenBlocks) {
  EXPECT_EQ(compute_final_k(0, 1, 11), 11);

  EXPECT_EQ(compute_final_k(0, 2, 11), 7);
  EXPECT_EQ(compute_final_k(1, 2, 11), 4);

  EXPECT_EQ(compute_final_k(0, 4, 11), 4);
  EXPECT_EQ(compute_final_k(1, 4, 11), 3);
  EXPECT_EQ(compute_final_k(2, 4, 11), 2);
  EXPECT_EQ(compute_final_k(3, 4, 11), 2);

  EXPECT_EQ(compute_final_k(0, 8, 11), 2);
  EXPECT_EQ(compute_final_k(1, 8, 11), 2);
  EXPECT_EQ(compute_final_k(2, 8, 11), 2);
  EXPECT_EQ(compute_final_k(3, 8, 11), 1);
  EXPECT_EQ(compute_final_k(4, 8, 11), 1);
  EXPECT_EQ(compute_final_k(5, 8, 11), 1);
  EXPECT_EQ(compute_final_k(6, 8, 11), 1);
  EXPECT_EQ(compute_final_k(7, 8, 11), 1);
}

TEST(PartitionUtilsTest, TwoBlocksLegacy) {
  EXPECT_EQ(compute_final_k_legacy(0, 1, 2), 2);
}

TEST(PartitionUtilsTest, FourBlocksLegacy) {
  EXPECT_EQ(compute_final_k_legacy(0, 1, 4), 4);

  EXPECT_EQ(compute_final_k_legacy(0, 2, 4), 2);
  EXPECT_EQ(compute_final_k_legacy(1, 2, 4), 2);
}

TEST(PartitionUtilsTest, ElevenBlocksLegacy) {
  EXPECT_EQ(compute_final_k_legacy(0, 1, 11), 11);

  EXPECT_EQ(compute_final_k_legacy(0, 2, 11), 6);
  EXPECT_EQ(compute_final_k_legacy(1, 2, 11), 5);

  EXPECT_EQ(compute_final_k_legacy(0, 4, 11), 3);
  EXPECT_EQ(compute_final_k_legacy(1, 4, 11), 3);
  EXPECT_EQ(compute_final_k_legacy(2, 4, 11), 3);
  EXPECT_EQ(compute_final_k_legacy(3, 4, 11), 2);

  EXPECT_EQ(compute_final_k_legacy(0, 8, 11), 2);
  EXPECT_EQ(compute_final_k_legacy(1, 8, 11), 1);
  EXPECT_EQ(compute_final_k_legacy(2, 8, 11), 2);
  EXPECT_EQ(compute_final_k_legacy(3, 8, 11), 1);
  EXPECT_EQ(compute_final_k_legacy(4, 8, 11), 2);
  EXPECT_EQ(compute_final_k_legacy(5, 8, 11), 1);
  EXPECT_EQ(compute_final_k_legacy(6, 8, 11), 1);
  EXPECT_EQ(compute_final_k_legacy(7, 8, 11), 1);
}

BlockID
compute_final_k_legacy_naive(const BlockID block, const BlockID current_k, const BlockID input_k) {
  if (current_k == 1) {
    return input_k;
  }
  if (current_k == input_k) {
    return 1;
  }

  if (block >= current_k / 2) {
    return compute_final_k_legacy(
        block - current_k / 2, current_k / 2, std::floor(1.0 * input_k / 2)
    );
  } else {
    return compute_final_k_legacy(block, current_k / 2, std::ceil(1.0 * input_k / 2));
  }
}

TEST(PartitionUtilsTest, TwoToTenThousendBlocksLegacy) {
  for (BlockID input_k = 2; input_k < 10'000; ++input_k) {
    for (BlockID current_k = 1; current_k < input_k; current_k *= 2) {
      for (BlockID block = 0; block < current_k; ++block) {
        EXPECT_EQ(
            compute_final_k_legacy_naive(block, current_k, input_k),
            compute_final_k_legacy(block, current_k, input_k)
        );
      }
    }
  }
}
} // namespace
