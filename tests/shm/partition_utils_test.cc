#include <gtest/gtest.h>

#include "kaminpar-shm/partition_utils.h"

namespace kaminpar::shm::test {
TEST(PartitionUtilsTest, block_count_computation_works) {
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

TEST(PartitionUtilsTest, legacy_block_count_computation_works) {
  EXPECT_EQ(compute_final_k(0, 1, 11), 11);

  EXPECT_EQ(compute_final_k(0, 2, 11), 6);
  EXPECT_EQ(compute_final_k(1, 2, 11), 5);

  EXPECT_EQ(compute_final_k(0, 4, 11), 3);
  EXPECT_EQ(compute_final_k(1, 4, 11), 3);
  EXPECT_EQ(compute_final_k(2, 4, 11), 3);
  EXPECT_EQ(compute_final_k(3, 4, 11), 2);

  EXPECT_EQ(compute_final_k(0, 8, 11), 2);
  EXPECT_EQ(compute_final_k(1, 8, 11), 1);
  EXPECT_EQ(compute_final_k(2, 8, 11), 2);
  EXPECT_EQ(compute_final_k(3, 8, 11), 1);
  EXPECT_EQ(compute_final_k(4, 8, 11), 2);
  EXPECT_EQ(compute_final_k(5, 8, 11), 1);
  EXPECT_EQ(compute_final_k(6, 8, 11), 1);
  EXPECT_EQ(compute_final_k(7, 8, 11), 1);
}
} // namespace kaminpar::shm::test
