/*******************************************************************************
 * @file:   distributed_math_test.cc
 *
 * @author: Daniel Seemaier
 * @date:   29.10.2021
 * @brief:  Unit tests for math utility functions only used for distributed
 * partitioning.
 ******************************************************************************/
#include <gmock/gmock.h>

#include "dkaminpar/utility/math.h"

using ::testing::AnyOf;
using ::testing::Eq;
using ::testing::Le;
using ::testing::Lt;
using ::testing::Pair;

namespace dkaminpar::test {
TEST(DistributedMathTest, TestLocalRangeComputationWithoutRemainer) {
  EXPECT_THAT(math::compute_local_range(10, 2, 0), Pair(Eq(0), Eq(5)));
  EXPECT_THAT(math::compute_local_range(10, 2, 1), Pair(Eq(5), Eq(10)));
}

TEST(DistributedMathTest, TestLocalRangeComputationWithRemainder) {
  EXPECT_THAT(math::compute_local_range(10, 3, 0), Pair(Eq(0), Eq(4)));
  EXPECT_THAT(math::compute_local_range(10, 3, 1), Pair(Eq(4), Eq(7)));
  EXPECT_THAT(math::compute_local_range(10, 3, 2), Pair(Eq(7), Eq(10)));
}

TEST(DistributedMathTest, TestLocalRangeComputationWithFewElements) {
  EXPECT_THAT(math::compute_local_range(3, 5, 0), Pair(Eq(0), Eq(1)));
  EXPECT_THAT(math::compute_local_range(3, 5, 1), Pair(Eq(1), Eq(2)));
  EXPECT_THAT(math::compute_local_range(3, 5, 2), Pair(Eq(2), Eq(3)));
  EXPECT_THAT(math::compute_local_range(3, 5, 3), Pair(Eq(3), Eq(3)));
  EXPECT_THAT(math::compute_local_range(3, 5, 4), Pair(Eq(3), Eq(3)));
}

TEST(DistributedMathTest, TestLocalRangeRankComptuationWithFewElements) {
  EXPECT_THAT(math::compute_local_range_rank(3, 5, 0), Eq(0));
  EXPECT_THAT(math::compute_local_range_rank(3, 5, 1), Eq(1));
  EXPECT_THAT(math::compute_local_range_rank(3, 5, 2), Eq(2));
}

TEST(DistributedMathTest, TestLocalRangeRankComputationWithoutRemainder) {
  EXPECT_THAT(math::compute_local_range_rank(10, 2, 0), Eq(0));
  EXPECT_THAT(math::compute_local_range_rank(10, 2, 1), Eq(0));
  EXPECT_THAT(math::compute_local_range_rank(10, 2, 2), Eq(0));
  EXPECT_THAT(math::compute_local_range_rank(10, 2, 3), Eq(0));
  EXPECT_THAT(math::compute_local_range_rank(10, 2, 4), Eq(0));

  EXPECT_THAT(math::compute_local_range_rank(10, 2, 5), Eq(1));
  EXPECT_THAT(math::compute_local_range_rank(10, 2, 6), Eq(1));
  EXPECT_THAT(math::compute_local_range_rank(10, 2, 7), Eq(1));
  EXPECT_THAT(math::compute_local_range_rank(10, 2, 8), Eq(1));
  EXPECT_THAT(math::compute_local_range_rank(10, 2, 9), Eq(1));
}

TEST(DistributedMathTest, TestLocalRangeRankComputationWithRemainder) {
  EXPECT_THAT(math::compute_local_range_rank(10, 3, 0), Eq(0));
  EXPECT_THAT(math::compute_local_range_rank(10, 3, 1), Eq(0));
  EXPECT_THAT(math::compute_local_range_rank(10, 3, 2), Eq(0));
  EXPECT_THAT(math::compute_local_range_rank(10, 3, 3), Eq(0));

  EXPECT_THAT(math::compute_local_range_rank(10, 3, 4), Eq(1));
  EXPECT_THAT(math::compute_local_range_rank(10, 3, 5), Eq(1));
  EXPECT_THAT(math::compute_local_range_rank(10, 3, 6), Eq(1));

  EXPECT_THAT(math::compute_local_range_rank(10, 3, 7), Eq(2));
  EXPECT_THAT(math::compute_local_range_rank(10, 3, 8), Eq(2));
  EXPECT_THAT(math::compute_local_range_rank(10, 3, 9), Eq(2));
}

TEST(DistributedMathTest, TestLocalRangeRankComputationWithRemainder2) {
  EXPECT_THAT(math::compute_local_range_rank(10, 6, 0), Eq(0));
  EXPECT_THAT(math::compute_local_range_rank(10, 6, 1), Eq(0));
  EXPECT_THAT(math::compute_local_range_rank(10, 6, 2), Eq(1));
  EXPECT_THAT(math::compute_local_range_rank(10, 6, 3), Eq(1));
  EXPECT_THAT(math::compute_local_range_rank(10, 6, 4), Eq(2));
  EXPECT_THAT(math::compute_local_range_rank(10, 6, 5), Eq(2));
  EXPECT_THAT(math::compute_local_range_rank(10, 6, 6), Eq(3));
  EXPECT_THAT(math::compute_local_range_rank(10, 6, 7), Eq(3));
  EXPECT_THAT(math::compute_local_range_rank(10, 6, 8), Eq(4));
  EXPECT_THAT(math::compute_local_range_rank(10, 6, 9), Eq(5));
}

TEST(DistributedMathTest, TestLocalRangeRankComputationWithRemainder_Exhaustive) {
  const int max_n = 100;

  for (int n = 1; n < max_n; ++n) {
    for (int size = 1; size <= n; ++size) {
      const int chunk = n / size;
      EXPECT_THAT(math::compute_local_range(n, size, 0).first, Eq(0));
      EXPECT_THAT(math::compute_local_range(n, size, size - 1).second, Eq(n));

      for (int pe = 0; pe < size; ++pe) {
        const auto [from, to] = math::compute_local_range(n, size, pe);
        EXPECT_THAT(to - from, AnyOf(Eq(chunk), Eq(chunk + 1)));

        for (int el = from; el < to; ++el) {
          EXPECT_THAT(math::compute_local_range_rank(n, size, el), Eq(pe));
        }
      }
    }
  }
}

TEST(DistributedMathTest, TestLocalRangeRankComputation_9_5) {
  EXPECT_THAT(math::compute_local_range_rank(9, 5, 0), Eq(0));
  EXPECT_THAT(math::compute_local_range_rank(9, 5, 1), Eq(0));
  EXPECT_THAT(math::compute_local_range_rank(9, 5, 2), Eq(1));
  EXPECT_THAT(math::compute_local_range_rank(9, 5, 3), Eq(1));
  EXPECT_THAT(math::compute_local_range_rank(9, 5, 4), Eq(2));
  EXPECT_THAT(math::compute_local_range_rank(9, 5, 5), Eq(2));
  EXPECT_THAT(math::compute_local_range_rank(9, 5, 6), Eq(3));
  EXPECT_THAT(math::compute_local_range_rank(9, 5, 7), Eq(3));
  EXPECT_THAT(math::compute_local_range_rank(9, 5, 8), Eq(4));
}

TEST(DistributedMathTest, Reg_TestLocalRangeComputation_ParHIPCrash) {
  // this set of arguments caused an overflow in ParHIP
  const std::uint32_t n = 1382867;
  const std::uint32_t size = 2048;

  const std::uint32_t chunk = n / size; // each range should have size chunk or chunk + 1
  for (std::uint32_t pe = 0; pe < size; ++pe) {
    const auto [from, to] = math::compute_local_range<std::uint32_t>(n, size, pe);
    EXPECT_THAT(to, Le(n));
    EXPECT_THAT(from, Lt(to));
    if (pe == 0) {
      EXPECT_THAT(from, Eq(0));
    } else if (pe + 1 == size) {
      EXPECT_THAT(to, Eq(n));
    }

    const std::uint32_t range_size = to - from;
    EXPECT_THAT(range_size, AnyOf(Eq(chunk), Eq(chunk + 1)));
  }
}

TEST(DistributedMathTest, Reg_7_3_Works) {
  EXPECT_THAT(math::compute_local_range_rank<std::uint64_t>(7, 3, 0), Eq(0));
  EXPECT_THAT(math::compute_local_range_rank<std::uint64_t>(7, 3, 1), Eq(0));
  EXPECT_THAT(math::compute_local_range_rank<std::uint64_t>(7, 3, 2), Eq(0));
  EXPECT_THAT(math::compute_local_range_rank<std::uint64_t>(7, 3, 3), Eq(1));
  EXPECT_THAT(math::compute_local_range_rank<std::uint64_t>(7, 3, 4), Eq(1));
  EXPECT_THAT(math::compute_local_range_rank<std::uint64_t>(7, 3, 5), Eq(2));
  EXPECT_THAT(math::compute_local_range_rank<std::uint64_t>(7, 3, 6), Eq(2));
}

TEST(DistributedMathTest, TestRoundRobinPermutation_1PerPEPerPE) {
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 4, 0), 0);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 4, 1), 4);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 4, 2), 8);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 4, 3), 12);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 4, 4), 1);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 4, 5), 5);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 4, 6), 9);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 4, 7), 13);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 4, 8), 2);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 4, 9), 6);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 4, 10), 10);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 4, 11), 14);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 4, 12), 3);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 4, 13), 7);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 4, 14), 11);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 4, 15), 15);
}

TEST(DistributedMathTest, TestRoundRobinPermutation_8PerPEPerPE) {
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 2, 0), 0);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 2, 1), 8);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 2, 2), 2);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 2, 3), 10);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 2, 4), 4);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 2, 5), 12);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 2, 6), 6);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 2, 7), 14);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 2, 8), 1);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 2, 9), 9);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 2, 10), 3);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 2, 11), 11);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 2, 12), 5);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 2, 13), 13);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 2, 14), 7);
  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 2, 15), 15);
}

//TEST(DistributedMathTest, TestRoundRobinPermutation_UnevenElementsPerPEPerPE) {
//  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 0), 0); // 0
//  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 1), 6); // 6
//  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 2), 12); // 11
//  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 3), 3); // 3
//  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 4), 9); // 9
//  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 5), 15); // 14
//  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 6), 1); // 1
//  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 7), 7); // 7
//  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 8), 13); // 12
//  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 9), 4); // 4
//  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 10), 10); // 10
//  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 11), 2); // 2
//  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 12), 8); // 8
//  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 13), 14); // 13
//  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 14), 5); // 5
//  EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 15), 11); // 11
//}
} // namespace dkaminpar::test