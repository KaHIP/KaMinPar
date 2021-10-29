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
} // namespace dkaminpar::test