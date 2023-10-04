#include <gmock/gmock.h>

#include "kaminpar-common/math.h"

using ::testing::AnyOf;
using ::testing::Eq;
using ::testing::Pair;

namespace kaminpar {
TEST(MathTest, FloorLog2) {
  EXPECT_EQ(0, math::floor_log2(1u));
  EXPECT_EQ(1, math::floor_log2(2u));
  EXPECT_EQ(1, math::floor_log2(3u));
  EXPECT_EQ(2, math::floor_log2(4u));
  EXPECT_EQ(2, math::floor_log2(5u));
  EXPECT_EQ(2, math::floor_log2(6u));
  EXPECT_EQ(2, math::floor_log2(7u));
  EXPECT_EQ(3, math::floor_log2(8u));
  EXPECT_EQ(4, math::floor_log2(16u));
}

TEST(MathTest, CeilLog2) {
  EXPECT_EQ(0, math::ceil_log2(1u));
  EXPECT_EQ(1, math::ceil_log2(2u));
  EXPECT_EQ(2, math::ceil_log2(3u));
  EXPECT_EQ(2, math::ceil_log2(4u));
  EXPECT_EQ(3, math::ceil_log2(5u));
  EXPECT_EQ(3, math::ceil_log2(6u));
  EXPECT_EQ(3, math::ceil_log2(7u));
  EXPECT_EQ(3, math::ceil_log2(8u));
}

TEST(MathTest, SplitNumberEvenlyWorks) {
  {
    const auto [k0, k1] = math::split_integral(0);
    EXPECT_EQ(k0, 0);
    EXPECT_EQ(k1, 0);
  }
  {
    const auto [k0, k1] = math::split_integral(1);
    EXPECT_THAT(k0, AnyOf(0, 1));
    EXPECT_THAT(k1, AnyOf(0, 1));
    EXPECT_NE(k0, k1);
  }
  {
    const auto [k0, k1] = math::split_integral(2);
    EXPECT_EQ(k0, 1);
    EXPECT_EQ(k1, 1);
  }
}

TEST(MathTest, SplitNumberWithRatioWorks) {
  {
    const auto [k0, k1] = math::split_integral(0, 0.1);
    EXPECT_EQ(k0, 0);
    EXPECT_EQ(k1, 0);
  }
  {
    const auto [k0, k1] = math::split_integral(1, 0.0);
    EXPECT_EQ(k0, 0);
    EXPECT_EQ(k1, 1);
  }
  {
    const auto [k0, k1] = math::split_integral(1, 1.0);
    EXPECT_EQ(k0, 1);
    EXPECT_EQ(k1, 0);
  }
  {
    const auto [k0, k1] = math::split_integral(10, 0.1);
    EXPECT_EQ(k0, 1);
    EXPECT_EQ(k1, 9);
  }
}

TEST(MathTest, RoundDownToPowerOfTwo) {
  EXPECT_EQ(math::floor2(1u), 1);
  EXPECT_EQ(math::floor2(2u), 2);
  EXPECT_EQ(math::floor2(3u), 2);
  EXPECT_EQ(math::floor2(4u), 4);
  EXPECT_EQ(math::floor2(5u), 4);
  EXPECT_EQ(math::floor2(6u), 4);
  EXPECT_EQ(math::floor2(7u), 4);
  EXPECT_EQ(math::floor2(8u), 8);
  EXPECT_EQ(math::floor2(1023u), 512);
  EXPECT_EQ(math::floor2(1024u), 1024);
  EXPECT_EQ(math::floor2(1025u), 1024);
}

TEST(MathTest, RoundUpToPowerOfTwo) {
  EXPECT_EQ(math::ceil2(1u), 1);
  EXPECT_EQ(math::ceil2(2u), 2);
  EXPECT_EQ(math::ceil2(3u), 4);
  EXPECT_EQ(math::ceil2(4u), 4);
  EXPECT_EQ(math::ceil2(5u), 8);
  EXPECT_EQ(math::ceil2(6u), 8);
  EXPECT_EQ(math::ceil2(7u), 8);
  EXPECT_EQ(math::ceil2(8u), 8);
  EXPECT_EQ(math::ceil2(1023u), 1024);
  EXPECT_EQ(math::ceil2(1024u), 1024);
  EXPECT_EQ(math::ceil2(1025u), 2048);
}

TEST(MathTest, TestLocalRangeComputationWithoutRemainer) {
  EXPECT_THAT(math::compute_local_range(10, 2, 0), Pair(Eq(0), Eq(5)));
  EXPECT_THAT(math::compute_local_range(10, 2, 1), Pair(Eq(5), Eq(10)));
}

TEST(MathTest, TestLocalRangeComputationWithRemainder) {
  EXPECT_THAT(math::compute_local_range(10, 3, 0), Pair(Eq(0), Eq(4)));
  EXPECT_THAT(math::compute_local_range(10, 3, 1), Pair(Eq(4), Eq(7)));
  EXPECT_THAT(math::compute_local_range(10, 3, 2), Pair(Eq(7), Eq(10)));
}

TEST(MathTest, TestLocalRangeComputationWithFewElements) {
  EXPECT_THAT(math::compute_local_range(3, 5, 0), Pair(Eq(0), Eq(1)));
  EXPECT_THAT(math::compute_local_range(3, 5, 1), Pair(Eq(1), Eq(2)));
  EXPECT_THAT(math::compute_local_range(3, 5, 2), Pair(Eq(2), Eq(3)));
  EXPECT_THAT(math::compute_local_range(3, 5, 3), Pair(Eq(3), Eq(3)));
  EXPECT_THAT(math::compute_local_range(3, 5, 4), Pair(Eq(3), Eq(3)));
}

TEST(MathTest, TestLocalRangeRankComptuationWithFewElements) {
  EXPECT_EQ(math::compute_local_range_rank(3, 5, 0), 0);
  EXPECT_EQ(math::compute_local_range_rank(3, 5, 1), 1);
  EXPECT_EQ(math::compute_local_range_rank(3, 5, 2), 2);
}

TEST(MathTest, TestLocalRangeRankComputationWithoutRemainder) {
  EXPECT_EQ(math::compute_local_range_rank(10, 2, 0), 0);
  EXPECT_EQ(math::compute_local_range_rank(10, 2, 1), 0);
  EXPECT_EQ(math::compute_local_range_rank(10, 2, 2), 0);
  EXPECT_EQ(math::compute_local_range_rank(10, 2, 3), 0);
  EXPECT_EQ(math::compute_local_range_rank(10, 2, 4), 0);

  EXPECT_EQ(math::compute_local_range_rank(10, 2, 5), 1);
  EXPECT_EQ(math::compute_local_range_rank(10, 2, 6), 1);
  EXPECT_EQ(math::compute_local_range_rank(10, 2, 7), 1);
  EXPECT_EQ(math::compute_local_range_rank(10, 2, 8), 1);
  EXPECT_EQ(math::compute_local_range_rank(10, 2, 9), 1);
}

TEST(MathTest, TestLocalRangeRankComputationWithRemainder) {
  EXPECT_EQ(math::compute_local_range_rank(10, 3, 0), 0);
  EXPECT_EQ(math::compute_local_range_rank(10, 3, 1), 0);
  EXPECT_EQ(math::compute_local_range_rank(10, 3, 2), 0);
  EXPECT_EQ(math::compute_local_range_rank(10, 3, 3), 0);

  EXPECT_EQ(math::compute_local_range_rank(10, 3, 4), 1);
  EXPECT_EQ(math::compute_local_range_rank(10, 3, 5), 1);
  EXPECT_EQ(math::compute_local_range_rank(10, 3, 6), 1);

  EXPECT_EQ(math::compute_local_range_rank(10, 3, 7), 2);
  EXPECT_EQ(math::compute_local_range_rank(10, 3, 8), 2);
  EXPECT_EQ(math::compute_local_range_rank(10, 3, 9), 2);
}

TEST(MathTest, TestLocalRangeRankComputationWithRemainder2) {
  EXPECT_EQ(math::compute_local_range_rank(10, 6, 0), 0);
  EXPECT_EQ(math::compute_local_range_rank(10, 6, 1), 0);
  EXPECT_EQ(math::compute_local_range_rank(10, 6, 2), 1);
  EXPECT_EQ(math::compute_local_range_rank(10, 6, 3), 1);
  EXPECT_EQ(math::compute_local_range_rank(10, 6, 4), 2);
  EXPECT_EQ(math::compute_local_range_rank(10, 6, 5), 2);
  EXPECT_EQ(math::compute_local_range_rank(10, 6, 6), 3);
  EXPECT_EQ(math::compute_local_range_rank(10, 6, 7), 3);
  EXPECT_EQ(math::compute_local_range_rank(10, 6, 8), 4);
  EXPECT_EQ(math::compute_local_range_rank(10, 6, 9), 5);
}

TEST(MathTest, TestLocalRangeRankComputationWithRemainder_Exhaustive) {
  const int max_n = 100;

  for (int n = 1; n < max_n; ++n) {
    for (int size = 1; size <= n; ++size) {
      const int chunk = n / size;
      EXPECT_EQ(math::compute_local_range(n, size, 0).first, 0);
      EXPECT_EQ(math::compute_local_range(n, size, size - 1).second, n);

      for (int pe = 0; pe < size; ++pe) {
        const auto [from, to] = math::compute_local_range(n, size, pe);
        EXPECT_THAT(to - from, AnyOf(Eq(chunk), Eq(chunk + 1)));

        for (int el = from; el < to; ++el) {
          EXPECT_EQ(math::compute_local_range_rank(n, size, el), pe);
        }
      }
    }
  }
}

TEST(MathTest, TestLocalRangeRankComputation_9_5) {
  EXPECT_EQ(math::compute_local_range_rank(9, 5, 0), 0);
  EXPECT_EQ(math::compute_local_range_rank(9, 5, 1), 0);
  EXPECT_EQ(math::compute_local_range_rank(9, 5, 2), 1);
  EXPECT_EQ(math::compute_local_range_rank(9, 5, 3), 1);
  EXPECT_EQ(math::compute_local_range_rank(9, 5, 4), 2);
  EXPECT_EQ(math::compute_local_range_rank(9, 5, 5), 2);
  EXPECT_EQ(math::compute_local_range_rank(9, 5, 6), 3);
  EXPECT_EQ(math::compute_local_range_rank(9, 5, 7), 3);
  EXPECT_EQ(math::compute_local_range_rank(9, 5, 8), 4);
}

TEST(MathTest, Reg_TestLocalRangeComputation_ParHIPCrash) {
  // this set of arguments caused an overflow in ParHIP
  const std::uint32_t n = 1382867;
  const std::uint32_t size = 2048;

  const std::uint32_t chunk = n / size; // each range should have size chunk or chunk + 1
  for (std::uint32_t pe = 0; pe < size; ++pe) {
    const auto [from, to] = math::compute_local_range<std::uint32_t>(n, size, pe);
    EXPECT_LE(to, n);
    EXPECT_LT(from, to);
    if (pe == 0) {
      EXPECT_EQ(from, 0);
    } else if (pe + 1 == size) {
      EXPECT_EQ(to, n);
    }

    const std::uint32_t range_size = to - from;
    EXPECT_THAT(range_size, AnyOf(Eq(chunk), Eq(chunk + 1)));
  }
}

TEST(MathTest, Reg_7_3_Works) {
  EXPECT_EQ(math::compute_local_range_rank<std::uint64_t>(7, 3, 0), 0);
  EXPECT_EQ(math::compute_local_range_rank<std::uint64_t>(7, 3, 1), 0);
  EXPECT_EQ(math::compute_local_range_rank<std::uint64_t>(7, 3, 2), 0);
  EXPECT_EQ(math::compute_local_range_rank<std::uint64_t>(7, 3, 3), 1);
  EXPECT_EQ(math::compute_local_range_rank<std::uint64_t>(7, 3, 4), 1);
  EXPECT_EQ(math::compute_local_range_rank<std::uint64_t>(7, 3, 5), 2);
  EXPECT_EQ(math::compute_local_range_rank<std::uint64_t>(7, 3, 6), 2);
}

TEST(MathTest, TestRoundRobinPermutation_1PerPEPerPE) {
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

TEST(MathTest, TestRoundRobinPermutation_8PerPEPerPE) {
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

// TEST(MathTest, TestRoundRobinPermutation_UnevenElementsPerPEPerPE) {
//   EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 0), 0); // 0
//   EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 1), 6); // 6
//   EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 2), 12); // 11
//   EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 3), 3); // 3
//   EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 4), 9); // 9
//   EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 5), 15); // 14
//   EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 6), 1); // 1
//   EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 7), 7); // 7
//   EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 8), 13); // 12
//   EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 9), 4); // 4
//   EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 10), 10); //
//   10 EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 11), 2); //
//   2 EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 12), 8); //
//   8 EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 13), 14); //
//   13 EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 14), 5); //
//   5 EXPECT_EQ(math::distribute_round_robin<std::uint64_t>(16, 3, 15), 11); //
//   11
// }

TEST(MathTest, encode_4x4_grid) {
  // 0  1  2  3
  // 4  5  6  7
  // 8  9  10 11
  // 12 13 14 15
  EXPECT_EQ(math::encode_grid_position(0, 0, 4), 0);
  EXPECT_EQ(math::encode_grid_position(0, 1, 4), 1);
  EXPECT_EQ(math::encode_grid_position(0, 2, 4), 2);
  EXPECT_EQ(math::encode_grid_position(0, 3, 4), 3);
  EXPECT_EQ(math::encode_grid_position(1, 0, 4), 4);
  EXPECT_EQ(math::encode_grid_position(1, 1, 4), 5);
  EXPECT_EQ(math::encode_grid_position(1, 2, 4), 6);
  EXPECT_EQ(math::encode_grid_position(1, 3, 4), 7);
  EXPECT_EQ(math::encode_grid_position(2, 0, 4), 8);
  EXPECT_EQ(math::encode_grid_position(2, 1, 4), 9);
  EXPECT_EQ(math::encode_grid_position(2, 2, 4), 10);
  EXPECT_EQ(math::encode_grid_position(2, 3, 4), 11);
  EXPECT_EQ(math::encode_grid_position(3, 0, 4), 12);
  EXPECT_EQ(math::encode_grid_position(3, 1, 4), 13);
  EXPECT_EQ(math::encode_grid_position(3, 2, 4), 14);
  EXPECT_EQ(math::encode_grid_position(3, 3, 4), 15);
}

TEST(MathTest, decode_4x4_grid) {
  // 0  1  2  3
  // 4  5  6  7
  // 8  9  10 11
  // 12 13 14 15
  EXPECT_EQ(math::decode_grid_position(0, 4), std::make_pair(0, 0));
  EXPECT_EQ(math::decode_grid_position(1, 4), std::make_pair(0, 1));
  EXPECT_EQ(math::decode_grid_position(2, 4), std::make_pair(0, 2));
  EXPECT_EQ(math::decode_grid_position(3, 4), std::make_pair(0, 3));
  EXPECT_EQ(math::decode_grid_position(4, 4), std::make_pair(1, 0));
  EXPECT_EQ(math::decode_grid_position(5, 4), std::make_pair(1, 1));
  EXPECT_EQ(math::decode_grid_position(6, 4), std::make_pair(1, 2));
  EXPECT_EQ(math::decode_grid_position(7, 4), std::make_pair(1, 3));
  EXPECT_EQ(math::decode_grid_position(8, 4), std::make_pair(2, 0));
  EXPECT_EQ(math::decode_grid_position(9, 4), std::make_pair(2, 1));
  EXPECT_EQ(math::decode_grid_position(10, 4), std::make_pair(2, 2));
  EXPECT_EQ(math::decode_grid_position(11, 4), std::make_pair(2, 3));
  EXPECT_EQ(math::decode_grid_position(12, 4), std::make_pair(3, 0));
  EXPECT_EQ(math::decode_grid_position(13, 4), std::make_pair(3, 1));
  EXPECT_EQ(math::decode_grid_position(14, 4), std::make_pair(3, 2));
  EXPECT_EQ(math::decode_grid_position(15, 4), std::make_pair(3, 3));
}
} // namespace kaminpar
