#include <algorithm>
#include <vector>

#include <gmock/gmock.h>

#include "kaminpar-common/datastructures/bitvector_rank.h"

namespace kaminpar {

TEST(RankCombinedBitVectorTest, RankCountsSetBitsBeforePosition) {
  constexpr std::size_t kLength = 1000;
  const std::vector<std::size_t> set_positions{0, 1, 63, 64, 497, 498, 511, 900, 999};

  RankCombinedBitVector<> bitvector(kLength, false);
  for (const std::size_t pos : set_positions) {
    bitvector.set(pos);
  }
  bitvector.update();

  for (std::size_t pos = 0; pos < kLength; ++pos) {
    const auto expected = static_cast<std::uint64_t>(
        std::count_if(set_positions.begin(), set_positions.end(), [&](const std::size_t set_pos) {
          return set_pos < pos;
        })
    );
    EXPECT_EQ(bitvector.rank(pos), expected) << "pos=" << pos;
  }
}

TEST(RankCombinedBitVectorTest, RankZeroDoesNotReadAnyBits) {
  RankCombinedBitVector<> bitvector(10, true);
  bitvector.update();

  EXPECT_EQ(bitvector.rank(0), 0);
}

} // namespace kaminpar
