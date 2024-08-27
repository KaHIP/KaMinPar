#include <numeric>
#include <vector>

#include <gmock/gmock.h>

#include "kaminpar-common/parallel/aligned_prefix_sum.h"

using namespace kaminpar;

using ::testing::ElementsAre;

namespace {

TEST(ParallelAlignedPrefixSumTest, works_with_zero_elements) {
  std::vector<int> storage(0);
  const std::size_t result = parallel::aligned_prefix_sum(storage.begin(), storage.end(), [](auto) {
    return std::make_pair(0, 0);
  });

  EXPECT_EQ(result, 0);
}

TEST(ParallelAlignedPrefixSumTest, works_with_single_element) {
  std::vector<int> storage(1);
  const std::size_t result = parallel::aligned_prefix_sum(storage.begin(), storage.end(), [](auto) {
    return std::make_pair(0, 0);
  });

  EXPECT_EQ(result, 0);
  EXPECT_EQ(storage[0], 0);
}

TEST(ParallelAlignedPrefixSumTest, works_with_aligned_values) {
  std::vector<int> storage(10);
  const std::size_t result = parallel::aligned_prefix_sum(storage.begin(), storage.end(), [](auto) {
    return std::make_pair(2, 2);
  });

  EXPECT_EQ(result, 20);
  for (std::size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(storage[i], 2 * i);
  }
}

TEST(ParallelAlignedPrefixSumTest, works_with_unaligned_values_4) {
  std::vector<int> storage(10);
  const std::size_t result =
      parallel::aligned_prefix_sum(storage.begin(), storage.end(), [](const std::size_t i) {
        return std::make_pair(4, i);
      });

  for (std::size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(storage[i] % 4, 0);
  }
}

TEST(ParallelAlignedPrefixSumTest, works_with_unaligned_values_8) {
  std::vector<int> storage(20);
  const std::size_t result =
      parallel::aligned_prefix_sum(storage.begin(), storage.end(), [](const std::size_t i) {
        return std::make_pair(8, i);
      });

  for (std::size_t i = 0; i < 20; ++i) {
    EXPECT_EQ(storage[i] % 8, 0);
  }
}

TEST(ParallelAlignedPrefixSumTest, works_with_multiple_alignments) {
  std::vector<int> storage(20);
  const std::size_t result =
      parallel::aligned_prefix_sum(storage.begin(), storage.end(), [](const std::size_t i) {
        return std::make_pair(2 * i, i);
      });

  for (std::size_t i = 0; i < 20; ++i) {
    EXPECT_EQ(storage[i] % (2 * i), 0);
  }
}

} // namespace
