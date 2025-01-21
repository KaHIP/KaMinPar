#include <vector>

#include <gmock/gmock.h>
#include <tbb/global_control.h>

#include "kaminpar-common/parallel/aligned_prefix_sum.h"

namespace {
using namespace kaminpar;

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

  for (std::size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(storage[i] % 2, 0);
  }
  EXPECT_GE(result, storage.back() + 2);
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
  EXPECT_GE(result, storage.back() + storage.size() - 1);
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
  EXPECT_GE(result, storage.back() + storage.size() - 1);
}

TEST(ParallelAlignedPrefixSumTest, works_with_multiple_alignments) {
  std::vector<int> storage(20);
  const std::size_t result =
      parallel::aligned_prefix_sum(storage.begin(), storage.end(), [](const std::size_t i) {
        return std::make_pair(1 << (i % 4), i);
      });

  for (std::size_t i = 0; i < 20; ++i) {
    EXPECT_EQ(storage[i] % (1 << (i % 4)), 0);
  }

  EXPECT_GE(result, storage.back() + storage.size() - 1);
}

} // namespace
