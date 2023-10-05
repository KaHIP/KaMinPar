#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "kaminpar-common/datastructures/preallocated_vector.h"

using namespace kaminpar;

TEST(PreallocatedAllocatorTest, view_other_vector) {
  constexpr std::size_t N = 10;

  std::vector<int> storage(N);
  std::iota(storage.begin(), storage.end(), 0);

  auto view = make_preallocated_vector(storage.data(), N);

  // Check initial state
  EXPECT_EQ(view.size(), N);
  for (std::size_t i = 0; i < storage.size(); ++i) {
    EXPECT_EQ(view[i], i);
  }

  // Make modification to underlaying vector
  std::swap(storage.front(), storage.back());

  EXPECT_EQ(view.front(), N - 1);
  for (std::size_t i = 1; i + 1 < storage.size(); ++i) {
    EXPECT_EQ(view[i], i);
  }
  EXPECT_EQ(view.back(), 0);
}
