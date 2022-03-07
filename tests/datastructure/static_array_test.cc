#include "kaminpar/datastructure/static_array.h"
#include "tests.h"

#include <gmock/gmock.h>

using ::testing::Eq;

namespace kaminpar {
TEST(StaticArrayTest, SimpleStorageTest) {
  StaticArray<int> array(10);
  for (std::size_t i = 0; i < 10; ++i) {
    array[i] = 10 * i;
  }
  for (std::size_t i = 0; i < 10; ++i) {
    EXPECT_THAT(array[i], Eq(10 * i));
  }
  EXPECT_THAT(array.size(), Eq(10));
  EXPECT_FALSE(array.empty());
}

TEST(StaticArrayTest, IteratorTest) {
  StaticArray<int> array(10);
  for (std::size_t i = 0; i < 10; ++i) {
    array[i] = 10 * i;
  }
  std::size_t i{0};

  for (const int &v : array) {
    EXPECT_THAT(v, i * 10);
    ++i;
  }
}
} // namespace kaminpar
