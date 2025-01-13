#include <iterator>

#include <gmock/gmock.h>

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar {

#if __cplusplus >= 202002L
using StaticIntArray = StaticArray<int>;
static_assert(std::random_access_iterator<StaticIntArray::iterator>);
static_assert(std::contiguous_iterator<StaticIntArray::iterator>);
#endif

TEST(StaticArrayTest, SimpleStorageTest) {
  StaticArray<int> array(10);
  for (std::size_t i = 0; i < 10; ++i) {
    array[i] = 10 * i;
  }
  for (std::size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(array[i], 10 * i);
  }
  EXPECT_EQ(array.size(), 10);
  EXPECT_FALSE(array.empty());
}

TEST(StaticArrayTest, IteratorTest) {
  StaticArray<int> array(10);
  for (std::size_t i = 0; i < 10; ++i) {
    array[i] = 10 * i;
  }
  std::size_t i{0};

  for (const int &v : array) {
    EXPECT_EQ(v, i * 10);
    ++i;
  }
}

} // namespace kaminpar
