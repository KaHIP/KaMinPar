#include <gmock/gmock.h>

#include "kaminpar-common/datastructures/compact_static_array.h"

namespace kaminpar {

TEST(CompactStaticArrayTest, single_byte_test) {
  CompactStaticArray<std::uint32_t> array(1, 10);
  for (std::size_t i = 0; i < 10; ++i) {
    array.write(i, 10 * i);
  }
  for (std::size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(array[i], 10 * i);
  }
  EXPECT_EQ(array.size(), 10);
  EXPECT_FALSE(array.empty());
}

TEST(CompactStaticArrayTest, single_byte_iterator_test) {
  CompactStaticArray<std::uint32_t> array(1, 10);
  for (std::size_t i = 0; i < 10; ++i) {
    array.write(i, 10 * i);
  }

  std::size_t i = 0;
  for (const int &v : array) {
    EXPECT_EQ(v, i * 10);
    ++i;
  }
}

TEST(CompactStaticArrayTest, two_bytes_test) {
  CompactStaticArray<std::uint32_t> array(2, 10);
  for (std::size_t i = 0; i < 10; ++i) {
    array.write(i, 1024 * i);
  }
  for (std::size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(array[i], 1024 * i);
  }
  EXPECT_EQ(array.size(), 10);
  EXPECT_FALSE(array.empty());
}

TEST(CompactStaticArrayTest, postfix_iterator_increment_advances_iterator) {
  CompactStaticArray<std::uint32_t> array(1, 2);
  array.write(0, 7);
  array.write(1, 9);

  auto it = array.begin();
  auto old = it++;

  EXPECT_EQ(*old, 7);
  EXPECT_EQ(*it, 9);
}

TEST(CompactStaticArrayTest, postfix_iterator_decrement_advances_iterator) {
  CompactStaticArray<std::uint32_t> array(1, 2);
  array.write(0, 7);
  array.write(1, 9);

  auto it = array.end();
  auto old = it--;

  EXPECT_EQ(old, array.end());
  EXPECT_EQ(*it, 9);
}

} // namespace kaminpar
