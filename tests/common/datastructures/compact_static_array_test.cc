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

} // namespace kaminpar
