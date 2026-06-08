#include <gmock/gmock.h>

#include "kaminpar-common/datastructures/fixed_size_sparse_map.h"
#include "kaminpar-common/datastructures/sparse_map.h"

namespace kaminpar {

TEST(SparseMapTest, AddRemoveAndLookupWork) {
  SparseMap<std::size_t, int> map(8);

  map.add(3, 30);
  map.add(5, 50);

  EXPECT_TRUE(map.contains(3));
  EXPECT_TRUE(map.contains(5));
  EXPECT_EQ(map.get(3), 30);
  EXPECT_EQ(map.get(5), 50);

  map.remove(3);

  EXPECT_FALSE(map.contains(3));
  EXPECT_TRUE(map.contains(5));
  EXPECT_EQ(map.size(), 1);
}

TEST(FixedSizeSparseMapTest, ExistingKeysAreQueryableWhenMapIsFull) {
  FixedSizeSparseMap<std::size_t, int, 4> map;

  map[0] = 10;
  map[1] = 20;
  map[2] = 30;
  map[3] = 40;

  EXPECT_EQ(map.size(), 4);
  EXPECT_TRUE(map.contains(0));
  EXPECT_TRUE(map.contains(1));
  EXPECT_TRUE(map.contains(2));
  EXPECT_TRUE(map.contains(3));
  EXPECT_EQ(map.get(0), 10);
  EXPECT_EQ(map.get(1), 20);
  EXPECT_EQ(map.get(2), 30);
  EXPECT_EQ(map.get(3), 40);

  map[2] += 1;
  EXPECT_EQ(map.get(2), 31);
}

TEST(FixedSizeSparseMapTest, MissingKeyIsNotContainedWhenMapIsFull) {
  FixedSizeSparseMap<std::size_t, int, 4> map;

  map[0] = 10;
  map[1] = 20;
  map[2] = 30;
  map[3] = 40;

  EXPECT_FALSE(map.contains(4));
}

TEST(FixedSizeSparseMapTest, MissingKeyIsNotContainedWhenMapIsFreed) {
  FixedSizeSparseMap<std::size_t, int, 4> map;
  map.freeInternalData();

  EXPECT_FALSE(map.contains(0));
}

} // namespace kaminpar
