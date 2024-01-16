#include <gmock/gmock.h>

#include "kaminpar-common/datastructures/dynamic_map.h"

using ::testing::ElementsAre;

namespace kaminpar {
TEST(DynamicMapTest, SmallInsertRetrieveNoGrowing) {
  DynamicFlatMap<int, int> map;

  map[0] = 0;
  map[100] = 1;
  map[200] = 2;

  EXPECT_EQ(map.size(), 3);

  EXPECT_TRUE(map.contains(0));
  EXPECT_EQ(map[0], 0);

  EXPECT_TRUE(map.contains(100));
  EXPECT_EQ(map[100], 1);

  EXPECT_TRUE(map.contains(200));
  EXPECT_EQ(map[200], 2);
}

TEST(DynamicMapTest, LargeInsertRetrieveWithGrowing) {
  constexpr int kNumElements = 200;

  DynamicFlatMap<int, int> map;

  for (int i = 0; i < kNumElements; ++i) {
    map[i * 100] = i;
  }

  EXPECT_EQ(map.size(), kNumElements);

  for (int i = 0; i < kNumElements; ++i) {
    // Test a few negative contains ...
    EXPECT_FALSE(map.contains(i * 100 - 2));
    EXPECT_FALSE(map.contains(i * 100 - 1));
    EXPECT_FALSE(map.contains(i * 100 + 1));
    EXPECT_FALSE(map.contains(i * 100 + 2));

    // ... before retrieving the actual value
    EXPECT_TRUE(map.contains(i * 100));
    EXPECT_EQ(map[i * 100], i);
  }
}

TEST(DynamicMapTest, ClearAfterFewInsertions) {
  DynamicFlatMap<int, int> map;

  map[0] = 0;
  map[100] = 1;
  map[200] = 2;

  EXPECT_EQ(map.size(), 3);

  map.clear();

  EXPECT_FALSE(map.contains(0));
  EXPECT_FALSE(map.contains(100));
  EXPECT_FALSE(map.contains(200));
}
} // namespace kaminpar
