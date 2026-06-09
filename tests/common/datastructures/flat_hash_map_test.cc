#include <cstdint>
#include <limits>
#include <vector>

#include <gmock/gmock.h>

#include "kaminpar-common/datastructures/flat_hash_map.h"

namespace kaminpar {

TEST(FlatHashMapTest, InsertRetrieveAndUpdate) {
  FlatHashMap<std::uint32_t, std::uint32_t> map;

  map[1] = 10;
  map[17] += 20;
  map[1] += 5;

  EXPECT_EQ(map.size(), 2);
  EXPECT_TRUE(map.contains(1));
  EXPECT_TRUE(map.contains(17));
  EXPECT_FALSE(map.contains(2));
  EXPECT_EQ(map.find(1)->second, 15);
  EXPECT_EQ(map.find(17)->second, 20);
}

TEST(FlatHashMapTest, GrowsAndKeepsEntriesIterable) {
  FlatHashMap<std::uint32_t, std::uint32_t> map;

  for (std::uint32_t key = 0; key < 1000; ++key) {
    map[key * 16] = key;
  }

  std::uint64_t sum = 0;
  for (const auto [key, value] : map.entries()) {
    EXPECT_EQ(value, key / 16);
    sum += value;
  }

  EXPECT_EQ(map.size(), 1000);
  EXPECT_EQ(sum, 999u * 1000u / 2u);
}

TEST(FlatHashMapTest, ClearRemovesEntriesButAllowsReuse) {
  FlatHashMap<std::uint32_t, std::uint32_t> map;

  map[1] = 10;
  map[2] = 20;
  map.clear();

  EXPECT_TRUE(map.empty());
  EXPECT_EQ(map.find(1), map.end());
  EXPECT_EQ(map.find(2), map.end());

  map[2] = 30;
  EXPECT_EQ(map.size(), 1);
  EXPECT_EQ(map.find(2)->second, 30);
}

TEST(FlatHashMapTest, SupportsValuesWithDestructors) {
  FlatHashMap<std::uint32_t, std::vector<std::uint32_t>> map;

  map[1].push_back(7);
  map[1].push_back(8);
  map[2].push_back(9);

  ASSERT_NE(map.find(1), map.end());
  EXPECT_THAT(map.find(1)->second, testing::ElementsAre(7, 8));

  map.clear();
  EXPECT_EQ(map.find(1), map.end());

  map[1].push_back(10);
  EXPECT_THAT(map.find(1)->second, testing::ElementsAre(10));
}

TEST(FlatHashMapTest, DoesNotReserveSentinelKeys) {
  FlatHashMap<std::uint32_t, std::uint32_t> map;
  constexpr std::uint32_t kMax = std::numeric_limits<std::uint32_t>::max();

  map[kMax] = 1;
  map[kMax - 1] = 2;

  EXPECT_EQ(map.find(kMax)->second, 1);
  EXPECT_EQ(map.find(kMax - 1)->second, 2);
}

} // namespace kaminpar
