#include <gmock/gmock.h>

#include "kaminpar-common/datastructures/compact_hash_map.h"

namespace kaminpar {
TEST(CompactHashMapTest, ObjectConstructionWorks) {
  std::vector<std::uint64_t> storage(16);
  CompactHashMap<std::uint64_t> ht(storage.data(), 16, 1);
}

TEST(CompactHashMapTest, SimpleIncreaseWithoutCollision) {
  std::vector<std::uint64_t> storage(4);
  CompactHashMap<std::uint64_t> ht(storage.data(), 4, 2);
  ht.increase_by(1, 1);
  ht.increase_by(2, 1);
  ht.increase_by(3, 1);
}

TEST(CompactHashMapTest, SimpleIncreaseWithCollision) {
  std::vector<std::uint64_t> storage(4);
  CompactHashMap<std::uint64_t> ht(storage.data(), 4, 3);
  ht.increase_by(0, 1);
  ht.increase_by(3, 1);
  ht.increase_by(6, 1);
}

TEST(CompactHashMapTest, SimpleIncreaseQueryWithoutCollision) {
  std::vector<std::uint64_t> storage(4);
  CompactHashMap<std::uint64_t> ht(storage.data(), 4, 2);
  ht.increase_by(1, 1);
  ht.increase_by(2, 4);
  ht.increase_by(3, 10);
  EXPECT_EQ(ht.get(1), 1);
  EXPECT_EQ(ht.get(2), 4);
  EXPECT_EQ(ht.get(3), 10);
}

TEST(CompactHashMapTest, SimpleIncreaseQueryWithCollision) {
  std::vector<std::uint64_t> storage(4);
  CompactHashMap<std::uint64_t> ht(storage.data(), 4, 3);
  ht.increase_by(0, 1);
  ht.increase_by(3, 4);
  ht.increase_by(6, 10);
  EXPECT_EQ(ht.get(0), 1);
  EXPECT_EQ(ht.get(3), 4);
  EXPECT_EQ(ht.get(6), 10);
}

TEST(CompactHashMapTest, SingleElementIncreaseDecreaseErase) {
  std::vector<std::uint64_t> storage(4);
  CompactHashMap<std::uint64_t> ht(storage.data(), 4, 2);
  ht.increase_by(1, 1);
  ht.increase_by(1, 4);
  ht.increase_by(1, 10);
  EXPECT_EQ(ht.get(1), 15);
  ht.decrease_by(1, 1);
  ht.decrease_by(1, 4);
  EXPECT_EQ(ht.get(1), 10);
  ht.decrease_by(1, 10);
  EXPECT_EQ(ht.get(1), 0);
}

TEST(CompactHashMapTest, InsertEraseSequence) {
  std::vector<std::uint64_t> storage(4);
  CompactHashMap<std::uint64_t> ht(storage.data(), 4, 4);
  ht.increase_by(7, 1);
  ht.increase_by(14, 1);
  ht.increase_by(6, 1);
  ht.decrease_by(6, 1);
  ht.increase_by(3, 1);
  ht.decrease_by(14, 1);
  EXPECT_EQ(ht.get(7), 1);
  EXPECT_EQ(ht.get(3), 1);
}

TEST(CompactHashMapTest, InsertEraseSequence2) {
  std::vector<std::uint64_t> storage(8);
  CompactHashMap<std::uint64_t> ht(storage.data(), 8, 4);
  ht.increase_by(2, 1);
  ht.increase_by(7, 1);
  ht.increase_by(0, 1);
  ht.increase_by(8, 1);
  ht.increase_by(15, 1);
  ht.increase_by(3, 1);
  ht.increase_by(3, 1);
  ht.decrease_by(8, 1);

  EXPECT_EQ(ht.get(2), 1);
  EXPECT_EQ(ht.get(7), 1);
  EXPECT_EQ(ht.get(0), 1);
  EXPECT_EQ(ht.get(15), 1);
  EXPECT_EQ(ht.get(3), 2);
  EXPECT_EQ(ht.get(8), 0);
}
} // namespace kaminpar
