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

TEST(CompactHashMapTest, FillCompletely) {
  std::vector<std::uint64_t> storage(4);
  CompactHashMap<std::uint64_t, true> ht(storage.data(), 4, 2);
  ht.increase_by(0, 1);
  ht.increase_by(1, 2);
  ht.increase_by(2, 3);
  ht.increase_by(3, 4);

  EXPECT_EQ(ht.get(0), 1);
  EXPECT_EQ(ht.get(1), 2);
  EXPECT_EQ(ht.get(2), 3);
  EXPECT_EQ(ht.get(3), 4);

  EXPECT_EQ(ht.count(), 4);
}

TEST(CompactHashMapTest, FillCompletelyWithCollisions) {
  std::vector<std::uint64_t> storage(4);
  CompactHashMap<std::uint64_t, true> ht(storage.data(), 4, 4);
  ht.increase_by(0, 1);
  ht.increase_by(4, 2);
  ht.increase_by(8, 3);
  ht.increase_by(12, 4);

  EXPECT_EQ(ht.get(0), 1);
  EXPECT_EQ(ht.get(4), 2);
  EXPECT_EQ(ht.get(8), 3);
  EXPECT_EQ(ht.get(12), 4);

  EXPECT_EQ(ht.count(), 4);
}

TEST(CompactHashMapTest, IncreaseOnFullMap) {
  std::vector<std::uint64_t> storage(4);
  CompactHashMap<std::uint64_t, true> ht(storage.data(), 4, 4);

  ht.increase_by(0, 1);
  ht.increase_by(1, 2);
  ht.increase_by(2, 3);
  ht.increase_by(5, 4);

  ht.increase_by(0, 1);
  ht.increase_by(1, 2);
  ht.increase_by(2, 3);
  ht.increase_by(5, 4);

  ht.increase_by(0, 1);
  ht.increase_by(1, 2);
  ht.increase_by(2, 3);
  ht.increase_by(5, 4);

  EXPECT_EQ(ht.get(0), 3);
  EXPECT_EQ(ht.get(1), 6);
  EXPECT_EQ(ht.get(2), 9);
  EXPECT_EQ(ht.get(5), 12);
}

TEST(CompactHashMapTest, RemoveFromFullMap) {
  std::vector<std::uint64_t> storage(4);
  CompactHashMap<std::uint64_t, true> ht(storage.data(), 4, 2);
  ht.increase_by(0, 1);
  ht.increase_by(1, 2);
  ht.increase_by(2, 3);
  ht.increase_by(3, 4);

  ht.decrease_by(0, 1);

  EXPECT_EQ(ht.get(1), 2);
  EXPECT_EQ(ht.get(2), 3);
  EXPECT_EQ(ht.get(3), 4);
}

TEST(CompactHashMapTest, RemoveFromFullMapWithCollisions) {
  std::vector<std::uint64_t> storage(4);
  CompactHashMap<std::uint64_t, true> ht(storage.data(), 4, 4);
  ht.increase_by(0, 1);
  ht.increase_by(4, 2);
  ht.increase_by(2, 3);
  ht.increase_by(6, 4);

  EXPECT_TRUE(ht.decrease_by(0, 1));

  EXPECT_EQ(ht.get(4), 2);
  EXPECT_EQ(ht.get(2), 3);
  EXPECT_EQ(ht.get(6), 4);

  EXPECT_EQ(ht.count(), 3);
}

TEST(CompactHashMapTest, RemoveFromFullMapAndReinsert) {
  std::vector<std::uint64_t> storage(4);
  CompactHashMap<std::uint64_t, true> ht(storage.data(), 4, 2);

  ht.increase_by(0, 1);
  ht.increase_by(1, 2);
  ht.increase_by(2, 3);
  ht.increase_by(3, 4);

  EXPECT_TRUE(ht.decrease_by(0, 1));
  EXPECT_EQ(ht.count(), 3);

  ht.increase_by(0, 5);

  EXPECT_EQ(ht.get(0), 5);
  EXPECT_EQ(ht.get(1), 2);
  EXPECT_EQ(ht.get(2), 3);
  EXPECT_EQ(ht.get(3), 4);
}

TEST(CompactHashMapTest, UnsuccessfulQueryOnFullMap) {
  std::vector<std::uint64_t> storage(4);
  CompactHashMap<std::uint64_t, true> ht(storage.data(), 4, 2);

  ht.increase_by(0, 1);
  ht.increase_by(1, 2);
  ht.increase_by(2, 3);
  ht.increase_by(3, 4);

  // We don't care about the value, but we should not crash or be stuck in an infinite loop
  [[maybe_unused]] auto _ = ht.get(5);
}

TEST(CompactHashMapTest, CountWorksOnEmptyMap) {
  std::vector<std::uint64_t> storage(4);
  CompactHashMap<std::uint64_t> ht(storage.data(), 4, 2);
  EXPECT_EQ(ht.capacity(), 4);
  EXPECT_EQ(ht.count(), 0);
}

TEST(CompactHashMapTest, CountWorksOnFullMap) {
  std::vector<std::uint64_t> storage(4);
  CompactHashMap<std::uint64_t, true> ht(storage.data(), 4, 5);
  EXPECT_EQ(ht.capacity(), 4);

  ht.increase_by(0, 1);
  ht.increase_by(5, 2);
  ht.increase_by(12, 3);
  ht.increase_by(18, 4);

  EXPECT_EQ(ht.count(), 4);
}

} // namespace kaminpar
