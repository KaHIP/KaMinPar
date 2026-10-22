#include <atomic>
#include <concepts>
#include <cstdint>
#include <thread>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "kaminpar-dist/datastructures/growt.h"

namespace kaminpar::dist {
namespace {

using Frozen = growt::StaticGhostNodeMapping;
using Builder = growt::MutableGhostNodeMapping;

template <typename T>
concept HasInsert = requires(T &map) { map.insert(1, 2); };
template <typename T>
concept HasHandle = requires(T &map) { map.get_handle(); };
template <typename T>
concept HasIterator = requires(T &map) { map.begin(); };
template <typename T>
concept CanFreezeLvalue = requires(T &map) { map.freeze(); };
static_assert(!HasInsert<Frozen> && !HasHandle<Frozen> && !HasIterator<Frozen>);
static_assert(!HasHandle<Builder> && !HasIterator<Builder> && !CanFreezeLvalue<Builder>);
static_assert(!std::is_copy_constructible_v<Frozen>);
static_assert(!std::is_copy_constructible_v<Builder>);
static_assert(
    std::same_as<decltype(std::declval<const Frozen &>().lookup(1)), std::optional<GlobalNodeID>>
);
static_assert(std::same_as<decltype(std::declval<Builder &&>().freeze()), Frozen>);

TEST(FrozenGhostMapTest, EmptyAndMoveOwnership) {
  Builder builder(0);
  auto frozen = std::move(builder).freeze();
  EXPECT_FALSE(frozen.contains(1));
  EXPECT_EQ(frozen.lookup(1), std::nullopt);
  const auto capacity = frozen.capacity();
  auto moved = std::move(frozen);
  EXPECT_EQ(moved.capacity(), capacity);
  EXPECT_FALSE(moved.contains(1));
  Frozen assigned;
  assigned = std::move(moved);
  EXPECT_FALSE(assigned.lookup(1).has_value());
}

TEST(FrozenGhostMapTest, LookupsReturnIndependentCopies) {
  Builder builder(8);
  ASSERT_TRUE(builder.insert(7, 0));
  ASSERT_TRUE(builder.insert(8, 123));
  auto frozen = std::move(builder).freeze();
  auto copy = frozen.lookup(8);
  ASSERT_TRUE(copy.has_value());
  *copy = 456;
  EXPECT_EQ(frozen.at(8), 123u);
  EXPECT_EQ(frozen.at(7), 0u);
  EXPECT_FALSE(frozen.lookup(9).has_value());
}

TEST(FrozenGhostMapTest, ConcurrentAtomicBuildThenConcurrentImmutableReads) {
  constexpr GlobalNodeID kItems = 4096;
  constexpr int kWorkers = 8;
  Builder builder(kItems);
  std::atomic<bool> valid{true};
  std::vector<std::thread> workers;
  for (int worker = 0; worker < kWorkers; ++worker) {
    workers.emplace_back([&, worker] {
      for (GlobalNodeID i = worker; i < kItems; i += kWorkers) {
        if (!builder.insert(i + 1, i * 17 + 3))
          valid.store(false, std::memory_order_relaxed);
      }
    });
  }
  for (auto &worker : workers)
    worker.join();
  // std::thread::join establishes the phase barrier required by freeze().
  const auto frozen = std::move(builder).freeze();
  workers.clear();
  for (int worker = 0; worker < kWorkers; ++worker) {
    workers.emplace_back([&, worker] {
      for (int repeat = 0; repeat < 16; ++repeat) {
        for (GlobalNodeID i = worker; i < kItems; i += kWorkers) {
          const auto value = frozen.lookup(i + 1);
          if (!value || *value != i * 17 + 3)
            valid.store(false, std::memory_order_relaxed);
        }
        if (frozen.contains(kItems + 1))
          valid.store(false, std::memory_order_relaxed);
      }
    });
  }
  for (auto &worker : workers)
    worker.join();
  EXPECT_TRUE(valid.load(std::memory_order_relaxed));
}

} // namespace
} // namespace kaminpar::dist
