#include <array>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "kaminpar-common/datastructures/compact_hash_map.h"

namespace kaminpar {
namespace {

template <typename T> class CompactHashMapConcurrentLookupTest : public ::testing::Test {};
using EntryWidths = ::testing::Types<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>;
TYPED_TEST_SUITE(CompactHashMapConcurrentLookupTest, EntryWidths);

TYPED_TEST(CompactHashMapConcurrentLookupTest, ReturnsOnlyTheValueOfTheObservedKey) {
  using Entry = TypeParam;
  std::array<Entry, 1> storage{};
  CompactHashMap<Entry, true> map(storage.data(), 1, 2);
  map.increase_by(Entry{1}, Entry{3});

  struct ReaderCounts {
    std::uint64_t reads = 0;
    std::uint64_t wrong_key1 = 0;
    std::uint64_t wrong_key2 = 0;
  };
  std::array<ReaderCounts, 3> counts{};
  std::atomic<bool> done{false};
  std::barrier start(4);
  std::vector<std::thread> readers;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    readers.emplace_back([&, i] {
      auto &local = counts[i];
      start.arrive_and_wait();
      do {
        for (unsigned batch = 0; batch < 1024; ++batch) {
          const Entry one = map.get(Entry{1});
          const Entry two = map.get(Entry{2});
          local.wrong_key1 += one != Entry{0} && one != Entry{3};
          local.wrong_key2 += two != Entry{0} && two != Entry{7};
          local.reads += 2;
        }
        std::this_thread::yield();
      } while (!done.load(std::memory_order_relaxed));
    });
  }

  // One writer uses only the actual public mutation API. No lookup hook or seam
  // controls the interleaving between the reader's key and value observations.
  start.arrive_and_wait();
  for (unsigned iteration = 0; iteration < 200000; ++iteration) {
    map.decrease_by(Entry{1}, Entry{3});
    map.increase_by(Entry{2}, Entry{7});
    map.decrease_by(Entry{2}, Entry{7});
    map.increase_by(Entry{1}, Entry{3});
    if (iteration % 256 == 0) {
      std::this_thread::yield();
    }
  }
  done.store(true, std::memory_order_relaxed);
  for (auto &reader : readers) {
    reader.join();
  }

  for (const auto &local : counts) {
    EXPECT_GT(local.reads, 0u);
    EXPECT_EQ(local.wrong_key1, 0u);
    EXPECT_EQ(local.wrong_key2, 0u);
  }
  EXPECT_EQ(map.get(Entry{1}), Entry{3});
  EXPECT_EQ(map.get(Entry{2}), Entry{0});
  EXPECT_EQ(map.count(), 1u);
}

} // namespace
} // namespace kaminpar
