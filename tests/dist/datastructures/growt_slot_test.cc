#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include <data-structures/element_types/simple_slot.hpp>
#include <gtest/gtest.h>

namespace kaminpar::dist {

template <bool Markable> void test_concurrent_slot_snapshots() {
  using Slot = ::growt::simple_slot<std::uint64_t, std::uint64_t, Markable>;
  using Snapshot = typename Slot::slot_type;
  typename Slot::atomic_slot_type slot(Snapshot(std::uint64_t{1}, ~std::uint64_t{1}));
  std::atomic<bool> start{false};
  std::atomic<std::uint64_t> next_key{2};
  std::atomic<std::size_t> inconsistent_snapshots{0};
  std::vector<std::thread> workers;

  for (int writer = 0; writer < 2; ++writer) {
    workers.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < 20000; ++i) {
        const std::uint64_t key = next_key.fetch_add(1, std::memory_order_relaxed);
        Snapshot expected = slot.load();
        const Snapshot goal(key, ~key);
        while (!slot.cas(expected, goal)) {
          expected = slot.load();
        }
      }
    });
  }
  for (int reader = 0; reader < 2; ++reader) {
    workers.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < 40000; ++i) {
        const Snapshot snapshot = slot.load();
        if (snapshot.get_mapped() != ~snapshot.get_key()) {
          inconsistent_snapshots.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }
  EXPECT_EQ(inconsistent_snapshots.load(), 0u);
}

TEST(GrowtSlotTest, AtomicSnapshotsStayCoherentDuringConcurrentUpdates) {
  test_concurrent_slot_snapshots<false>();
}

TEST(GrowtSlotTest, MarkableAtomicSnapshotsStayCoherentDuringConcurrentUpdates) {
  test_concurrent_slot_snapshots<true>();
}

} // namespace kaminpar::dist
