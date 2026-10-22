#include <atomic>
#include <chrono>
#include <cstddef>
#include <limits>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "kaminpar-common/datastructures/multi_queue.h"

namespace kaminpar {

template <typename Queue> class MultiQueueTest : public ::testing::Test {};

using QueueTypes = ::testing::Types<
    MinMultiQueue<std::size_t, int>,
    MaxMultiQueue<std::size_t, int>,
    MinMultiQueue<std::size_t, float>,
    MaxMultiQueue<std::size_t, float>>;
TYPED_TEST_SUITE(MultiQueueTest, QueueTypes);

TYPED_TEST(MultiQueueTest, EmptyQueueReturnsNoHandle) {
  TypeParam queue;
  EXPECT_FALSE(queue.lock_pop_pq());
}

TYPED_TEST(MultiQueueTest, FindsSparselyPopulatedQueueAfterRandomSampling) {
  // With many queues, random sampling usually misses the sole item. The complete scan must
  // still find it for both comparator directions and for keys equal to the empty sentinel.
  using Key = decltype(std::declval<typename TypeParam::PQ>().peek_key());
  for (const Key key :
       {Key{7}, std::numeric_limits<Key>::lowest(), std::numeric_limits<Key>::max()}) {
    for (int seed = 0; seed < 8; ++seed) {
      TypeParam queue(1024, seed);
      auto push = queue.lock_push_pq();
      push->push(42, key);
      queue.unlock(std::move(push));

      auto pop = queue.lock_pop_pq();
      ASSERT_TRUE(pop) << "key=" << key << ", seed=" << seed;
      EXPECT_EQ(pop->peek_id(), 42u);
      EXPECT_EQ(pop->peek_key(), key);
      pop->pop();
      queue.unlock(std::move(pop));
      EXPECT_FALSE(queue.lock_pop_pq());
    }
  }
}

TYPED_TEST(MultiQueueTest, ResetAndFreeDiscardCachedQueueState) {
  TypeParam queue;
  auto push = queue.lock_push_pq();
  push->push(1, 1);
  queue.unlock(std::move(push));
  queue.reset();
  EXPECT_FALSE(queue.lock_pop_pq());
  queue.free();
  queue.reset(2);
  EXPECT_FALSE(queue.lock_pop_pq());

  push = queue.lock_push_pq();
  push->push(2, 2);
  queue.unlock(std::move(push));
  auto pop = queue.lock_pop_pq();
  ASSERT_TRUE(pop);
  EXPECT_EQ(pop->peek_id(), 2u);
  pop->pop();
  queue.unlock(std::move(pop));
  EXPECT_FALSE(queue.lock_pop_pq());
}

TYPED_TEST(MultiQueueTest, ConcurrentProducersAndConsumersPreserveEveryItem) {
  constexpr std::size_t kNumProducers = 4;
  constexpr std::size_t kNumConsumers = 4;
  constexpr std::size_t kItemsPerProducer = 2000;
  constexpr std::size_t kNumItems = kNumProducers * kItemsPerProducer;

  TypeParam queue(kNumProducers);
  std::atomic<bool> start{false};
  std::atomic<std::size_t> producers_done{0};
  std::atomic<std::size_t> consumed{0};
  std::atomic<std::size_t> invalid_ids{0};
  std::vector<unsigned int> visits(kNumItems);
  std::vector<std::thread> workers;

  for (std::size_t producer = 0; producer < kNumProducers; ++producer) {
    workers.emplace_back([&, producer] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (std::size_t i = 0; i < kItemsPerProducer; ++i) {
        const std::size_t id = producer * kItemsPerProducer + i;
        auto handle = queue.lock_push_pq();
        handle->push(id, static_cast<int>(id % 17) - 8);
        queue.unlock(std::move(handle));
      }
      producers_done.fetch_add(1, std::memory_order_release);
    });
  }
  for (std::size_t consumer = 0; consumer < kNumConsumers; ++consumer) {
    workers.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
      while (producers_done.load(std::memory_order_acquire) != kNumProducers ||
             consumed.load(std::memory_order_relaxed) < kNumItems) {
        auto handle = queue.lock_pop_pq();
        if (!handle) {
          if (std::chrono::steady_clock::now() >= deadline) {
            return;
          }
          std::this_thread::yield();
          continue;
        }
        const std::size_t id = handle->peek_id();
        handle->pop();
        queue.unlock(std::move(handle));
        if (id < kNumItems) {
          std::atomic_ref<unsigned int>(visits[id]).fetch_add(1, std::memory_order_relaxed);
        } else {
          invalid_ids.fetch_add(1, std::memory_order_relaxed);
        }
        consumed.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }

  EXPECT_EQ(invalid_ids.load(), 0u);
  EXPECT_EQ(consumed.load(), kNumItems);
  for (std::size_t id = 0; id < kNumItems; ++id) {
    EXPECT_EQ(visits[id], 1u) << "id=" << id;
  }
  EXPECT_FALSE(queue.lock_pop_pq());
}

} // namespace kaminpar
