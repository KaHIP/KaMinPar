#include <algorithm>
#include <array>
#include <barrier>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "kaminpar-shm/refinement/flow/util/buffered_vector.h"

namespace kaminpar::shm {

TEST(BufferedVectorTest, ConcurrentAppendsReserveDisjointSlots) {
  constexpr int kEntriesPerWorker = 2048;
  constexpr int kNumWorkers = 4;
  BufferedVector<int> entries;
  entries.reserve(kEntriesPerWorker * kNumWorkers);
  std::barrier start(kNumWorkers);
  std::vector<std::thread> workers;

  for (int worker = 0; worker < kNumWorkers; ++worker) {
    workers.emplace_back([&, worker] {
      start.arrive_and_wait();
      for (int i = 0; i < kEntriesPerWorker; i += 4) {
        const int first = worker * kEntriesPerWorker + i;
        entries.atomic_push_back(first);
        const std::array batch = {first + 1, first + 2, first + 3};
        entries.atomic_push_back(std::span<const int>(batch));
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }

  ASSERT_EQ(entries.size(), kEntriesPerWorker * kNumWorkers);
  std::vector<int> values(entries.view().begin(), entries.view().end());
  std::sort(values.begin(), values.end());
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(values[i], i);
  }
}

} // namespace kaminpar::shm
