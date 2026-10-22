#include <barrier>
#include <thread>

#include <gtest/gtest.h>

#include "kaminpar-shm/refinement/fm/node_tracker.h"

namespace kaminpar::shm::fm {

TEST(NodeTrackerTest, OwnershipHandoffPublishesProtectedData) {
  NodeTracker tracker(1);
  int protected_value = 0;
  std::barrier start(2);

  auto worker = [&](const int id) {
    start.arrive_and_wait();
    for (int i = 0; i < 4000; ++i) {
      while (!tracker.lock(0, id)) {
        std::this_thread::yield();
      }
      ++protected_value;
      tracker.set(0, NodeTracker::UNLOCKED);
    }
  };

  std::thread first_worker(worker, 1);
  std::thread second_worker(worker, 2);
  first_worker.join();
  second_worker.join();

  EXPECT_EQ(protected_value, 8000);
  EXPECT_EQ(tracker.owner(0), NodeTracker::UNLOCKED);
}

} // namespace kaminpar::shm::fm
