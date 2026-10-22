#include <array>
#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "kaminpar-dist/distributed_label_propagation.h"

namespace kaminpar::dist {

TEST(OwnedRelaxedClusterWeightVectorTest, ConcurrentMovesConserveClusterWeights) {
  constexpr NodeID kWorkers = 4;
  constexpr NodeWeight kMaxWeight = 8;
  OwnedRelaxedClusterWeightVector<NodeID, NodeWeight> weights;
  weights.allocate_cluster_weights(kWorkers + 1);
  for (NodeID cluster = 0; cluster <= kWorkers; ++cluster) {
    weights.init_cluster_weight(cluster, 1);
  }

  std::atomic<bool> start{false};
  std::array<bool, kWorkers> failed{};
  std::vector<std::thread> workers;
  for (NodeID worker = 0; worker < kWorkers; ++worker) {
    workers.emplace_back([&, worker] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      // Each worker owns one source cluster and repeatedly transfers its unit
      // through the shared hub. Hub capacity checks overlap other workers' moves.
      const NodeID source = worker + 1;
      for (int iteration = 0; iteration < 50000; ++iteration) {
        if (!weights.move_cluster_weight(source, 0, 1, kMaxWeight) ||
            !weights.move_cluster_weight(0, source, 1, kMaxWeight)) {
          failed[worker] = true;
          break;
        }
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }

  NodeWeight total_weight = 0;
  for (NodeID cluster = 0; cluster <= kWorkers; ++cluster) {
    const auto weight = weights.cluster_weight(cluster);
    EXPECT_EQ(weight, 1u);
    EXPECT_LE(weight, kMaxWeight);
    total_weight += weight;
  }
  EXPECT_EQ(total_weight, kWorkers + 1);
  for (const bool move_failed : failed) {
    EXPECT_FALSE(move_failed);
  }
}

} // namespace kaminpar::dist
