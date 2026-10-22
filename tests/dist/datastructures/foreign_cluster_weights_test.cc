#include <array>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <tbb/global_control.h>

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/datastructures/foreign_cluster_weights.h"

namespace kaminpar::dist {

TEST(ForeignClusterWeightsTest, ConcurrentSignedUpdatesPreserveEveryPayload) {
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 4);
  ForeignClusterWeights weights;
  weights.reset(3);
  weights.initialize_entry(0, 0, 0);
  weights.initialize_entry(1, 42, -7);
  weights.initialize_entry(2, GlobalNodeID{1} << 40, 11);
  weights.finish_initialization();

  constexpr int kWorkers = 4;
  constexpr int kIterations = 10000;
  std::atomic<bool> start{false};
  std::array<std::thread, kWorkers> workers;
  for (int worker = 0; worker < kWorkers; ++worker) {
    workers[worker] = std::thread([&, worker] {
      while (!start.load(std::memory_order_acquire)) {
      }
      for (int i = 0; i < kIterations; ++i) {
        weights.add(42, worker + 1);
        weights.subtract(GlobalNodeID{1} << 40, worker + 1);
        weights.add(0, weights.weight(42) >= -7 ? 1 : 0);
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }
  EXPECT_EQ(weights.weight(42), -7 + kIterations * 10);
  EXPECT_EQ(weights.weight(GlobalNodeID{1} << 40), 11 - kIterations * 10);
  EXPECT_EQ(weights.weight(0), kWorkers * kIterations);
}

TEST(ForeignClusterWeightsTest, JoinedGrowthDeduplicatesKeysAndKeepsHistoricalWeights) {
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 4);
  ForeignClusterWeights weights;
  weights.reset(1);
  weights.initialize_entry(0, 0, 19);
  weights.finish_initialization();
  EXPECT_EQ(weights.weight(0), 19); // Keep a cached lookup handle across actual Growt migration.

  // Exceeds the initial Growt table's fill threshold and repeats every candidate.
  constexpr GlobalNodeID kNewKeys = 50000;
  std::vector<GlobalNodeID> keys;
  keys.reserve(2 * kNewKeys);
  for (GlobalNodeID key = 1; key <= kNewKeys; ++key) {
    keys.push_back(key);
    keys.push_back(key);
  }
  weights.prepare_keys(keys);
  ASSERT_EQ(weights.size(), kNewKeys + 1);
  EXPECT_EQ(weights.weight(0), 19);
  for (GlobalNodeID key = 1; key <= kNewKeys; ++key) {
    EXPECT_EQ(weights.weight(key), 0);
    weights.add(key, static_cast<GlobalNodeWeight>(key % 7) - 3);
  }

  // A later empty/existing-only batch must neither duplicate nor erase invisible keys.
  weights.prepare_keys({});
  weights.prepare_keys(keys);
  EXPECT_EQ(weights.size(), kNewKeys + 1);
  for (GlobalNodeID key = 1; key <= kNewKeys; ++key) {
    EXPECT_EQ(weights.weight(key), static_cast<GlobalNodeWeight>(key % 7) - 3);
  }
  weights.subtract(0, 19);
  weights.prepare_keys(std::array<GlobalNodeID, 3>{0, GlobalNodeID{1} << 48, 0});
  weights.add(0, 5);
  EXPECT_EQ(weights.weight(0), 5);
  EXPECT_EQ(weights.weight(GlobalNodeID{1} << 48), 0);
  EXPECT_EQ(weights.size(), kNewKeys + 2);

  ForeignClusterWeights moved(std::move(weights));
  EXPECT_EQ(moved.weight(0), 5);
  weights.reset(1);
  weights.initialize_entry(0, 55, -9);
  weights.finish_initialization();
  EXPECT_EQ(weights.weight(55), -9);

  ForeignClusterWeights occupied;
  occupied.reset(1);
  occupied.initialize_entry(0, 77, 23);
  occupied.finish_initialization();
  EXPECT_EQ(occupied.weight(77), 23); // Cache a lookup handle into the destination's old table.
  occupied = std::move(moved);
  EXPECT_EQ(occupied.weight(0), 5);
  EXPECT_EQ(occupied.size(), kNewKeys + 2);

  moved.reset(1);
  moved.initialize_entry(0, 99, -2);
  moved.finish_initialization();
  EXPECT_EQ(moved.size(), 1u);
  EXPECT_EQ(moved.weight(99), -2);
}

TEST(ForeignClusterWeightsTest, MpiBatchesIntroduceAndReviveForeignIdsBetweenJoinedScans) {
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 4);
  const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
  ForeignClusterWeights weights;
  weights.reset(0);
  weights.finish_initialization();
  std::unordered_map<GlobalNodeID, GlobalNodeWeight> reference;

  for (int chunk = 0; chunk < 6; ++chunk) {
    // The second and third label suffixes are absent from the initial domain. The
    // fourth batch revives old IDs; all joins precede another capacity/index mutation.
    const GlobalNodeID label =
        GlobalNodeID{100000} * ((rank + chunk + 1) % size) + 7 + 17 * (chunk % 3);
    const GlobalNodeWeight delta = rank % 2 == 0 ? rank + 1 : -rank - 1;
    const auto labels = mpi::allgather(label, MPI_COMM_WORLD);
    const auto deltas = mpi::allgather(delta, MPI_COMM_WORLD);
    std::vector<GlobalNodeID> candidates(labels.begin(), labels.end());
    candidates.insert(candidates.end(), labels.begin(), labels.end());
    weights.prepare_keys(candidates);
    for (std::size_t i = 0; i < labels.size(); ++i) {
      reference[labels[i]] += 100 * deltas[i];
    }
    tbb::parallel_for<std::size_t>(0, 100 * labels.size(), [&](const std::size_t i) {
      weights.add(labels[i % labels.size()], deltas[i % deltas.size()]);
    });
    ASSERT_EQ(weights.size(), reference.size());
    for (const auto [key, value] : reference) {
      EXPECT_EQ(weights.weight(key), value);
    }
  }
}

} // namespace kaminpar::dist
