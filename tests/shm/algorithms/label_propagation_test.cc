#include <array>
#include <atomic>
#include <cstdint>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/algorithms/iteration_order.h"
#include "kaminpar-shm/algorithms/label_propagation/active_set.h"
#include "kaminpar-shm/algorithms/label_propagation/adaptive_rating_map.h"
#include "kaminpar-shm/algorithms/label_propagation/balanced_selector.h"
#include "kaminpar-shm/algorithms/label_propagation/clustering_selector.h"
#include "kaminpar-shm/algorithms/label_propagation/fixed_capacity_rating_map.h"
#include "kaminpar-shm/algorithms/label_propagation/linear_rating_map.h"
#include "kaminpar-shm/algorithms/label_propagation/neighborhood_ratings.h"
#include "kaminpar-shm/algorithms/label_propagation/parallel_rating_map.h"
#include "kaminpar-shm/algorithms/label_propagation/positive_gain_selector.h"
#include "kaminpar-shm/algorithms/label_propagation/rating_map_pool.h"
#include "kaminpar-shm/algorithms/label_propagation/uniform_tie_set.h"

#include "kaminpar-common/datastructures/fixed_size_sparse_map.h"

namespace kaminpar::shm::testing {

namespace {

class CoverageKernel {
public:
  explicit CoverageKernel(std::vector<std::atomic<int>> &hits, const bool stop = false)
      : _hits(hits),
        _stop(stop) {}

  [[nodiscard]] bool should_stop() const {
    return _stop;
  }

  class Local {
  public:
    Local(std::vector<std::atomic<int>> &hits, std::atomic<int> &stop_checks)
        : _hits(hits),
          _stop_checks(stop_checks) {}

    bool operator()(const NodeID u) {
      _hits[u].fetch_add(1, std::memory_order_relaxed);
      return true;
    }

    [[nodiscard]] bool should_stop(const NodeID) {
      _stop_checks.fetch_add(1, std::memory_order_relaxed);
      return false;
    }

    void finish() {}

  private:
    std::vector<std::atomic<int>> &_hits;
    std::atomic<int> &_stop_checks;
  };

  [[nodiscard]] Local make_local(Random &) {
    return Local(_hits, _stop_checks);
  }

  [[nodiscard]] int stop_checks() const {
    return _stop_checks.load(std::memory_order_relaxed);
  }

private:
  std::vector<std::atomic<int>> &_hits;
  bool _stop;
  std::atomic<int> _stop_checks{0};
};

} // namespace

TEST(IterationOrderTest, InOrderVisitsTheRequestedRangeExactlyOnce) {
  constexpr NodeID kNumNodes = 128;
  std::vector<std::atomic<int>> hits(kNumNodes);
  for (auto &hit : hits) {
    hit.store(0);
  }
  InOrderIterationOrder order;
  order.initialize(kNumNodes, 17, 103);
  CoverageKernel kernel(hits);

  order.for_each(kernel);

  for (NodeID u = 0; u < kNumNodes; ++u) {
    EXPECT_EQ(hits[u].load(), 17 <= u && u < 103 ? 1 : 0);
  }
  EXPECT_GT(kernel.stop_checks(), 0);
}

TEST(IterationOrderTest, ChunkShuffledVisitsEveryNonIsolatedNodeExactlyOnce) {
  Graph graph = make_path_graph(256);
  std::vector<std::atomic<int>> hits(graph.n());
  for (auto &hit : hits) {
    hit.store(0);
  }
  ChunkShuffledIterationOrder::Permutations permutations;
  ChunkShuffledIterationOrder order(permutations);
  order.initialize(graph.csr_graph());
  CoverageKernel kernel(hits);

  order.for_each(kernel);

  for (const NodeID u : graph.csr_graph().nodes()) {
    EXPECT_EQ(hits[u].load(), 1);
  }
  EXPECT_EQ(kernel.stop_checks(), 0);
}

TEST(IterationOrderTest, StopRequestIsCheckedBeforeStartingWorkUnits) {
  std::vector<std::atomic<int>> hits(64);
  for (auto &hit : hits) {
    hit.store(0);
  }
  InOrderIterationOrder order;
  order.initialize(hits.size());
  CoverageKernel kernel(hits, true);

  order.for_each(kernel);

  for (const auto &hit : hits) {
    EXPECT_EQ(hit.load(), 0);
  }
}

TEST(ActiveSetTest, SupportsParallelAlgorithmLifecycle) {
  lp::ActiveSet active;
  active.reset(4);

  EXPECT_TRUE(active.contains(2));
  active.deactivate(2);
  EXPECT_FALSE(active.contains(2));
  active.activate(2);
  EXPECT_TRUE(active.contains(2));
}

TEST(ClusteringStateViewTest, MutationsAreVisibleInTheOwningState) {
  Graph graph = make_graph({0, 1, 2}, {1, 0});
  StaticArray<NodeID> clustering(graph.n(), static_array::noinit);
  lp::ClusteringState state;
  state.set_max_cluster_weight(2);
  state.reset(clustering, graph.csr_graph());
  lp::ClusteringStateView view = state.view();

  view.deactivate(0);
  EXPECT_FALSE(view.is_active(0));

  const lp::MoveResult result = view.commit(graph.csr_graph(), 0, 0, 1, 1);
  EXPECT_TRUE(result.moved);
  EXPECT_TRUE(result.emptied_cluster);
  EXPECT_EQ(state.cluster(0), 1);
  EXPECT_EQ(state.cluster_weight(0), 0);
  EXPECT_EQ(state.cluster_weight(1), 2);
}

TEST(NeighborhoodRatingsTest, AccumulatesRatingsByLabel) {
  Graph graph = make_graph({0, 2, 3, 4}, {1, 2, 0, 0}, {1, 1, 1}, {2, 3, 2, 3});
  FixedSizeSparseMap<NodeID, EdgeWeight, 128> ratings;
  const std::vector<NodeID> labels = {0, 7, 7};

  lp::NeighborhoodRatings::accumulate(
      graph.csr_graph(),
      0,
      ratings,
      [&](const NodeID v) { return labels[v]; },
      [](const NodeID) { return true; }
  );

  EXPECT_EQ(ratings.size(), 1);
  EXPECT_EQ(ratings[7], 5);
}

TEST(NeighborhoodRatingsTest, StopsAtTheDistinctLabelCapacity) {
  Graph graph = make_graph({0, 2, 3, 4}, {1, 2, 0, 0}, {1, 1, 1}, {2, 3, 2, 3});
  FixedSizeSparseMap<NodeID, EdgeWeight, 128> ratings;
  const std::vector<NodeID> labels = {0, 1, 2};

  const bool full = lp::NeighborhoodRatings::accumulate_until_full(
      graph.csr_graph(),
      0,
      ratings,
      1,
      [&](const NodeID v) { return labels[v]; },
      [](const NodeID) { return true; }
  );

  EXPECT_TRUE(full);
  EXPECT_EQ(ratings.size(), 1);
}

TEST(NeighborhoodRatingsTest, AccumulatesWithoutCapacityChecksWhenTheBoundFits) {
  Graph graph = make_graph({0, 2, 3, 4}, {1, 2, 0, 0}, {1, 1, 1}, {2, 3, 2, 3});
  lp::LinearRatingMap<NodeID, EdgeWeight, 2> ratings;
  const std::vector<NodeID> labels = {0, 7, 7};

  const bool full = lp::NeighborhoodRatings::accumulate_with_capacity(
      graph.csr_graph(),
      0,
      ratings,
      1,
      ratings.capacity(),
      [&](const NodeID v) { return labels[v]; },
      [](const NodeID) { return true; }
  );

  EXPECT_FALSE(full);
  EXPECT_EQ(ratings.size(), 1);
  EXPECT_EQ(ratings.get(7), 5);
}

TEST(NeighborhoodRatingsTest, ChecksTheActualDistinctLabelsWhenTheBoundCanFillTheMap) {
  Graph graph = make_graph({0, 2, 3, 4}, {1, 2, 0, 0}, {1, 1, 1}, {2, 3, 2, 3});
  lp::LinearRatingMap<NodeID, EdgeWeight, 2> ratings;
  const std::vector<NodeID> labels = {0, 7, 7};

  const bool full = lp::NeighborhoodRatings::accumulate_with_capacity(
      graph.csr_graph(),
      0,
      ratings,
      2,
      ratings.capacity(),
      [&](const NodeID v) { return labels[v]; },
      [](const NodeID) { return true; }
  );

  EXPECT_FALSE(full);
  EXPECT_EQ(ratings.size(), 1);
  EXPECT_EQ(ratings.get(7), 5);
}

TEST(NeighborhoodRatingsTest, StopsWhenTheCapacityIsActuallyReached) {
  Graph graph = make_graph({0, 2, 3, 4}, {1, 2, 0, 0}, {1, 1, 1}, {2, 3, 2, 3});
  lp::LinearRatingMap<NodeID, EdgeWeight, 1> ratings;
  const std::vector<NodeID> labels = {0, 1, 2};

  const bool full = lp::NeighborhoodRatings::accumulate_with_capacity(
      graph.csr_graph(),
      0,
      ratings,
      2,
      ratings.capacity(),
      [&](const NodeID v) { return labels[v]; },
      [](const NodeID) { return true; }
  );

  EXPECT_TRUE(full);
  EXPECT_EQ(ratings.size(), 1);
}

TEST(NeighborhoodRatingsTest, UsesTheFixedCapacityAccumulatorUntilFull) {
  Graph graph = make_graph({0, 2, 3, 4}, {1, 2, 0, 0}, {1, 1, 1}, {2, 3, 2, 3});
  lp::FixedCapacityRatingMap<NodeID, EdgeWeight, 4> ratings;
  const std::vector<NodeID> labels = {0, 1, 2};

  const bool full = lp::NeighborhoodRatings::accumulate_until_full(
      graph.csr_graph(),
      0,
      ratings,
      1,
      [&](const NodeID v) { return labels[v]; },
      [](const NodeID) { return true; }
  );

  EXPECT_TRUE(full);
  EXPECT_EQ(ratings.size(), 1);
  EXPECT_EQ(ratings.get(1), 2);
}

TEST(LinearRatingMapTest, AggregatesInFirstInsertionOrderAndResets) {
  lp::LinearRatingMap<NodeID, EdgeWeight, 4> ratings(3);

  ratings[7] += 2;
  ratings[2] += 4;
  ratings[7] += 1;

  EXPECT_EQ(ratings.size(), 2);
  EXPECT_EQ(ratings.capacity(), 4);
  EXPECT_TRUE(ratings.contains(7));
  EXPECT_FALSE(ratings.contains(9));
  EXPECT_EQ(ratings.get(7), 6);
  EXPECT_EQ(ratings.get(2), 7);

  std::vector<std::pair<NodeID, EdgeWeight>> entries;
  for (const auto [key, value] : ratings.entries()) {
    entries.emplace_back(key, value);
  }
  const std::vector<std::pair<NodeID, EdgeWeight>> expected = {{7, 6}, {2, 7}};
  EXPECT_EQ(entries, expected);

  ratings.clear();
  EXPECT_EQ(ratings.size(), 0);
  EXPECT_FALSE(ratings.contains(7));

  ratings[2] += 5;
  EXPECT_EQ(ratings.size(), 1);
  EXPECT_EQ(ratings.get(2), 8);
}

TEST(FixedCapacityRatingMapTest, HandlesCollisionsAndPreservesInsertionOrder) {
  lp::FixedCapacityRatingMap<NodeID, EdgeWeight, 16> ratings(3);
  ratings.set_capacity(5);

  EXPECT_EQ(ratings.capacity(), 8);

  // With the production hash and capacity 8, these keys have the same initial slot.
  ratings[1] += 2;
  ratings[9] += 4;
  ratings[17] += 1;
  ratings[9] += 3;

  EXPECT_EQ(ratings.size(), 3);
  EXPECT_EQ(ratings.get(1), 5);
  EXPECT_EQ(ratings.get(9), 10);
  EXPECT_EQ(ratings.get(17), 4);
  EXPECT_FALSE(ratings.contains(25));

  std::vector<std::pair<NodeID, EdgeWeight>> entries;
  for (const auto [key, value] : ratings.entries()) {
    entries.emplace_back(key, value);
  }
  const std::vector<std::pair<NodeID, EdgeWeight>> expected = {{1, 5}, {9, 10}, {17, 4}};
  EXPECT_EQ(entries, expected);

  ratings.clear();
  EXPECT_EQ(ratings.size(), 0);
  EXPECT_FALSE(ratings.contains(1));

  ratings.set_capacity(1);
  EXPECT_EQ(ratings.capacity(), 2);

  ratings.set_capacity(4);
  ratings[4] = 11;
  EXPECT_EQ(ratings.capacity(), 4);
  EXPECT_EQ(ratings.get(4), 11);

  ratings.clear();
  ratings.set_capacity(16);
  EXPECT_FALSE(ratings.contains(1));
  EXPECT_FALSE(ratings.contains(4));
  ratings[9] = -3;
  EXPECT_EQ(ratings.get(9), -3);
}

TEST(FixedCapacityRatingMapTest, ReportsMissingKeysWhenTheTableIsFull) {
  lp::FixedCapacityRatingMap<NodeID, EdgeWeight, 4> ratings;

  ratings[0] = 1;
  ratings[1] = 2;
  ratings[2] = 3;
  ratings[3] = 4;

  EXPECT_EQ(ratings.size(), ratings.capacity());
  EXPECT_FALSE(ratings.contains(99));
}

TEST(FixedCapacityRatingMapTest, AccumulatorAddsRatingsAndTracksSize) {
  lp::FixedCapacityRatingMap<NodeID, EdgeWeight, 8> ratings(3);
  auto accumulator = ratings.accumulator();
  accumulator.add(2, 4);
  accumulator.add(7, 5);
  accumulator.add(2, 6);

  EXPECT_EQ(accumulator.size(), 2);
  EXPECT_EQ(ratings.size(), 2);
  EXPECT_EQ(ratings.get(2), 13);
  EXPECT_EQ(ratings.get(7), 8);
}

TEST(AdaptiveRatingMapTest, SelectsProductionMapsAtTheTierBoundaries) {
  using AdaptiveMap = lp::AdaptiveRatingMap<NodeID, EdgeWeight, lp::adaptive_rating_map::SparseMap>;
  using LinearMap = typename AdaptiveMap::LinearMap;
  using HashMap = typename AdaptiveMap::HashMap;
  using DirectMap = lp::adaptive_rating_map::SparseMap<NodeID, EdgeWeight>;

  enum class SelectedMap {
    kLinear,
    kHash,
    kDirect
  };

  AdaptiveMap ratings(20000);
  const auto inspect = [&](const std::size_t upper_bound) {
    std::pair<SelectedMap, std::size_t> result;
    ratings.execute(upper_bound, [&](auto &map) {
      using Map = std::remove_cvref_t<decltype(map)>;
      if constexpr (std::is_same_v<Map, LinearMap>) {
        result.first = SelectedMap::kLinear;
      } else if constexpr (std::is_same_v<Map, HashMap>) {
        result.first = SelectedMap::kHash;
      } else {
        static_assert(std::is_same_v<Map, DirectMap>);
        result.first = SelectedMap::kDirect;
      }
      result.second = map.capacity();
      map.clear();
    });
    return result;
  };

  EXPECT_EQ(inspect(AdaptiveMap::kLinearMapCapacity).first, SelectedMap::kLinear);

  const auto first_hash = inspect(AdaptiveMap::kLinearMapCapacity + 1);
  EXPECT_EQ(first_hash.first, SelectedMap::kHash);
  EXPECT_EQ(first_hash.second, AdaptiveMap::kMinHashCapacity);

  const auto larger_hash = inspect(42);
  EXPECT_EQ(larger_hash.first, SelectedMap::kHash);
  EXPECT_EQ(larger_hash.second, 512);

  const auto largest_hash = inspect(AdaptiveMap::kMaxHashSize);
  EXPECT_EQ(largest_hash.first, SelectedMap::kHash);
  EXPECT_EQ(largest_hash.second, AdaptiveMap::kMaxHashCapacity);

  const auto direct = inspect(AdaptiveMap::kMaxHashSize + 1);
  EXPECT_EQ(direct.first, SelectedMap::kDirect);
  EXPECT_EQ(direct.second, ratings.max_size());
}

TEST(AdaptiveRatingMapTest, UsesTheDirectMapForSmallKeyUniverses) {
  using AdaptiveMap = lp::AdaptiveRatingMap<NodeID, EdgeWeight, lp::adaptive_rating_map::SparseMap>;
  using DirectMap = lp::adaptive_rating_map::SparseMap<NodeID, EdgeWeight>;

  AdaptiveMap ratings(64);
  EXPECT_FALSE(ratings.hash_map().is_allocated());

  bool used_direct_map = false;
  ratings.execute(1, [&](auto &map) {
    used_direct_map = std::is_same_v<std::remove_cvref_t<decltype(map)>, DirectMap>;
    EXPECT_EQ(map.capacity(), ratings.max_size());
    map.clear();
  });

  EXPECT_TRUE(used_direct_map);
  EXPECT_FALSE(ratings.hash_map().is_allocated());
}

TEST(AdaptiveRatingMapTest, SwitchesAtTheDirectMapUniverseBoundary) {
  using AdaptiveMap = lp::AdaptiveRatingMap<NodeID, EdgeWeight, lp::adaptive_rating_map::SparseMap>;
  using HashMap = typename AdaptiveMap::HashMap;
  using DirectMap = lp::adaptive_rating_map::SparseMap<NodeID, EdgeWeight>;
  constexpr std::size_t kUpperBound = AdaptiveMap::kLinearMapCapacity + 1;

  AdaptiveMap at_boundary(AdaptiveMap::kDirectMapThreshold);
  at_boundary.execute(kUpperBound, [&](auto &map) {
    EXPECT_TRUE((std::is_same_v<std::remove_cvref_t<decltype(map)>, DirectMap>));
    map.clear();
  });
  EXPECT_FALSE(at_boundary.hash_map().is_allocated());

  AdaptiveMap above_boundary(AdaptiveMap::kDirectMapThreshold + 1);
  above_boundary.execute(kUpperBound, [&](auto &map) {
    EXPECT_TRUE((std::is_same_v<std::remove_cvref_t<decltype(map)>, HashMap>));
    EXPECT_EQ(map.capacity(), AdaptiveMap::kMinHashCapacity);
    map.clear();
  });
}

TEST(AdaptiveRatingMapTest, ChangesRepresentationWhenTheKeyUniverseChanges) {
  using AdaptiveMap = lp::AdaptiveRatingMap<NodeID, EdgeWeight, lp::adaptive_rating_map::SparseMap>;
  using HashMap = typename AdaptiveMap::HashMap;
  using DirectMap = lp::adaptive_rating_map::SparseMap<NodeID, EdgeWeight>;
  constexpr std::size_t kUpperBound = AdaptiveMap::kLinearMapCapacity + 1;

  AdaptiveMap ratings(AdaptiveMap::kDirectMapThreshold);
  ratings.change_max_size(AdaptiveMap::kDirectMapThreshold + 1);
  ratings.execute(kUpperBound, [&](auto &map) {
    EXPECT_TRUE((std::is_same_v<std::remove_cvref_t<decltype(map)>, HashMap>));
    map.clear();
  });
  EXPECT_TRUE(ratings.hash_map().is_allocated());

  ratings.change_max_size(AdaptiveMap::kDirectMapThreshold);
  ratings.execute(kUpperBound, [&](auto &map) {
    EXPECT_TRUE((std::is_same_v<std::remove_cvref_t<decltype(map)>, DirectMap>));
    EXPECT_EQ(map.capacity(), AdaptiveMap::kDirectMapThreshold);
    map.clear();
  });

  AdaptiveMap hash_only(AdaptiveMap::kDirectMapThreshold + 1);
  hash_only.change_max_size(AdaptiveMap::kDirectMapThreshold);
  hash_only.execute(kUpperBound, [&](auto &map) {
    EXPECT_TRUE((std::is_same_v<std::remove_cvref_t<decltype(map)>, DirectMap>));
    EXPECT_EQ(map.capacity(), AdaptiveMap::kDirectMapThreshold);
    map.clear();
  });
}

TEST(RatingMapPoolTest, ReusesAdaptiveMapsAcrossCapacityChanges) {
  lp::RatingMapPool<NodeID, EdgeWeight> maps;
  maps.ensure_capacity(64);
  EXPECT_EQ(maps.local().ratings.max_size(), 64);

  maps.ensure_capacity(16);
  EXPECT_EQ(maps.local().ratings.max_size(), 16);

  maps.ensure_capacity(128);
  EXPECT_EQ(maps.local().ratings.max_size(), 128);
}

TEST(RatingMapPoolTest, NewWorkersObserveCapacityShrinks) {
  using RatingMaps = lp::RatingMapPool<NodeID, EdgeWeight>;
  using RatingMap = typename RatingMaps::RatingMap;
  using DirectMap = lp::adaptive_rating_map::FastResetArray<NodeID, EdgeWeight>;
  RatingMaps maps;
  maps.ensure_capacity(RatingMap::kDirectMapThreshold + 1);
  EXPECT_EQ(maps.local().ratings.max_size(), RatingMap::kDirectMapThreshold + 1);

  maps.ensure_capacity(RatingMap::kDirectMapThreshold);
  std::size_t worker_capacity = 0;
  bool worker_uses_direct_map = false;
  std::thread worker([&] {
    auto &ratings = maps.local().ratings;
    worker_capacity = ratings.max_size();
    ratings.execute(RatingMap::kLinearMapCapacity + 1, [&](auto &map) {
      worker_uses_direct_map = std::is_same_v<std::remove_cvref_t<decltype(map)>, DirectMap>;
      map.clear();
    });
  });
  worker.join();

  EXPECT_EQ(worker_capacity, RatingMap::kDirectMapThreshold);
  EXPECT_TRUE(worker_uses_direct_map);
}

TEST(ParallelRatingMapTest, FlushesAndCombinesHighDegreeRatings) {
  constexpr NodeID kNumNeighbors = lp::ParallelRatingMap<NodeID, EdgeWeight>::kFlushThreshold + 1;
  Graph graph = make_star_graph(kNumNeighbors);
  lp::RatingMapPool<NodeID, EdgeWeight> local_maps;
  local_maps.ensure_capacity(graph.n());
  lp::ParallelRatingMap<NodeID, EdgeWeight> ratings;
  ratings.ensure_capacity(graph.n());

  ratings.accumulate(
      graph.csr_graph(),
      0,
      local_maps,
      [](const NodeID v) { return v; },
      [](const NodeID) { return true; }
  );

  std::vector<std::atomic<int>> hits(graph.n());
  for (auto &hit : hits) {
    hit.store(0);
  }
  std::atomic<bool> valid_entries = true;
  ratings.for_each_partition_and_reset([&](const std::size_t, const auto &entries) {
    for (const auto [key, rating] : entries) {
      if (key == 0 || key >= hits.size() || rating != 1) {
        valid_entries.store(false, std::memory_order_relaxed);
        continue;
      }
      hits[key].fetch_add(1, std::memory_order_relaxed);
    }
  });

  EXPECT_TRUE(valid_entries.load(std::memory_order_relaxed));
  EXPECT_EQ(hits[0].load(std::memory_order_relaxed), 0);
  for (NodeID u = 1; u < graph.n(); ++u) {
    EXPECT_EQ(hits[u].load(std::memory_order_relaxed), 1);
  }

  std::atomic<std::size_t> remaining_entries = 0;
  ratings.for_each_partition_and_reset([&](const std::size_t, const auto &entries) {
    for ([[maybe_unused]] const auto &entry : entries) {
      remaining_entries.fetch_add(1, std::memory_order_relaxed);
    }
  });
  EXPECT_EQ(remaining_entries.load(std::memory_order_relaxed), 0);
}

TEST(UniformTieSetTest, PreservesTheEstablishedFallbackSemantics) {
  ScalableVector<NodeID> storage;
  lp::UniformTieSet ties(storage);
  Random &random = Random::instance();

  ties.add(4);
  EXPECT_EQ(ties.select_or(2, random), 2);

  ties.add(7);
  const NodeID selected = ties.select_or(2, random);
  EXPECT_TRUE(selected == 4 || selected == 7);
}

TEST(PositiveGainSelectorTest, DoesNotInsertAMissingSourceIntoAFullMap) {
  using Gain = std::int64_t;
  lp::LinearRatingMap<BlockID, Gain, 8> ratings;
  for (BlockID block = 1; block <= 8; ++block) {
    ratings[block] = block;
  }
  ScalableVector<BlockID> ties;
  Random random;
  random.reinit(123);

  const auto [target, gain] =
      lp::select_best_positive_gain<Gain>(BlockID{0}, ratings.entries(), random, ties);

  EXPECT_EQ(target, 8);
  EXPECT_EQ(gain, 8);
  EXPECT_EQ(ratings.size(), 8);
  EXPECT_FALSE(ratings.contains(0));
  EXPECT_TRUE(ties.empty());
}

TEST(PositiveGainSelectorTest, ComputesGainWhenTheSourceEntryAppearsLast) {
  using Gain = std::int64_t;
  lp::LinearRatingMap<BlockID, Gain, 4> ratings;
  ratings[1] = 6;
  ratings[2] = 7;
  ratings[0] = 5;
  ScalableVector<BlockID> ties;
  Random random;
  random.reinit(123);

  EXPECT_EQ(
      lp::select_best_positive_gain<Gain>(BlockID{0}, ratings.entries(), random, ties),
      (std::pair<BlockID, Gain>{2, 2})
  );
  EXPECT_TRUE(ties.empty());
}

TEST(PositiveGainSelectorTest, DoesNotDrawRandomnessWithoutATie) {
  using Gain = std::int64_t;
  ScalableVector<BlockID> ties;
  Random random;
  Random unchanged;
  random.reinit(123);
  unchanged.reinit(123);

  lp::LinearRatingMap<BlockID, Gain, 2> non_positive;
  non_positive[1] = 4;
  non_positive[0] = 4;
  EXPECT_EQ(
      lp::select_best_positive_gain<Gain>(BlockID{0}, non_positive.entries(), random, ties),
      (std::pair<BlockID, Gain>{0, 0})
  );
  EXPECT_EQ(random.generator(), unchanged.generator());
  EXPECT_TRUE(ties.empty());

  lp::LinearRatingMap<BlockID, Gain, 2> unique_positive;
  unique_positive[1] = 6;
  unique_positive[0] = 4;
  EXPECT_EQ(
      lp::select_best_positive_gain<Gain>(BlockID{0}, unique_positive.entries(), random, ties),
      (std::pair<BlockID, Gain>{1, 2})
  );
  EXPECT_EQ(random.generator(), unchanged.generator());
  EXPECT_TRUE(ties.empty());
}

TEST(PositiveGainSelectorTest, ReturnsTheSourceWithoutAForeignTarget) {
  using Gain = std::int64_t;
  lp::LinearRatingMap<BlockID, Gain, 1> ratings;
  ratings[0] = 9;
  ScalableVector<BlockID> ties;
  Random random;
  Random unchanged;
  random.reinit(123);
  unchanged.reinit(123);

  EXPECT_EQ(
      lp::select_best_positive_gain<Gain>(BlockID{0}, ratings.entries(), random, ties),
      (std::pair<BlockID, Gain>{0, 0})
  );
  EXPECT_EQ(random.generator(), unchanged.generator());
  EXPECT_TRUE(ties.empty());
}

TEST(PositiveGainSelectorTest, BreaksPositiveTiesInInsertionOrderWithOneDraw) {
  using Gain = std::int64_t;
  lp::LinearRatingMap<BlockID, Gain, 4> ratings;
  ratings[4] = 8;
  ratings[7] = 8;
  ratings[0] = 3;
  ScalableVector<BlockID> ties;
  Random random;
  Random expected_random;
  random.reinit(789);
  expected_random.reinit(789);
  const std::array<BlockID, 2> expected_targets = {4, 7};
  const BlockID expected_target =
      expected_targets[expected_random.random_index(0, expected_targets.size())];

  EXPECT_EQ(
      lp::select_best_positive_gain<Gain>(BlockID{0}, ratings.entries(), random, ties),
      (std::pair<BlockID, Gain>{expected_target, 5})
  );
  EXPECT_EQ(random.generator(), expected_random.generator());
  EXPECT_TRUE(ties.empty());
}

TEST(ClusteringSelectorTest, SelectsAndCommitsTheStrongestFeasibleCluster) {
  Graph graph = make_graph({0, 1, 2}, {1, 0});
  StaticArray<NodeID> clustering(graph.n(), static_array::noinit);
  lp::ClusteringState state;
  state.set_max_cluster_weight(2);
  state.reset(clustering, graph.csr_graph());

  FixedSizeSparseMap<NodeID, EdgeWeight, 128> ratings;
  ratings[1] = 5;
  ScalableVector<NodeID> ties;
  ScalableVector<NodeID> favored_ties;
  lp::ClusteringSelector selector(state.view());
  const auto selection =
      selector.select(0, 1, true, ratings, Random::instance(), ties, favored_ties);

  EXPECT_EQ(selection.cluster, 1);
  EXPECT_EQ(selection.favored_cluster, 1);
  const lp::MoveResult result = state.view().commit(graph.csr_graph(), 0, 0, selection.cluster, 1);
  EXPECT_TRUE(result.moved);
  EXPECT_TRUE(result.emptied_cluster);
  EXPECT_EQ(state.cluster(0), 1);
}

TEST(ClusteringSelectorTest, RejectsClustersFromOtherCommunities) {
  Graph graph = make_graph({0, 1, 2}, {1, 0});
  StaticArray<NodeID> clustering(graph.n(), static_array::noinit);
  const std::vector<NodeID> communities = {0, 1};
  lp::ClusteringState state;
  state.set_max_cluster_weight(2);
  state.set_communities(communities);
  state.reset(clustering, graph.csr_graph());

  FixedSizeSparseMap<NodeID, EdgeWeight, 128> ratings;
  ratings[1] = 5;
  ScalableVector<NodeID> ties;
  ScalableVector<NodeID> favored_ties;
  lp::ClusteringSelector selector(state.view());

  EXPECT_EQ(
      selector.select(0, 1, true, ratings, Random::instance(), ties, favored_ties).cluster, 0
  );
}

TEST(BalancedSelectorTest, SelectsAndCommitsTheStrongestFeasibleBlock) {
  Graph graph = make_path_graph(3);
  PartitionedGraph p_graph = make_p_graph(graph, 2, {0, 1, 0});
  Context ctx = create_default_context();
  ctx.partition.setup(graph, 2, 0.03);

  lp::BalancedState state;
  state.reset(p_graph, ctx.partition, {});
  FixedSizeSparseMap<BlockID, EdgeWeight, 128> ratings;
  ratings[1] = 5;
  ScalableVector<BlockID> ties;
  lp::BalancedSelector selector(state);

  const BlockID selected = selector.select(0, 1, ratings, Random::instance(), ties);
  EXPECT_EQ(selected, 1);
  EXPECT_TRUE(state.commit(graph.csr_graph(), 0, 0, selected, 1).moved);
  EXPECT_EQ(p_graph.block(0), 1);
}

} // namespace kaminpar::shm::testing
