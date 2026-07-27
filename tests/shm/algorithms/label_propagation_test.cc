#include <atomic>
#include <vector>

#include <gmock/gmock.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/algorithms/iteration_order.h"
#include "kaminpar-shm/algorithms/label_propagation/active_set.h"
#include "kaminpar-shm/algorithms/label_propagation/balanced_selector.h"
#include "kaminpar-shm/algorithms/label_propagation/clustering_selector.h"
#include "kaminpar-shm/algorithms/label_propagation/neighborhood_ratings.h"
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

TEST(RatingMapPoolTest, ReusesAdaptiveMapsAcrossCapacityChanges) {
  lp::RatingMapPool<EdgeWeight, NodeID> maps;
  maps.ensure_capacity(64);
  EXPECT_EQ(maps.local().ratings.max_size(), 64);

  maps.ensure_capacity(16);
  EXPECT_EQ(maps.local().ratings.max_size(), 16);

  maps.ensure_capacity(128);
  EXPECT_EQ(maps.local().ratings.max_size(), 128);
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
