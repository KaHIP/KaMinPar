#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <tbb/global_control.h>

#include "tests/shm/graph_builder.h"
#include "tests/shm/graph_factories.h"

#include "kaminpar-shm/algorithms/label_propagation/global_two_hop_clustering.h"
#include "kaminpar-shm/algorithms/label_propagation/threadwise_two_hop_clustering.h"
#include "kaminpar-shm/algorithms/label_propagation/two_hop_candidates.h"
#include "kaminpar-shm/context.h"
#include "kaminpar-shm/presets.h"

namespace kaminpar::shm::testing {

namespace {

using ::testing::ElementsAre;

Graph make_weighted_star(
    const NodeID num_leaves, const NodeWeight center_weight = 10, const NodeWeight leaf_weight = 1
) {
  GraphBuilder builder;
  builder.new_node(center_weight);
  for (NodeID leaf = 1; leaf <= num_leaves; ++leaf) {
    builder.new_edge(leaf);
  }

  for (NodeID leaf = 1; leaf <= num_leaves; ++leaf) {
    builder.new_node(leaf_weight);
    builder.new_edge(0);
  }

  return builder.build();
}

class TwoHopTestState {
public:
  TwoHopTestState(Graph input_graph, const NodeWeight max_cluster_weight)
      : graph(std::move(input_graph)),
        clustering(graph.n(), static_array::noinit) {
    state.set_max_cluster_weight(max_cluster_weight);
    state.reset(clustering, graph.csr_graph());
  }

  void assign_leaf_group(const NodeID group = 0) {
    for (NodeID u = 1; u < graph.n(); ++u) {
      state.set_favored_cluster(u, group);
    }
  }

  [[nodiscard]] std::vector<std::size_t> sorted_cluster_sizes() const {
    std::unordered_map<NodeID, std::size_t> sizes;
    for (NodeID u = 0; u < graph.n(); ++u) {
      ++sizes[state.cluster(u)];
    }

    std::vector<std::size_t> result;
    result.reserve(sizes.size());
    for (const auto &[cluster, size] : sizes) {
      (void)cluster;
      result.push_back(size);
    }
    std::ranges::sort(result);
    return result;
  }

  void expect_consistent_cluster_weights() const {
    std::vector<NodeWeight> expected(graph.n(), 0);
    for (NodeID u = 0; u < graph.n(); ++u) {
      expected[state.cluster(u)] += graph.csr_graph().node_weight(u);
    }
    for (NodeID cluster = 0; cluster < graph.n(); ++cluster) {
      EXPECT_EQ(state.cluster_weight(cluster), expected[cluster]);
    }
  }

  Graph graph;
  StaticArray<NodeID> clustering;
  lp::ClusteringState state;
};

} // namespace

TEST(TwoHopCandidatesTest, SelectsOnlyUnchangedLightNonIsolatedSingletons) {
  TwoHopTestState test_state(make_weighted_star(4), 2);
  lp::TwoHopCandidates candidates(test_state.graph.csr_graph(), test_state.state);

  EXPECT_FALSE(candidates.contains(0));
  EXPECT_TRUE(candidates.contains(1));

  test_state.state.move_node(1, 0);
  EXPECT_FALSE(candidates.contains(1));

  ASSERT_TRUE(test_state.state.move_cluster_weight(2, 3, 1));
  test_state.state.move_node(2, 3);
  EXPECT_FALSE(candidates.contains(2));
  EXPECT_FALSE(candidates.contains(3));
  EXPECT_TRUE(candidates.contains(4));
}

TEST(TwoHopCandidatesTest, RejectsIsolatedAndOverweightSingletons) {
  TwoHopTestState isolated(make_empty_graph(1), 2);
  lp::TwoHopCandidates isolated_candidates(isolated.graph.csr_graph(), isolated.state);
  EXPECT_FALSE(isolated_candidates.contains(0));

  TwoHopTestState overweight(make_weighted_star(1, 10, 2), 3);
  lp::TwoHopCandidates overweight_candidates(overweight.graph.csr_graph(), overweight.state);
  EXPECT_FALSE(overweight_candidates.contains(1));
}

TEST(GlobalTwoHopClusteringTest, MatchBuildsGlobalPairs) {
  TwoHopTestState test_state(make_weighted_star(4), 2);
  test_state.assign_leaf_group();

  lp::GlobalTwoHopClustering(test_state.graph.csr_graph(), test_state.state).match();

  EXPECT_THAT(test_state.sorted_cluster_sizes(), ElementsAre(1, 2, 2));
  EXPECT_EQ(test_state.state.num_clusters(), 3);
  test_state.expect_consistent_cluster_weights();
}

TEST(GlobalTwoHopClusteringTest, MatchCoordinatesOneLargeGroupAcrossWorkers) {
  constexpr NodeID kNumLeaves = 4096;
  const tbb::global_control four_workers(tbb::global_control::max_allowed_parallelism, 4);
  TwoHopTestState test_state(make_weighted_star(kNumLeaves), 2);
  test_state.assign_leaf_group();

  lp::GlobalTwoHopClustering(test_state.graph.csr_graph(), test_state.state).match();

  const std::vector<std::size_t> sizes = test_state.sorted_cluster_sizes();
  EXPECT_EQ(sizes.size(), 1 + kNumLeaves / 2);
  EXPECT_EQ(std::ranges::count(sizes, 1), 1);
  EXPECT_EQ(std::ranges::count(sizes, 2), kNumLeaves / 2);
  EXPECT_EQ(test_state.state.num_clusters(), 1 + kNumLeaves / 2);
  test_state.expect_consistent_cluster_weights();
}

TEST(GlobalTwoHopClusteringTest, DirectlyMergesCandidatesThatFavorEachOther) {
  const tbb::global_control one_worker(tbb::global_control::max_allowed_parallelism, 1);
  TwoHopTestState test_state(make_weighted_star(2), 2);
  test_state.state.set_favored_cluster(1, 2);
  test_state.state.set_favored_cluster(2, 0);

  lp::GlobalTwoHopClustering(test_state.graph.csr_graph(), test_state.state).match();

  EXPECT_EQ(test_state.state.cluster(1), 2);
  EXPECT_EQ(test_state.state.num_clusters(), 2);
  test_state.expect_consistent_cluster_weights();
}

TEST(GlobalTwoHopClusteringTest, DoesNotCountASelfFavoriteAsAMerge) {
  const tbb::global_control one_worker(tbb::global_control::max_allowed_parallelism, 1);
  TwoHopTestState test_state(make_weighted_star(1), 2);

  lp::GlobalTwoHopClustering(test_state.graph.csr_graph(), test_state.state).cluster();

  EXPECT_EQ(test_state.state.cluster(1), 1);
  EXPECT_EQ(test_state.state.num_clusters(), 2);
  test_state.expect_consistent_cluster_weights();
}

TEST(GlobalTwoHopClusteringTest, ClusterFillsAGroupUntilTheWeightLimit) {
  const tbb::global_control one_worker(tbb::global_control::max_allowed_parallelism, 1);
  TwoHopTestState test_state(make_weighted_star(4), 3);
  test_state.assign_leaf_group();

  lp::GlobalTwoHopClustering(test_state.graph.csr_graph(), test_state.state).cluster();

  EXPECT_THAT(test_state.sorted_cluster_sizes(), ElementsAre(1, 1, 3));
  EXPECT_EQ(test_state.state.num_clusters(), 3);
  test_state.expect_consistent_cluster_weights();
}

TEST(ThreadwiseTwoHopClusteringTest, MatchBuildsWorkerLocalPairs) {
  const tbb::global_control one_worker(tbb::global_control::max_allowed_parallelism, 1);
  TwoHopTestState test_state(make_weighted_star(4), 2);
  test_state.assign_leaf_group();

  lp::ThreadwiseTwoHopClustering(test_state.graph.csr_graph(), test_state.state).match();

  EXPECT_THAT(test_state.sorted_cluster_sizes(), ElementsAre(1, 2, 2));
  test_state.expect_consistent_cluster_weights();
}

TEST(ThreadwiseTwoHopClusteringTest, ClusterRollsOverToANewRepresentativeAtTheWeightLimit) {
  const tbb::global_control one_worker(tbb::global_control::max_allowed_parallelism, 1);
  TwoHopTestState test_state(make_weighted_star(4), 3);
  test_state.assign_leaf_group();

  lp::ThreadwiseTwoHopClustering(test_state.graph.csr_graph(), test_state.state).cluster();

  EXPECT_THAT(test_state.sorted_cluster_sizes(), ElementsAre(1, 1, 3));
  test_state.expect_consistent_cluster_weights();
}

TEST(ThreadwiseTwoHopClusteringTest, ClusterMaintainsConsistentWeightsAcrossWorkers) {
  constexpr NodeID kNumLeaves = 4096;
  const tbb::global_control four_workers(tbb::global_control::max_allowed_parallelism, 4);
  TwoHopTestState test_state(make_weighted_star(kNumLeaves), 3);
  test_state.assign_leaf_group();

  lp::ThreadwiseTwoHopClustering(test_state.graph.csr_graph(), test_state.state).cluster();

  const std::vector<std::size_t> sizes = test_state.sorted_cluster_sizes();
  EXPECT_LT(sizes.size(), test_state.graph.n());
  EXPECT_TRUE(std::ranges::all_of(sizes, [](const std::size_t size) { return size <= 3; }));
  test_state.expect_consistent_cluster_weights();
}

TEST(LabelPropagationConfigurationTest, UsesRatingAggregationTerminologyAndLegacyValueAliases) {
  const auto aggregations = get_lp_rating_aggregations();
  EXPECT_EQ(aggregations.at("local"), LabelPropagationRatingAggregation::LOCAL);
  EXPECT_EQ(
      aggregations.at("deferred-parallel"), LabelPropagationRatingAggregation::DEFERRED_PARALLEL
  );
  EXPECT_EQ(aggregations.at("single-phase"), LabelPropagationRatingAggregation::LOCAL);
  EXPECT_EQ(aggregations.at("two-phase"), LabelPropagationRatingAggregation::DEFERRED_PARALLEL);

  std::ostringstream out;
  out << LabelPropagationRatingAggregation::LOCAL << " "
      << LabelPropagationRatingAggregation::DEFERRED_PARALLEL;
  EXPECT_EQ(out.str(), "local deferred-parallel");

  // Preserve the values emitted by existing generated configuration files.
  EXPECT_EQ(static_cast<int>(LabelPropagationRatingAggregation::LOCAL), 0);
  EXPECT_EQ(static_cast<int>(LabelPropagationRatingAggregation::DEFERRED_PARALLEL), 1);
}

TEST(LabelPropagationConfigurationTest, PresetsRetainTheirRatingAggregationBehavior) {
  EXPECT_EQ(
      create_default_context().coarsening.clustering.lp.rating_aggregation,
      LabelPropagationRatingAggregation::DEFERRED_PARALLEL
  );
  EXPECT_EQ(
      create_esa21_smallk_context().coarsening.clustering.lp.rating_aggregation,
      LabelPropagationRatingAggregation::LOCAL
  );
}

TEST(LabelPropagationConfigurationTest, ExposesAllFourTwoHopAlgorithms) {
  const auto strategies = get_two_hop_strategies();
  EXPECT_EQ(strategies.at("match"), TwoHopStrategy::MATCH);
  EXPECT_EQ(strategies.at("match-threadwise"), TwoHopStrategy::MATCH_THREADWISE);
  EXPECT_EQ(strategies.at("cluster"), TwoHopStrategy::CLUSTER);
  EXPECT_EQ(strategies.at("cluster-threadwise"), TwoHopStrategy::CLUSTER_THREADWISE);

  // The CLI writes these numeric values to generated configuration files.
  EXPECT_EQ(static_cast<int>(TwoHopStrategy::MATCH), 1);
  EXPECT_EQ(static_cast<int>(TwoHopStrategy::MATCH_THREADWISE), 2);
  EXPECT_EQ(static_cast<int>(TwoHopStrategy::CLUSTER), 3);
  EXPECT_EQ(static_cast<int>(TwoHopStrategy::CLUSTER_THREADWISE), 4);
}

} // namespace kaminpar::shm::testing
