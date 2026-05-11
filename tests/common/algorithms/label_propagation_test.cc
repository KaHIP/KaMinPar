#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <tbb/task_arena.h>

#include "kaminpar-common/algorithms/label_propagation.h"
#include "kaminpar-common/datastructures/concurrent_fast_reset_array.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/rating_map.h"
#include "kaminpar-common/datastructures/static_array.h"

using ::testing::Eq;

namespace kaminpar {

namespace {

class TestGraph {
public:
  using NodeID = std::uint32_t;
  using NodeWeight = std::int32_t;
  using EdgeID = std::uint32_t;
  using EdgeWeight = std::int32_t;

  struct Edge {
    NodeID target;
    EdgeWeight weight;
  };

  TestGraph(std::vector<std::vector<Edge>> adjacency, std::vector<NodeWeight> node_weights)
      : _adjacency(std::move(adjacency)),
        _node_weights(std::move(node_weights)) {}

  [[nodiscard]] NodeID n() const {
    return static_cast<NodeID>(_adjacency.size());
  }

  [[nodiscard]] EdgeID m() const {
    EdgeID edges = 0;
    for (const auto &neighbors : _adjacency) {
      edges += static_cast<EdgeID>(neighbors.size());
    }
    return edges;
  }

  [[nodiscard]] EdgeID degree(const NodeID node) const {
    return static_cast<EdgeID>(_adjacency[node].size());
  }

  [[nodiscard]] NodeWeight node_weight(const NodeID node) const {
    return _node_weights[node];
  }

  template <typename Visitor> void adjacent_nodes(const NodeID node, Visitor &&visitor) const {
    for (const Edge edge : _adjacency[node]) {
      if (call_visitor(visitor, edge)) {
        break;
      }
    }
  }

  template <typename Visitor>
  void adjacent_nodes(const NodeID node, const NodeID max_neighbors, Visitor &&visitor) const {
    NodeID count = 0;
    for (const Edge edge : _adjacency[node]) {
      if (count++ >= max_neighbors) {
        break;
      }
      if (call_visitor(visitor, edge)) {
        break;
      }
    }
  }

  template <typename Callback>
  void pfor_adjacent_nodes(
      const NodeID node, const NodeID max_neighbors, const NodeID, Callback &&callback
  ) const {
    callback([&](auto &&visitor) {
      adjacent_nodes(node, max_neighbors, std::forward<decltype(visitor)>(visitor));
    });
  }

private:
  template <typename Visitor> static bool call_visitor(Visitor &visitor, const Edge edge) {
    if constexpr (std::is_invocable_r_v<bool, Visitor, NodeID, EdgeWeight>) {
      return visitor(edge.target, edge.weight);
    } else if constexpr (std::is_invocable_v<Visitor, NodeID, EdgeWeight>) {
      visitor(edge.target, edge.weight);
      return false;
    } else if constexpr (std::is_invocable_r_v<bool, Visitor, NodeID>) {
      return visitor(edge.target);
    } else {
      visitor(edge.target);
      return false;
    }
  }

  std::vector<std::vector<Edge>> _adjacency;
  std::vector<NodeWeight> _node_weights;
};

using TestNodeID = TestGraph::NodeID;
using TestClusterID = TestGraph::NodeID;
using TestNodeWeight = TestGraph::NodeWeight;
using TestEdgeWeight = TestGraph::EdgeWeight;
using TestRatingMap = RatingMap<TestEdgeWeight, TestClusterID>;
using TestGrowingRatingMap = DynamicRememberingFlatMap<TestClusterID, TestEdgeWeight>;
using TestConcurrentRatingMap = ConcurrentFastResetArray<TestEdgeWeight, TestClusterID>;
using TestWorkspace = lp::Workspace<
    TestNodeID,
    TestClusterID,
    TestEdgeWeight,
    TestRatingMap,
    TestGrowingRatingMap,
    TestConcurrentRatingMap,
    true>;

class TestWeights : public lp::RelaxedClusterWeightVector<TestClusterID, TestNodeWeight> {
public:
  void set_initial_weights(std::vector<TestNodeWeight> weights) {
    _initial_weights = std::move(weights);
  }

  void set_max_cluster_weight(const TestNodeWeight max_cluster_weight) {
    _max_cluster_weight = max_cluster_weight;
  }

  [[nodiscard]] TestNodeWeight initial_cluster_weight(const TestClusterID cluster) const {
    return _initial_weights[cluster];
  }

  [[nodiscard]] TestNodeWeight max_cluster_weight(const TestClusterID) const {
    return _max_cluster_weight;
  }

private:
  std::vector<TestNodeWeight> _initial_weights;
  TestNodeWeight _max_cluster_weight = std::numeric_limits<TestNodeWeight>::max();
};

class TestSelector {
public:
  explicit TestSelector(TestWeights &weights) : _weights(weights) {}

  template <lp::TieBreakingStrategy TieBreaking, typename Context, typename RatingMap>
  auto select(
      const Context &context,
      RatingMap &map,
      ScalableVector<TestClusterID> &tie_breaking_clusters,
      ScalableVector<TestClusterID> &tie_breaking_favored_clusters
  ) {
    return lp::choose_cluster<TieBreaking>(
        context, map, *this, tie_breaking_clusters, tie_breaking_favored_clusters
    );
  }

  [[nodiscard]] TestNodeWeight cluster_weight(const TestClusterID cluster) const {
    return _weights.cluster_weight(cluster);
  }

  template <typename Context, typename Candidate, typename Choice>
  [[nodiscard]] bool
  is_feasible(const Context &context, const Candidate &candidate, const Choice &) const {
    return candidate.weight + context.node_weight <=
               _weights.max_cluster_weight(candidate.cluster) ||
           candidate.cluster == context.initial_cluster;
  }

  template <typename Context, typename Candidate, typename Choice>
  [[nodiscard]] lp::CandidateComparison
  compare(const Context &, const Candidate &candidate, const Choice &choice) const {
    return lp::compare_by_gain(candidate.gain, choice.best_gain);
  }

private:
  TestWeights &_weights;
};

struct TestNeighborPolicy {
  std::optional<TestNodeID> rejected_neighbor;
  std::vector<std::uint8_t> skipped_nodes;

  [[nodiscard]] bool accept(const TestNodeID, const TestNodeID v) const {
    return !rejected_neighbor.has_value() || *rejected_neighbor != v;
  }

  [[nodiscard]] bool activate(const TestNodeID) const {
    return true;
  }

  [[nodiscard]] bool skip(const TestNodeID u) const {
    return u < skipped_nodes.size() && skipped_nodes[u] != 0;
  }
};

struct KernelFixture {
  explicit KernelFixture(TestGraph graph, lp::PassConfig<TestNodeID, TestClusterID> config = {})
      : graph(std::move(graph)),
        labels_array(this->graph.n()),
        selector(weights),
        kernel(this->graph, labels, weights, selector, neighbors, workspace, config) {
    labels.init(labels_array);
    weights.allocate(this->graph.n());

    std::vector<TestNodeWeight> initial_weights(this->graph.n());
    for (TestNodeID u = 0; u < this->graph.n(); ++u) {
      initial_weights[u] = this->graph.node_weight(u);
    }
    weights.set_initial_weights(std::move(initial_weights));

    kernel.initialize({
        .num_nodes = this->graph.n(),
        .num_active_nodes = this->graph.n(),
        .num_clusters = this->graph.n(),
    });
  }

  TestGraph graph;
  StaticArray<TestClusterID> labels_array;
  lp::ExternalLabelArray<TestNodeID, TestClusterID> labels;
  TestWeights weights;
  TestWorkspace workspace;
  TestSelector selector;
  TestNeighborPolicy neighbors;
  lp::LabelPropagationKernel<
      TestGraph,
      lp::ExternalLabelArray<TestNodeID, TestClusterID>,
      TestWeights,
      TestSelector,
      TestNeighborPolicy,
      TestWorkspace>
      kernel;
};

struct TestOrder {
  std::vector<TestNodeID> nodes;

  template <typename Visitor> void parallel_for_each(Visitor &&visitor) {
    for (const TestNodeID node : nodes) {
      visitor(node);
    }
  }
};

TestGraph weighted_star() {
  return TestGraph{
      {{{1, 1}, {2, 5}, {3, 7}}, {}, {}, {}},
      {1, 1, 1, 1},
  };
}

} // namespace

TEST(LabelPropagationKernelTest, manual_single_phase_pass_moves_node_to_highest_rated_cluster) {
  KernelFixture fixture(weighted_star());

  lp::SinglePhasePass<
      decltype(fixture.kernel),
      lp::ActiveSetStrategy::NONE,
      lp::TieBreakingStrategy::GEOMETRIC>
      pass(fixture.kernel);
  pass.handle_next_node(0);
  const auto result = pass.finish();

  EXPECT_THAT(fixture.labels.cluster(0), Eq(3));
  EXPECT_THAT(result.processed_nodes, Eq(1));
  EXPECT_THAT(result.moved_nodes, Eq(1));
}

TEST(LabelPropagationKernelTest, neighbor_filter_and_max_neighbors_limit_rating_accumulation) {
  lp::PassConfig<TestNodeID, TestClusterID> config;
  config.nodes.max_neighbors = 2;
  KernelFixture fixture(weighted_star(), config);
  fixture.neighbors.rejected_neighbor = 2;

  lp::SinglePhasePass<
      decltype(fixture.kernel),
      lp::ActiveSetStrategy::NONE,
      lp::TieBreakingStrategy::GEOMETRIC>
      pass(fixture.kernel);
  pass.handle_next_node(0);
  (void)pass.finish();

  EXPECT_THAT(fixture.labels.cluster(0), Eq(1));
}

TEST(LabelPropagationKernelTest, inactive_nodes_are_skipped) {
  lp::PassConfig<TestNodeID, TestClusterID> config;
  config.active_set.strategy = lp::ActiveSetStrategy::GLOBAL;
  KernelFixture fixture(weighted_star(), config);
  fixture.workspace.active_set.flags[0] = 0;

  lp::SinglePhasePass<
      decltype(fixture.kernel),
      lp::ActiveSetStrategy::GLOBAL,
      lp::TieBreakingStrategy::GEOMETRIC>
      pass(fixture.kernel);
  pass.handle_next_node(0);
  const auto result = pass.finish();

  EXPECT_THAT(fixture.labels.cluster(0), Eq(0));
  EXPECT_THAT(result.processed_nodes, Eq(0));
  EXPECT_THAT(result.moved_nodes, Eq(0));
}

TEST(LabelPropagationKernelTest, isolated_node_clustering_reuses_kernel_postprocessing) {
  KernelFixture fixture(TestGraph{{{}, {}, {}}, {1, 1, 1}});
  fixture.weights.set_max_cluster_weight(3);

  tbb::task_arena arena(1);
  arena.execute([&] { lp::cluster_isolated_nodes(fixture.kernel); });

  EXPECT_THAT(fixture.labels.cluster(0), Eq(0));
  EXPECT_THAT(fixture.labels.cluster(1), Eq(0));
  EXPECT_THAT(fixture.labels.cluster(2), Eq(0));
}

TEST(LabelPropagationKernelTest, two_hop_clustering_uses_favored_clusters) {
  lp::PassConfig<TestNodeID, TestClusterID> config;
  config.selection.track_favored_clusters = true;
  KernelFixture fixture(TestGraph{{{{2, 1}}, {{2, 1}}, {}}, {1, 1, 1}}, config);
  fixture.weights.set_max_cluster_weight(3);
  fixture.workspace.postprocessing.favored_clusters[0] = 2;
  fixture.workspace.postprocessing.favored_clusters[1] = 2;
  fixture.workspace.postprocessing.favored_clusters[2] = 2;

  tbb::task_arena arena(1);
  arena.execute([&] { lp::cluster_two_hop_nodes(fixture.kernel); });

  EXPECT_THAT(fixture.labels.cluster(0), Eq(0));
  EXPECT_THAT(fixture.labels.cluster(1), Eq(0));
  EXPECT_THAT(fixture.labels.cluster(2), Eq(2));
}

TEST(LabelPropagationPassTest, rating_map_passes_produce_same_move_without_deferral) {
  KernelFixture single_phase(weighted_star());
  KernelFixture growing_hash_tables(weighted_star());
  KernelFixture two_phase(weighted_star());

  lp::SinglePhasePass<
      decltype(single_phase.kernel),
      lp::ActiveSetStrategy::NONE,
      lp::TieBreakingStrategy::GEOMETRIC>
      single_phase_pass(single_phase.kernel);
  lp::GrowingHashTablePass<
      decltype(growing_hash_tables.kernel),
      lp::ActiveSetStrategy::NONE,
      lp::TieBreakingStrategy::GEOMETRIC>
      growing_pass(growing_hash_tables.kernel);
  lp::TwoPhasePass<
      decltype(two_phase.kernel),
      lp::ActiveSetStrategy::NONE,
      lp::TieBreakingStrategy::GEOMETRIC>
      two_phase_pass(
          two_phase.kernel,
          {.strategy = lp::RatingMapStrategy::TWO_PHASE, .large_map_threshold = 10000}
      );

  single_phase_pass.handle_next_node(0);
  growing_pass.handle_next_node(0);
  two_phase_pass.handle_next_node(0);

  EXPECT_THAT(single_phase_pass.finish().moved_nodes, Eq(1));
  EXPECT_THAT(growing_pass.finish().moved_nodes, Eq(1));
  EXPECT_THAT(two_phase_pass.finish().moved_nodes, Eq(1));
  EXPECT_THAT(single_phase.labels.cluster(0), Eq(3));
  EXPECT_THAT(growing_hash_tables.labels.cluster(0), Eq(3));
  EXPECT_THAT(two_phase.labels.cluster(0), Eq(3));
}

TEST(LabelPropagationPassTest, two_phase_pass_defers_large_nodes_to_finish) {
  KernelFixture fixture(weighted_star());

  lp::TwoPhasePass<
      decltype(fixture.kernel),
      lp::ActiveSetStrategy::NONE,
      lp::TieBreakingStrategy::GEOMETRIC>
      pass(
          fixture.kernel, {.strategy = lp::RatingMapStrategy::TWO_PHASE, .large_map_threshold = 1}
      );
  pass.handle_next_node(0);

  EXPECT_THAT(fixture.labels.cluster(0), Eq(0));
  EXPECT_THAT(pass.finish().moved_nodes, Eq(1));
  EXPECT_THAT(fixture.labels.cluster(0), Eq(3));
}

TEST(LabelPropagationRunTest, manual_iteration_and_run_iteration_are_equivalent) {
  KernelFixture manual(weighted_star());
  KernelFixture utility(weighted_star());

  lp::SinglePhasePass<
      decltype(manual.kernel),
      lp::ActiveSetStrategy::NONE,
      lp::TieBreakingStrategy::GEOMETRIC>
      pass(manual.kernel);
  for (const TestNodeID node : std::vector<TestNodeID>{0}) {
    pass.handle_next_node(node);
  }
  const auto manual_result = pass.finish();

  TestOrder order{{0}};
  const auto utility_result =
      lp::run_iteration(order, utility.kernel, {.strategy = lp::RatingMapStrategy::SINGLE_PHASE});

  EXPECT_THAT(manual.labels.cluster(0), Eq(utility.labels.cluster(0)));
  EXPECT_THAT(manual_result.moved_nodes, Eq(utility_result.moved_nodes));
}

} // namespace kaminpar
