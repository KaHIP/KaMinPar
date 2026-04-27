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
    false>;

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

  template <typename State, typename RatingMap>
  TestClusterID select(
      const bool,
      const TestEdgeWeight gain_delta,
      State &state,
      RatingMap &map,
      ScalableVector<TestClusterID> &,
      ScalableVector<TestClusterID> &
  ) {
    TestClusterID favored_cluster = state.initial_cluster;
    TestEdgeWeight favored_gain = 0;

    for (const auto [cluster, rating] : map.entries()) {
      if (rating > favored_gain) {
        favored_gain = rating;
        favored_cluster = cluster;
      }

      state.current_cluster = cluster;
      state.current_gain = rating - gain_delta;
      state.current_cluster_weight = _weights.cluster_weight(cluster);

      if (state.current_gain > state.best_gain &&
          (state.current_cluster_weight + state.u_weight <=
               _weights.max_cluster_weight(state.current_cluster) ||
           state.current_cluster == state.initial_cluster)) {
        state.best_cluster = state.current_cluster;
        state.best_cluster_weight = state.current_cluster_weight;
        state.best_gain = state.current_gain;
      }
    }

    return favored_cluster;
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

struct CoreFixture {
  explicit CoreFixture(TestGraph graph, lp::Options<TestNodeID, TestClusterID> options = {})
      : graph(std::move(graph)),
        labels_array(this->graph.n()),
        selector(weights),
        core(this->graph, labels, weights, selector, neighbors, workspace, options) {
    labels.init(labels_array);
    weights.allocate(this->graph.n());

    std::vector<TestNodeWeight> initial_weights(this->graph.n());
    for (TestNodeID u = 0; u < this->graph.n(); ++u) {
      initial_weights[u] = this->graph.node_weight(u);
    }
    weights.set_initial_weights(std::move(initial_weights));

    core.initialize({
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
  lp::LabelPropagationCore<
      TestGraph,
      lp::ExternalLabelArray<TestNodeID, TestClusterID>,
      TestWeights,
      TestSelector,
      TestNeighborPolicy,
      TestWorkspace>
      core;
};

TestGraph weighted_star() {
  return TestGraph{
      {{{1, 1}, {2, 5}, {3, 7}}, {}, {}, {}},
      {1, 1, 1, 1},
  };
}

} // namespace

TEST(LabelPropagationCoreTest, manual_pass_moves_node_to_highest_rated_cluster) {
  CoreFixture fixture(weighted_star());

  auto pass = fixture.core.begin_pass();
  pass.handle_next_node(0);
  const auto result = pass.finish();

  EXPECT_THAT(fixture.labels.cluster(0), Eq(3));
  EXPECT_THAT(result.processed_nodes, Eq(1));
  EXPECT_THAT(result.moved_nodes, Eq(1));
}

TEST(LabelPropagationCoreTest, neighbor_filter_and_max_neighbors_limit_rating_accumulation) {
  lp::Options<TestNodeID, TestClusterID> options;
  options.max_num_neighbors = 2;
  CoreFixture fixture(weighted_star(), options);
  fixture.neighbors.rejected_neighbor = 2;

  auto pass = fixture.core.begin_pass();
  pass.handle_next_node(0);
  (void)pass.finish();

  EXPECT_THAT(fixture.labels.cluster(0), Eq(1));
}

TEST(LabelPropagationCoreTest, inactive_nodes_are_skipped) {
  lp::Options<TestNodeID, TestClusterID> options;
  options.active_set_strategy = lp::ActiveSetStrategy::GLOBAL;
  CoreFixture fixture(weighted_star(), options);
  fixture.workspace.active[0] = 0;

  auto pass = fixture.core.begin_pass();
  pass.handle_next_node(0);
  const auto result = pass.finish();

  EXPECT_THAT(fixture.labels.cluster(0), Eq(0));
  EXPECT_THAT(result.processed_nodes, Eq(0));
  EXPECT_THAT(result.moved_nodes, Eq(0));
}

TEST(LabelPropagationCoreTest, isolated_node_clustering_reuses_core_postprocessing) {
  CoreFixture fixture(TestGraph{{{}, {}, {}}, {1, 1, 1}});
  fixture.weights.set_max_cluster_weight(3);

  tbb::task_arena arena(1);
  arena.execute([&] { fixture.core.cluster_isolated_nodes(); });

  EXPECT_THAT(fixture.labels.cluster(0), Eq(0));
  EXPECT_THAT(fixture.labels.cluster(1), Eq(0));
  EXPECT_THAT(fixture.labels.cluster(2), Eq(0));
}

TEST(LabelPropagationCoreTest, two_hop_clustering_uses_favored_clusters) {
  lp::Options<TestNodeID, TestClusterID> options;
  options.use_two_hop_clustering = true;
  CoreFixture fixture(TestGraph{{{{2, 1}}, {{2, 1}}, {}}, {1, 1, 1}}, options);
  fixture.weights.set_max_cluster_weight(3);
  fixture.workspace.favored_clusters[0] = 2;
  fixture.workspace.favored_clusters[1] = 2;
  fixture.workspace.favored_clusters[2] = 2;

  tbb::task_arena arena(1);
  arena.execute([&] { fixture.core.cluster_two_hop_nodes(); });

  EXPECT_THAT(fixture.labels.cluster(0), Eq(0));
  EXPECT_THAT(fixture.labels.cluster(1), Eq(0));
  EXPECT_THAT(fixture.labels.cluster(2), Eq(2));
}

} // namespace kaminpar
