#include <gtest/gtest.h>

#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/refinement/flow/flow_cutter/flow_cutter.h"
#include "kaminpar-shm/refinement/flow/flow_network/flow_network.h"

namespace {
using namespace kaminpar;
using namespace kaminpar::shm;
using namespace kaminpar::shm::testing;

class FlowMinBlockWeightsTest : public ::testing::TestWithParam<bool> {
protected:
  void SetUp() override {
    // The two possible cuts of weight one leave block weights (3, 5) or (7, 1).
    // The initial cut has weight ten and block weights (5, 3).
    _graph = make_graph({0, 1, 2, 4, 6}, {2, 3, 0, 3, 1, 2}, {3, 1, 2, 2}, {1, 1, 1, 10, 1, 10});

    _p_graph = make_p_graph(_graph, 2, {0, 1, 0, 1});
    _ctx = create_default_context();
    _ctx.partition.setup(_graph, 2, 0.75);
    _ctx.refinement.twoway_flow.flow_cutter.small_flow_network_threshold = 0;

    _border_region.initialize(0, 1, 2, 2);
    _border_region.insert_border_node_into_region1(2, 2);
    _border_region.insert_border_node_into_region2(3, 2);
  }

  std::pair<EdgeWeight, std::vector<BlockWeight>> refine(std::vector<BlockWeight> min_weights) {
    if (!min_weights.empty()) {
      _ctx.partition.setup_min_block_weights(std::move(min_weights));
    }

    auto p_csr_graph = _p_graph.csr_view();
    FlowNetworkConstructor constructor(
        _ctx.refinement.twoway_flow.construction, p_csr_graph, _graph.csr_graph()
    );
    const auto network = constructor.construct_flow_network(_border_region, 5, 3, GetParam());
    FlowCutter cutter(_ctx.partition, _ctx.refinement.twoway_flow.flow_cutter);
    const auto result = cutter.compute_cut(_border_region, network, GetParam());

    std::vector<BlockWeight> weights = {5, 3};
    for (const auto &move : result.moves) {
      const auto weight = _graph.csr_graph().node_weight(move.node);
      weights[move.old_block] -= weight;
      weights[move.new_block] += weight;
    }
    return {result.gain, weights};
  }

  Graph _graph;
  PartitionedGraph _p_graph;
  Context _ctx;
  BorderRegion _border_region;
};

TEST_P(FlowMinBlockWeightsTest, FindsImprovementWithoutLowerBounds) {
  const auto [gain, weights] = refine({});
  EXPECT_EQ(gain, 9);
  EXPECT_TRUE(
      weights == std::vector<BlockWeight>({3, 5}) || weights == std::vector<BlockWeight>({7, 1})
  );
}

TEST_P(FlowMinBlockWeightsTest, RejectsCutsThatUnderloadEitherBlock) {
  const auto [gain, weights] = refine({5, 3});
  EXPECT_EQ(gain, 0);
  EXPECT_EQ(weights, std::vector<BlockWeight>({5, 3}));
}

TEST_P(FlowMinBlockWeightsTest, FindsImprovementWithinAsymmetricLowerBounds) {
  const auto [gain, weights] = refine({5, 1});
  EXPECT_EQ(gain, 9);
  EXPECT_EQ(weights, std::vector<BlockWeight>({7, 1}));
}

TEST_P(FlowMinBlockWeightsTest, FindsImprovementPreservingSinkMinimum) {
  const auto [gain, weights] = refine({1, 3});
  EXPECT_EQ(gain, 9);
  EXPECT_EQ(weights, std::vector<BlockWeight>({3, 5}));
}

TEST_P(FlowMinBlockWeightsTest, ReturnsNoMovesForInfeasiblePairBounds) {
  const auto [gain, weights] = refine({6, 3});
  EXPECT_EQ(gain, 0);
  EXPECT_EQ(weights, std::vector<BlockWeight>({5, 3}));
}

INSTANTIATE_TEST_SUITE_P(SequentialAndParallel, FlowMinBlockWeightsTest, ::testing::Bool());
} // namespace
