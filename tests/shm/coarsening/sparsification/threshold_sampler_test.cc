#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"
#include "tests/shm/matchers.h"

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/coarsening/sparsification/independent_random_sampler.h"
#include "kaminpar-shm/coarsening/sparsification/threshold_sampler.h"
#include "kaminpar-shm/graphutils/permutator.h"
#include "kaminpar-shm/graphutils/subgraph_extractor.h"

#include "kaminpar-common/datastructures/static_array.h"

using ::testing::AllOf;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::Le;
using ::testing::Lt;
using ::testing::UnorderedElementsAre;

namespace kaminpar::shm::testing {

std::vector<EdgeID> nodes{0, 2, 5, 7, 10};
std::vector<NodeID> edges = {2, 4, 1, 3, 5, 2, 4, 1, 3, 5, 2, 4};
std::vector<NodeWeight> node_weights = {1, 1, 1, 1};
std::vector<EdgeWeight> edge_weights = {1, 1, 1, 1, 5, 1, 2, 1, 2, 10, 5, 10};

CSRGraph testgraph(
    StaticArray<EdgeID>(nodes.begin(), nodes.end()),
    StaticArray<NodeID>(edges.begin(), edges.end()),
    StaticArray<NodeWeight>(node_weights.begin(), node_weights.end()),
    StaticArray<EdgeWeight>(edge_weights.begin(), edge_weights.end())
);

class WeightFunction : public sparsification::ScoreFunction<EdgeWeight> {
public:
  StaticArray<EdgeWeight> scores(const CSRGraph &g) override {
    return StaticArray<EdgeWeight>(g.raw_edge_weights().begin(), g.raw_edge_weights().end());
  }
};

sparsification::ThresholdSampler<EdgeWeight> t_weight(std::make_unique<WeightFunction>());

TEST(ThresholdTest, Testgraph) {
  EXPECT_EQ(edges.size(), 12);
  EXPECT_EQ(testgraph.m(), 12);
  EXPECT_EQ(testgraph.n(), 4);
}

EdgeID number_of_edges_in_sample(StaticArray<EdgeWeight> &sample, const CSRGraph &g) {
  EdgeID edges_in_sample = 0;
  sparsification::utils::for_upward_edges(g, [&](auto e) {
    if (sample[e])
      edges_in_sample++;
  });
  return edges_in_sample;
}

TEST(ThresholdTest, KeepOnlyHaviestEdge) {
  auto sample = t_weight.sample(testgraph, 2);
  ASSERT_EQ(number_of_edges_in_sample(sample, testgraph), 1);
  ASSERT_EQ(sample[9], 10);
}

TEST(ThresholdTest, KeepThreeHaviestEdges) {
  auto sample = t_weight.sample(testgraph, 6);
  ASSERT_EQ(number_of_edges_in_sample(sample, testgraph), 3);
  ASSERT_EQ(sample[4], 5);
  ASSERT_EQ(sample[6], 2);
  ASSERT_EQ(sample[9], 10);
}

TEST(ThresholdTest, KeepAllEdges) {
  auto sample = t_weight.sample(testgraph, 12);
  ASSERT_EQ(number_of_edges_in_sample(sample, testgraph), 6);
  sparsification::utils::for_upward_edges(testgraph, [&](auto e) {
    ASSERT_EQ(sample[e], edge_weights[e]);
  });
}

} // namespace kaminpar::shm::testing
