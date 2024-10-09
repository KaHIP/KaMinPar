#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"
#include "tests/shm/matchers.h"

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/coarsening/sparsification/IndependentRandomSampler.h"
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
    std::move(StaticArray<EdgeID>(nodes.begin(), nodes.end())),
    std::move(StaticArray<NodeID>(edges.begin(), edges.end())),
    std::move(StaticArray<NodeWeight>(node_weights.begin(), node_weights.end())),
    std::move(StaticArray<EdgeWeight>(edge_weights.begin(), edge_weights.end()))
);
class WeightFunction : public sparsification::ScoreFunction<EdgeWeight> {
public:
  StaticArray<EdgeWeight> scores(const CSRGraph &g) override {
    return StaticArray<EdgeWeight>(g.raw_edge_weights().begin(), g.raw_edge_weights().end());
  };
};
sparsification::IndependentRandomSampler<EdgeWeight> ir_weight(std::make_unique<WeightFunction>());
TEST(IRTest, Testgraph) {
  EXPECT_EQ(edges.size(), 12);
  EXPECT_EQ(testgraph.m(), 12);
  EXPECT_EQ(testgraph.n(), 4);
}

TEST(IRTest, NormalizationFactorForSmallGraph1) {
  EXPECT_NEAR(ir_weight.exactNormalizationFactor(testgraph, testgraph.raw_edge_weights(), 6), 0.2, 1e-6);
}
TEST(IRTest, NormalizationFactorForSmallGraph2) {
  EXPECT_NEAR(ir_weight.exactNormalizationFactor(testgraph, testgraph.raw_edge_weights(), 4), 0.1, 1e-6);
}
} // namespace kaminpar::shm::testing