#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <gmock/gmock.h>

#include "kaminpar-common/graph_compression/compressed_neighborhoods_builder.h"

using ::testing::ElementsAre;
using ::testing::Pair;

namespace kaminpar {

TEST(CompressedNeighborhoodsTest, WeightedStreamVByteNeighborhoodContributesToTotalEdgeWeight) {
  using NodeID = std::uint32_t;
  using EdgeID = std::uint32_t;
  using EdgeWeight = std::uint32_t;

  std::vector<std::pair<NodeID, EdgeWeight>> neighborhood{
      {10, 5},
      {12, 7},
      {15, 11},
      {18, 13},
      {22, 17},
  };

  CompressedNeighborhoodsBuilder<NodeID, EdgeID, EdgeWeight> builder(1, neighborhood.size(), true);
  builder.add(0, std::span(neighborhood));
  auto compressed = builder.build();

  EXPECT_EQ(compressed.total_edge_weight(), 53);

  std::vector<std::pair<NodeID, EdgeWeight>> decoded;
  compressed.adjacent_nodes(0, [&](const NodeID v, const EdgeWeight w) {
    decoded.emplace_back(v, w);
  });

  EXPECT_THAT(
      decoded, ElementsAre(Pair(10, 5), Pair(12, 7), Pair(15, 11), Pair(18, 13), Pair(22, 17))
  );
}

} // namespace kaminpar
