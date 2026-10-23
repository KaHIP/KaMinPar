/*******************************************************************************
 * @file:   gain_cache_test.cc
 * @author: Daniel Seemaier
 * @date:   16.08.2024
 ******************************************************************************/
#include <gmock/gmock.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/gains/compact_hashing_gain_cache.h"
#include "kaminpar-shm/refinement/gains/dense_gain_cache.h"
#include "kaminpar-shm/refinement/gains/hashing_gain_cache.h"
#include "kaminpar-shm/refinement/gains/on_the_fly_gain_cache.h"
#include "kaminpar-shm/refinement/gains/sparse_gain_cache.h"

namespace {
using namespace kaminpar::shm;
using namespace kaminpar::shm::testing;

template <typename GainCacheType> class GainCacheTest : public ::testing::Test {
public:
  void init(const PartitionedGraph &p_graph) {
    _ctx.partition.setup(p_graph.graph(), p_graph.k(), 0.03);

    this->_gain_cache = std::make_unique<GainCacheType>(this->_ctx, p_graph.n(), p_graph.k());
    this->_gain_cache->initialize(p_graph.graph().csr_graph(), p_graph);
  }

  void free() {
    this->_gain_cache->free();
    this->_gain_cache = nullptr;
  }

  void move(PartitionedGraph &p_graph, const NodeID node, const BlockID to) {
    this->_gain_cache->move(node, p_graph.block(node), to);
    p_graph.set_block(node, to);
  }

  std::unique_ptr<GainCacheType> _gain_cache = nullptr;
  Context _ctx = create_default_context();
};

using GainCacheTypes = ::testing::Types<
    NormalDenseGainCache<CSRGraph>,
    NormalHashingGainCache<CSRGraph>,
    OnTheFlyGainCache<CSRGraph>,
    NormalSparseGainCache<CSRGraph>,
    NormalCompactHashingGainCache<CSRGraph>>;

TYPED_TEST_SUITE(GainCacheTest, GainCacheTypes);

TYPED_TEST(GainCacheTest, ScalarAndPairQueriesSupportAliasedBlocks) {
  const Graph graph = make_complete_graph(8);
  auto p_graph = make_p_graph(graph, 4, {0, 0, 1, 1, 2, 2, 3, 3});
  this->init(p_graph);
  auto check = [&] {
    for (BlockID from = 0; from < 4; ++from) {
      const EdgeWeight source = this->_gain_cache->conn(0, from);
      for (BlockID first = 0; first < 4; ++first) {
        const EdgeWeight expected_first = this->_gain_cache->conn(0, first) - source;
        EXPECT_EQ(this->_gain_cache->gain(0, from, first), expected_first);
        for (BlockID second = 0; second < 4; ++second) {
          const auto gains = this->_gain_cache->gain(0, from, {first, second});
          EXPECT_EQ(gains.first, expected_first);
          EXPECT_EQ(gains.second, this->_gain_cache->conn(0, second) - source);
        }
      }
    }
  };
  check();
  this->move(p_graph, 1, 3);
  check();
}

TEST(OnTheFlyGainCacheTest, ReinitializationCanGrowTheBlockRange) {
  const Graph graph = make_complete_graph(6);
  Context ctx = create_default_context();
  auto small = make_p_graph(graph, 2, {0, 0, 1, 1, 0, 1});
  ctx.partition.setup(graph, 2, .03);
  OnTheFlyGainCache<CSRGraph> cache(ctx, graph.n(), 2);
  cache.initialize(graph.csr_graph(), small);
  cache.gains(0, 0, [](BlockID, auto &&) {});

  constexpr BlockID k = 1024;
  auto large = make_p_graph(graph, k, {0, 1, 2, 254, 511, 1023});
  ctx.partition.setup(graph, k, .03);
  cache.initialize(graph.csr_graph(), large);
  std::vector<unsigned> visits(k);
  cache.gains(0, 0, [&](const BlockID to, auto &&gain) {
    ASSERT_LT(to, k);
    EXPECT_EQ(gain(), cache.conn(0, to) - cache.conn(0, 0));
    ++visits[to];
  });
  EXPECT_EQ(visits[0], 0u);
  for (BlockID to = 1; to < k; ++to) {
    EXPECT_EQ(visits[to], 1u);
  }
}

TYPED_TEST(GainCacheTest, ObjectConstructionWorks) {
  this->_gain_cache = std::make_unique<TypeParam>(this->_ctx, 0, 0);
}

TYPED_TEST(GainCacheTest, InitWorksOnEmptyGraphWith0Nodes) {
  auto empty = make_empty_graph(0);
  auto p_empty = make_p_graph(empty, 2, {});

  this->init(p_empty);
  this->free();
}

TYPED_TEST(GainCacheTest, InitWorksOnGraphWithSingleNode) {
  auto single_node_graph = make_empty_graph(1);
  auto p_single_node_graph = make_p_graph(single_node_graph, 1, {0});

  this->init(p_single_node_graph);

  EXPECT_EQ(this->_gain_cache->conn(0, 0), 0);
  EXPECT_EQ(this->_gain_cache->gain(0, 0, 0), 0);

  this->free();
}

TYPED_TEST(GainCacheTest, InitWorksOnEmptyGraphWith4Nodes) {
  auto empty = make_empty_graph(4);
  auto p_empty = make_p_graph(empty, 2, {0, 0, 1, 1});

  this->init(p_empty);

  EXPECT_EQ(this->_gain_cache->conn(0, 0), 0);
  EXPECT_EQ(this->_gain_cache->conn(0, 1), 0);
  EXPECT_EQ(this->_gain_cache->conn(1, 0), 0);
  EXPECT_EQ(this->_gain_cache->conn(1, 1), 0);
  EXPECT_EQ(this->_gain_cache->conn(2, 0), 0);
  EXPECT_EQ(this->_gain_cache->conn(2, 1), 0);
  EXPECT_EQ(this->_gain_cache->conn(3, 0), 0);
  EXPECT_EQ(this->_gain_cache->conn(3, 1), 0);

  EXPECT_EQ(this->_gain_cache->gain(0, 0, 1), 0);
  EXPECT_EQ(this->_gain_cache->gain(1, 0, 1), 0);
  EXPECT_EQ(this->_gain_cache->gain(2, 1, 0), 0);
  EXPECT_EQ(this->_gain_cache->gain(2, 1, 0), 0);

  this->_gain_cache->gains(0, 0, [&](BlockID, auto &&gain) { EXPECT_EQ(gain(), 0); });

  this->free();
}

TYPED_TEST(GainCacheTest, InitWorksOnBipartiteStarGraphWith4Nodes) {
  const Graph abstract_graph = make_star_graph(3);
  const CSRGraph &star = abstract_graph.csr_graph();
  EXPECT_EQ(star.degree(0), 3); // Node 0 is the center of the star

  auto p_star = make_p_graph(abstract_graph, 2, {0, 1, 1, 1});

  this->init(p_star);

  EXPECT_EQ(this->_gain_cache->conn(0, 0), 0);
  EXPECT_EQ(this->_gain_cache->conn(0, 1), 3);

  EXPECT_EQ(this->_gain_cache->conn(1, 0), 1);
  EXPECT_EQ(this->_gain_cache->conn(1, 1), 0);
  EXPECT_EQ(this->_gain_cache->conn(2, 0), 1);
  EXPECT_EQ(this->_gain_cache->conn(2, 1), 0);
  EXPECT_EQ(this->_gain_cache->conn(3, 0), 1);
  EXPECT_EQ(this->_gain_cache->conn(3, 1), 0);

  this->free();
}

TYPED_TEST(GainCacheTest, InitWorksOn4PartiteStarGraphWith4Nodes) {
  const Graph abstract_graph = make_star_graph(3);
  const CSRGraph &star = abstract_graph.csr_graph();
  EXPECT_EQ(star.degree(0), 3); // Node 0 is the center of the star

  auto p_star = make_p_graph(abstract_graph, 4, {0, 1, 2, 3});

  this->init(p_star);

  EXPECT_EQ(this->_gain_cache->conn(0, 0), 0);
  EXPECT_EQ(this->_gain_cache->conn(0, 1), 1);
  EXPECT_EQ(this->_gain_cache->conn(0, 2), 1);
  EXPECT_EQ(this->_gain_cache->conn(0, 3), 1);
  this->_gain_cache->gains(0, 0, [&](const BlockID to, auto &&gain) {
    EXPECT_EQ(gain(), to == 0 ? 0 : 1);
  });

  for (const BlockID leaf : {1, 2, 3}) {
    EXPECT_EQ(this->_gain_cache->conn(leaf, 0), 1);
    EXPECT_EQ(this->_gain_cache->conn(leaf, 1), 0);
    EXPECT_EQ(this->_gain_cache->conn(leaf, 2), 0);
    EXPECT_EQ(this->_gain_cache->conn(leaf, 3), 0);

    this->_gain_cache->gains(leaf, leaf, [&](const BlockID to, auto &&gain) {
      EXPECT_EQ(gain(), to == 0 ? 1 : 0);
    });
  }

  this->free();
}

TYPED_TEST(GainCacheTest, InitWorksOnGraphWithTwoConnectedNodes) {
  auto two_node_graph = make_path_graph(2);
  auto p_two_node_graph = make_p_graph(two_node_graph, 2, {0, 1});

  this->init(p_two_node_graph);

  EXPECT_EQ(this->_gain_cache->conn(0, 0), 0);
  EXPECT_EQ(this->_gain_cache->conn(0, 1), 1);
  EXPECT_EQ(this->_gain_cache->conn(1, 0), 1);
  EXPECT_EQ(this->_gain_cache->conn(1, 1), 0);

  this->free();
}

TYPED_TEST(GainCacheTest, MoveWorksOn4PartiteStarGraphWith4Nodes) {
  const Graph abstract_graph = make_star_graph(3);
  const CSRGraph &star = abstract_graph.csr_graph();
  EXPECT_EQ(star.degree(0), 3); // Node 0 is the center of the star

  auto p_star = make_p_graph(abstract_graph, 4, {0, 1, 2, 3});

  this->init(p_star);

  // Move center of the star to block 1 == block of node 1
  this->move(p_star, 0, 1);

  EXPECT_EQ(this->_gain_cache->conn(0, 0), 0);
  EXPECT_EQ(this->_gain_cache->conn(0, 1), 1);
  EXPECT_EQ(this->_gain_cache->conn(0, 2), 1);
  EXPECT_EQ(this->_gain_cache->conn(0, 3), 1);

  for (const auto leaf : {1, 2, 3}) {
    EXPECT_EQ(this->_gain_cache->conn(leaf, 0), 0);
    EXPECT_EQ(this->_gain_cache->conn(leaf, 1), 1);
    EXPECT_EQ(this->_gain_cache->conn(leaf, 2), 0);
    EXPECT_EQ(this->_gain_cache->conn(leaf, 3), 0);
  }

  this->free();
}

TYPED_TEST(GainCacheTest, MoveWorksOnGraphWithTwoConnectedNodes) {
  auto two_node_graph = make_path_graph(2);
  auto p_two_node_graph = make_p_graph(two_node_graph, 2, {0, 1});

  this->init(p_two_node_graph);

  this->move(p_two_node_graph, 0, 1);

  EXPECT_EQ(this->_gain_cache->conn(0, 0), 0);
  EXPECT_EQ(this->_gain_cache->conn(0, 1), 1);
  EXPECT_EQ(this->_gain_cache->conn(1, 0), 0);
  EXPECT_EQ(this->_gain_cache->conn(1, 1), 1);

  this->free();
}

TYPED_TEST(GainCacheTest, MoveWorksOnGraphWithDisconnectedNodes) {
  auto disconnected_graph = make_empty_graph(4);
  auto p_disconnected_graph = make_p_graph(disconnected_graph, 4, {0, 1, 2, 3});

  this->init(p_disconnected_graph);

  this->move(p_disconnected_graph, 0, 1);

  EXPECT_EQ(this->_gain_cache->conn(0, 0), 0);
  EXPECT_EQ(this->_gain_cache->conn(0, 1), 0);
  EXPECT_EQ(this->_gain_cache->conn(0, 2), 0);
  EXPECT_EQ(this->_gain_cache->conn(0, 3), 0);

  this->free();
}

TYPED_TEST(GainCacheTest, InitWorksOnCompleteGraph) {
  auto complete_graph = make_complete_graph(5); // Create a complete graph with 5 nodes
  auto p_complete_graph = make_p_graph(complete_graph, 3, {0, 1, 2, 0, 1});

  this->init(p_complete_graph);

  EXPECT_EQ(this->_gain_cache->conn(0, 0), 1);
  EXPECT_EQ(this->_gain_cache->conn(0, 1), 2);
  EXPECT_EQ(this->_gain_cache->conn(0, 2), 1);
  EXPECT_EQ(this->_gain_cache->conn(1, 0), 2);
  EXPECT_EQ(this->_gain_cache->conn(1, 1), 1);
  EXPECT_EQ(this->_gain_cache->conn(1, 2), 1);
  EXPECT_EQ(this->_gain_cache->conn(2, 0), 2);
  EXPECT_EQ(this->_gain_cache->conn(2, 1), 2);
  EXPECT_EQ(this->_gain_cache->conn(2, 2), 0);
  EXPECT_EQ(this->_gain_cache->conn(3, 0), 1);
  EXPECT_EQ(this->_gain_cache->conn(3, 1), 2);
  EXPECT_EQ(this->_gain_cache->conn(3, 2), 1);
  EXPECT_EQ(this->_gain_cache->conn(4, 0), 2);
  EXPECT_EQ(this->_gain_cache->conn(4, 1), 1);
  EXPECT_EQ(this->_gain_cache->conn(4, 2), 1);

  this->free();
}

TYPED_TEST(GainCacheTest, MoveWorksOnCompleteGraph) {
  auto complete_graph = make_complete_graph(5); // Create a complete graph with 5 nodes
  auto p_complete_graph = make_p_graph(complete_graph, 3, {0, 1, 2, 0, 1});

  this->init(p_complete_graph);

  this->move(p_complete_graph, 0, 1);
  this->move(p_complete_graph, 1, 2);
  this->move(p_complete_graph, 2, 0);
  this->move(p_complete_graph, 3, 1);
  this->move(p_complete_graph, 4, 2);

  EXPECT_EQ(this->_gain_cache->conn(0, 0), 1);
  EXPECT_EQ(this->_gain_cache->conn(0, 1), 1);
  EXPECT_EQ(this->_gain_cache->conn(0, 2), 2);
  EXPECT_EQ(this->_gain_cache->conn(1, 0), 1);
  EXPECT_EQ(this->_gain_cache->conn(1, 1), 2);
  EXPECT_EQ(this->_gain_cache->conn(1, 2), 1);
  EXPECT_EQ(this->_gain_cache->conn(2, 0), 0);
  EXPECT_EQ(this->_gain_cache->conn(2, 1), 2);
  EXPECT_EQ(this->_gain_cache->conn(2, 2), 2);
  EXPECT_EQ(this->_gain_cache->conn(3, 0), 1);
  EXPECT_EQ(this->_gain_cache->conn(3, 1), 1);
  EXPECT_EQ(this->_gain_cache->conn(3, 2), 2);
  EXPECT_EQ(this->_gain_cache->conn(4, 0), 1);
  EXPECT_EQ(this->_gain_cache->conn(4, 1), 2);
  EXPECT_EQ(this->_gain_cache->conn(4, 2), 1);

  this->free();
}

} // namespace
