#include <barrier>
#include <thread>

#include <gtest/gtest.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/refinement/gains/compact_hashing_gain_cache.h"
#include "kaminpar-shm/refinement/gains/dense_gain_cache.h"
#include "kaminpar-shm/refinement/gains/hashing_gain_cache.h"
#include "kaminpar-shm/refinement/gains/sparse_gain_cache.h"

namespace {
using namespace kaminpar::shm;
using namespace kaminpar::shm::testing;

template <typename GainCache> class ConcurrentGainCacheTest : public ::testing::Test {};

using ConcurrentGainCacheTypes = ::testing::Types<
    NormalDenseGainCache<CSRGraph>,
    NormalHashingGainCache<CSRGraph>,
    NormalSparseGainCache<CSRGraph>,
    NormalCompactHashingGainCache<CSRGraph>>;
TYPED_TEST_SUITE(ConcurrentGainCacheTest, ConcurrentGainCacheTypes);

TYPED_TEST(ConcurrentGainCacheTest, BorderQueriesCanOverlapNeighborMoves) {
  // Equal degrees satisfy the degree-bucket ordering. With k > degree, all three hash-based
  // caches use per-node hash tables whose locks share storage with the weighted degree.
  GraphBuilder builder;
  for (NodeID node = 0; node < 8; ++node) {
    builder.new_node();
    for (NodeID neighbor = 0; neighbor < 8; ++neighbor) {
      if (node != neighbor) {
        builder.new_edge(neighbor);
      }
    }
  }
  const Graph graph = builder.build(true);
  auto p_graph = make_p_graph(graph, 16, {0, 1, 1, 1, 1, 1, 1, 1});
  Context ctx = create_default_context();
  ctx.partition.setup(graph, p_graph.k(), 0.03);
  TypeParam gain_cache(ctx, graph.n(), p_graph.k());
  gain_cache.initialize(graph.csr_graph(), p_graph);

  std::barrier start(3);
  auto move_neighbor = [&](const NodeID node) {
    start.arrive_and_wait();
    for (std::size_t i = 0; i < 4000; ++i) {
      gain_cache.move(node, 1, 2);
      gain_cache.move(node, 2, 1);
    }
  };
  std::thread first_writer(move_neighbor, 1);
  std::thread second_writer(move_neighbor, 2);
  start.arrive_and_wait();

  // Node 0 never has a neighbor in its own block, so its border status stays true even while
  // connectivity to blocks 1 and 2 changes. TSAN checks the overlapping weighted-degree reads.
  for (std::size_t i = 0; i < 8000; ++i) {
    EXPECT_TRUE(gain_cache.is_border_node(0, 0));
  }

  first_writer.join();
  second_writer.join();
  for (const NodeID node : graph.csr_graph().nodes()) {
    EXPECT_EQ(gain_cache.conn(node, 0), node == 0 ? 0 : 1);
    EXPECT_EQ(gain_cache.conn(node, 1), node == 0 ? 7 : 6);
    EXPECT_EQ(gain_cache.conn(node, 2), 0);
  }
}
} // namespace
