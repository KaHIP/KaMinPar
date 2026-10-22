/*******************************************************************************
 * @file:   jet_gain_cache_test.cc
 * @author: Daniel Seemaier
 * @date:   22.10.2026
 ******************************************************************************/
#include <atomic>
#include <limits>
#include <random>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/refinement/gains/jet_gain_cache.h"

namespace kaminpar::shm {
namespace {

using namespace testing;

void expect_exact(const JetGainCache<CSRGraph> &cache, const PartitionedGraph &p_graph) {
  for (const NodeID u : p_graph.graph().csr_graph().nodes()) {
    std::vector<EdgeWeight> expected(p_graph.k(), 0);
    p_graph.graph().csr_graph().adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
      expected[p_graph.block(v)] += weight;
    });
    std::vector<EdgeWeight> actual(p_graph.k(), 0);
    cache.for_each_connection(u, [&](const BlockID block, const EdgeWeight weight) {
      ASSERT_LT(block, p_graph.k());
      EXPECT_EQ(actual[block], 0);
      EXPECT_GT(weight, 0);
      actual[block] = weight;
    });
    EXPECT_EQ(actual, expected);
    const BlockID from = p_graph.block(u);
    bool border = false;
    for (const BlockID to : p_graph.blocks()) {
      EXPECT_EQ(cache.conn(u, to), expected[to]);
      EXPECT_EQ(cache.gain(u, from, to), expected[to] - expected[from]);
      border |= to != from && expected[to] > 0;
    }
    EXPECT_EQ(cache.is_border_node(u, from), border);
    std::vector<EdgeWeight> adjacent(p_graph.k(), 0);
    cache.gains(u, from, [&](const BlockID to, auto &&conn) {
      EXPECT_NE(to, from);
      adjacent[to] = conn();
    });
    expected[from] = 0;
    EXPECT_EQ(adjacent, expected);
  }
}

TEST(JetGainCacheTest, SizesRowsFromObservedConnectionsAndGrowsWithoutLosingUpdates) {
  const auto graph = make_star_graph(100);
  std::vector<BlockID> partition(graph.n(), 0);
  partition[0] = 1;
  auto p_graph = make_p_graph(graph, 128, partition);
  const Context ctx = create_default_context();
  JetGainCache<CSRGraph> cache(ctx, graph.n(), p_graph.k());
  cache.initialize(graph.csr_graph(), p_graph);

  EXPECT_EQ(cache.row_capacity(0), 2);
  EXPECT_LT(cache.row_capacity(0), std::min<std::size_t>(graph.csr_graph().degree(0), p_graph.k()));
  EXPECT_EQ(cache.active_entries(), 102);
  EXPECT_EQ(cache.allocated_entries(), 102);
  expect_exact(cache, p_graph);

  for (NodeID u = 1; u < graph.n(); ++u) {
    const BlockID from = p_graph.block(u);
    p_graph.set_block(u, u);
    cache.move(u, from, u);
  }
  EXPECT_EQ(cache.row_capacity(0), 100);
  EXPECT_EQ(cache.active_entries(), 200);
  EXPECT_EQ(cache.allocated_entries(), 202); // Includes the abandoned initial two-slot row.
  expect_exact(cache, p_graph);
}

TEST(JetGainCacheTest, WeightedMovesPreserveProbeChainsIncludingFullRows) {
  const auto graph = make_complete_graph(19, [](const NodeID u, const NodeID v) {
    return static_cast<EdgeWeight>(1 + (std::min(u, v) * 7 + std::max(u, v) * 11) % 29);
  });
  auto p_graph = make_p_graph(graph, 17, std::vector<BlockID>(graph.n(), 0));
  const Context ctx = create_default_context();
  JetGainCache<CSRGraph> cache(ctx, graph.n(), p_graph.k());
  cache.initialize(graph.csr_graph(), p_graph);
  std::mt19937 random(42);

  for (int step = 0; step < 2000; ++step) {
    const NodeID u = random() % graph.n();
    const BlockID to = random() % p_graph.k();
    const BlockID from = p_graph.block(u);
    cache.move(u, from, to);
    p_graph.set_block(u, to);
    if (step % 31 == 0) {
      expect_exact(cache, p_graph);
    }
  }
  expect_exact(cache, p_graph);
}

TEST(JetGainCacheTest, ConcurrentGrowthDeletionAndQueriesKeepEachRowConsistent) {
  const tbb::global_control control(tbb::global_control::max_allowed_parallelism, 4);
  const auto graph = make_star_graph(256);
  auto p_graph = make_p_graph(graph, 257, std::vector<BlockID>(graph.n(), 0));
  const Context ctx = create_default_context();
  JetGainCache<CSRGraph> cache(ctx, graph.n(), p_graph.k());
  cache.initialize(graph.csr_graph(), p_graph);
  std::atomic<bool> done = false;
  std::atomic<int> invalid_rows = 0;
  std::thread reader([&] {
    while (!done.load(std::memory_order_relaxed)) {
      EdgeWeight sum = 0;
      cache.for_each_connection(0, [&](const BlockID block, const EdgeWeight weight) {
        if (block >= p_graph.k() || weight <= 0) {
          invalid_rows.fetch_add(1, std::memory_order_relaxed);
        }
        sum += weight;
      });
      if (sum != 256) {
        invalid_rows.fetch_add(1, std::memory_order_relaxed);
      }
      const EdgeWeight gain = cache.gain(0, 0, 1);
      if (gain < -256 || gain > 256) {
        invalid_rows.fetch_add(1, std::memory_order_relaxed);
      }
    }
  });
  for (int iteration = 0; iteration < 12; ++iteration) {
    tbb::parallel_for<NodeID>(1, graph.n(), [&](const NodeID u) {
      const BlockID to = iteration % 2 == 0 ? u : 0;
      const BlockID from = p_graph.block(u);
      p_graph.set_block(u, to);
      cache.move(u, from, to);
    });
  }
  done.store(true, std::memory_order_relaxed);
  reader.join();
  EXPECT_EQ(invalid_rows.load(), 0);
  expect_exact(cache, p_graph);
}

TEST(JetGainCacheTest, DoesNotPackOrTruncateEdgeWeights) {
  constexpr EdgeWeight weight = std::numeric_limits<EdgeWeight>::max() / 4;
  const auto graph = make_star_graph(2, [](NodeID, NodeID) { return weight; });
  auto p_graph = make_p_graph(graph, 1024, {0, 1023, 1023});
  const Context ctx = create_default_context();
  JetGainCache<CSRGraph> cache(ctx, graph.n(), p_graph.k());
  cache.initialize(graph.csr_graph(), p_graph);
  EXPECT_EQ(cache.conn(0, 1023), 2 * weight);
  cache.move(1, 1023, 511);
  p_graph.set_block(1, 511);
  expect_exact(cache, p_graph);
}

TEST(JetGainCacheTest, ReinitializesAfterRollbackAndGraphOrBlockCountChanges) {
  const Context ctx = create_default_context();
  JetGainCache<CSRGraph> cache(ctx, 0, 0);
  const auto star = make_star_graph(20);
  auto p_star = make_p_graph(star, 25, std::vector<BlockID>(star.n(), 0));
  cache.initialize(star.csr_graph(), p_star);
  for (NodeID u = 1; u < star.n(); ++u) {
    cache.move(u, 0, u);
    p_star.set_block(u, u);
  }
  for (const NodeID u : star.csr_graph().nodes()) {
    p_star.set_block(u, 0);
  }
  cache.initialize(star.csr_graph(), p_star);
  EXPECT_EQ(cache.row_capacity(0), 2);
  expect_exact(cache, p_star);

  const auto empty = make_empty_graph(4);
  const auto p_empty = make_p_graph(empty, 1, {0, 0, 0, 0});
  cache.initialize(empty.csr_graph(), p_empty);
  EXPECT_EQ(cache.active_entries(), 0);
  expect_exact(cache, p_empty);

  const auto no_nodes = make_empty_graph(0);
  const auto p_no_nodes = make_p_graph(no_nodes, 2, {});
  cache.initialize(no_nodes.csr_graph(), p_no_nodes);
  EXPECT_EQ(cache.active_entries(), 0);
  cache.free();
  cache.initialize(star.csr_graph(), p_star);
  expect_exact(cache, p_star);
}

} // namespace
} // namespace kaminpar::shm
