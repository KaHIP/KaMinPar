/*******************************************************************************
 * @file:   jet_gain_cache_test.cc
 * @author: Daniel Seemaier
 * @date:   22.10.2026
 ******************************************************************************/
#include <atomic>
#include <limits>
#include <numeric>
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

template <bool lock_queries>
void expect_exact(
    const JetGainCache<CSRGraph, lock_queries> &cache, const PartitionedGraph &p_graph
) {
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

  EXPECT_EQ(cache.row_capacity(0), 4);
  EXPECT_LT(cache.row_capacity(0), std::min<std::size_t>(graph.csr_graph().degree(0), p_graph.k()));
  EXPECT_EQ(cache.active_entries(), 104);
  EXPECT_EQ(cache.allocated_entries(), 200);
  expect_exact(cache, p_graph);

  for (NodeID u = 1; u < graph.n(); ++u) {
    const BlockID from = p_graph.block(u);
    p_graph.set_block(u, u);
    cache.move(u, from, u);
  }
  EXPECT_EQ(cache.row_capacity(0), 100);
  EXPECT_EQ(cache.active_entries(), 200);
  EXPECT_EQ(cache.allocated_entries(), 200); // Growth reuses the reserved row slice.
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
  ASSERT_EQ(graph.csr_graph().edge_weight(0), weight);
  ASSERT_EQ(graph.csr_graph().edge_weight(1), weight);
  auto p_graph = make_p_graph(graph, 1024, {0, 1023, 1023});
  const Context ctx = create_default_context();
  JetGainCache<CSRGraph> cache(ctx, graph.n(), p_graph.k());
  cache.initialize(graph.csr_graph(), p_graph);
  EXPECT_EQ(cache.conn(0, 1023), 2 * weight);
  cache.move(1, 1023, 511);
  p_graph.set_block(1, 511);
  expect_exact(cache, p_graph);
}

TEST(JetGainCacheTest, UnlockedQueriesAreExactBetweenParallelMovePhases) {
  const tbb::global_control control(tbb::global_control::max_allowed_parallelism, 4);
  const auto graph = make_complete_bipartite_graph(8, 128, [](const NodeID u, const NodeID v) {
    return static_cast<EdgeWeight>(1 + (u + v) % 7);
  });
  auto p_graph = make_p_graph(graph, 67, std::vector<BlockID>(graph.n(), 0));
  const Context ctx = create_default_context();
  SynchronousJetGainCache<CSRGraph> cache(ctx, graph.n(), p_graph.k());
  cache.initialize(graph.csr_graph(), p_graph);

  for (int iteration = 0; iteration < 12; ++iteration) {
    // The blocking parallel_for separates all writes (including row growth) from queries.
    graph.csr_graph().pfor_nodes([&](const NodeID u) {
      const BlockID to = iteration % 2 == 0 ? (u + iteration) % p_graph.k() : 0;
      const BlockID from = p_graph.block(u);
      p_graph.set_block(u, to);
      cache.move(u, from, to);
    });
    expect_exact(cache, p_graph);
  }
}

TEST(JetGainCacheTest, SmallBatchesHandleCollisionsAndOverflowWithoutDuplicateKeys) {
  const tbb::global_control control(tbb::global_control::max_allowed_parallelism, 8);
  const auto graph = make_star_graph(1000, [](const NodeID u, const NodeID v) {
    return static_cast<EdgeWeight>(1 + (u + v) % 19);
  });
  auto p_graph = make_p_graph(graph, 127, std::vector<BlockID>(graph.n(), 0));
  const Context ctx = create_default_context();
  SynchronousJetGainCache<CSRGraph> cache(ctx, graph.n(), p_graph.k());
  cache.initialize(graph.csr_graph(), p_graph);
  ASSERT_EQ(cache.row_capacity(0), 4);
  const std::size_t reserved = cache.allocated_entries();

  for (int round = 0; round < 40; ++round) {
    std::vector<JetGainCacheMove> moves;
    for (NodeID u = 1; u <= 64; ++u) {
      // Start with many distinct targets to force overflow, then alternate shared targets,
      // zero-connectivity tombstones, and colliding new keys under heavy row contention.
      const BlockID to = round % 3 == 0 ? u : (round % 3 == 1 ? 0 : 1 + (u % 2) * 81);
      moves.push_back({u, p_graph.block(u), to});
      p_graph.set_block(u, to);
    }
    ASSERT_LE(moves.size(), graph.n() / 10);
    cache.move_batch(moves, p_graph);
    EXPECT_EQ(cache.allocated_entries(), reserved);
    expect_exact(cache, p_graph);
  }
  EXPECT_GT(cache.row_capacity(0), 4);
}

TEST(JetGainCacheTest, BatchInsertionFindsAnExistingKeyBehindATombstone) {
  const auto graph = make_star_graph(100);
  auto p_graph = make_p_graph(graph, 17, std::vector<BlockID>(graph.n(), 0));
  const Context ctx = create_default_context();
  SynchronousJetGainCache<CSRGraph> cache(ctx, graph.n(), p_graph.k());
  cache.initialize(graph.csr_graph(), p_graph);
  ASSERT_EQ(cache.row_capacity(0), 4);
  const auto move_one = [&](const NodeID u, const BlockID to) {
    const JetGainCacheMove move{u, p_graph.block(u), to};
    p_graph.set_block(u, to);
    cache.move_batch(std::span(&move, 1), p_graph);
    expect_exact(cache, p_graph);
  };
  move_one(1, 4); // Keys 0, 4, 8 collide at position zero in the four-slot table.
  move_one(2, 8);
  move_one(1, 0); // Removing key 4 leaves a tombstone before the existing key 8.
  move_one(3, 8);
  EXPECT_EQ(cache.conn(0, 8), 2);
  EXPECT_EQ(cache.row_capacity(0), 4);
}

TEST(JetGainCacheTest, LargeBatchesRebuildAndShrinkOnlyAffectedRows) {
  const auto graph = make_star_graph(100);
  auto p_graph = make_p_graph(graph, 128, std::vector<BlockID>(graph.n(), 0));
  const Context ctx = create_default_context();
  SynchronousJetGainCache<CSRGraph> cache(ctx, graph.n(), p_graph.k());
  cache.initialize(graph.csr_graph(), p_graph);
  const std::size_t reserved = cache.allocated_entries();

  std::vector<JetGainCacheMove> moves;
  for (NodeID u = 1; u <= 20; ++u) {
    moves.push_back({u, 0, u});
    p_graph.set_block(u, u);
  }
  ASSERT_GT(moves.size(), graph.n() / 10);
  cache.move_batch(moves, p_graph);
  EXPECT_EQ(cache.row_capacity(0), 26);
  expect_exact(cache, p_graph);

  for (auto &move : moves) {
    std::swap(move.from, move.to);
    p_graph.set_block(move.node, move.to);
  }
  cache.move_batch(moves, p_graph);
  EXPECT_EQ(cache.row_capacity(0), 4);
  EXPECT_EQ(cache.allocated_entries(), reserved);
  expect_exact(cache, p_graph);

  cache.move_batch({}, p_graph);
  moves = {{1, 0, 0}};
  cache.move_batch(moves, p_graph);
  expect_exact(cache, p_graph);
}

TEST(JetGainCacheTest, DirectRowsKeepFixedKeysWhenConnectivityDisappearsAndReturns) {
  const tbb::global_control control(tbb::global_control::max_allowed_parallelism, 8);
  const auto graph = make_star_graph(1000, [](const NodeID u, const NodeID v) {
    return static_cast<EdgeWeight>(1 + (u + v) % 13);
  });
  std::vector<BlockID> partition(graph.n(), 0);
  for (NodeID u = 1; u <= 64; ++u) {
    partition[u] = 1 + u % 6;
  }
  auto p_graph = make_p_graph(graph, 7, partition);
  const Context ctx = create_default_context();
  SynchronousJetGainCache<CSRGraph> cache(ctx, graph.n(), p_graph.k());
  cache.initialize(graph.csr_graph(), p_graph);
  ASSERT_EQ(cache.row_capacity(0), 7);

  for (int round = 0; round < 30; ++round) {
    std::vector<JetGainCacheMove> moves;
    for (NodeID u = 1; u <= 64; ++u) {
      const BlockID to = round % 2 == 0 ? 0 : 1 + u % 6;
      moves.push_back({u, p_graph.block(u), to});
      p_graph.set_block(u, to);
    }
    cache.move_batch(moves, p_graph);
    EXPECT_EQ(cache.row_capacity(0), 7);
    expect_exact(cache, p_graph);
  }
}

TEST(JetGainCacheTest, MixedBatchSizesRemainExactForAdjacentMovedVertices) {
  const tbb::global_control control(tbb::global_control::max_allowed_parallelism, 8);
  const auto graph = make_complete_bipartite_graph(16, 256, [](const NodeID u, const NodeID v) {
    return static_cast<EdgeWeight>(1 + (u + v) % 29);
  });
  auto p_graph = make_p_graph(graph, 17, std::vector<BlockID>(graph.n(), 0));
  const Context ctx = create_default_context();
  SynchronousJetGainCache<CSRGraph> cache(ctx, graph.n(), p_graph.k());
  cache.initialize(graph.csr_graph(), p_graph);
  std::mt19937 random(42);
  std::vector<NodeID> nodes(graph.n());
  std::iota(nodes.begin(), nodes.end(), 0);

  for (int round = 0; round < 60; ++round) {
    std::shuffle(nodes.begin(), nodes.end(), random);
    const std::size_t count = round % 2 == 0 ? 20 : 100;
    std::vector<JetGainCacheMove> moves;
    for (std::size_t i = 0; i < count; ++i) {
      const NodeID u = nodes[i];
      const BlockID to = random() % p_graph.k();
      moves.push_back({u, p_graph.block(u), to});
      p_graph.set_block(u, to);
    }
    cache.move_batch(moves, p_graph);
    expect_exact(cache, p_graph);
  }
}

TEST(JetGainCacheTest, SmallBatchDoesNotPackWeightsDuringExchange) {
  constexpr EdgeWeight weight = std::numeric_limits<EdgeWeight>::max() / 4;
  // Two heavy neighbors exchange blocks through the atomic update path.
  const auto graph = make_star_graph(20, [](const NodeID u, const NodeID v) {
    return std::max(u, v) <= 2 ? weight : EdgeWeight{1};
  });
  std::vector<BlockID> partition(graph.n(), 0);
  partition[1] = 1023;
  partition[2] = 511;
  auto p_graph = make_p_graph(graph, 1024, partition);
  const Context ctx = create_default_context();
  SynchronousJetGainCache<CSRGraph> cache(ctx, graph.n(), p_graph.k());
  cache.initialize(graph.csr_graph(), p_graph);
  const std::vector<JetGainCacheMove> moves = {{1, 1023, 511}, {2, 511, 1023}};
  p_graph.set_block(1, 511);
  p_graph.set_block(2, 1023);
  cache.move_batch(moves, p_graph);
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
  EXPECT_EQ(cache.row_capacity(0), 4);
  expect_exact(cache, p_star);

  const auto empty = make_empty_graph(4);
  const auto p_empty = make_p_graph(empty, 1, {0, 0, 0, 0});
  const std::size_t reserved = cache.allocated_entries();
  cache.initialize(empty.csr_graph(), p_empty);
  EXPECT_EQ(cache.allocated_entries(), reserved);
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
