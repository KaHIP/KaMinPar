/*******************************************************************************
 * @file:   largek_delta_gain_cache_test.cc
 * @author: Daniel Seemaier
 * @date:   22.10.2026
 ******************************************************************************/
#include <algorithm>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/refinement/gains/compact_hashing_gain_cache.h"
#include "kaminpar-shm/refinement/gains/fm_jet_gain_cache.h"

namespace kaminpar::shm {
namespace {

using namespace testing;

using ExactCompactGainCache =
    CompactHashingGainCache<CSRGraph, LargeKGenericDeltaGainCache, false, true>;

template <typename GainCache> class LargeKDeltaGainCacheTest : public ::testing::Test {
public:
  using DeltaGainCache = typename GainCache::DeltaGainCache;

  void expect_exact(
      const GainCache &shared, const DeltaGainCache &delta, const DeltaPartitionedGraph &d_graph
  ) {
    for (const NodeID u : d_graph.graph().csr_graph().nodes()) {
      std::vector<EdgeWeight> expected(d_graph.k(), 0);
      d_graph.graph().csr_graph().adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
        expected[d_graph.block(v)] += weight;
      });

      const BlockID from = d_graph.block(u);
      std::vector<unsigned int> visits(d_graph.k(), 0);
      delta.gains(u, from, [&](const BlockID to, auto &&value) {
        ASSERT_LT(to, d_graph.k());
        EXPECT_NE(to, from);
        EXPECT_EQ(++visits[to], 1);
        EXPECT_GT(expected[to], 0);
        EXPECT_EQ(value(), expected[to] - (GainCache::kIteratesExactGains ? expected[from] : 0));
        // A consumer may query the shared cache after the row snapshot has completed.
        // With Jet this would deadlock if gains() invoked us while holding its row lock.
        EXPECT_GE(shared.conn(u, to), 0);
      });

      for (const BlockID to : d_graph.blocks()) {
        EXPECT_EQ(delta.conn(u, to), expected[to]);
        EXPECT_EQ(delta.gain(u, from, to), expected[to] - expected[from]);
        EXPECT_EQ(visits[to], static_cast<unsigned int>(to != from && expected[to] > 0));
      }
    }
  }

  Context ctx = create_default_context();
};

using GainCacheTypes = ::testing::
    Types<LargeKCompactHashingGainCache<CSRGraph>, FMJetGainCache<CSRGraph>, ExactCompactGainCache>;
TYPED_TEST_SUITE(LargeKDeltaGainCacheTest, GainCacheTypes);

TYPED_TEST(LargeKDeltaGainCacheTest, NewLocalTargetUsesConnectivityOrExactGainAsRequested) {
  const auto graph = make_star_graph(3, [](const NodeID u, const NodeID v) {
    const NodeID leaf = std::max(u, v);
    return static_cast<EdgeWeight>(leaf == 1 ? 5 : (leaf == 2 ? 2 : 3));
  });
  auto p_graph = make_p_graph(graph, 4, {0, 0, 1, 0});
  TypeParam shared(this->ctx, graph.n(), p_graph.k());
  shared.initialize(graph.csr_graph(), p_graph);
  DeltaPartitionedGraph d_graph(&p_graph);
  typename TypeParam::DeltaGainCache delta(shared, d_graph);

  delta.move(3, 0, 2);
  d_graph.set_block(3, 2);
  this->expect_exact(shared, delta, d_graph);
  EXPECT_EQ(delta.conn(0, 0), 5);
  EXPECT_EQ(delta.conn(0, 1), 2);
  EXPECT_EQ(delta.conn(0, 2), 3);
}

TYPED_TEST(LargeKDeltaGainCacheTest, KeepsLocalTargetWhenLastSharedConnectionDisappears) {
  const auto graph = make_star_graph(2, [](const NodeID u, const NodeID v) {
    return static_cast<EdgeWeight>(std::max(u, v));
  });
  auto p_graph = make_p_graph(graph, 4, {0, 1, 0});
  TypeParam shared(this->ctx, graph.n(), p_graph.k());
  shared.initialize(graph.csr_graph(), p_graph);
  DeltaPartitionedGraph d_graph(&p_graph);
  typename TypeParam::DeltaGainCache delta(shared, d_graph);

  // Block 1 is initially adjacent, so recording only newly adjacent blocks loses it.
  delta.move(2, 0, 1);
  d_graph.set_block(2, 1);
  shared.move(1, 1, 2);
  p_graph.set_block(1, 2);

  EXPECT_EQ(shared.conn(0, 1), 0);
  EXPECT_EQ(delta.conn(0, 1), 2);
  this->expect_exact(shared, delta, d_graph);
}

TYPED_TEST(LargeKDeltaGainCacheTest, EmitsEachSharedAndLocalTargetOnce) {
  const auto graph = make_star_graph(3);
  auto p_graph = make_p_graph(graph, 4, {0, 0, 0, 0});
  TypeParam shared(this->ctx, graph.n(), p_graph.k());
  shared.initialize(graph.csr_graph(), p_graph);
  DeltaPartitionedGraph d_graph(&p_graph);
  typename TypeParam::DeltaGainCache delta(shared, d_graph);

  delta.move(1, 0, 1);
  d_graph.set_block(1, 1);
  shared.move(2, 0, 1);
  p_graph.set_block(2, 1);
  this->expect_exact(shared, delta, d_graph);

  // Cancellation must remove the target from iteration even though its delta was touched.
  delta.move(1, 1, 0);
  d_graph.set_block(1, 0);
  shared.move(2, 1, 0);
  p_graph.set_block(2, 0);
  this->expect_exact(shared, delta, d_graph);

  // The same delta-map entry can become nonzero again after cancelling to zero.
  delta.move(1, 0, 1);
  d_graph.set_block(1, 1);
  this->expect_exact(shared, delta, d_graph);

  delta.clear();
  d_graph.clear();
  this->expect_exact(shared, delta, d_graph);
}

TYPED_TEST(LargeKDeltaGainCacheTest, WeightedLocalAndSharedMovesMatchPartitionOracle) {
  const auto graph = make_complete_graph(41, [](const NodeID u, const NodeID v) {
    return static_cast<EdgeWeight>(1 + (std::min(u, v) * 7 + std::max(u, v) * 13) % 31);
  });
  auto p_graph = make_p_graph(graph, 67, std::vector<BlockID>(graph.n(), 0));
  TypeParam shared(this->ctx, graph.n(), p_graph.k());
  shared.initialize(graph.csr_graph(), p_graph);
  DeltaPartitionedGraph d_graph(&p_graph);
  typename TypeParam::DeltaGainCache delta(shared, d_graph);
  std::mt19937 random(20261022);

  // Disjoint ownership mirrors independent FM workers: one keeps local moves while the
  // other commits shared moves. Their incident gain-cache rows can still overlap.
  for (unsigned int step = 0; step < 700; ++step) {
    const NodeID u = random() % graph.n();
    const BlockID to = random() % p_graph.k();
    if (u < graph.n() / 2) {
      const BlockID from = d_graph.block(u);
      delta.move(u, from, to);
      d_graph.set_block(u, to);
    } else {
      const BlockID from = p_graph.block(u);
      shared.move(u, from, to);
      p_graph.set_block(u, to);
    }

    if (step % 17 == 0) {
      this->expect_exact(shared, delta, d_graph);
    }
    if (step % 101 == 100) {
      d_graph.for_each([&](const NodeID node, const BlockID block) {
        shared.move(node, p_graph.block(node), block);
        p_graph.set_block(node, block);
      });
      delta.clear();
      d_graph.clear();
      this->expect_exact(shared, delta, d_graph);
    }
  }
  this->expect_exact(shared, delta, d_graph);
}

} // namespace
} // namespace kaminpar::shm
