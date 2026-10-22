#include <array>
#include <barrier>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <tbb/task_arena.h>

#include "tests/shm/graph_factories.h"
#include "tests/shm/graph_helpers.h"

#include "kaminpar-shm/graphutils/compressed_graph_builder.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/multi_queue_overload_balancer.h"
#include "kaminpar-shm/refinement/gains/compact_hashing_gain_cache.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm::testing {
namespace {

using Cache = LargeKCompactHashingGainCache<const CSRGraph>;

#ifndef KAMINPAR_TEST_C6_REFERENCE

struct WidthCase {
  const char *name;
  BlockID k;
  std::uint64_t edge_weight;
};

class CompactGainQueryWidthTest : public ::testing::TestWithParam<WidthCase> {};

template <template <typename> typename CacheType> void check_width_case(const WidthCase &item) {
  const auto edge_weight = static_cast<EdgeWeight>(item.edge_weight);
  GraphBuilder builder;
  for (NodeID u = 0; u < 3; ++u) {
    builder.new_node();
    for (NodeID v = 0; v < 3; ++v) {
      if (u != v) {
        builder.new_edge(v, edge_weight);
      }
    }
  }
  const Graph graph = builder.build(true);
  auto p_graph = make_p_graph(graph, item.k, {0, 0, 1});
  Context ctx = create_default_context();
  ctx.partition.setup(graph, item.k, 0.03);
  CacheType<CSRGraph> cache(ctx, graph.n(), item.k);
  cache.initialize(graph.csr_graph(), p_graph);

  auto check = [&](auto &&query) {
    for (BlockID from = 0; from < item.k; ++from) {
      for (BlockID to = 0; to < item.k; ++to) {
        EdgeWeight from_weight = 0;
        EdgeWeight to_weight = 0;
        graph.csr_graph().adjacent_nodes(0, [&](const NodeID v, const EdgeWeight weight) {
          from_weight += p_graph.block(v) == from ? weight : 0;
          to_weight += p_graph.block(v) == to ? weight : 0;
        });
        EXPECT_EQ(query(from, to), to_weight - from_weight);
        EXPECT_EQ(query(from, to), cache.gain(0, from, to));
      }
    }
  };

  unsigned calls = 0;
  cache.with_gain_query(0, [&](auto &&query) {
    ++calls;
    check(query);
    cache.move(1, 0, 1);
    p_graph.set_block(1, 1);
    // Both source and target connections changed while the same scoped query stays alive.
    check(query);
    EXPECT_EQ(query(0, 1), 2 * edge_weight);
    EXPECT_EQ(query(1, 0), -2 * edge_weight);
    cache.move(1, 1, 0);
    p_graph.set_block(1, 0);
    check(query);
  });
  EXPECT_EQ(calls, 1u);

  // Reinitialization and release/reuse must retain the same layout and unlocked payload.
  cache.initialize(graph.csr_graph(), p_graph);
  cache.move(1, 0, 1);
  p_graph.set_block(1, 1);
  cache.with_gain_query(0, check);
  cache.free();
  cache.initialize(graph.csr_graph(), p_graph);
  cache.move(1, 1, 0);
  p_graph.set_block(1, 0);
  cache.with_gain_query(0, check);
}

TEST_P(CompactGainQueryWidthTest, PreservesEveryGainAndRefreshesConnectionsWithinScope) {
  if (GetParam().edge_weight >
      static_cast<std::uint64_t>(std::numeric_limits<EdgeWeight>::max()) / 2) {
    GTEST_SKIP() << "Two incident edges exceed the configured signed edge-weight range";
  }
  check_width_case<NormalCompactHashingGainCache>(GetParam());
  check_width_case<LargeKCompactHashingGainCache>(GetParam());
}

// Each queried node has degree2. Full tables have k2 and no key bits; hash tables
// have k16 and4 key bits. The selected total incident weights exercise widths0/1/2/4/8.
INSTANTIATE_TEST_SUITE_P(
    FullAndHashWidths,
    CompactGainQueryWidthTest,
    ::testing::Values(
        WidthCase{"Full0", 2, 0},
        WidthCase{"Full1", 2, 1},
        WidthCase{"Full2", 2, EdgeWeight{1} << 7},
        WidthCase{"Full4", 2, EdgeWeight{1} << 15},
        WidthCase{"Full8", 2, std::uint64_t{1} << 31},
        WidthCase{"Hash0", 16, 0},
        WidthCase{"Hash1", 16, 1},
        WidthCase{"Hash2", 16, EdgeWeight{1} << 3},
        WidthCase{"Hash4", 16, EdgeWeight{1} << 11},
        WidthCase{"Hash8", 16, EdgeWeight{1} << 27}
    ),
    [](const auto &info) { return info.param.name; }
);

TEST(CompactGainQueryConcurrencyTest, ScopedQueryCanOverlapHashTableNeighborMoves) {
  const Graph graph = make_complete_graph(8);
  auto p_graph = make_p_graph(graph, 16, {0, 1, 1, 1, 1, 1, 1, 1});
  Context ctx = create_default_context();
  ctx.partition.setup(graph, p_graph.k(), 0.03);
  Cache cache(ctx, graph.n(), p_graph.k());
  cache.initialize(graph.csr_graph(), p_graph);

  cache.with_gain_query(0, [&](auto &&query) {
    std::barrier start(3);
    auto move_neighbor = [&](const NodeID node) {
      start.arrive_and_wait();
      for (unsigned i = 0; i < 4000; ++i) {
        cache.move(node, 1, 2);
        cache.move(node, 2, 1);
      }
    };
    std::thread first(move_neighbor, 1);
    std::thread second(move_neighbor, 2);
    start.arrive_and_wait();
    for (unsigned i = 0; i < 8000; ++i) {
      const EdgeWeight gain = query(0, 1);
      EXPECT_GE(gain, 5);
      EXPECT_LE(gain, 7);
    }
    first.join();
    second.join();
    EXPECT_EQ(query(0, 1), 7);
    EXPECT_EQ(query(0, 2), 0);
  });
}

template <typename GraphType> Graph make_unit_degree_fixture(const EdgeWeight weight) {
  GraphBuilder builder;
  for (NodeID u = 0; u < 4; ++u) {
    builder.new_node();
    if (u < 3) {
      for (NodeID v = 0; v < 3; ++v) {
        if (u != v) {
          builder.new_edge(v, weight);
        }
      }
    }
  }
  Graph graph = builder.build(true);
  if constexpr (std::is_same_v<GraphType, CompressedGraph>) {
    return compress(graph.csr_graph());
  } else {
    EXPECT_FALSE(graph.csr_graph().raw_edge_weights().empty());
    return graph;
  }
}

template <typename GraphType> class CompactUnitDegreeTest : public ::testing::Test {};

using UnitDegreeGraphTypes = ::testing::Types<CSRGraph, CompressedGraph>;
TYPED_TEST_SUITE(CompactUnitDegreeTest, UnitDegreeGraphTypes);

template <typename GraphType, template <typename> typename CacheType>
void check_unit_degree_lifecycle(const BlockID k) {
  Context ctx = create_default_context();
  CacheType<GraphType> cache(ctx, 0, k);
  // The same cache switches representation eligibility repeatedly. Zero-weight
  // edges retain nonzero degree and must keep the weighted width-zero path.
  for (const EdgeWeight weight : {1, 2, 0, 16, 1}) {
    Graph graph = make_unit_degree_fixture<GraphType>(weight);
    const GraphType &concrete = as_concrete_graph<GraphType>(graph);
    ASSERT_EQ(graph.is_edge_weighted(), weight != 1);
    auto p_graph = make_p_graph(graph, k, {0, 0, 1, 0});
    ctx.partition.setup(graph, k, 0.03);
    cache.initialize(concrete, p_graph);

    auto check = [&](const NodeID u, auto &&query) {
      EdgeWeight total = 0;
      std::vector<EdgeWeight> connections(k);
      concrete.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight edge_weight) {
        total += edge_weight;
        connections[p_graph.block(v)] += edge_weight;
      });
      EXPECT_EQ(cache.is_border_node(u, p_graph.block(u)), total != connections[p_graph.block(u)]);
      for (BlockID from = 0; from < k; ++from) {
        for (BlockID to = 0; to < k; ++to) {
          EXPECT_EQ(query(from, to), connections[to] - connections[from]);
          EXPECT_EQ(cache.gain(u, from, to), connections[to] - connections[from]);
        }
      }
    };

    for (const NodeID u : concrete.nodes()) {
      unsigned calls = 0;
      cache.with_gain_query(u, [&](auto &&query) {
        ++calls;
        check(u, query);
        cache.move(1, 0, 1);
        p_graph.set_block(1, 1);
        check(u, query);
        cache.move(1, 1, 0);
        p_graph.set_block(1, 0);
        check(u, query);
      });
      EXPECT_EQ(calls, 1u);
    }
    if (weight == 0) {
      cache.free();
    }
  }
}

TYPED_TEST(CompactUnitDegreeTest, ReinitializationRetainsExactUnitWeightedAndIsolatedQueries) {
  for (const BlockID k : {2u, 16u}) {
    check_unit_degree_lifecycle<TypeParam, NormalCompactHashingGainCache>(k);
    check_unit_degree_lifecycle<TypeParam, LargeKCompactHashingGainCache>(k);
  }
}

TYPED_TEST(CompactUnitDegreeTest, UnitQueriesCanOverlapPackedHashWriterLocks) {
  Graph graph = make_complete_graph(8);
  if constexpr (std::is_same_v<TypeParam, CompressedGraph>) {
    graph = compress(graph.csr_graph());
  }
  const TypeParam &concrete = as_concrete_graph<TypeParam>(graph);
  ASSERT_FALSE(graph.is_edge_weighted());
  auto p_graph = make_p_graph(graph, 16, {0, 1, 1, 1, 1, 1, 1, 1});
  Context ctx = create_default_context();
  ctx.partition.setup(graph, p_graph.k(), 0.03);
  LargeKCompactHashingGainCache<TypeParam> cache(ctx, graph.n(), p_graph.k());
  cache.initialize(concrete, p_graph);
  cache.with_gain_query(0, [&](auto &&query) {
    std::barrier start(3);
    auto move_neighbor = [&](const NodeID node) {
      start.arrive_and_wait();
      for (unsigned i = 0; i < 4000; ++i) {
        cache.move(node, 1, 2);
        cache.move(node, 2, 1);
      }
    };
    std::thread first(move_neighbor, 1);
    std::thread second(move_neighbor, 2);
    start.arrive_and_wait();
    for (unsigned i = 0; i < 8000; ++i) {
      EXPECT_TRUE(cache.is_border_node(0, 0));
      EXPECT_GE(query(0, 1), 5);
      EXPECT_LE(query(0, 1), 7);
    }
    first.join();
    second.join();
    EXPECT_EQ(query(0, 1), 7);
  });
}

#endif

class CompactGainQueryBalancerTest : public ::testing::TestWithParam<unsigned> {};

TEST_P(CompactGainQueryBalancerTest, SerialRebalanceRetainsTargetsMovesAndRandomState) {
  const unsigned topology = GetParam();
  GraphBuilder builder;
  for (NodeID u = 0; u < 64; ++u) {
    builder.new_node();
    if (topology == 1) {
      builder.new_edge((u + 1) % 64);
      builder.new_edge((u + 63) % 64);
    } else if (topology == 2) {
      for (NodeID v = 0; v < 64; ++v) {
        if (v != u) {
          builder.new_edge(v);
        }
      }
    }
  }
  const Graph graph = builder.build(true);
  auto p_graph = make_p_graph(graph, 16, std::vector<BlockID>(64, 0));
  Context ctx = create_default_context();
  ctx.parallel.num_threads = 1;
  ctx.partition.setup(graph, p_graph.k(), 0.03);
  Cache cache(ctx, graph.n(), p_graph.k());
  cache.initialize(graph.csr_graph(), p_graph);
  Random::instance().reinit(17);

  std::vector<std::tuple<NodeID, BlockID, BlockID>> moves;
  MultiQueueOverloadBalancer balancer(ctx);
  balancer.track_moves([&](const NodeID u, const BlockID from, const BlockID to) {
    cache.move(u, from, to);
    moves.emplace_back(u, from, to);
  });
  tbb::task_arena arena(1);
  arena.execute([&] {
    EXPECT_TRUE(balancer.refine_with_gain_cache(p_graph, ctx.partition, graph.csr_graph(), cache));
  });
  EXPECT_FALSE(moves.empty());
  EXPECT_TRUE(metrics::is_balanced(p_graph, ctx.partition));
  for (const auto &[node, from, to] : moves) {
    EXPECT_EQ(from, 0u);
    EXPECT_GT(to, 0u);
    EXPECT_LT(to, p_graph.k());
    EXPECT_LT(node, graph.n());
  }
  std::array<bool, 32> random_bits;
  for (auto &bit : random_bits) {
    bit = Random::instance().random_bool();
  }
  if (std::getenv("KAMINPAR_TEST_DUMP_COMPACT_QUERY") != nullptr) {
    std::cout << "COMPACT_QUERY topology=" << topology << " moves=";
    for (const auto &[node, from, to] : moves) {
      std::cout << node << ':' << from << ':' << to << ',';
    }
    std::cout << " labels=";
    for (const NodeID u : graph.csr_graph().nodes()) {
      std::cout << p_graph.block(u) << ',';
    }
    std::cout << " rng=";
    for (const bool bit : random_bits) {
      std::cout << bit;
    }
    std::cout << '\n';
  }
}

INSTANTIATE_TEST_SUITE_P(ZeroHashFull, CompactGainQueryBalancerTest, ::testing::Values(0u, 1u, 2u));

} // namespace
} // namespace kaminpar::shm::testing
