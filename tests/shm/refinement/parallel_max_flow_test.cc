#include <array>
#include <cstdint>
#include <limits>
#include <random>

#include <gtest/gtest.h>
#include <tbb/task_arena.h>

#include "tests/shm/graph_builder.h"

#include "kaminpar-shm/refinement/flow/max_flow/parallel_blocked_preflow_push_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/parallel_preflow_push_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/preflow_push_algorithm.h"
#include "kaminpar-shm/refinement/flow/util/reverse_edge_index.h"

namespace kaminpar::shm::testing {
namespace {

template <typename Algorithm> class ParallelMaxFlowTest : public ::testing::Test {};
using ParallelMaxFlowAlgorithms =
    ::testing::Types<ParallelPreflowPushAlgorithm, ParallelBlockedPreflowPushAlgorithm>;
TYPED_TEST_SUITE(ParallelMaxFlowTest, ParallelMaxFlowAlgorithms);

TYPED_TEST(ParallelMaxFlowTest, FindsTheLayeredNetworkMinimumCut) {
  // All nodes in each layer share neighbors, exercising concurrent BFS discovery and pushes.
  // There are 64 edge-disjoint source-to-sink paths and the source cut has capacity 64.
  constexpr NodeID kWidth = 64;
  constexpr NodeID kSource = 0;
  constexpr NodeID kSink = 2 * kWidth + 1;
  EdgeBasedGraphBuilder builder;
  for (NodeID left = 1; left <= kWidth; ++left) {
    builder.add_edge(kSource, left);
    for (NodeID right = kWidth + 1; right <= 2 * kWidth; ++right) {
      builder.add_edge(left, right);
    }
  }
  for (NodeID right = kWidth + 1; right <= 2 * kWidth; ++right) {
    builder.add_edge(right, kSink);
  }
  const Graph graph = builder.build();
  const CSRGraph &csr = graph.csr_graph();
  const auto reverse_edges = compute_reverse_edge_index(csr);
  auto ctx = create_default_context().refinement.twoway_flow.flow_cutter.flow;
  ctx.sequential_discharge_threshold = 0;
  TypeParam algorithm(ctx);

  tbb::task_arena arena(4);
  arena.execute([&] {
    algorithm.initialize(csr, reverse_edges, kSource, kSink);
    const auto [value, flow] = algorithm.compute_max_preflow();
    EXPECT_EQ(value, kWidth);
    EXPECT_TRUE(debug::is_valid_preflow(csr, algorithm.node_status(), flow));
    EXPECT_TRUE(debug::is_max_flow(csr, algorithm.node_status(), flow));
  });
}

EdgeWeight exhaustive_cut(const CSRGraph &graph, const NodeID source, const NodeID sink) {
  const NodeID n = graph.n();
  EdgeWeight best = std::numeric_limits<EdgeWeight>::max();
  for (std::uint64_t mask = 0; mask < (std::uint64_t{1} << n); ++mask) {
    if ((mask & (std::uint64_t{1} << source)) == 0 || (mask & (std::uint64_t{1} << sink)) != 0) {
      continue;
    }
    EdgeWeight cut = 0;
    for (NodeID u = 0; u < n; ++u) {
      if ((mask & (std::uint64_t{1} << u)) != 0) {
        graph.neighbors(u, [&](EdgeID, const NodeID v, const EdgeWeight capacity) {
          if ((mask & (std::uint64_t{1} << v)) == 0) {
            cut += capacity;
          }
        });
      }
    }
    best = std::min(best, cut);
  }
  return best;
}

TEST(ParallelInactiveTargetTest, ConcurrentPushesToInitiallyInactiveLowerIdTargetMatchSerialCut) {
  // All active u have height1 and ID>v. After their sink edge saturates they relabel to2.
  // The old active-active tie rule rejects u->v, although v starts this round with no excess.
  constexpr NodeID kWidth = 64;
  constexpr NodeID kSource = 0;
  constexpr NodeID kInactiveTarget = 1;
  constexpr NodeID kSink = kWidth + 2;
  EdgeBasedGraphBuilder builder;
  for (NodeID u = 2; u < kSink; ++u) {
    builder.add_edge(kSource, u, 5);
    builder.add_edge(u, kSink, 1);
    builder.add_edge(u, kInactiveTarget, 4);
  }
  builder.add_edge(kInactiveTarget, kSink, 4 * kWidth);
  const Graph graph = builder.build();
  const CSRGraph &csr = graph.csr_graph();
  const auto reverse_edges = compute_reverse_edge_index(csr);
  auto ctx = create_default_context().refinement.twoway_flow.flow_cutter.flow;
  ctx.sequential_discharge_threshold = 0;
  PreflowPushAlgorithm serial(ctx);
  serial.initialize(csr, reverse_edges, kSource, kSink);
  const auto [expected, serial_flow] = serial.compute_max_preflow();
  ASSERT_EQ(expected, 5 * kWidth);
  ASSERT_TRUE(debug::is_max_flow(csr, serial.node_status(), serial_flow));
  for (const int threads : {2, 4, 8}) {
    tbb::task_arena arena(threads);
    arena.execute([&] {
      ParallelPreflowPushAlgorithm parallel(ctx);
      for (int repeat = 0; repeat < 4; ++repeat) {
        parallel.initialize(csr, reverse_edges, kSource, kSink);
        const auto [actual, flow] = parallel.compute_max_preflow();
        EXPECT_EQ(actual, expected);
        EXPECT_TRUE(debug::is_valid_preflow(csr, parallel.node_status(), flow));
        EXPECT_TRUE(debug::is_max_flow(csr, parallel.node_status(), flow));
      }
    });
  }
}

TEST(
    ParallelInactiveTargetTest, WeightedRandomNetworksAndIncrementalTerminalsMatchSerialReference
) {
  auto ctx = create_default_context().refinement.twoway_flow.flow_cutter.flow;
  ctx.sequential_discharge_threshold = 0;
  for (std::uint32_t seed = 0; seed < 32; ++seed) {
    SCOPED_TRACE(::testing::Message() << "network seed=" << seed);
    const NodeID n = 8 + seed % 5;
    const NodeID source = 0;
    const NodeID sink = n - 1;
    std::mt19937 random(seed);
    EdgeBasedGraphBuilder builder;
    for (NodeID u = 0; u < n; ++u) {
      for (NodeID v = u + 1; v < n; ++v) {
        if (v == u + 1 || random() % 4 == 0) {
          builder.add_edge(u, v, 1 + random() % 13);
        }
      }
    }
    const Graph graph = builder.build();
    const CSRGraph &csr = graph.csr_graph();
    const auto reverse_edges = compute_reverse_edge_index(csr);
    const EdgeWeight oracle = exhaustive_cut(csr, source, sink);
    for (const int threads : {2, 4, 8}) {
      tbb::task_arena arena(threads);
      arena.execute([&] {
        PreflowPushAlgorithm serial(ctx);
        ParallelPreflowPushAlgorithm parallel(ctx);
        serial.initialize(csr, reverse_edges, source, sink);
        parallel.initialize(csr, reverse_edges, source, sink);
        for (int phase = 0; phase < 3; ++phase) {
          const auto [expected, serial_flow] = serial.compute_max_preflow();
          const auto [actual, flow] = parallel.compute_max_preflow();
          if (phase == 0) {
            EXPECT_EQ(expected, oracle);
          }
          EXPECT_EQ(actual, expected);
          EXPECT_TRUE(debug::is_valid_preflow(csr, serial.node_status(), serial_flow));
          EXPECT_TRUE(debug::is_max_flow(csr, serial.node_status(), serial_flow));
          EXPECT_TRUE(debug::is_valid_preflow(csr, parallel.node_status(), flow));
          EXPECT_TRUE(debug::is_max_flow(csr, parallel.node_status(), flow));
          if (phase < 2) {
            const NodeID next_terminal = phase == 0 ? 1 : n - 2;
            const std::array<NodeID, 1> terminals{next_terminal};
            serial.pierce_nodes(phase == 0, terminals);
            parallel.pierce_nodes(phase == 0, terminals);
          }
        }
      });
    }
  }
}

} // namespace
} // namespace kaminpar::shm::testing
