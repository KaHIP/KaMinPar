#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <tbb/global_control.h>

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/metrics.h"
#include "kaminpar-dist/refinement/balancer/node_balancer.h"
#include "kaminpar-dist/refinement/gains/on_the_fly_gain_cache.h"

#include "kaminpar-common/random.h"

namespace kaminpar::dist {
namespace {

DistributedGraph make_complete_graph(const NodeID n, const std::vector<NodeWeight> &weights) {
  const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
  std::vector<GlobalNodeID> node_distribution(size + 1);
  std::vector<GlobalEdgeID> edge_distribution(size + 1);
  for (PEID rank = 0; rank <= size; ++rank) {
    node_distribution[rank] = static_cast<GlobalNodeID>(rank) * n;
    edge_distribution[rank] = static_cast<GlobalEdgeID>(rank) * n * (n - 1);
  }
  std::vector<EdgeID> nodes;
  std::vector<NodeID> edges;
  for (NodeID u = 0; u < n; ++u) {
    nodes.push_back(edges.size());
    for (NodeID v = 0; v < n; ++v) {
      if (u != v) {
        edges.push_back(v);
      }
    }
  }
  nodes.push_back(edges.size());
  return DistributedGraph(
      std::make_unique<DistributedCSRGraph>(
          static_array::create(node_distribution),
          static_array::create(edge_distribution),
          static_array::create(nodes),
          static_array::create(edges),
          static_array::create(weights),
          StaticArray<EdgeWeight>{},
          StaticArray<PEID>{},
          StaticArray<GlobalNodeID>{},
          growt::StaticGhostNodeMapping{},
          false,
          MPI_COMM_WORLD
      )
  );
}

void expect_same_gain(const MaxGainer &a, const MaxGainer &b) {
  EXPECT_EQ(a.int_degree, b.int_degree);
  EXPECT_EQ(a.ext_degree, b.ext_degree);
  EXPECT_EQ(a.block, b.block);
  EXPECT_EQ(a.weight, b.weight);
}

std::array<bool, 32> random_tail() {
  std::array<bool, 32> tail;
  for (auto &value : tail) {
    value = Random::instance().random_bool();
  }
  return tail;
}

} // namespace

class CapacityMaskEquivalenceTest : public ::testing::TestWithParam<std::tuple<BlockID, bool>> {};

TEST_P(CapacityMaskEquivalenceTest, PreservesScalarGainsCapacityBoundaryAndTieSequence) {
  const auto [k, weighted] = GetParam();
  const NodeID n = std::max<NodeID>(8, 2 * k + 4);
  std::vector<NodeWeight> weights;
  if (weighted) {
    weights.resize(n);
    for (NodeID u = 0; u < n; ++u) {
      weights[u] = 3 + u % 5;
    }
    weights[0] = 0;
    weights[1] = 1;
    weights[2] = 2;
    weights[3] = 7;
  }
  auto graph = make_complete_graph(n, weights);
  ASSERT_EQ(graph.is_node_weighted(), weighted);
  StaticArray<BlockID> partition(n);
  for (NodeID u = 0; u < n; ++u) {
    partition[u] = u % k;
  }
  DistributedPartitionedGraph p_graph(&graph, k, std::move(partition));
  Context ctx = create_default_context();
  ctx.partition.k = k;
  OnTheFlyGainCache<DistributedCSRGraph> cache(ctx);
  cache.init(graph.csr_graph(), p_graph);

  for (NodeID u = 0; u < 8; ++u) {
    const NodeWeight weight = graph.node_weight(u);
    for (int pattern = 0; pattern < 5; ++pattern) {
      SCOPED_TRACE(::testing::Message() << "node=" << u << " pattern=" << pattern);
      std::vector<BlockWeight> limits(k);
      for (BlockID block = 0; block < k; ++block) {
        const BlockWeight after = p_graph.block_weight(block) + weight;
        // All fit, none fit, alternating exact-capacity/one-below, only last fits,
        // and all have one unit of headroom. Internal edges must never be filtered.
        const bool fits = pattern == 0 || pattern == 4 || (pattern == 2 && block % 2 == 0) ||
                          (pattern == 3 && block + 1 == k);
        limits[block] = fits ? after + (pattern == 4) : after - 1;
      }
      PartitionContext p_ctx;
      p_ctx.setup(graph.csr_graph(), limits);
      std::uint64_t mask = 0;
      std::vector<bool> fits(k);
      for (BlockID block = 0; block < k; ++block) {
        fits[block] = p_graph.block_weight(block) + weight <= p_ctx.max_block_weight(block);
        if (fits[block] && k <= 64) {
          mask |= std::uint64_t{1} << block;
        }
      }
      EdgeWeight internal = 0;
      std::vector<EdgeWeight> connections(k);
      graph.csr_graph().adjacent_nodes(u, [&](const NodeID v, const EdgeWeight edge_weight) {
        const BlockID block = p_graph.block(v);
        if (block == p_graph.block(u)) {
          internal += edge_weight;
        } else if (fits[block]) {
          connections[block] += edge_weight;
        }
      });
      const EdgeWeight external = *std::max_element(connections.begin(), connections.end());
      for (int seed = 0; seed < 4; ++seed) {
        Random::instance().reinit(seed);
        const auto original = cache.compute_max_gainer(u, p_ctx);
        const auto original_tail = random_tail();
        EXPECT_EQ(original.int_degree, internal);
        EXPECT_EQ(original.ext_degree, external);
        EXPECT_EQ(original.weight, weight);
        if (external == 0) {
          EXPECT_EQ(original.block, p_graph.block(u));
        } else {
          EXPECT_TRUE(fits[original.block]);
          EXPECT_NE(original.block, p_graph.block(u));
          EXPECT_EQ(connections[original.block], external);
        }
        if (k <= 64) {
          Random::instance().reinit(seed);
          const auto masked = cache.compute_max_gainer_for_blocks(u, mask);
          expect_same_gain(original, masked);
          EXPECT_EQ(random_tail(), original_tail);
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    BlockBoundariesAndWeights,
    CapacityMaskEquivalenceTest,
    ::testing::Combine(::testing::Values(1u, 16u, 64u, 65u), ::testing::Bool())
);

class CapacityMaskLifecycleTest : public ::testing::TestWithParam<std::tuple<BlockID, bool>> {};

TEST_P(CapacityMaskLifecycleTest, RefreshesAfterMovesAndKeepsWeightedAndLargeBlockFallbacks) {
  const auto [k, weighted] = GetParam();
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 4);
  const NodeID n = 2 * k;
  auto graph =
      make_complete_graph(n, weighted ? std::vector<NodeWeight>(n, 2) : std::vector<NodeWeight>{});
  ASSERT_EQ(graph.is_node_weighted(), weighted);
  StaticArray<BlockID> partition(n);
  for (NodeID u = 0; u < n; ++u) {
    partition[u] = u % k;
  }
  DistributedPartitionedGraph p_graph(&graph, k, std::move(partition));
  Context ctx = create_default_context();
  ctx.partition.setup(graph.csr_graph(), k, 0.0);
  ctx.refinement.node_balancer.enable_sequential_balancing = false;
  ctx.refinement.node_balancer.enable_parallel_balancing = true;
  ctx.refinement.node_balancer.max_num_rounds = 100;
  NodeBalancerFactory factory(ctx);
  auto balancer = factory.create(p_graph, ctx.partition);

  ASSERT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
  balancer->initialize();
  balancer->refine();
  for (int direction = 0; direction < 2; ++direction) {
    // Restore a known feasible partition, then reverse which block has room. A mask
    // retained across calls would select the wrong destination on the second pass.
    for (NodeID u = 0; u < n; ++u) {
      p_graph.set_block<false>(u, u % k);
    }
    p_graph.reinit_block_weights();
    balancer->initialize();
    balancer->refine();
    ASSERT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
    const NodeID u = direction == 0 ? k - 1 : 0;
    p_graph.set_block<false>(u, direction == 0 ? 0 : k - 1);
    p_graph.reinit_block_weights();
    ASSERT_FALSE(metrics::is_feasible(p_graph, ctx.partition));
    balancer->initialize();
    balancer->refine();
    ASSERT_TRUE(metrics::is_feasible(p_graph, ctx.partition));
  }
}

INSTANTIATE_TEST_SUITE_P(
    UnitFastPathAndFallbacks,
    CapacityMaskLifecycleTest,
    ::testing::Combine(::testing::Values(16u, 64u, 65u), ::testing::Bool())
);

} // namespace kaminpar::dist
