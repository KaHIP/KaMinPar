#include <barrier>
#include <cstdint>
#include <thread>

#include <gtest/gtest.h>

#include "tests/dist/distributed_graph_builder.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/refinement/gains/compact_hashing_gain_cache.h"
#include "kaminpar-dist/refinement/gains/lazy_compact_hashing_gain_cache.h"

namespace kaminpar::dist {

template <typename Cache> class DistributedGainCacheConcurrencyTest : public ::testing::Test {};
using ConcurrentCacheTypes = ::testing::Types<
    CompactHashingGainCache<DistributedCSRGraph>,
    LazyCompactHashingGainCache<DistributedCSRGraph>>;
TYPED_TEST_SUITE(DistributedGainCacheConcurrencyTest, ConcurrentCacheTypes);

TYPED_TEST(DistributedGainCacheConcurrencyTest, GainQueriesCanOverlapNeighborMoves) {
  constexpr NodeID kNumNodes = 8;
  constexpr EdgeWeight kWeight = sizeof(EdgeWeight) == sizeof(std::uint64_t)
                                     ? static_cast<EdgeWeight>(std::uint64_t{1} << 33)
                                     : EdgeWeight{7};
  graph::Builder builder(MPI_COMM_WORLD);
  builder.initialize(kNumNodes);
  const GlobalNodeID offset = mpi::get_comm_rank(MPI_COMM_WORLD) * kNumNodes;
  for (NodeID node = 0; node < kNumNodes; ++node) {
    builder.create_node(1);
    for (NodeID neighbor = 0; neighbor < kNumNodes; ++neighbor) {
      if (neighbor != node) {
        builder.create_edge(kWeight, offset + neighbor);
      }
    }
  }
  DistributedGraph graph(std::make_unique<DistributedCSRGraph>(builder.finalize()));
  StaticArray<BlockID> partition(graph.total_n(), BlockID{1});
  partition[0] = 0;
  DistributedPartitionedGraph p_graph(&graph, 16, std::move(partition));
  Context ctx = create_default_context();
  TypeParam cache(ctx);
  cache.init(graph.csr_graph(), p_graph);

  // Degree 7 < k=16 selects a compact hash table. The cache lock shares storage with the
  // weighted degree, which readers consult to determine the hash entry width on each query.
  std::barrier start(3);
  auto move_neighbor = [&](const NodeID node) {
    start.arrive_and_wait();
    for (int i = 0; i < 4000; ++i) {
      cache.move(node, 1, 2);
      cache.move(node, 2, 1);
    }
  };
  std::thread first_writer(move_neighbor, 1);
  std::thread second_writer(move_neighbor, 2);
  start.arrive_and_wait();
  for (int i = 0; i < 8000; ++i) {
    const auto gain = cache.compute_max_gainer(0);
    EXPECT_EQ(gain.int_degree, 0);
    EXPECT_GE(gain.ext_degree, 5 * kWeight);
    EXPECT_LE(gain.ext_degree, 7 * kWeight);
  }
  first_writer.join();
  second_writer.join();

  for (NodeID node = 0; node < kNumNodes; ++node) {
    const auto gain = cache.compute_max_gainer(node);
    EXPECT_EQ(gain.int_degree, node == 0 ? 0 : 6 * kWeight);
    EXPECT_EQ(gain.ext_degree, node == 0 ? 7 * kWeight : kWeight);
    EXPECT_EQ(gain.block, node == 0 ? 1u : 0u);
  }
}

} // namespace kaminpar::dist
