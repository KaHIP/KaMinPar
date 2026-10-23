#include <atomic>
#include <cstddef>
#include <limits>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "tests/shm/graph_helpers.h"

namespace kaminpar::shm::testing {
namespace {

constexpr BlockWeight kUnbounded = std::numeric_limits<BlockWeight>::max();

template <typename Before, typename Worker, typename After>
void parallel_rounds(
    const int num_threads, const int num_rounds, Before &&before, Worker &&worker, After &&after
) {
  // Explicit publication/join also exposes every arrival to ThreadSanitizer on
  // standard libraries whose barrier arrival tree is not instrumented.
  std::atomic<int> generation = 0;
  std::vector<std::atomic<int>> completed(num_threads);
  std::vector<std::thread> threads;
  for (int thread = 0; thread < num_threads; ++thread) {
    threads.emplace_back([&, thread] {
      for (int round = 0; round < num_rounds; ++round) {
        while (generation.load(std::memory_order_acquire) != round + 1) {
          std::this_thread::yield();
        }
        worker(thread, round);
        completed[thread].store(round + 1, std::memory_order_release);
      }
    });
  }
  for (int round = 0; round < num_rounds; ++round) {
    before(round);
    generation.store(round + 1, std::memory_order_release);
    for (auto &done : completed) {
      while (done.load(std::memory_order_acquire) != round + 1) {
        std::this_thread::yield();
      }
    }
    after(round);
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void expect_consistent_weights(
    const PartitionedGraph &p_graph,
    const PartitionedCSRGraph &view,
    const std::vector<BlockWeight> &minima,
    const std::vector<BlockWeight> &maxima
) {
  std::vector<BlockWeight> actual(p_graph.k(), 0);
  for (NodeID node = 0; node < p_graph.n(); ++node) {
    const BlockID block = p_graph.block(node);
    ASSERT_LT(block, p_graph.k());
    EXPECT_EQ(view.block(node), block);
    actual[block] += p_graph.node_weight(node);
  }
  BlockWeight total = 0;
  for (const BlockID block : p_graph.blocks()) {
    EXPECT_EQ(p_graph.block_weight(block), actual[block]) << "block=" << block;
    EXPECT_EQ(view.block_weight(block), actual[block]) << "block=" << block;
    EXPECT_GE(actual[block], minima[block]) << "block=" << block;
    EXPECT_LE(actual[block], maxima[block]) << "block=" << block;
    total += actual[block];
  }
  EXPECT_EQ(total, p_graph.graph().total_node_weight());
}

class PartitionedGraphMinWeightTransactionsTest : public ::testing::TestWithParam<BlockID> {};

TEST_P(
    PartitionedGraphMinWeightTransactionsTest, SequentialWeightedRejectionsLeaveBothSidesUnchanged
) {
  const BlockID k = GetParam();
  const Graph graph = make_graph(std::vector<EdgeID>(5, 0), {}, {4, 2, 1, 3}, {});
  auto p_graph = make_p_graph(graph, k, {0, 0, 1, 2});
  auto view = p_graph.csr_view();
  std::vector<BlockWeight> minima(k, 0);
  std::vector<BlockWeight> maxima(k, 0);
  minima[0] = 4;
  minima[2] = 2;
  maxima[0] = 8;
  maxima[1] = 2;
  maxima[2] = 7;

  // The source cannot lose the heavy node, and the other destination cannot fit the light one.
  EXPECT_FALSE(p_graph.move(0, 0, 2, maxima[2], minima[0]));
  EXPECT_FALSE(view.move(1, 0, 1, maxima[1], minima[0]));
  EXPECT_EQ(p_graph.block_weight(0), 6);
  EXPECT_EQ(p_graph.block_weight(1), 1);
  EXPECT_EQ(p_graph.block_weight(2), 3);
  EXPECT_EQ(p_graph.block(0), 0);
  EXPECT_EQ(p_graph.block(1), 0);
  expect_consistent_weights(p_graph, view, minima, maxima);

  // A weight-only transaction changes the labels only when its caller commits them.
  ASSERT_TRUE(view.move_block_weight(0, 2, 2, maxima[2], minima[0]));
  EXPECT_EQ(p_graph.block(1), 0);
  EXPECT_EQ(p_graph.block_weight(0), 4);
  EXPECT_EQ(p_graph.block_weight(2), 5);
  view.set_block<false>(1, 2);
  expect_consistent_weights(p_graph, view, minima, maxima);

  // Mix a zero-minimum source with a positive-minimum source through the shared view.
  EXPECT_TRUE(p_graph.move(2, 1, 0, maxima[0], minima[1]));
  EXPECT_TRUE(view.move(3, 2, 0, maxima[0], minima[2]));
  EXPECT_EQ(p_graph.block_weight(0), 8);
  EXPECT_EQ(p_graph.block_weight(1), 0);
  EXPECT_EQ(p_graph.block_weight(2), 2);
  EXPECT_FALSE(view.move_block_weight(0, 1, 4, maxima[1], minima[0]));
  expect_consistent_weights(p_graph, view, minima, maxima);
}

TEST_P(PartitionedGraphMinWeightTransactionsTest, MixedDestinationsShareTheSourceLockAcrossViews) {
  constexpr int kNumThreads = 16;
  constexpr int kNumRounds = 128;
  constexpr NodeID kNumNodes = 32;
  const BlockID k = GetParam();
  for (const NodeWeight weight : {1, 3}) {
    SCOPED_TRACE(weight);
    const Graph graph = make_graph(
        std::vector<EdgeID>(kNumNodes + 1, 0), {}, std::vector<NodeWeight>(kNumNodes, weight), {}
    );
    auto p_graph = make_p_graph(graph, k, std::vector<BlockID>(kNumNodes, 0));
    auto view = p_graph.csr_view();
    auto second_view = view.csr_view();
    std::vector<BlockWeight> minima(k, 0);
    std::vector<BlockWeight> maxima(k, 0);
    minima[0] = (kNumNodes - 1) * weight;
    maxima[0] = kNumNodes * weight;
    maxima[1] = kUnbounded;
    maxima[2] = weight;
    std::atomic<NodeID> num_moves = 0;

    parallel_rounds(
        kNumThreads,
        kNumRounds,
        [&](const int) {
          for (NodeID node = 0; node < kNumNodes; ++node) {
            p_graph.set_block<false>(node, 0);
          }
          p_graph.recompute_block_weights(true);
          num_moves.store(0, std::memory_order_relaxed);
        },
        [&](const int thread, const int) {
          for (NodeID node = thread; node < kNumNodes; node += kNumThreads) {
            const BlockID to = 1 + node % 2;
            bool moved;
            if (thread % 3 == 0) {
              moved = p_graph.move(node, 0, to, maxima[to], minima[0]);
            } else if (thread % 3 == 1) {
              moved = view.move(node, 0, to, maxima[to], minima[0]);
            } else {
              moved = second_view.move(node, 0, to, maxima[to], minima[0]);
            }
            num_moves.fetch_add(moved, std::memory_order_relaxed);
          }
        },
        [&](const int round) {
          SCOPED_TRACE(round);
          // A precheck outside the source lock can admit multiple donors at this last slot.
          // An unbounded-target CAS that bypasses the bounded mover's lock has the same race.
          EXPECT_EQ(num_moves.load(std::memory_order_relaxed), 1u);
          EXPECT_EQ(p_graph.block_weight(0), minima[0]);
          EXPECT_EQ(p_graph.block_weight(1) + p_graph.block_weight(2), weight);
          expect_consistent_weights(p_graph, view, minima, maxima);
        }
    );
  }
}

TEST_P(
    PartitionedGraphMinWeightTransactionsTest, CyclicWeightedTrafficPreservesAllBoundsAndLabels
) {
  constexpr int kNumThreads = 16;
  constexpr int kNumRounds = 64;
  constexpr NodeID kNumNodes = 768;
  const BlockID k = GetParam();
  std::vector<NodeWeight> weights(kNumNodes);
  std::vector<BlockID> partition(kNumNodes);
  for (NodeID node = 0; node < kNumNodes; ++node) {
    weights[node] = 1 + (node / 4) % 3;
    partition[node] = node % 4;
  }
  const Graph graph = make_graph(std::vector<EdgeID>(kNumNodes + 1, 0), {}, weights, {});
  auto p_graph = make_p_graph(graph, k, partition);
  auto view = p_graph.csr_view();
  std::vector<BlockWeight> minima(k, 0);
  std::vector<BlockWeight> maxima(k, 0);
  minima[0] = 368;
  minima[1] = 0;
  minima[2] = 374;
  minima[3] = 360;
  maxima[0] = maxima[1] = maxima[3] = 400;
  maxima[2] = kUnbounded;
  std::atomic<NodeID> accepted = 0;
  std::atomic<NodeID> rejected = 0;

  parallel_rounds(
      kNumThreads,
      kNumRounds,
      [](const int) {},
      [&](const int thread, const int round) {
        // Every worker owns distinct node IDs for the entire test, including across rounds.
        for (NodeID node = thread; node < kNumNodes; node += kNumThreads) {
          const BlockID from = p_graph.block(node);
          const BlockID to = (from + 1 + (node + round) % 2) % 4;
          bool moved;
          if ((node + round) % 3 == 0) {
            moved = view.move_block_weight(from, to, weights[node], maxima[to], minima[from]);
            if (moved) {
              view.set_block<false>(node, to);
            }
          } else if (thread % 2 == 0) {
            moved = p_graph.move(node, from, to, maxima[to], minima[from]);
          } else {
            moved = view.move(node, from, to, maxima[to], minima[from]);
          }
          if (moved) {
            accepted.fetch_add(1, std::memory_order_relaxed);
          } else {
            rejected.fetch_add(1, std::memory_order_relaxed);
          }
        }
      },
      [&](const int round) {
        SCOPED_TRACE(round);
        expect_consistent_weights(p_graph, view, minima, maxima);
      }
  );
  EXPECT_GT(accepted.load(std::memory_order_relaxed), 0u);
  EXPECT_GT(rejected.load(std::memory_order_relaxed), 0u);
}

TEST_P(PartitionedGraphMinWeightTransactionsTest, ConcurrentFullTargetsDoNotWithdrawSourceWeight) {
  constexpr int kNumThreads = 16;
  constexpr NodeID kNumNodes = 65;
  const BlockID k = GetParam();
  const Graph graph = make_graph(
      std::vector<EdgeID>(kNumNodes + 1, 0), {}, std::vector<NodeWeight>(kNumNodes, 2), {}
  );
  std::vector<BlockID> partition(kNumNodes, 0);
  partition.back() = 1;
  auto p_graph = make_p_graph(graph, k, partition);
  auto view = p_graph.csr_view();
  std::vector<BlockWeight> minima(k, 0);
  std::vector<BlockWeight> maxima(k, 0);
  minima[0] = 64;
  minima[1] = 2;
  maxima[0] = 128;
  maxima[1] = 2;
  std::atomic<NodeID> num_moves = 0;

  parallel_rounds(
      kNumThreads,
      16,
      [](const int) {},
      [&](const int thread, const int) {
        for (NodeID node = thread; node + 1 < kNumNodes; node += kNumThreads) {
          const bool moved = thread % 2 == 0 ? p_graph.move(node, 0, 1, maxima[1], minima[0])
                                             : view.move(node, 0, 1, maxima[1], minima[0]);
          num_moves.fetch_add(moved, std::memory_order_relaxed);
        }
      },
      [&](const int round) {
        SCOPED_TRACE(round);
        EXPECT_EQ(num_moves.load(std::memory_order_relaxed), 0u);
        EXPECT_EQ(p_graph.block_weight(0), 128);
        EXPECT_EQ(p_graph.block_weight(1), 2);
        expect_consistent_weights(p_graph, view, minima, maxima);
      }
  );
}

INSTANTIATE_TEST_SUITE_P(
    AlignedAndDenseWeights,
    PartitionedGraphMinWeightTransactionsTest,
    ::testing::Values(16, 256, 257)
);

} // namespace
} // namespace kaminpar::shm::testing
