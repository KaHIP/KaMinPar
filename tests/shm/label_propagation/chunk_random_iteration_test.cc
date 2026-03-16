/*******************************************************************************
 * Unit tests for the ChunkRandomIterator and InOrderIterator building blocks.
 *
 * @file:   chunk_random_iteration_test.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include <atomic>
#include <vector>

#include <gmock/gmock.h>

#include "tests/shm/graph_factories.h"

#include "kaminpar-shm/label_propagation/chunk_random_iteration.h"
#include "kaminpar-shm/label_propagation/config.h"
#include "kaminpar-shm/label_propagation/in_order_iteration.h"

#include "kaminpar-common/parallel/atomic.h"

namespace kaminpar::lp::testing {

using namespace kaminpar::shm::testing;
using Graph = shm::Graph;
using CSRGraph = shm::CSRGraph;

// Minimal config for iterator tests. Uses default values from LabelPropagationConfig
// and adds the required ClusterID / ClusterWeight type aliases.
struct IterTestConfig : public LabelPropagationConfig {
  using ClusterID = shm::NodeID;
  using ClusterWeight = shm::NodeWeight;
};

using NodeID = shm::NodeID;

// =============================================================================
// ChunkRandomIterator
// =============================================================================

class ChunkRandomIteratorTest : public ::testing::Test {
protected:
  // Build a path graph with `n` nodes (degree: 1 for endpoints, 2 for inner).
  // max_degree is set above the highest degree so all nodes are processed.
  static constexpr NodeID kMaxDegree = 100;
  static constexpr NodeID kNumNodes = 50;

  ChunkRandomIteratorTest()
      : graph(make_path_graph(kNumNodes)),
        csr(graph.csr_graph()),
        iter(perms) {}

  Graph graph;
  const CSRGraph &csr;
  ChunkRandomIterator<IterTestConfig>::Permutations perms;
  ChunkRandomIterator<IterTestConfig> iter;
};

TEST_F(ChunkRandomIteratorTest, VisitsEveryNodeExactlyOnce) {
  iter.prepare(csr, 0, csr.n(), kMaxDegree);

  std::vector<std::atomic<int>> visited(kNumNodes);
  for (auto &v : visited) {
    v = 0;
  }

  parallel::Atomic<NodeID> num_clusters = kNumNodes;
  iter.iterate(
      csr,
      kMaxDegree,
      [&](const NodeID u) {
        visited[u].fetch_add(1, std::memory_order_relaxed);
        return std::make_pair(false, false);
      },
      [] { return false; },
      [](const NodeID) { return true; },
      num_clusters
  );

  for (NodeID u = 0; u < kNumNodes; ++u) {
    EXPECT_EQ(visited[u].load(), 1) << "node " << u << " visited " << visited[u].load() << " times";
  }
}

TEST_F(ChunkRandomIteratorTest, CountsMoves) {
  iter.prepare(csr, 0, csr.n(), kMaxDegree);

  parallel::Atomic<NodeID> num_clusters = kNumNodes;
  const NodeID moved = iter.iterate(
      csr,
      kMaxDegree,
      [&](const NodeID) { return std::make_pair(true, false); },
      [] { return false; },
      [](const NodeID) { return true; },
      num_clusters
  );

  EXPECT_EQ(moved, kNumNodes);
}

TEST_F(ChunkRandomIteratorTest, SkipsInactiveNodes) {
  iter.init_chunks(csr, 0, csr.n(), kMaxDegree);

  std::vector<std::atomic<int>> visited(kNumNodes);
  for (auto &v : visited) {
    v = 0;
  }

  parallel::Atomic<NodeID> num_clusters = kNumNodes;
  iter.iterate(
      csr,
      kMaxDegree,
      [&](const NodeID u) {
        visited[u].fetch_add(1, std::memory_order_relaxed);
        return std::make_pair(false, false);
      },
      [] { return false; },
      [](const NodeID) { return false; }, // all inactive
      num_clusters
  );

  for (NodeID u = 0; u < kNumNodes; ++u) {
    EXPECT_EQ(visited[u].load(), 0);
  }
}

TEST_F(ChunkRandomIteratorTest, StopsEarlyWhenRequested) {
  iter.init_chunks(csr, 0, csr.n(), kMaxDegree);

  std::atomic<int> total_visited = 0;
  parallel::Atomic<NodeID> num_clusters = kNumNodes;
  // should_stop returns true immediately after the first chunk is started
  iter.iterate(
      csr,
      kMaxDegree,
      [&](const NodeID) {
        total_visited.fetch_add(1, std::memory_order_relaxed);
        return std::make_pair(false, false);
      },
      [] { return true; }, // stop immediately
      [](const NodeID) { return true; },
      num_clusters
  );

  EXPECT_LT(total_visited.load(), static_cast<int>(kNumNodes));
}

TEST_F(ChunkRandomIteratorTest, FirstPhaseVisitsNodesWithLowDegree) {
  iter.init_chunks(csr, 0, csr.n(), kMaxDegree);

  std::vector<std::atomic<int>> visited(kNumNodes);
  for (auto &v : visited) {
    v = 0;
  }

  parallel::Atomic<NodeID> num_clusters = kNumNodes;
  const auto [processed, moved] = iter.iterate_first_phase(
      csr,
      kMaxDegree,
      [&](const NodeID u) {
        visited[u].fetch_add(1, std::memory_order_relaxed);
        return std::make_pair(false, false);
      },
      [] { return false; },
      [](const NodeID) { return true; },
      num_clusters
  );

  for (NodeID u = 0; u < kNumNodes; ++u) {
    EXPECT_EQ(visited[u].load(), 1);
  }
  EXPECT_EQ(processed, kNumNodes);
  EXPECT_EQ(moved, 0u);
}

TEST_F(ChunkRandomIteratorTest, EmptyAfterClear) {
  iter.init_chunks(csr, 0, csr.n(), kMaxDegree);
  EXPECT_FALSE(iter.empty());
  iter.clear();
  EXPECT_TRUE(iter.empty());
}

TEST_F(ChunkRandomIteratorTest, ReleasedAndRestoredDataStructures) {
  iter.init_chunks(csr, 0, csr.n(), kMaxDegree);

  // Do a first iteration
  parallel::Atomic<NodeID> num_clusters = kNumNodes;
  iter.iterate(
      csr, kMaxDegree, [](NodeID) { return std::make_pair(false, false); },
      [] { return false; }, [](NodeID) { return true; }, num_clusters
  );

  // Release and restore
  auto ds = iter.release();
  EXPECT_TRUE(iter.empty());

  ChunkRandomIterator<IterTestConfig> iter2(perms);
  iter2.setup(std::move(ds));
  iter2.init_chunks(csr, 0, csr.n(), kMaxDegree);

  std::vector<std::atomic<int>> visited(kNumNodes);
  for (auto &v : visited) {
    v = 0;
  }
  iter2.iterate(
      csr, kMaxDegree,
      [&](NodeID u) {
        visited[u].fetch_add(1, std::memory_order_relaxed);
        return std::make_pair(false, false);
      },
      [] { return false; }, [](NodeID) { return true; }, num_clusters
  );
  for (NodeID u = 0; u < kNumNodes; ++u) {
    EXPECT_EQ(visited[u].load(), 1);
  }
}

// =============================================================================
// InOrderIterator
// =============================================================================

class InOrderIteratorTest : public ::testing::Test {
protected:
  static constexpr NodeID kMaxDegree = 100;
  static constexpr NodeID kNumNodes = 40;

  InOrderIteratorTest() : graph(make_path_graph(kNumNodes)), csr(graph.csr_graph()) {}

  Graph graph;
  const CSRGraph &csr;
  InOrderIterator<IterTestConfig> iter;
};

TEST_F(InOrderIteratorTest, VisitsEveryNodeExactlyOnce) {
  iter.prepare(csr, 0, csr.n(), kMaxDegree);

  std::vector<std::atomic<int>> visited(kNumNodes);
  for (auto &v : visited) {
    v = 0;
  }

  parallel::Atomic<NodeID> num_clusters = kNumNodes;
  iter.iterate(
      csr,
      kMaxDegree,
      [&](const NodeID u) {
        visited[u].fetch_add(1, std::memory_order_relaxed);
        return std::make_pair(false, false);
      },
      [] { return false; },
      [](const NodeID) { return true; },
      num_clusters
  );

  for (NodeID u = 0; u < kNumNodes; ++u) {
    EXPECT_EQ(visited[u].load(), 1) << "node " << u;
  }
}

TEST_F(InOrderIteratorTest, SkipsInactiveNodes) {
  iter.prepare(csr, 0, csr.n(), kMaxDegree);

  std::vector<std::atomic<int>> visited(kNumNodes);
  for (auto &v : visited) {
    v = 0;
  }

  parallel::Atomic<NodeID> num_clusters = kNumNodes;
  iter.iterate(
      csr,
      kMaxDegree,
      [&](const NodeID u) {
        visited[u].fetch_add(1, std::memory_order_relaxed);
        return std::make_pair(false, false);
      },
      [] { return false; },
      [](const NodeID) { return false; }, // all inactive
      num_clusters
  );

  for (NodeID u = 0; u < kNumNodes; ++u) {
    EXPECT_EQ(visited[u].load(), 0);
  }
}

TEST_F(InOrderIteratorTest, CountsMoves) {
  iter.prepare(csr, 0, csr.n(), kMaxDegree);

  parallel::Atomic<NodeID> num_clusters = kNumNodes;
  const NodeID moved = iter.iterate(
      csr,
      kMaxDegree,
      [](const NodeID) { return std::make_pair(true, false); },
      [] { return false; },
      [](const NodeID) { return true; },
      num_clusters
  );

  EXPECT_EQ(moved, kNumNodes);
}

TEST_F(InOrderIteratorTest, PrepareIsPersistentAcrossIterations) {
  // prepare() is called once; iterate() can be called multiple times
  iter.prepare(csr, 0, csr.n(), kMaxDegree);

  for (int round = 0; round < 3; ++round) {
    std::vector<std::atomic<int>> visited(kNumNodes);
    for (auto &v : visited) {
      v = 0;
    }

    parallel::Atomic<NodeID> num_clusters = kNumNodes;
    iter.iterate(
        csr,
        kMaxDegree,
        [&](const NodeID u) {
          visited[u].fetch_add(1, std::memory_order_relaxed);
          return std::make_pair(false, false);
        },
        [] { return false; },
        [](const NodeID) { return true; },
        num_clusters
    );

    for (NodeID u = 0; u < kNumNodes; ++u) {
      EXPECT_EQ(visited[u].load(), 1) << "round " << round << " node " << u;
    }
  }
}

TEST_F(InOrderIteratorTest, ClearResetsRange) {
  iter.prepare(csr, 0, csr.n(), kMaxDegree);
  iter.clear();

  // After clear(), iterate() visits nothing (range is [0, 0))
  std::atomic<int> total = 0;
  parallel::Atomic<NodeID> num_clusters = kNumNodes;
  iter.iterate(
      csr,
      kMaxDegree,
      [&](const NodeID) {
        total.fetch_add(1, std::memory_order_relaxed);
        return std::make_pair(false, false);
      },
      [] { return false; },
      [](const NodeID) { return true; },
      num_clusters
  );

  EXPECT_EQ(total.load(), 0);
}

} // namespace kaminpar::lp::testing
