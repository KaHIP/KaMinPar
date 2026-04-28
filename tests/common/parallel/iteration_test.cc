#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>

#include "kaminpar-common/parallel/iteration.h"

using ::testing::ElementsAre;
using ::testing::Eq;

namespace kaminpar {

namespace {

class BucketedTestGraph {
public:
  using NodeID = std::uint32_t;
  using EdgeID = std::uint32_t;

  BucketedTestGraph(std::vector<EdgeID> degrees, std::vector<NodeID> bucket_starts)
      : _degrees(std::move(degrees)),
        _bucket_starts(std::move(bucket_starts)) {}

  [[nodiscard]] NodeID n() const {
    return static_cast<NodeID>(_degrees.size());
  }

  [[nodiscard]] EdgeID m() const {
    return std::accumulate(_degrees.begin(), _degrees.end(), EdgeID{0});
  }

  [[nodiscard]] std::size_t number_of_buckets() const {
    return _bucket_starts.size();
  }

  [[nodiscard]] NodeID first_node_in_bucket(const std::size_t bucket) const {
    return _bucket_starts[bucket];
  }

  [[nodiscard]] NodeID bucket_size(const std::size_t bucket) const {
    const NodeID next = bucket + 1 < _bucket_starts.size() ? _bucket_starts[bucket + 1] : n();
    return next - _bucket_starts[bucket];
  }

  [[nodiscard]] EdgeID degree(const NodeID node) const {
    return _degrees[node];
  }

private:
  std::vector<EdgeID> _degrees;
  std::vector<NodeID> _bucket_starts;
};

std::vector<std::uint32_t> sorted(std::vector<std::uint32_t> nodes) {
  std::sort(nodes.begin(), nodes.end());
  return nodes;
}

std::vector<std::uint32_t> expected_range(const std::uint32_t from, const std::uint32_t to) {
  std::vector<std::uint32_t> nodes(to - from);
  std::iota(nodes.begin(), nodes.end(), from);
  return nodes;
}

BucketedTestGraph make_bucketed_graph_with_isolated_nodes() {
  return BucketedTestGraph{{0, 0, 3, 0, 2, 0, 1, 4, 0, 0, 2, 0}, {0, 5, 8}};
}

} // namespace

TEST(IterationTest, natural_order_visits_range_in_order) {
  iteration::NaturalNodeOrder<std::uint32_t> order({2, 7});

  std::vector<std::uint32_t> nodes;
  order.for_each([&](const std::uint32_t node) { nodes.push_back(node); });

  EXPECT_THAT(nodes, ElementsAre(2, 3, 4, 5, 6));
}

TEST(IterationTest, natural_parallel_order_covers_range_once) {
  iteration::NaturalNodeOrder<std::uint32_t> order({2, 11});

  tbb::concurrent_vector<std::uint32_t> nodes;
  order.parallel_for_each([&](const std::uint32_t node) { nodes.push_back(node); });
  std::vector<std::uint32_t> copied_nodes(nodes.begin(), nodes.end());

  EXPECT_THAT(sorted(copied_nodes), expected_range(2, 11));
}

TEST(IterationTest, chunk_random_order_covers_full_range_once) {
  const auto graph = make_bucketed_graph_with_isolated_nodes();
  iteration::ChunkRandomNodeOrderWorkspace<std::uint32_t, 4, 4> workspace;
  iteration::ChunkRandomNodeOrder order(
      graph, workspace, iteration::NodeRange<std::uint32_t>{0, graph.n()}, 2
  );

  std::vector<std::uint32_t> nodes;
  order.for_each([&](const std::uint32_t node) { nodes.push_back(node); });

  EXPECT_THAT(sorted(nodes), expected_range(0, graph.n()));
}

TEST(IterationTest, chunk_random_order_covers_subrange_once_across_buckets) {
  const auto graph = make_bucketed_graph_with_isolated_nodes();
  iteration::ChunkRandomNodeOrderWorkspace<std::uint32_t, 4, 4> workspace;
  iteration::ChunkRandomNodeOrder order(
      graph, workspace, iteration::NodeRange<std::uint32_t>{2, 10}, 2
  );

  std::vector<std::uint32_t> nodes;
  order.for_each([&](const std::uint32_t node) { nodes.push_back(node); });

  EXPECT_THAT(sorted(nodes), expected_range(2, 10));
}

TEST(IterationTest, chunk_random_parallel_order_covers_subrange_once) {
  const auto graph = make_bucketed_graph_with_isolated_nodes();
  iteration::ChunkRandomNodeOrderWorkspace<std::uint32_t, 4, 4> workspace;
  iteration::ChunkRandomNodeOrder order(
      graph, workspace, iteration::NodeRange<std::uint32_t>{1, 11}, 2
  );

  tbb::concurrent_vector<std::uint32_t> nodes;
  order.parallel_for_each([&](const std::uint32_t node) { nodes.push_back(node); });
  std::vector<std::uint32_t> copied_nodes(nodes.begin(), nodes.end());

  EXPECT_THAT(sorted(copied_nodes), expected_range(1, 11));
}

TEST(IterationTest, chunk_random_order_respects_bucket_limit) {
  const auto graph = make_bucketed_graph_with_isolated_nodes();
  iteration::ChunkRandomNodeOrderWorkspace<std::uint32_t, 4, 4> workspace;
  iteration::ChunkRandomNodeOrder order(
      graph, workspace, iteration::NodeRange<std::uint32_t>{0, graph.n()}, 2, 2
  );

  std::vector<std::uint32_t> nodes;
  order.for_each([&](const std::uint32_t node) { nodes.push_back(node); });

  EXPECT_THAT(sorted(nodes), expected_range(0, 8));
}

TEST(IterationTest, chunk_random_workspace_rebuilds_when_bucket_limit_changes) {
  const auto graph = make_bucketed_graph_with_isolated_nodes();
  iteration::ChunkRandomNodeOrderWorkspace<std::uint32_t, 4, 4> workspace;

  iteration::ChunkRandomNodeOrder first_order(
      graph, workspace, iteration::NodeRange<std::uint32_t>{0, graph.n()}, 2, 1
  );
  std::vector<std::uint32_t> first_nodes;
  first_order.for_each([&](const std::uint32_t node) { first_nodes.push_back(node); });

  iteration::ChunkRandomNodeOrder second_order(
      graph, workspace, iteration::NodeRange<std::uint32_t>{0, graph.n()}, 2, 3
  );
  std::vector<std::uint32_t> second_nodes;
  second_order.for_each([&](const std::uint32_t node) { second_nodes.push_back(node); });

  EXPECT_THAT(sorted(first_nodes), expected_range(0, 5));
  EXPECT_THAT(sorted(second_nodes), expected_range(0, graph.n()));
}

TEST(IterationTest, parallel_order_reuses_local_state_for_worker_chunk) {
  struct LocalState {
    tbb::concurrent_vector<std::size_t> *batch_sizes;
    std::size_t num_nodes = 0;

    ~LocalState() {
      batch_sizes->push_back(num_nodes);
    }
  };

  const auto graph = make_bucketed_graph_with_isolated_nodes();
  iteration::ChunkRandomNodeOrderWorkspace<std::uint32_t, 4, 4> workspace;
  iteration::ChunkRandomNodeOrder order(
      graph, workspace, iteration::NodeRange<std::uint32_t>{0, graph.n()}, 2
  );

  tbb::concurrent_vector<std::uint32_t> nodes;
  tbb::concurrent_vector<std::size_t> batch_sizes;
  order.parallel_for_each_with_local(
      [&] { return LocalState{&batch_sizes}; },
      [&](LocalState &local, const std::uint32_t node) {
        ++local.num_nodes;
        nodes.push_back(node);
      }
  );
  std::vector<std::uint32_t> copied_nodes(nodes.begin(), nodes.end());

  EXPECT_THAT(sorted(copied_nodes), expected_range(0, graph.n()));
  EXPECT_TRUE(std::any_of(batch_sizes.begin(), batch_sizes.end(), [](const std::size_t size) {
    return size > 1;
  }));
}

TEST(IterationTest, chunk_random_workspace_rebuilds_when_range_changes) {
  const auto graph = make_bucketed_graph_with_isolated_nodes();
  iteration::ChunkRandomNodeOrderWorkspace<std::uint32_t, 4, 4> workspace;

  iteration::ChunkRandomNodeOrder first_order(
      graph, workspace, iteration::NodeRange<std::uint32_t>{0, 4}, 2
  );
  std::vector<std::uint32_t> first_nodes;
  first_order.for_each([&](const std::uint32_t node) { first_nodes.push_back(node); });

  iteration::ChunkRandomNodeOrder second_order(
      graph, workspace, iteration::NodeRange<std::uint32_t>{7, 12}, 2
  );
  std::vector<std::uint32_t> second_nodes;
  second_order.for_each([&](const std::uint32_t node) { second_nodes.push_back(node); });

  EXPECT_THAT(sorted(first_nodes), expected_range(0, 4));
  EXPECT_THAT(sorted(second_nodes), expected_range(7, 12));
}

TEST(IterationTest, chunk_random_order_is_seed_reproducible) {
  const auto graph = make_bucketed_graph_with_isolated_nodes();
  tbb::task_arena arena(1);

  Random::reseed(0);
  iteration::ChunkRandomNodeOrderWorkspace<std::uint32_t, 4, 4> first_workspace;
  iteration::ChunkRandomNodeOrder first_order(
      graph, first_workspace, iteration::NodeRange<std::uint32_t>{0, graph.n()}, 2
  );
  std::vector<std::uint32_t> first_nodes;
  arena.execute([&] {
    first_order.for_each([&](const std::uint32_t node) { first_nodes.push_back(node); });
  });

  Random::reseed(0);
  iteration::ChunkRandomNodeOrderWorkspace<std::uint32_t, 4, 4> second_workspace;
  iteration::ChunkRandomNodeOrder second_order(
      graph, second_workspace, iteration::NodeRange<std::uint32_t>{0, graph.n()}, 2
  );
  std::vector<std::uint32_t> second_nodes;
  arena.execute([&] {
    second_order.for_each([&](const std::uint32_t node) { second_nodes.push_back(node); });
  });

  EXPECT_THAT(first_nodes, Eq(second_nodes));
}

} // namespace kaminpar
