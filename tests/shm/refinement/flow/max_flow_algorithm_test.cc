/*******************************************************************************
 * @file:   max_flow_algorithm_test
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#include <memory>
#include <queue>
#include <span>
#include <unordered_set>

#include <gtest/gtest.h>

#include "tests/shm/graph_builder.h"

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/max_flow/edmond_karp_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/fifo_preflow_push_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_flow_algorithm.h"
#include "kaminpar-shm/refinement/flow/util/reverse_edge_index.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::testing {

class MaxFlowAlgorithmTest : public ::testing::TestWithParam<MaxFlowAlgorithm *> {
protected:
  void check_algorithm(
      const Graph &graph,
      const NodeID source,
      const NodeID sink,
      const EdgeWeight expected_cut_value
  ) const {
    const CSRGraph &csr_graph = graph.csr_graph();
    StaticArray<NodeID> reverse_edges = compute_reverse_edge_index(csr_graph);

    MaxFlowAlgorithm &max_flow_algorithm = *GetParam();
    max_flow_algorithm.initialize(csr_graph, reverse_edges, source, sink);
    const auto [flow_value, flow] = max_flow_algorithm.compute_max_flow();

    const std::unordered_set<NodeID> sources{source};
    const std::unordered_set<NodeID> sinks{sink};
    ASSERT_TRUE(debug::is_valid_flow(csr_graph, sources, sinks, flow));
    ASSERT_TRUE(debug::is_max_flow(csr_graph, sources, sinks, flow));

    std::unordered_set<NodeID> reachable_nodes = compute_reachable_nodes(csr_graph, sources, flow);
    const EdgeWeight cut_value = compute_cut_value(csr_graph, reachable_nodes);
    ASSERT_EQ(flow_value, cut_value);
    ASSERT_EQ(cut_value, expected_cut_value);
  }

private:
  [[nodiscard]] static std::unordered_set<NodeID> compute_reachable_nodes(
      const CSRGraph &graph,
      const std::unordered_set<NodeID> &terminals,
      std::span<const EdgeWeight> flow
  ) {
    std::unordered_set<NodeID> reachable_nodes;

    std::queue<NodeID> bfs_queue;
    for (const NodeID terminal : terminals) {
      reachable_nodes.insert(terminal);
      bfs_queue.push(terminal);
    }

    while (!bfs_queue.empty()) {
      const NodeID u = bfs_queue.front();
      bfs_queue.pop();

      graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
        if (reachable_nodes.contains(v) || flow[e] == c) {
          return;
        }

        reachable_nodes.insert(v);
        bfs_queue.push(v);
      });
    }

    return reachable_nodes;
  }

  [[nodiscard]] static EdgeWeight
  compute_cut_value(const CSRGraph &graph, const std::unordered_set<NodeID> &source_side_nodes) {
    EdgeWeight cut_value = 0;

    for (const NodeID u : source_side_nodes) {
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight c) {
        if (source_side_nodes.contains(v)) {
          return;
        }

        cut_value += c;
      });
    }

    return cut_value;
  }
};

TEST_P(MaxFlowAlgorithmTest, Graph1) {
  EdgeBasedGraphBuilder graph_builder;
  graph_builder.add_edge(0, 1, 1);

  Graph graph = graph_builder.build();
  this->check_algorithm(graph, 0, 1, 1);
}

TEST_P(MaxFlowAlgorithmTest, Graph2) {
  EdgeBasedGraphBuilder graph_builder;
  graph_builder.add_edge(0, 1, 1000);
  graph_builder.add_edge(0, 2, 1000);
  graph_builder.add_edge(1, 2, 1);
  graph_builder.add_edge(1, 3, 1000);
  graph_builder.add_edge(2, 3, 1000);

  Graph graph = graph_builder.build();
  this->check_algorithm(graph, 0, 3, 2000);
}

TEST_P(MaxFlowAlgorithmTest, Graph3) {
  EdgeBasedGraphBuilder graph_builder;
  graph_builder.add_edge(0, 1, 3);
  graph_builder.add_edge(0, 2, 1);
  graph_builder.add_edge(1, 3, 3);
  graph_builder.add_edge(2, 3, 5);
  graph_builder.add_edge(2, 4, 4);
  graph_builder.add_edge(4, 5, 2);
  graph_builder.add_edge(3, 6, 2);
  graph_builder.add_edge(5, 6, 3);

  Graph graph = graph_builder.build();
  this->check_algorithm(graph, 0, 6, 4);
}

TEST_P(MaxFlowAlgorithmTest, Graph4) {
  EdgeBasedGraphBuilder graph_builder;
  graph_builder.add_edge(0, 2, 2);
  graph_builder.add_edge(0, 3, 1);
  graph_builder.add_edge(3, 4, 1);
  graph_builder.add_edge(4, 1, 1);
  graph_builder.add_edge(2, 1, 2);
  graph_builder.add_edge(1, 5, 2);

  Graph graph = graph_builder.build();
  this->check_algorithm(graph, 0, 5, 2);
}

static const std::unique_ptr<MaxFlowAlgorithm> edmond_karp =
    std::make_unique<EdmondsKarpAlgorithm>();

INSTANTIATE_TEST_SUITE_P(EdmondKarp, MaxFlowAlgorithmTest, ::testing::Values(edmond_karp.get()));

static const std::unique_ptr<MaxFlowAlgorithm> fifo_preflow_push =
    std::make_unique<FIFOPreflowPushAlgorithm>(FIFOPreflowPushContext(false, 1));

static const std::unique_ptr<MaxFlowAlgorithm> fifo_preflow_push_with_global_relabeling_heurstic =
    std::make_unique<FIFOPreflowPushAlgorithm>(FIFOPreflowPushContext(true, 1));

INSTANTIATE_TEST_SUITE_P(
    FIFOPreflowPush,
    MaxFlowAlgorithmTest,
    ::testing::Values(
        fifo_preflow_push.get(), fifo_preflow_push_with_global_relabeling_heurstic.get()
    )
);

} // namespace kaminpar::shm::testing
