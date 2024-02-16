/*******************************************************************************
 * @file:   graph_rearrangement.cc
 * @author: Daniel Seemaier
 * @date:   17.11.2021
 * @brief:  Algorithms to rearrange graphs.
 ******************************************************************************/
#include "kaminpar-shm/graphutils/permutator.h"

#include <cmath>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::graph {
namespace {
std::pair<NodeID, NodeWeight> find_isolated_nodes_info(
    const StaticArray<EdgeID> &nodes, const StaticArray<NodeWeight> &node_weights
) {
  KASSERT((node_weights.empty() || node_weights.size() + 1 == nodes.size()));

  tbb::enumerable_thread_specific<NodeID> isolated_nodes;
  tbb::enumerable_thread_specific<NodeWeight> isolated_nodes_weights;
  const bool is_weighted = !node_weights.empty();

  const NodeID n = nodes.size() - 1;
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](const tbb::blocked_range<NodeID> &r) {
    NodeID &local_isolated_nodes = isolated_nodes.local();
    NodeWeight &local_isolated_weights = isolated_nodes_weights.local();

    for (NodeID u = r.begin(); u != r.end(); ++u) {
      if (nodes[u] == nodes[u + 1]) {
        ++local_isolated_nodes;
        local_isolated_weights += is_weighted ? node_weights[u] : 1;
      }
    }
  });

  return {isolated_nodes.combine(std::plus{}), isolated_nodes_weights.combine(std::plus{})};
}
} // namespace

NodePermutations<StaticArray> rearrange_graph(
    PartitionContext &p_ctx,
    StaticArray<EdgeID> &nodes,
    StaticArray<NodeID> &edges,
    StaticArray<NodeWeight> &node_weights,
    StaticArray<EdgeWeight> &edge_weights
) {
  START_TIMER("Allocation");
  StaticArray<EdgeID> tmp_nodes(nodes.size());
  StaticArray<NodeID> tmp_edges(edges.size());
  StaticArray<NodeWeight> tmp_node_weights(node_weights.size());
  StaticArray<EdgeWeight> tmp_edge_weights(edge_weights.size());
  STOP_TIMER();

  // if we are about to remove all isolated nodes, we place them to the end of
  // the graph data structure this way, we can just cut them off without doing
  // further work
  START_TIMER("Rearrange input graph");
  NodePermutations<StaticArray> permutations = sort_by_degree_buckets<>(nodes);
  build_permuted_graph(
      nodes,
      edges,
      node_weights,
      edge_weights,
      permutations,
      tmp_nodes,
      tmp_edges,
      tmp_node_weights,
      tmp_edge_weights
  );
  std::swap(nodes, tmp_nodes);
  std::swap(edges, tmp_edges);
  std::swap(node_weights, tmp_node_weights);
  std::swap(edge_weights, tmp_edge_weights);
  STOP_TIMER();

  const NodeWeight total_node_weight =
      node_weights.empty() ? nodes.size() - 1 : parallel::accumulate(node_weights, 0);
  const auto [isolated_nodes, isolated_nodes_weight] =
      find_isolated_nodes_info(nodes, node_weights);

  const NodeID old_n = nodes.size() - 1;
  const NodeID new_n = old_n - isolated_nodes;
  const NodeWeight new_weight = total_node_weight - isolated_nodes_weight;

  const BlockID k = p_ctx.k;
  const double old_max_block_weight = (1 + p_ctx.epsilon) * std::ceil(1.0 * total_node_weight / k);
  const double new_epsilon = old_max_block_weight / std::ceil(1.0 * new_weight / k) - 1;
  p_ctx.epsilon = new_epsilon;
  p_ctx.n = new_n;
  p_ctx.total_node_weight = new_weight;

  nodes.restrict(new_n + 1);
  if (!node_weights.empty()) {
    node_weights.restrict(new_n);
  }

  return permutations;
}

NodeID integrate_isolated_nodes(Graph &graph, const double epsilon, Context &ctx) {
  const NodeID num_nonisolated_nodes = graph.n(); // this becomes the first isolated node
  graph.raw_nodes().unrestrict();
  graph.raw_node_weights().unrestrict();
  graph.update_total_node_weight();
  const NodeID num_isolated_nodes = graph.n() - num_nonisolated_nodes;

  // note: max block weights should not change
  ctx.partition.epsilon = epsilon;
  ctx.setup(graph);

  return num_isolated_nodes;
}

PartitionedGraph assign_isolated_nodes(
    PartitionedGraph p_graph, const NodeID num_isolated_nodes, const PartitionContext &p_ctx
) {
  const Graph &graph = p_graph.graph();
  const NodeID num_nonisolated_nodes = graph.n() - num_isolated_nodes;

  StaticArray<BlockID> partition(graph.n()); // n() should include isolated nodes now
  // copy partition of non-isolated nodes
  tbb::parallel_for(
      static_cast<NodeID>(0),
      static_cast<NodeID>(num_nonisolated_nodes),
      [&](const NodeID u) { partition[u] = p_graph.block(u); }
  );

  // now append the isolated ones
  const BlockID k = p_graph.k();
  auto block_weights = p_graph.take_raw_block_weights();
  BlockID b = 0;

  // TODO parallelize this
  for (NodeID u = num_nonisolated_nodes; u < num_nonisolated_nodes + num_isolated_nodes; ++u) {
    while (b + 1 < k && block_weights[b] + graph.node_weight(u) > p_ctx.block_weights.max(b)) {
      ++b;
    }
    partition[u] = b;
    block_weights[b] += graph.node_weight(u);
  }

  return {graph, k, std::move(partition)};
}

Graph rearrange_by_degree_buckets(Context &ctx, Graph old_graph) {
  auto nodes = old_graph.take_raw_nodes();
  auto edges = old_graph.take_raw_edges();
  auto node_weights = old_graph.take_raw_node_weights();
  auto edge_weights = old_graph.take_raw_edge_weights();

  auto node_permutations =
      graph::rearrange_graph(ctx.partition, nodes, edges, node_weights, edge_weights);

  KASSERT(
      [&] {
        if (!node_weights.empty() && node_weights.size() + 1 < nodes.size()) {
          LOG_WARNING << "node weights array is not empty, but smaller than the number of nodes";
          return false;
        }
        if (!edge_weights.empty() && edge_weights.size() < edges.size()) {
          LOG_WARNING << "edge weights array is not empty, but smaller than the number of edges";
          return false;
        }
        for (NodeID u = 0; u + 1 < nodes.size(); ++u) {
          if (nodes[u] > nodes[u + 1] || nodes[u + 1] > edges.size()) {
            LOG_WARNING << "invalid nodes[] entry for node " << u;
            return false;
          }
          for (EdgeID e = nodes[u]; e < nodes[u + 1]; ++e) {
            const NodeID v = edges[e];
            if (v + 1 > nodes.size()) {
              LOG_WARNING << "neighbor " << v << " of node " << u << " is out of range";
              return false;
            }
          }
        }
        return true;
      }(),
      "graph permutation produced invalid CSR graph",
      assert::heavy
  );

  Graph new_graph(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), true
  );
  new_graph.set_permutation(std::move(node_permutations.old_to_new));
  return new_graph;
}
} // namespace kaminpar::shm::graph
