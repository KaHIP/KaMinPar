/*******************************************************************************
 * @file:   graph_rearrangement.cc
 * @author: Daniel Seemaier
 * @date:   17.11.2021
 * @brief:  Algorithms to rearrange graphs.
 ******************************************************************************/
#include "kaminpar-shm/graphutils/permutator.h"

#include <algorithm>
#include <cmath>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/parallel/algorithm.h"

namespace kaminpar::shm::graph {

NodePermutations rearrange_graph(
    StaticArray<EdgeID> &nodes,
    StaticArray<NodeID> &edges,
    StaticArray<NodeWeight> &node_weights,
    StaticArray<EdgeWeight> &edge_weights
) {
  START_HEAP_PROFILER("Temporal nodes and edges allocation");
  RECORD("tmp_nodes")
  StaticArray<EdgeID> tmp_nodes(nodes.size(), static_array::noinit);
  RECORD("tmp_edges")
  StaticArray<NodeID> tmp_edges(edges.size(), static_array::noinit);
  RECORD("tmp_node_weights")
  StaticArray<NodeWeight> tmp_node_weights(node_weights.size(), static_array::noinit);
  RECORD("tmp_edge_weights")
  StaticArray<EdgeWeight> tmp_edge_weights(edge_weights.size(), static_array::noinit);
  STOP_HEAP_PROFILER();

  // if we are about to remove all isolated nodes, we place them to the end of
  // the graph data structure this way, we can just cut them off without doing
  // further work
  START_HEAP_PROFILER("Rearrange input graph");
  NodePermutations permutations = compute_node_permutation_by_degree_buckets(nodes);
  START_TIMER("Rearrange input graph");
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
  STOP_HEAP_PROFILER();

  START_TIMER("Deallocation");
  tbb::parallel_invoke(
      [&] { tmp_nodes.free(); },
      [&] { tmp_edges.free(); },
      [&] { tmp_node_weights.free(); },
      [&] { tmp_edge_weights.free(); }
  );
  STOP_TIMER();

  return permutations;
}

Graph rearrange_by_degree_buckets(CSRGraph &old_graph) {
  SCOPED_TIMER("Rearrange by degree-buckets");

  [[maybe_unused]] const NodeID n = old_graph.n();
  auto nodes = old_graph.take_raw_nodes();
  auto edges = old_graph.take_raw_edges();
  auto node_weights = old_graph.take_raw_node_weights();
  auto edge_weights = old_graph.take_raw_edge_weights();

  auto node_permutations = graph::rearrange_graph(nodes, edges, node_weights, edge_weights);

  KASSERT(
      debug::validate_graph(n, nodes, edges, node_weights, edge_weights),
      "graph permutation produced an invalid CSR graph",
      assert::heavy
  );

  Graph new_graph(std::make_unique<CSRGraph>(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), true
  ));
  new_graph.set_permutation(std::move(node_permutations.old_to_new));
  return new_graph;
}

// See https://devblogs.microsoft.com/oldnewthing/20170102-00/?p=95095
template <typename S, typename T, typename U, typename V>
static void apply_permutation(S *u, T *v, U &indices, V size) {
  for (V i = 0; i < size; ++i) {
    V current = i;

    while (i != indices[current]) {
      V next = indices[current];
      std::swap(u[current], u[next]);
      std::swap(v[current], v[next]);
      indices[current] = current;
      current = next;
    }

    indices[current] = current;
  }
}

static void sort_by_compression(
    NodeID *edges_begin,
    NodeID *edges_end,
    bool store_edge_weights,
    tbb::enumerable_thread_specific<std::vector<EdgeID>> &permutation_ets,
    EdgeWeight *edge_weights
) {
  const auto permutate = [&](NodeID *edges_begin, NodeID *edges_end, EdgeWeight *edge_weights) {
    if constexpr (CompressedGraph::kIntervalEncoding) {
      const auto local_degree = static_cast<NodeID>(edges_end - edges_begin);

      if (local_degree < 2) {
        return;
      }

      NodeID interval_len = 1;
      NodeID prev_adjacent_node = *edges_begin;
      NodeID *rot_begin = edges_begin;

      EdgeWeight *rot_edge_weight_begin = edge_weights;
      EdgeWeight *rot_edge_weight_end = edge_weights + 1;

      for (NodeID *iter = edges_begin + 1; iter != edges_end; ++iter) {
        const NodeID adjacent_node = *iter;

        if (prev_adjacent_node + 1 == adjacent_node) {
          interval_len++;

          // The interval ends if there are no more nodes or the next node is not the increment of
          // the current node.
          if (iter + 1 == edges_end || *(iter + 1) != adjacent_node + 1) {
            if (interval_len >= CompressedGraph::kIntervalLengthTreshold) {
              NodeID *rot_end = iter + 1;
              std::rotate(
                  std::reverse_iterator(rot_end),
                  std::reverse_iterator(rot_end) + interval_len,
                  std::reverse_iterator(rot_begin)
              );

              if (store_edge_weights) {
                std::rotate(
                    std::reverse_iterator(rot_edge_weight_end + 1),
                    std::reverse_iterator(rot_edge_weight_end + 1) + interval_len,
                    std::reverse_iterator(rot_edge_weight_begin)
                );
              }

              rot_begin += interval_len;
              rot_edge_weight_begin += interval_len;
            }

            interval_len = 1;
          }
        }

        prev_adjacent_node = adjacent_node;
        rot_edge_weight_end++;
      }
    };
  };

  const auto degree = static_cast<NodeID>(edges_end - edges_begin);

  if (store_edge_weights) {
    auto &permutation = permutation_ets.local();
    permutation.clear();
    permutation.resize(degree);

    for (EdgeID i = 0; i != degree; ++i) {
      permutation[i] = i;
    }

    std::sort(permutation.begin(), permutation.end(), [&](const EdgeID u, const EdgeID v) {
      return edges_begin[u] < edges_begin[v];
    });

    apply_permutation(edges_begin, edge_weights, permutation, static_cast<EdgeID>(degree));
  } else {
    std::sort(edges_begin, edges_end);
  }

  const bool split_neighbourhood = degree >= CompressedGraph::kHighDegreeThreshold;
  if (split_neighbourhood) {
    NodeID part_count = ((degree % CompressedGraph::kHighDegreePartLength) == 0)
                            ? (degree / CompressedGraph::kHighDegreePartLength)
                            : ((degree / CompressedGraph::kHighDegreePartLength) + 1);
    NodeID last_part_length = ((degree % CompressedGraph::kHighDegreePartLength) == 0)
                                  ? CompressedGraph::kHighDegreePartLength
                                  : (degree % CompressedGraph::kHighDegreePartLength);

    for (NodeID i = 0; i < part_count; ++i) {
      NodeID *part_edges = edges_begin + i * CompressedGraph::kHighDegreePartLength;
      EdgeWeight *part_edge_weights = edge_weights + i * CompressedGraph::kHighDegreePartLength;

      const bool last_part = i + 1 == part_count;
      NodeID part_length = last_part ? last_part_length : CompressedGraph::kHighDegreePartLength;

      permutate(part_edges, part_edges + part_length, part_edge_weights);
    }
  } else {
    permutate(edges_begin, edges_end, edge_weights);
  }
}

void reorder_edges_by_compression(CSRGraph &graph) {
  SCOPED_HEAP_PROFILER("Reorder edges of input graph");
  SCOPED_TIMER("Reorder edges of input");

  StaticArray<EdgeID> &raw_nodes = graph.raw_nodes();
  StaticArray<NodeID> &raw_edges = graph.raw_edges();
  StaticArray<EdgeWeight> &raw_edge_weights = graph.raw_edge_weights();

  tbb::enumerable_thread_specific<std::vector<EdgeID>> permutation_ets;
  graph.pfor_nodes([&](const NodeID node) {
    NodeID *edges_begin = raw_edges.data() + raw_nodes[node];
    NodeID *edges_end = raw_edges.data() + raw_nodes[node + 1];

    const bool store_edge_weights = graph.is_edge_weighted();
    EdgeWeight *edge_weights =
        store_edge_weights ? (raw_edge_weights.data() + raw_nodes[node]) : nullptr;

    sort_by_compression(edges_begin, edges_end, store_edge_weights, permutation_ets, edge_weights);
  });
}

template <typename NodeContainer, typename NodeWeightContainer>
std::pair<NodeID, NodeWeight>
find_isolated_nodes_info(const NodeContainer &nodes, const NodeWeightContainer &node_weights) {
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

template <typename Graph>
void remove_isolated_nodes_generic_graph(Graph &graph, PartitionContext &p_ctx) {
  auto &nodes = graph.raw_nodes();
  auto &node_weights = graph.raw_node_weights();

  const NodeWeight total_node_weight =
      node_weights.empty() ? nodes.size() - 1 : parallel::accumulate(node_weights, 0);
  const auto [isolated_nodes, isolated_nodes_weight] =
      find_isolated_nodes_info(nodes, node_weights);

  const NodeID old_n = nodes.size() - 1;
  const NodeID new_n = old_n - isolated_nodes;
  const NodeWeight new_weight = total_node_weight - isolated_nodes_weight;

  const BlockID k = p_ctx.k;
  const double old_max_block_weight = (1 + p_ctx.epsilon) * std::ceil(1.0 * total_node_weight / k);
  const double new_epsilon =
      new_weight > 0 ? old_max_block_weight / std::ceil(1.0 * new_weight / k) - 1 : 0.0;
  p_ctx.epsilon = new_epsilon;
  p_ctx.n = new_n;
  p_ctx.total_node_weight = new_weight;

  graph.remove_isolated_nodes(isolated_nodes);
}

void remove_isolated_nodes(Graph &graph, PartitionContext &p_ctx) {
  SCOPED_TIMER("Remove isolated nodes");
  graph.reified([&](auto &graph) { remove_isolated_nodes_generic_graph(graph, p_ctx); });
}

template <typename Graph>
NodeID integrate_isolated_nodes_generic_graph(Graph &graph, const double epsilon, Context &ctx) {
  const NodeID num_nonisolated_nodes = graph.n(); // this becomes the first isolated node

  graph.integrate_isolated_nodes();

  const NodeID num_isolated_nodes = graph.n() - num_nonisolated_nodes;

  // note: max block weights should not change
  ctx.partition.epsilon = epsilon;

  return num_isolated_nodes;
}

NodeID integrate_isolated_nodes(Graph &graph, double epsilon, Context &ctx) {
  NodeID num_isolated_nodes = graph.reified([&](auto &graph) {
    return integrate_isolated_nodes_generic_graph(graph, epsilon, ctx);
  });

  ctx.setup(graph);
  return num_isolated_nodes;
}

PartitionedGraph assign_isolated_nodes(
    PartitionedGraph p_graph, const NodeID num_isolated_nodes, const PartitionContext &p_ctx
) {
  const Graph &graph = p_graph.graph();
  const NodeID num_nonisolated_nodes = graph.n() - num_isolated_nodes;

  // The following call graph.n() should include isolated nodes now
  RECORD("partition")
  StaticArray<BlockID> partition(graph.n(), static_array::noinit);

  // copy partition of non-isolated nodes
  tbb::parallel_for<NodeID>(0, num_nonisolated_nodes, [&](const NodeID u) {
    partition[u] = p_graph.block(u);
  });

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

} // namespace kaminpar::shm::graph
