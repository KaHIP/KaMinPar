/*******************************************************************************
 * @file:   graph_permutation.cc
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Computes graph permutations and builds the permuted graph.
 ******************************************************************************/
#include "kaminpar/algorithm/graph_permutation.h"

#include "kaminpar/utility/timer.h"

namespace kaminpar::graph {
/*
 * Builds a node permutation perm[x] such that the following condition is satisfied:
 * let
 * - n0 be the number of nodes with degree zero
 * - and ni be the number of nodes with degree in 2^(i - 1)..(2^i)-1
 * then
 * - perm[0..n0-1] contains all nodes with degree zero
 * - and perm[ni..n(i + 1)-1] contains all nodes with degree 2^(i - 1)..(2^i)-1
 */
NodePermutations sort_by_degree_buckets(const StaticArray<EdgeID> &nodes, const bool deg0_position) {
  auto find_bucket = [&](const Degree deg) {
    return (deg0_position && deg == 0) ? kNumberOfDegreeBuckets - 1 : degree_bucket(deg);
  };

  const NodeID n = nodes.size() - 1;
  const int p = std::min<int>(tbb::this_task_arena::max_concurrency(), n);

  START_TIMER("Allocation");
  StaticArray<NodeID> permutation{n};
  StaticArray<NodeID> inverse_permutation{n};
  STOP_TIMER();

  using Buckets = std::array<NodeID, kNumberOfDegreeBuckets + 1>;
  std::vector<Buckets, tbb::cache_aligned_allocator<Buckets>> local_buckets(p + 1);

  tbb::parallel_for(static_cast<int>(0), p, [&](const int id) {
    if (id >= p) { return; }

    // TODO is there a nicer way to do this?
    auto &my_buckets = local_buckets[id + 1];
    const NodeID chunk = n / p;
    const NodeID rem = n % p;
    const NodeID from = id * chunk + std::min(id, static_cast<int>(rem));
    const NodeID to = from + ((id < static_cast<int>(rem)) ? chunk + 1 : chunk);

    for (NodeID u = from; u < to; ++u) {
      const Degree bucket = find_bucket(nodes[u + 1] - nodes[u]);
      permutation[u] = my_buckets[bucket]++;
    }
  });

  // Build a table of prefix numbers to correct the position of each node in the final permutation
  // After the previous loop, permutation[u] contains the position of u in the thread-local bucket.
  // (i) account for smaller buckets --> add prefix computed in global_buckets
  // (ii) account for the same bucket in smaller processor IDs --> add prefix computed in local_buckets
  Buckets global_buckets{};
  for (int id = 1; id < p + 1; ++id) {
    for (std::size_t i = 0; i + 1 < global_buckets.size(); ++i) { global_buckets[i + 1] += local_buckets[id][i]; }
  }
  parallel::prefix_sum(global_buckets.begin(), global_buckets.end(), global_buckets.begin());
  for (std::size_t i = 0; i < global_buckets.size(); ++i) {
    for (int id = 0; id + 1 < p; ++id) { local_buckets[id + 1][i] += local_buckets[id][i]; }
  }

  START_TIMER("Build permutation");
  tbb::parallel_for(static_cast<int>(0), p, [&](const int id) {
    if (id >= p) { return; }
    auto &my_buckets = local_buckets[id];
    const NodeID chunk = n / p;
    const NodeID rem = n % p;
    const NodeID from = id * chunk + std::min(id, static_cast<int>(rem));
    const NodeID to = from + ((id < static_cast<int>(rem)) ? chunk + 1 : chunk);
    for (NodeID u = from; u < to; ++u) {
      const Degree bucket = find_bucket(nodes[u + 1] - nodes[u]);
      permutation[u] += global_buckets[bucket] + my_buckets[bucket];
    }
  });
  STOP_TIMER();

  START_TIMER("Invert permutation");
  tbb::parallel_for(static_cast<std::size_t>(1), nodes.size(), [&](const NodeID u_plus_one) {
    const NodeID u = u_plus_one - 1;
    inverse_permutation[permutation[u]] = u;
  });
  STOP_TIMER();

  return {std::move(permutation), std::move(inverse_permutation)};
}

/*
 * Applies a node permutation `permutation` to a graph given as adjacency array.
 */
void build_permuted_graph(const StaticArray<EdgeID> &old_nodes, const StaticArray<NodeID> &old_edges,
                          const StaticArray<NodeWeight> &old_node_weights,
                          const StaticArray<EdgeWeight> &old_edge_weights, const NodePermutations &permutations,
                          StaticArray<EdgeID> &new_nodes, StaticArray<NodeID> &new_edges,
                          StaticArray<NodeWeight> &new_node_weights, StaticArray<EdgeWeight> &new_edge_weights) {
  const bool is_node_weighted = old_node_weights.size() + 1 == old_nodes.size();
  const bool is_edge_weighted = old_edge_weights.size() == old_edges.size();

  const NodeID n = old_nodes.size() - 1;
  ASSERT(n + 1 == new_nodes.size());

  // Build p_nodes, p_node_weights
  tbb::parallel_for(static_cast<NodeID>(0), n, [&](const NodeID u) {
    const NodeID old_u = permutations.new_to_old[u];

    new_nodes[u] = old_nodes[old_u + 1] - old_nodes[old_u];
    if (is_node_weighted) { new_node_weights[u] = old_node_weights[old_u]; }
  });
  parallel::prefix_sum(new_nodes.begin(), new_nodes.end(), new_nodes.begin());

  // Build p_edges, p_edge_weights
  tbb::parallel_for(static_cast<NodeID>(0), n, [&](const NodeID u) {
    const NodeID old_u = permutations.new_to_old[u];

    for (EdgeID e = old_nodes[old_u]; e < old_nodes[old_u + 1]; ++e) {
      const NodeID v = old_edges[e];
      const EdgeID p_e = --new_nodes[u];
      new_edges[p_e] = permutations.old_to_new[v];
      if (is_edge_weighted) { new_edge_weights[p_e] = old_edge_weights[e]; }
    }
  });
}

std::pair<NodeID, NodeWeight> find_isolated_nodes_info(const StaticArray<EdgeID> &nodes,
                                                       const StaticArray<NodeWeight> &node_weights) {
  ASSERT(node_weights.empty() || node_weights.size() + 1 == nodes.size());

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

NodePermutations rearrange_and_remove_isolated_nodes(const bool remove_isolated_nodes, PartitionContext &p_ctx,
                                                     StaticArray<EdgeID> &nodes, StaticArray<NodeID> &edges,
                                                     StaticArray<NodeWeight> &node_weights,
                                                     StaticArray<EdgeWeight> &edge_weights,
                                                     NodeWeight total_node_weight) {
  START_TIMER("Allocation");
  StaticArray<EdgeID> tmp_nodes(nodes.size());
  StaticArray<NodeID> tmp_edges(edges.size());
  StaticArray<NodeWeight> tmp_node_weights(node_weights.size());
  StaticArray<EdgeWeight> tmp_edge_weights(edge_weights.size());
  STOP_TIMER();

  // if we are about to remove all isolated nodes, we place them to the end of the graph data structure
  // this way, we can just cut them off without doing further work
  START_TIMER("Rearrange input graph");
  NodePermutations permutations = sort_by_degree_buckets(nodes, remove_isolated_nodes);
  build_permuted_graph(nodes, edges, node_weights, edge_weights, permutations, tmp_nodes, tmp_edges, tmp_node_weights,
                       tmp_edge_weights);
  std::swap(nodes, tmp_nodes);
  std::swap(edges, tmp_edges);
  std::swap(node_weights, tmp_node_weights);
  std::swap(edge_weights, tmp_edge_weights);
  STOP_TIMER();

  if (remove_isolated_nodes) {
    if (total_node_weight == -1) {
      if (node_weights.size() == 0) {
        total_node_weight = nodes.size() - 1;
      } else {
        total_node_weight = parallel::accumulate(node_weights);
      }
    }

    const auto [isolated_nodes, isolated_nodes_weight] = find_isolated_nodes_info(nodes, node_weights);

    const NodeID old_n = nodes.size() - 1;
    const NodeID new_n = old_n - isolated_nodes;
    const NodeWeight new_weight = total_node_weight - isolated_nodes_weight;

    const BlockID k = p_ctx.k;
    const double old_max_block_weight = (1 + p_ctx.epsilon) * std::ceil(1.0 * total_node_weight / k);
    const double new_epsilon = old_max_block_weight / std::ceil(1.0 * new_weight / k) - 1;
    p_ctx.epsilon = new_epsilon;

    nodes.restrict(new_n + 1);
    if (!node_weights.empty()) { node_weights.restrict(new_n); }
  }

  return permutations;
}

PartitionedGraph revert_isolated_nodes_removal(PartitionedGraph p_graph, const NodeID num_isolated_nodes,
                                               const PartitionContext &p_ctx) {
  const Graph &graph = p_graph.graph();
  const NodeID num_nonisolated_nodes = graph.n() - num_isolated_nodes;

  StaticArray<parallel::IntegralAtomicWrapper<BlockID>> partition(graph.n()); // n() should include isolated nodes now
  // copy partition of non-isolated nodes
  tbb::parallel_for(static_cast<NodeID>(0), static_cast<NodeID>(num_nonisolated_nodes),
                    [&](const NodeID u) { partition[u] = p_graph.block(u); });

  // now append the isolated ones
  const BlockID k = p_graph.k();
  auto block_weights = p_graph.take_block_weights();
  BlockID b = 0;

  // TODO parallelize this
  for (NodeID u = num_nonisolated_nodes; u < num_nonisolated_nodes + num_isolated_nodes; ++u) {
    while (b + 1 < k && block_weights[b] + graph.node_weight(u) > p_ctx.max_block_weight(b)) { ++b; }
    partition[u] = b;
    block_weights[b] += graph.node_weight(u);
  }

  return {graph, k, std::move(partition)};
}
} // namespace kaminpar::graph