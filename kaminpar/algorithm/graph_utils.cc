#include "graph_utils.h"

#include "datastructure/graph.h"
#include "datastructure/marker.h"
#include "datastructure/static_array.h"
#include "definitions.h"
#include "parallel.h"
#include "utility/timer.h"

#include <mutex>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

namespace kaminpar {
bool validate_graph(const Graph &graph) {
  LOG << "Validate n=" << graph.n() << " m=" << graph.m();

  for (NodeID u = 0; u < graph.n(); ++u) {
    ALWAYS_ASSERT(graph.raw_nodes()[u] <= graph.raw_nodes()[u + 1])
    << V(u) << V(graph.raw_nodes()[u]) << V(graph.raw_nodes()[u + 1]);
  }

  for (const NodeID u : graph.nodes()) {
    for (const auto [e, v] : graph.neighbors(u)) {
      ALWAYS_ASSERT(v < graph.n());
      bool found_reverse = false;
      for (const auto [e_prime, u_prime] : graph.neighbors(v)) {
        ALWAYS_ASSERT(u_prime < graph.n());
        if (u != u_prime) { continue; }
        ALWAYS_ASSERT(graph.edge_weight(e) == graph.edge_weight(e_prime))
        << V(e) << V(graph.edge_weight(e)) << V(e_prime) << V(graph.edge_weight(e_prime)) << " Edge from " << u
        << " --> " << v << " --> " << u_prime;
        found_reverse = true;
        break;
      }
      ALWAYS_ASSERT(found_reverse) << u << " --> " << v << " exists with edge " << e << " but no reverse edge found!";
    }
  }
  return true;
}

namespace {
void fill_final_k(scalable_vector<BlockID> &data, const BlockID b0, const BlockID final_k, const BlockID k) {
  const auto [final_k1, final_k2] = math::split_integral(final_k);
  std::array<BlockID, 2> ks{std::clamp<BlockID>(std::ceil(k * 1.0 * final_k1 / final_k), 1, k - 1),
                            std::clamp<BlockID>(std::floor(k * 1.0 * final_k2 / final_k), 1, k - 1)};
  std::array<BlockID, 2> b{b0, b0 + ks[0]};
  data[b[0]] = final_k1;
  data[b[1]] = final_k2;

  if (ks[0] > 1) { fill_final_k(data, b[0], final_k1, ks[0]); }
  if (ks[1] > 1) { fill_final_k(data, b[1], final_k2, ks[1]); }
}
} // namespace

void copy_subgraph_partitions(PartitionedGraph &p_graph,
                              const scalable_vector<StaticArray<BlockID>> &p_subgraph_partitions, const BlockID k_prime,
                              const BlockID input_k, const scalable_vector<NodeID> &mapping) {
  scalable_vector<BlockID> k0(p_graph.k() + 1, k_prime / p_graph.k());
  k0[0] = 0;

  scalable_vector<BlockID> final_ks(k_prime, 1);

  // we are done partitioning? --> use final_ks
  if (k_prime == input_k) { std::copy(p_graph.final_ks().begin(), p_graph.final_ks().end(), k0.begin() + 1); }

  parallel::prefix_sum(k0.begin(), k0.end(), k0.begin()); // blocks of old block i start at k0[i]

  // we are not done partitioning?
  if (k_prime != input_k) {
    ALWAYS_ASSERT(math::is_power_of_2(k_prime));
    const BlockID k_per_block = k_prime / p_graph.k();
    tbb::parallel_for(static_cast<BlockID>(0), p_graph.k(),
                      [&](const BlockID b) { fill_final_k(final_ks, k0[b], p_graph.final_k(b), k_per_block); });
  }

  p_graph.change_k(k_prime);
  tbb::parallel_for(static_cast<NodeID>(0), p_graph.n(), [&](const NodeID &u) {
    const BlockID b = p_graph._partition[u];
    const NodeID s_u = mapping[u];
    p_graph._partition[u] = k0[b] + p_subgraph_partitions[b][s_u];
  });

  p_graph.set_final_ks(std::move(final_ks));
  p_graph.reinit_block_weights();
}

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

  START_TIMER(TIMER_ALLOCATION);
  StaticArray<NodeID> permutation{n};
  StaticArray<NodeID> inverse_permutation{n};
  STOP_TIMER();

  using Buckets = std::array<NodeID, kNumberOfDegreeBuckets + 1>;
  std::vector<Buckets, tbb::cache_aligned_allocator<Buckets>> local_buckets(p + 1);

  tbb::parallel_for(static_cast<int>(0), p, [&](const int id) {
    if (id >= p) { return; }

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

  START_TIMER("Part 2");
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
  ASSERT((old_node_weights.empty() && old_edge_weights.empty()) ||
         (old_node_weights.size() + 1 == old_nodes.size() && old_edge_weights.size() == old_edges.size()));
  const bool is_weighted = old_node_weights.size() + 1 == old_nodes.size();

  const NodeID n = old_nodes.size() - 1;
  ASSERT(n + 1 == new_nodes.size());

  // Build p_nodes, p_node_weights
  tbb::parallel_for(static_cast<NodeID>(0), n, [&](const NodeID u) {
    const NodeID old_u = permutations.new_to_old[u];

    new_nodes[u] = old_nodes[old_u + 1] - old_nodes[old_u];
    if (is_weighted) { new_node_weights[u] = old_node_weights[old_u]; }
  });
  parallel::prefix_sum(new_nodes.begin(), new_nodes.end(), new_nodes.begin());

  // Build p_edges, p_edge_weights
  tbb::parallel_for(static_cast<NodeID>(0), n, [&](const NodeID u) {
    const NodeID old_u = permutations.new_to_old[u];

    for (EdgeID e = old_nodes[old_u]; e < old_nodes[old_u + 1]; ++e) {
      const NodeID v = old_edges[e];
      const EdgeID p_e = --new_nodes[u];
      new_edges[p_e] = permutations.old_to_new[v];
      if (is_weighted) { new_edge_weights[p_e] = old_edge_weights[e]; }
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
  START_TIMER(TIMER_ALLOCATION);
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

  StaticArray<BlockID> partition(graph.n()); // n() should include isolated nodes now
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

std::pair<NodeID, NodeID> find_furthest_away_node(const Graph &graph, const NodeID start_node, Queue<NodeID> &queue,
                                                  Marker<> &marker) {
  queue.push_tail(start_node);
  marker.set<true>(start_node);

  NodeID current_distance = 0;
  NodeID last_node = start_node;
  NodeID remaining_nodes_in_level = 1;
  NodeID nodes_in_next_level = 0;

  while (!queue.empty()) {
    const NodeID u = queue.head();
    queue.pop_head();
    last_node = u;

    for (const NodeID v : graph.adjacent_nodes(u)) {
      if (marker.get(v)) continue;
      queue.push_tail(v);
      marker.set<true>(v);
      ++nodes_in_next_level;
    }

    // keep track of distance from start_node
    ASSERT(remaining_nodes_in_level > 0);
    --remaining_nodes_in_level;
    if (remaining_nodes_in_level == 0) {
      ++current_distance;
      remaining_nodes_in_level = nodes_in_next_level;
      nodes_in_next_level = 0;
    }
  }
  ASSERT(current_distance > 0);
  --current_distance;

  // bfs did not scan the whole graph, i.e., we have disconnected components
  if (marker.first_unmarked_element() < graph.n()) {
    last_node = marker.first_unmarked_element();
    current_distance = std::numeric_limits<NodeID>::max(); // infinity
  }

  marker.reset();
  queue.clear();
  return {last_node, current_distance};
}
} // namespace kaminpar
