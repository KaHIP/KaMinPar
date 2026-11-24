/*******************************************************************************
 * @file:   graph_rearrangement.cc
 * @author: Daniel Seemaier
 * @date:   17.11.2021
 * @brief:  Algorithms to rearrange graphs.
 ******************************************************************************/
#include "kaminpar-shm/graphutils/permutator.h"

#include <algorithm>
#include <numeric>
#include <set>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_invoke.h>
#include <tbb/task_arena.h>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/heap_profiler.h"

namespace kaminpar::shm::graph {

NodePermutations rearrange_graph(
    StaticArray<EdgeID> &nodes,
    StaticArray<NodeID> &edges,
    StaticArray<NodeWeight> &node_weights,
    StaticArray<EdgeWeight> &edge_weights
) {
  START_HEAP_PROFILER("Temporal nodes and edges allocation");
  StaticArray<EdgeID> tmp_nodes(nodes.size(), static_array::noinit);
  StaticArray<NodeID> tmp_edges(edges.size(), static_array::noinit);
  StaticArray<NodeWeight> tmp_node_weights(node_weights.size(), static_array::noinit);
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

  const NodeID n = old_graph.n();
  const CSRGraph::PrecomputedStats stats{
      .total_node_weight = old_graph.total_node_weight(),
      .max_node_weight = old_graph.max_node_weight(),
      .total_edge_weight = old_graph.total_edge_weight(),
      .max_edge_weight = old_graph.max_edge_weight(),
      .max_degree = old_graph.max_degree(),
  };

  auto nodes = old_graph.take_raw_nodes();
  auto edges = old_graph.take_raw_edges();
  auto node_weights = old_graph.take_raw_node_weights();
  auto edge_weights = old_graph.take_raw_edge_weights();

  constexpr std::size_t kNumBuckets = kNumberOfDegreeBuckets<NodeID>;
  std::vector<NodeID> csr_buckets(kNumBuckets + 1, 0);
  std::size_t number_of_buckets = 0;

  if (n == 0) {
    CSRGraph new_graph(
        std::move(nodes),
        std::move(edges),
        std::move(node_weights),
        std::move(edge_weights),
        true,
        stats,
        std::move(csr_buckets),
        0
    );
    return Graph(std::make_unique<CSRGraph>(std::move(new_graph)));
  }

  START_HEAP_PROFILER("Rearrange input graph");
  const std::size_t cpus = std::min<std::size_t>(tbb::this_task_arena::max_concurrency(), n);
  CacheAlignedVector<std::vector<NodeID>> local_perm_counts(
      cpus + 1, std::vector<NodeID>(kNumBuckets + 1, 0)
  );
  CacheAlignedVector<std::vector<NodeID>> local_csr_counts(
      cpus, std::vector<NodeID>(kNumBuckets + 1, 0)
  );
  struct CpuOrderInfo {
    bool sorted = true;
    bool non_empty = false;
    NodeID first_b = 0;
    NodeID last_b = 0;
  };
  CacheAlignedVector<CpuOrderInfo> cpu_order(cpus);

  const EdgeID *__restrict__ nodes_ptr = nodes.data();
  {
    SCOPED_TIMER("Sort nodes by integer buckets");
    parallel::deterministic_for<NodeID>(
        0, n, [&](const NodeID from, const NodeID to, const std::size_t cpu) {
          NodeID *__restrict__ my_perm = local_perm_counts[cpu + 1].data();
          NodeID *__restrict__ my_csr = local_csr_counts[cpu].data();
          auto &order = cpu_order[cpu];

          if (from < to) {
            order.non_empty = true;
            NodeID prev_b = 0;
            for (NodeID u = from; u < to; ++u) {
              const NodeID deg = static_cast<NodeID>(nodes_ptr[u + 1] - nodes_ptr[u]);
              const std::size_t csr_b = degree_bucket(deg);
              const NodeID perm_b = (deg == 0) ? (kNumBuckets - 1) : static_cast<NodeID>(csr_b);
              ++my_perm[perm_b];
              ++my_csr[csr_b + 1];

              if (u == from) {
                order.first_b = perm_b;
              } else if (perm_b < prev_b) {
                order.sorted = false;
              }
              prev_b = perm_b;
            }
            order.last_b = prev_b;
          }
        }
    );

    for (std::size_t cpu = 0; cpu < cpus; ++cpu) {
      for (std::size_t i = 0; i < kNumBuckets + 1; ++i) {
        csr_buckets[i] += local_csr_counts[cpu][i];
      }
    }
    auto last_nonempty_bucket =
        std::find_if(csr_buckets.rbegin(), csr_buckets.rend(), [](const auto c) { return c > 0; });
    number_of_buckets =
        (last_nonempty_bucket == csr_buckets.rend())
            ? 0
            : std::distance(csr_buckets.begin(), (last_nonempty_bucket + 1).base());
    std::partial_sum(csr_buckets.begin(), csr_buckets.end(), csr_buckets.begin());
  }

  bool already_sorted = true;
  NodeID prev_cpu_last_b = 0;
  bool seen_non_empty = false;
  for (std::size_t cpu = 0; cpu < cpus; ++cpu) {
    if (!cpu_order[cpu].non_empty) {
      continue;
    }
    if (!cpu_order[cpu].sorted || (seen_non_empty && cpu_order[cpu].first_b < prev_cpu_last_b)) {
      already_sorted = false;
      break;
    }
    prev_cpu_last_b = cpu_order[cpu].last_b;
    seen_non_empty = true;
  }

  if (already_sorted) {
    STOP_HEAP_PROFILER();
    CSRGraph new_graph(
        std::move(nodes),
        std::move(edges),
        std::move(node_weights),
        std::move(edge_weights),
        true,
        stats,
        std::move(csr_buckets),
        number_of_buckets
    );
    return Graph(std::make_unique<CSRGraph>(std::move(new_graph)));
  }

  NodePermutations node_permutations;
  {
    SCOPED_TIMER("Sort nodes by integer buckets");
    node_permutations.old_to_new = StaticArray<NodeID>(n, static_array::noinit);
    node_permutations.new_to_old = StaticArray<NodeID>(n, static_array::noinit);

    std::vector<NodeID> global_buckets(kNumBuckets + 1, 0);
    for (std::size_t id = 1; id < cpus + 1; ++id) {
      for (std::size_t i = 0; i < kNumBuckets; ++i) {
        global_buckets[i + 1] += local_perm_counts[id][i];
      }
    }
    parallel::prefix_sum(global_buckets.begin(), global_buckets.end(), global_buckets.begin());
    for (std::size_t i = 0; i < global_buckets.size(); ++i) {
      for (std::size_t id = 0; id + 1 < cpus; ++id) {
        local_perm_counts[id + 1][i] += local_perm_counts[id][i];
      }
      for (std::size_t id = 0; id < cpus; ++id) {
        local_perm_counts[id][i] += global_buckets[i];
      }
    }

    NodeID *__restrict__ perm_ptr = node_permutations.old_to_new.data();
    NodeID *__restrict__ inv_ptr = node_permutations.new_to_old.data();
    parallel::deterministic_for<NodeID>(
        0, n, [&](const NodeID from, const NodeID to, const std::size_t cpu) {
          NodeID *__restrict__ my_offsets = local_perm_counts[cpu].data();
          for (NodeID u = from; u < to; ++u) {
            const NodeID deg = static_cast<NodeID>(nodes_ptr[u + 1] - nodes_ptr[u]);
            const NodeID perm_b =
                (deg == 0) ? (kNumBuckets - 1) : static_cast<NodeID>(degree_bucket(deg));
            const NodeID new_u = my_offsets[perm_b]++;
            perm_ptr[u] = new_u;
            inv_ptr[new_u] = u;
          }
        }
    );
  }
  STOP_HEAP_PROFILER();

  START_HEAP_PROFILER("Temporal nodes and edges allocation");
  StaticArray<EdgeID> tmp_nodes(nodes.size(), static_array::noinit);
  StaticArray<NodeID> tmp_edges(edges.size(), static_array::noinit);
  StaticArray<NodeWeight> tmp_node_weights(node_weights.size(), static_array::noinit);
  StaticArray<EdgeWeight> tmp_edge_weights(edge_weights.size(), static_array::noinit);
  STOP_HEAP_PROFILER();

  START_HEAP_PROFILER("Rearrange input graph");
  START_TIMER("Rearrange input graph");
  build_permuted_graph(
      nodes,
      edges,
      node_weights,
      edge_weights,
      node_permutations,
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

  KASSERT(
      debug::validate_graph(n, nodes, edges, node_weights, edge_weights),
      "graph permutation produced an invalid CSR graph",
      assert::heavy
  );

  CSRGraph new_graph(
      std::move(nodes),
      std::move(edges),
      std::move(node_weights),
      std::move(edge_weights),
      true,
      stats,
      std::move(csr_buckets),
      number_of_buckets
  );
  new_graph.set_permutation(std::move(node_permutations.old_to_new));

  return Graph(std::make_unique<CSRGraph>(std::move(new_graph)));
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
                  std::reverse_iterator<NodeID *>(rot_end),
                  std::reverse_iterator<NodeID *>(rot_end) + interval_len,
                  std::reverse_iterator<NodeID *>(rot_begin)
              );

              if (store_edge_weights) {
                std::rotate(
                    std::reverse_iterator<EdgeWeight *>(rot_edge_weight_end + 1),
                    std::reverse_iterator<EdgeWeight *>(rot_edge_weight_end + 1) + interval_len,
                    std::reverse_iterator<EdgeWeight *>(rot_edge_weight_begin)
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

std::vector<BlockWeight> compute_nonisolated_min_block_weights(
    const PartitionContext &p_ctx, const BlockWeight isolated_weight
) {
  KASSERT(isolated_weight >= 0);

  std::vector<BlockWeight> min_weights(p_ctx.k);
  __uint128_t total_min_weight = 0;
  for (BlockID b = 0; b < p_ctx.k; ++b) {
    total_min_weight += p_ctx.min_block_weight(b);
  }
  if (total_min_weight == 0) {
    return min_weights;
  }

  const __uint128_t reserved_weight =
      std::min(total_min_weight, static_cast<__uint128_t>(isolated_weight));
  __uint128_t prefix_min_weight = 0;
  __uint128_t previous_reservation = 0;
  for (BlockID b = 0; b < p_ctx.k; ++b) {
    prefix_min_weight += p_ctx.min_block_weight(b);
    const __uint128_t reservation = prefix_min_weight * reserved_weight / total_min_weight;
    min_weights[b] =
        p_ctx.min_block_weight(b) - static_cast<BlockWeight>(reservation - previous_reservation);
    previous_reservation = reservation;
  }
  return min_weights;
}

PartitionedGraph assign_isolated_nodes(
    PartitionedGraph p_graph, const NodeID num_isolated_nodes, const PartitionContext &p_ctx
) {
  if (num_isolated_nodes == 0) {
    return p_graph;
  }

  const Graph &graph = p_graph.graph();
  const NodeID num_nonisolated_nodes = graph.n() - num_isolated_nodes;

  // The following call graph.n() should include isolated nodes now
  StaticArray<BlockID> partition(graph.n(), static_array::noinit);

  // Copy partition of non-isolated nodes
  tbb::parallel_for<NodeID>(0, num_nonisolated_nodes, [&](const NodeID u) {
    partition[u] = p_graph.block(u);
  });

  // Now append the isolated ones
  const BlockID k = p_graph.k();
  auto block_weights = p_graph.take_raw_block_weights();
  const auto node_weights = reified(graph, [](const auto &concrete_graph) {
    return concrete_graph.raw_node_weights().view();
  });
  const auto node_weight = [&](const NodeID u) {
    return node_weights.empty() ? NodeWeight{1} : node_weights[u];
  };
  NodeID next_isolated_node = num_nonisolated_nodes;

  if (p_ctx.has_min_block_weights() && graph.max_node_weight() > 1) {
    // Weighted nodes may not fit the next underloaded block. Keep all feasible targets
    // available for later, lighter nodes instead of discarding their remaining capacity.
    using Capacity = std::pair<BlockWeight, BlockID>;
    std::set<Capacity> capacities;
    std::set<Capacity> underloaded_capacities;
    for (BlockID b = 0; b < k; ++b) {
      const Capacity entry = {p_ctx.max_block_weight(b) - block_weights[b], b};
      capacities.insert(entry);
      if (block_weights[b] < p_ctx.min_block_weight(b)) {
        underloaded_capacities.insert(entry);
      }
    }

    for (NodeID u = next_isolated_node; u < graph.n(); ++u) {
      const NodeWeight weight = node_weight(u);
      auto target = underloaded_capacities.lower_bound({weight, 0});
      BlockID to;
      if (target != underloaded_capacities.end()) {
        to = target->second;
      } else {
        const auto feasible = capacities.lower_bound({weight, 0});
        // Indivisible weighted nodes need not admit a feasible greedy packing. If none fits,
        // choose the block with the most capacity, as a best-effort fallback.
        to = feasible != capacities.end() ? feasible->second : capacities.rbegin()->second;
      }

      const Capacity old_entry = {p_ctx.max_block_weight(to) - block_weights[to], to};
      capacities.erase(old_entry);
      underloaded_capacities.erase(old_entry);
      partition[u] = to;
      block_weights[to] += weight;
      const Capacity new_entry = {p_ctx.max_block_weight(to) - block_weights[to], to};
      capacities.insert(new_entry);
      if (block_weights[to] < p_ctx.min_block_weight(to)) {
        underloaded_capacities.insert(new_entry);
      }
    }
    return {graph, k, std::move(partition), std::move(block_weights)};
  }

  if (p_ctx.has_min_block_weights()) {
    // Unit-weight nodes can fill every deficit exactly in linear time.
    for (BlockID b = 0; b < k && next_isolated_node < graph.n(); ++b) {
      while (next_isolated_node < graph.n() && block_weights[b] < p_ctx.min_block_weight(b) &&
             block_weights[b] + node_weight(next_isolated_node) <= p_ctx.max_block_weight(b)) {
        partition[next_isolated_node] = b;
        block_weights[b] += node_weight(next_isolated_node);
        ++next_isolated_node;
      }
    }
  }

  BlockID b = 0;

  for (NodeID u = next_isolated_node; u < num_nonisolated_nodes + num_isolated_nodes; ++u) {
    while (b + 1 < k && block_weights[b] + node_weight(u) > p_ctx.max_block_weight(b)) {
      ++b;
    }
    partition[u] = b;
    block_weights[b] += node_weight(u);
  }

  return {graph, k, std::move(partition), std::move(block_weights)};
}

NodeID count_isolated_nodes(const Graph &graph) {
  if (graph.sorted()) {
    if (const auto *csr = dynamic_cast<const CSRGraph *>(graph.underlying_graph());
        csr != nullptr) {
      const NodeID n = csr->n();
      if (n == 0) {
        return 0;
      }
      const EdgeID *__restrict__ nodes = csr->raw_nodes().data();
      const EdgeID m = nodes[n];
      if (nodes[n - 1] < m) {
        return 0;
      }
      const EdgeID *first_isolated = std::lower_bound(nodes, nodes + n, m);
      return static_cast<NodeID>(n - (first_isolated - nodes));
    }
  }

  tbb::enumerable_thread_specific<NodeID> isolated_nodes_ets;

  reified(graph, [&](const auto &graph) {
    graph.pfor_nodes_range([&](const auto &range) {
      auto &isolated_nodes = isolated_nodes_ets.local();

      for (NodeID u = range.begin(); u != range.end(); ++u) {
        if (graph.degree(u) == 0) {
          ++isolated_nodes;
        }
      }
    });
  });

  return isolated_nodes_ets.combine(std::plus{});
}

} // namespace kaminpar::shm::graph
