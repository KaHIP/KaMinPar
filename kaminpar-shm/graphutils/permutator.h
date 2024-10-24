/*******************************************************************************
 * Algorithms to rearrange graphs.
 *
 * @file:   permutator.h
 * @author: Daniel Seemaier
 * @date:   17.11.2021
 ******************************************************************************/
#pragma once

#include <utility>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/cache_aligned_vector.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/parallel/loops.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::graph {

struct NodePermutations {
  StaticArray<NodeID> old_to_new;
  StaticArray<NodeID> new_to_old;
};

template <std::integral Bucket, typename Lambda>
NodePermutations compute_node_permutation_by_generic_buckets(
    const NodeID n, const Bucket num_buckets, Lambda &&find_bucket
) {
  static_assert(std::is_invocable_r_v<NodeID, Lambda, NodeID>);
  SCOPED_TIMER("Sort nodes by integer buckets");

  const std::size_t cpus = std::min<std::size_t>(tbb::this_task_arena::max_concurrency(), n);

  RECORD("permutation") StaticArray<NodeID> permutation(n);
  RECORD("inverse_permutation") StaticArray<NodeID> inverse_permutation(n);

  // local_buckets[cpu][bucket]: thread-local bucket sizes
  CacheAlignedVector<std::vector<NodeID>> local_buckets(
      cpus + 1, std::vector<NodeID>(num_buckets + 1)
  );

  parallel::deterministic_for<NodeID>(
      0,
      n,
      [&](const NodeID from, const NodeID to, const std::size_t cpu) {
        KASSERT(cpu < cpus);

        for (NodeID u = from; u < to; ++u) {
          const auto bucket = find_bucket(u);
          permutation[u] = local_buckets[cpu + 1][bucket]++;
        }
      }
  );

  // Build a table of prefix numbers to correct the position of each node in the
  // final permutation After the previous loop, permutation[u] contains the
  // position of u in the thread-local bucket. (i) account for smaller buckets
  // --> add prefix computed in global_buckets (ii) account for the same bucket
  // in smaller processor IDs --> add prefix computed in local_buckets
  std::vector<NodeID> global_buckets(num_buckets + 1);
  for (std::size_t id = 1; id < cpus + 1; ++id) {
    for (std::size_t i = 0; i + 1 < global_buckets.size(); ++i) {
      global_buckets[i + 1] += local_buckets[id][i];
    }
  }
  parallel::prefix_sum(global_buckets.begin(), global_buckets.end(), global_buckets.begin());
  for (std::size_t i = 0; i < global_buckets.size(); ++i) {
    for (std::size_t id = 0; id + 1 < cpus; ++id) {
      local_buckets[id + 1][i] += local_buckets[id][i];
    }
  }

  // Apply offsets to obtain global permutation
  parallel::deterministic_for<NodeID>(
      0,
      n,
      [&](const NodeID from, const NodeID to, const std::size_t cpu) {
        KASSERT(cpu < cpus);

        for (NodeID u = from; u < to; ++u) {
          const auto bucket = find_bucket(u);
          permutation[u] += global_buckets[bucket] + local_buckets[cpu][bucket];
        }
      }
  );

  // Compute inverse permutation
  tbb::parallel_for<std::size_t>(0, n, [&](const NodeID u) noexcept {
    inverse_permutation[permutation[u]] = u;
  });

  return {std::move(permutation), std::move(inverse_permutation)};
}

/*!
 * Computes a permutation on the nodes of the graph such that nodes are sorted
 * by their exponentially spaced degree buckets. Isolated nodes moved to the
 * back of the graph.
 *
 * @tparam put_deg0_at_end Whether isolated nodes are moved to the back
 * @param n The number of nodes.
 * @param degrees Function that returns the degree of a node.
 * @return Bidirectional node permutation.
 */
template <typename Lambda>
NodePermutations compute_node_permutation_by_degree_buckets(
    const NodeID n, Lambda &&degrees, const bool put_deg0_at_end = true
) {
  static_assert(std::is_invocable_r_v<NodeID, Lambda, NodeID>);
  return compute_node_permutation_by_generic_buckets(
      n,
      kNumberOfDegreeBuckets<NodeID>,
      [&](const NodeID u) {
        const NodeID deg = degrees(u);

        return deg == 0 ? (put_deg0_at_end ? kNumberOfDegreeBuckets<NodeID> - 1 : 0)
                        : degree_bucket(deg);
      }
  );
}

/*!
 * Computes a permutation on the nodes of the graph such that nodes are sorted
 * by their exponentially spaced degree buckets. Isolated nodes moved to the
 * back of the graph.
 *
 * @tparam Container
 * @param nodes Nodes array of a static graph.
 * @return Bidirectional node permutation.
 */
inline NodePermutations compute_node_permutation_by_degree_buckets(
    const StaticArray<EdgeID> &nodes, const bool put_deg0_at_end = true
) {
  return compute_node_permutation_by_degree_buckets(
      nodes.size() - 1,
      [&](const NodeID u) {
        const NodeID degree = nodes[u + 1] - nodes[u];
        return degree;
      },
      put_deg0_at_end
  );
}

/*!
 * Creates a permuted copy of a graph.
 *
 * @tparam Container
 * @tparam has_ghost_nodes If true, edge targets may not exist. These are not
 * permuted.
 * @param old_nodes Original nodes array of a static graph.
 * @param old_edges Original edges array of a static graph.
 * @param old_node_weights Original node weights, may be empty.
 * @param old_edge_weights Original edge weights, may be empty.
 * @param permutations Node permutation.
 * @param new_nodes New nodes array, must already be allocated.
 * @param new_edges New edges array, must already be allocated.
 * @param new_node_weights New node weights, may be empty iff. the old node
 * weights array is empty.
 * @param new_edge_weights New edge weights, may be empty empty iff. the old
 * edge weights array is empty.
 */
template <
    bool has_ghost_nodes = false,
    typename GraphNodeID = NodeID,
    typename GraphEdgeID = EdgeID,
    typename GraphNodeWeight = NodeWeight,
    typename GraphEdgeWeight = EdgeWeight>
void build_permuted_graph(
    const StaticArray<GraphEdgeID> &old_nodes,
    const StaticArray<GraphNodeID> &old_edges,
    const StaticArray<GraphNodeWeight> &old_node_weights,
    const StaticArray<GraphEdgeWeight> &old_edge_weights,
    const NodePermutations &permutations,
    StaticArray<GraphEdgeID> &new_nodes,
    StaticArray<GraphNodeID> &new_edges,
    StaticArray<GraphNodeWeight> &new_node_weights,
    StaticArray<GraphEdgeWeight> &new_edge_weights
) {
  // >= for ghost nodes in a distributed graph
  const bool is_node_weighted = old_node_weights.size() + 1 >= old_nodes.size();
  const bool is_edge_weighted = old_edge_weights.size() == old_edges.size();

  const GraphNodeID n = old_nodes.size() - 1;
  KASSERT(n + 1 == new_nodes.size());

  // Build p_nodes, p_node_weights
  tbb::parallel_for<GraphNodeID>(0, n, [&](const GraphNodeID u) {
    const auto old_u = permutations.new_to_old[u];

    new_nodes[u] = old_nodes[old_u + 1] - old_nodes[old_u];
    if (is_node_weighted) {
      new_node_weights[u] = old_node_weights[old_u];
    }
  });

  parallel::prefix_sum(new_nodes.begin(), new_nodes.end(), new_nodes.begin());
  new_nodes.back() = n > 0 ? new_nodes[n - 1] : 0;

  // Build p_edges, p_edge_weights
  tbb::parallel_for<GraphNodeID>(0, n, [&](const GraphNodeID u) {
    const NodeID old_u = permutations.new_to_old[u];

    for (auto e = old_nodes[old_u]; e < old_nodes[old_u + 1]; ++e) {
      const auto v = old_edges[e];
      const auto p_e = --new_nodes[u];
      new_edges[p_e] = (!has_ghost_nodes || v < n) ? permutations.old_to_new[v] : v;
      if (is_edge_weighted) {
        new_edge_weights[p_e] = old_edge_weights[e];
      }
    }
  });
}

NodePermutations rearrange_graph(
    StaticArray<EdgeID> &nodes,
    StaticArray<NodeID> &edges,
    StaticArray<NodeWeight> &node_weights,
    StaticArray<EdgeWeight> &edge_weights
);

CSRGraph rearrange_graph(const CSRGraph &graph, const NodePermutations &permutations);

/*!
 * Rearranges the nodes of the graph such that nodes are sorted by their exponentially spaced degree
 * buckets and the isolated nodes are moved to the back of the graph.
 *
 * @param graph The graph to rearrange.
 * @return The rearranged graph.
 */
Graph rearrange_by_degree_buckets(CSRGraph &graph);

/*!
 * Rearrange the neighborhood of each node in a graph, so that the ordering is the same as in the
 * compressed version of the graph.
 *
 * @param graph The graph to rearrange
 */
void reorder_edges_by_compression(CSRGraph &graph);

/*!
 * Removes the isolated nodes of a graph which are located at the back of the graph.
 *
 * @param graph The graph whose isolated nodes to remove.
 * @param p_ctx The parition context to update.
 */
void remove_isolated_nodes(Graph &graph, PartitionContext &p_ctx);

/*!
 * Integrates the isolated nodes of a graph that have been removed.
 *
 * @param graph The graph whose isolated nodes to integrate.
 * @param epsilon The epsilon value before removing the integrated nodes.
 * @param ctx The context to update.
 * @return The number of isolated nodes integrated.
 */
NodeID integrate_isolated_nodes(Graph &graph, double epsilon, Context &ctx);

/*!
 * Assignes isolated nodes to a partition.
 *
 * @param p_graph The partitioned graph whose isolated nodes to assign.
 * @param num_isolated_nodes the number of isolated nodes.
 * @param p_ctx The partition context of the graph.
 * @return The updated partitioned graph.
 */
PartitionedGraph assign_isolated_nodes(
    PartitionedGraph p_graph, const NodeID num_isolated_nodes, const PartitionContext &p_ctx
);

} // namespace kaminpar::shm::graph
