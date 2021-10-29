/*******************************************************************************
 * @file:   contraction_helper.h
 *
 * @author: Daniel Seemaier
 * @date:   29.10.2021
 * @brief:  Utility functions for contracting distributed graphs.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/utility/math.h"
#include "kaminpar/datastructure/ts_navigable_linked_list.h"

#include <algorithm>

namespace dkaminpar::coarsening::helper {
template <typename T> scalable_vector<T> compute_distribution(const T c_global_n, MPI_Comm comm) {
  const auto size = mpi::get_comm_size(comm);

  scalable_vector<GlobalNodeID> distribution(size + 1);
  for (PEID pe = 0; pe < size; ++pe) {
    distribution[pe + 1] = math::compute_local_range<T>(c_global_n, size, pe).second;
  }

  return distribution;
}

/**
 * Constructs a distributed graph from an edge list.
 * @param edge_list List of edges with the following fields: \c u, \c v and \c weight, where \c u is a local node ID and
 * \c v is a global node ID.
 * @return Distributed graph built from the edge list.
 */
template <typename NodeWeightLambda>
DistributedGraph build_distributed_graph_from_edge_list(const auto &edge_list,
                                                        scalable_vector<GlobalNodeID> node_distribution, MPI_Comm comm,
                                                        NodeWeightLambda &&node_weight_lambda) {
  const PEID size = mpi::get_comm_size(comm);
  const PEID rank = mpi::get_comm_rank(comm);
  const NodeID n = node_distribution[rank + 1] - node_distribution[rank];

  // Bucket-sort edge list
  scalable_vector<shm::parallel::IntegralAtomicWrapper<NodeID>> bucket_index(n + 1);
  tbb::parallel_for<std::size_t>(0, edge_list.size(), [&](const std::size_t i) {
    bucket_index[edge_list[i].u].fetch_add(1, std::memory_order_relaxed);
  });
  shm::parallel::prefix_sum(bucket_index.begin(), bucket_index.end(), bucket_index.begin());
  scalable_vector<std::size_t> buckets(edge_list.size());
  tbb::parallel_for<std::size_t>(0, edge_list.size(), [&](const std::size_t i) {
    buckets[bucket_index[edge_list[i].u].fetch_sub(1, std::memory_order_relaxed) - 1] = i;
  });

  // Assertion:
  // Buckets of node u go from bucket_index[u]..bucket_index[u + 1]
  // edge_list[buckets[bucket_index[u] + i]] is the i-th outgoing edge from node u

  // Construct the edges of the graph in thread-local buffers
  struct Edge {
    GlobalNodeID v;
    EdgeWeight weight;
  };
  shm::NavigableLinkedList<NodeID, Edge> edge_buffer_ets;

  scalable_vector<EdgeID> nodes(n + 1);

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](const auto r) {
    auto &edge_buffer = edge_buffer_ets.local();

    for (NodeID u = r.begin(); u != r.end(); ++u) {
      edge_buffer.mark(u);

      const std::size_t u_bucket_start = bucket_index[u];
      const std::size_t u_bucket_end = bucket_index[u + 1];

      // Sort outgoing edges from u by target node
      std::sort(buckets.begin() + u_bucket_start, buckets.begin() + u_bucket_end,
                [&](const auto &lhs, const auto &rhs) { return edge_list[lhs].v < edge_list[rhs].v; });

      // Construct outgoing edges
      EdgeID degree = 0;
      GlobalNodeID current_v = kInvalidGlobalNodeID;
      EdgeWeight current_weight = 0;
      for (std::size_t i = u_bucket_start; i < u_bucket_end; ++i) {
        const GlobalNodeID v = edge_list[buckets[i]].v;
        const EdgeWeight weight = edge_list[buckets[i]].weight;
        if (i > u_bucket_start && v != current_v) {
          edge_buffer.push_back({current_v, current_weight});
          current_v = v;
          current_weight = 0;
          ++degree;
        }

        current_weight += weight;
      }

      if (current_v != kInvalidGlobalNodeID) { // finish last edge if there was at least one edge
        edge_buffer.push_back({current_v, current_weight});
        ++degree;
      }

      nodes[u + 1] = degree;
    }
  });

  shm::parallel::prefix_sum(nodes.begin(), nodes.end(), nodes.begin());
  const auto all_buffered_nodes = shm::ts_navigable_list::combine<NodeID, Edge>(edge_buffer_ets);

  const EdgeID m = nodes.back();
  scalable_vector<NodeID> edges(m);
  scalable_vector<EdgeWeight> edge_weights(m);

  const GlobalNodeID from = node_distribution[rank];
  const GlobalNodeID to = node_distribution[rank + 1];

  // Remap ghost nodes to local nodes
  // TODO parallelize this part
  std::unordered_map<GlobalNodeID, NodeID> global_to_ghost;
  scalable_vector<GlobalNodeID> ghost_to_global;
  scalable_vector<PEID> ghost_owner;

  for (std::size_t i = 0; i < edge_list.size(); ++i) {
    const auto v = edge_list[i].v;
    if ((v < from || v >= to) && !global_to_ghost.contains(v)) { // new ghost node?
      NodeID v_local = ghost_to_global.size();
      global_to_ghost[v] = v_local;
      ghost_to_global[v_local] = v;
      ghost_owner[v_local] = math::compute_local_range_rank<GlobalNodeID>(node_distribution.back(), size, v);
    }
  }
  const NodeID ghost_n = ghost_owner.size();

  // Now construct the coarse graph
  tbb::parallel_for<NodeID>(0, n, [&](const NodeID i) {
    const auto &marker = all_buffered_nodes[i];
    const auto *list = marker.local_list;
    const NodeID u = marker.key;

    // Copy edges to edges + edge weights arrays
    const EdgeID u_degree = nodes[u + 1] - nodes[u];
    const EdgeID first_src_index = marker.position;
    const EdgeID first_dst_index = nodes[u];

    for (EdgeID j = 0; j < u_degree; ++j) {
      const auto to = first_dst_index + j;
      const auto [v, weight] = list->get(first_src_index + j);

      if (v >= from && v < to) { // local node
        edges[to] = static_cast<NodeID>(v - from);
      } else { // ghost node
        edges[to] = global_to_ghost[v];
      }

      edge_weights[to] = weight;
    }
  });

  // node weights for ghost nodes must be computed afterwards
  scalable_vector<NodeWeight> node_weights(n + ghost_n);
  tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) {
    node_weights[u] = node_weight_lambda(u);
  });

  return {std::move(node_distribution),
          compute_distribution<GlobalEdgeID>(m, comm),
          std::move(nodes),
          std::move(edges),
          std::move(node_weights),
          std::move(edge_weights),
          std::move(ghost_owner),
          std::move(ghost_to_global),
          std::move(global_to_ghost),
          comm};
}
} // namespace dkaminpar::coarsening::helper
