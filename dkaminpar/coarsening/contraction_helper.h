/*******************************************************************************
 * @file:   contraction_helper.h
 *
 * @author: Daniel Seemaier
 * @date:   29.10.2021
 * @brief:  Utility functions for contracting distributed graphs.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dkaminpar/utility/math.h"
#include "kaminpar/datastructure/ts_navigable_linked_list.h"
#include "kaminpar/utility/timer.h"

#include <algorithm>

namespace dkaminpar::coarsening::helper {
namespace {
SET_DEBUG(false);
}

struct LocalToGlobalEdge {
  NodeID u;
  EdgeWeight weight;
  GlobalNodeID v;
};

struct DeduplicateEdgeListMemoryContext {
  scalable_vector<Atomic<NodeID>> bucket_index;
  scalable_vector<NodeID> deduplicated_bucket_index;
  scalable_vector<LocalToGlobalEdge> buffer_list;
};

inline std::pair<scalable_vector<LocalToGlobalEdge>, DeduplicateEdgeListMemoryContext>
deduplicate_edge_list(scalable_vector<LocalToGlobalEdge> edge_list, const NodeID n,
                      DeduplicateEdgeListMemoryContext m_ctx) {
  auto &bucket_index = m_ctx.bucket_index;
  auto &deduplicated_bucket_index = m_ctx.deduplicated_bucket_index;
  auto &buffer_list = m_ctx.buffer_list;

  TIMED_SCOPE("Allocation") {
    if (bucket_index.size() < n + 1) {
      bucket_index.resize(n + 1);
    }
    if (deduplicated_bucket_index.size() < n + 1) {
      deduplicated_bucket_index.resize(n + 1);
    }
    if (buffer_list.size() < edge_list.size() + 1) {
      buffer_list.resize(edge_list.size() + 1);
    }
  };

  // sort edges by u and store result in compressed_edge_list
  tbb::parallel_for<NodeID>(0, n + 1, [&](const NodeID u) { bucket_index[u] = 0; });
  tbb::parallel_for<std::size_t>(0, edge_list.size(), [&](const std::size_t i) {
    ASSERT(edge_list[i].u < n);
    bucket_index[edge_list[i].u].fetch_add(1, std::memory_order_relaxed);
  });
  shm::parallel::prefix_sum(bucket_index.begin(), bucket_index.end(), bucket_index.begin());
  tbb::parallel_for<std::size_t>(0, edge_list.size(), [&](const std::size_t i) {
    const std::size_t j = bucket_index[edge_list[i].u].fetch_sub(1, std::memory_order_relaxed) - 1;
    buffer_list[j] = edge_list[i];
  });
  buffer_list.back().v = kInvalidGlobalNodeID; // dummy element

  Atomic<std::size_t> next_edge_list_index = 0;

  // sort outgoing edges for each node and collapse duplicated edges
  tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) {
    const EdgeID first_edge_id = bucket_index[u];
    const EdgeID first_invalid_edge_id = bucket_index[u + 1];

    std::sort(buffer_list.begin() + first_edge_id, buffer_list.begin() + first_invalid_edge_id,
              [&](const auto &lhs, const auto &rhs) { return lhs.v < rhs.v; });

    EdgeID compressed_degree = 0;
    EdgeID current_accumulate = buffer_list.size() - 1; // dummy element

    // count number of compressed edges
    for (EdgeID edge_id = first_edge_id; edge_id < first_invalid_edge_id; ++edge_id) {
      ASSERT(buffer_list[edge_id].u == u);
      if (buffer_list[edge_id].v != buffer_list[current_accumulate].v) {
        ++compressed_degree;
        current_accumulate = edge_id;
      }
    }

    // reset to degree, so that after another prefix sum, it indexes the compressed edge list
    deduplicated_bucket_index[u + 1] = compressed_degree;

    // reserve memory in edge_list -- +1 in first iteration
    std::size_t copy_to_index = next_edge_list_index.fetch_add(compressed_degree, std::memory_order_relaxed) - 1;

    // compress and copy edges to edge_list
    GlobalNodeID current_v = kInvalidGlobalNodeID;
    for (EdgeID edge_id = first_edge_id; edge_id < first_invalid_edge_id; ++edge_id) {
      const auto &edge = buffer_list[edge_id];
      if (edge.v == current_v) {
        ASSERT(edge_list[copy_to_index].u == edge.u);
        ASSERT(edge_list[copy_to_index].v == edge.v);
        edge_list[copy_to_index].weight += edge.weight;
      } else {
        copy_to_index++;
        current_v = edge.v;
        edge_list[copy_to_index] = edge;
      }
    }
  });

  deduplicated_bucket_index[0] = 0;
  shm::parallel::prefix_sum(deduplicated_bucket_index.begin(), deduplicated_bucket_index.begin() + n + 1,
                            deduplicated_bucket_index.begin());
  edge_list.resize(deduplicated_bucket_index[n]);

  return {std::move(edge_list), std::move(m_ctx)};
}

template <typename T>
inline scalable_vector<T> create_perfect_distribution_from_global_count(const T global_count, MPI_Comm comm) {
  const auto size = mpi::get_comm_size(comm);

  scalable_vector<T> distribution(size + 1);
  for (PEID pe = 0; pe < size; ++pe) {
    distribution[pe + 1] = math::compute_local_range<T>(global_count, size, pe).second;
  }

  return distribution;
}

template <typename T>
inline scalable_vector<T> create_distribution_from_local_count(const T local_count, MPI_Comm comm) {
  const auto [size, rank] = mpi::get_comm_info(comm);

  scalable_vector<T> distribution(size + 1);
  mpi::allgather(&local_count, 1, distribution.data() + 1, 1, comm);
  shm::parallel::prefix_sum(distribution.begin(), distribution.end(), distribution.begin());
  distribution.front() = 0;

  return distribution;
}

/**
 * Constructs a distributed graph from an edge list.
 * @param edge_list List of edges with the following fields: \c u, \c v and \c weight, where \c u is a local node ID and
 * \c v is a global node ID.
 * @return Distributed graph built from the edge list.
 */
template <typename NodeWeightLambda, typename FindGhostNodeOwnerLambda>
inline DistributedGraph build_distributed_graph_from_edge_list(const auto &edge_list,
                                                               scalable_vector<GlobalNodeID> node_distribution,
                                                               MPI_Comm comm, NodeWeightLambda &&node_weight_lambda,
                                                               FindGhostNodeOwnerLambda && /* find_ghost_node_owner */) {
  SCOPED_TIMER("Build graph from edge list", TIMER_FINE);

  const PEID rank = mpi::get_comm_rank(comm);
  const NodeID n = node_distribution[rank + 1] - node_distribution[rank];

  // Bucket-sort edge list
  START_TIMER("Bucket sort edge list", TIMER_FINE);
  scalable_vector<Atomic<NodeID>> bucket_index(n + 1);
  tbb::parallel_for<std::size_t>(0, edge_list.size(), [&](const std::size_t i) {
    bucket_index[edge_list[i].u].fetch_add(1, std::memory_order_relaxed);
  });
  shm::parallel::prefix_sum(bucket_index.begin(), bucket_index.end(), bucket_index.begin());
  scalable_vector<std::size_t> buckets(edge_list.size());
  tbb::parallel_for<std::size_t>(0, edge_list.size(), [&](const std::size_t i) {
    buckets[bucket_index[edge_list[i].u].fetch_sub(1, std::memory_order_relaxed) - 1] = i;
  });
  STOP_TIMER(TIMER_FINE);

  // Assertion:
  // Buckets of node u go from bucket_index[u]..bucket_index[u + 1]
  // edge_list[buckets[bucket_index[u] + i]] is the i-th outgoing edge from node u

  // Construct the edges of the graph in thread-local buffers
  struct Edge {
    GlobalNodeID v;
    EdgeWeight weight;
  };
  shm::NavigableLinkedList<NodeID, Edge> edge_buffer_ets;

  START_TIMER("Allocation", TIMER_FINE);
  scalable_vector<EdgeID> nodes(n + 1);
  STOP_TIMER(TIMER_FINE);

  START_TIMER("Build coarse edges", TIMER_FINE);
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

        if (v != current_v) {
          if (current_v != kInvalidGlobalNodeID) {
            edge_buffer.push_back({current_v, current_weight});
            ++degree;
          }

          current_v = v;
          current_weight = 0;
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
  STOP_TIMER(TIMER_FINE);

  START_TIMER("Allocation", TIMER_FINE);
  const EdgeID m = nodes.back();
  scalable_vector<NodeID> edges(m);
  scalable_vector<EdgeWeight> edge_weights(m);
  STOP_TIMER(TIMER_FINE);

  const GlobalNodeID from = node_distribution[rank];
  const GlobalNodeID to = node_distribution[rank + 1];

  // Now construct the coarse graph
  START_TIMER("Construct coarse graph", TIMER_FINE);
  graph::GhostNodeMapper mapper{node_distribution, comm};

  tbb::parallel_for<NodeID>(0, n, [&](const NodeID i) {
    const auto &marker = all_buffered_nodes[i];
    const auto *list = marker.local_list;
    const NodeID u = marker.key;

    // Copy edges to edges + edge weights arrays
    const EdgeID u_degree = nodes[u + 1] - nodes[u];
    const EdgeID first_src_index = marker.position;
    const EdgeID first_dst_index = nodes[u];

    for (EdgeID j = 0; j < u_degree; ++j) {
      const auto dst_index = first_dst_index + j;
      const auto src_index = first_src_index + j;

      const auto [v, weight] = list->get(src_index);

      if (from <= v && v < to) {
        edges[dst_index] = static_cast<NodeID>(v - from);
      } else {
        edges[dst_index] = mapper.new_ghost_node(v);
      }
      edge_weights[dst_index] = weight;
    }
  });

  auto [global_to_ghost, ghost_to_global, ghost_owner] = mapper.finalize();
  const NodeID ghost_n = ghost_to_global.size();
  STOP_TIMER(TIMER_FINE);

  // node weights for ghost nodes must be computed afterwards
  START_TIMER("Construct coarse node weights", TIMER_FINE);
  scalable_vector<NodeWeight> node_weights(n + ghost_n);
  tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) { node_weights[u] = node_weight_lambda(u); });
  STOP_TIMER(TIMER_FINE);

  return {std::move(node_distribution),
          create_distribution_from_local_count<GlobalEdgeID>(m, comm),
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
