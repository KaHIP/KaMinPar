/*******************************************************************************
 * @file:   distributed_graph_helpers.h
 * @author: Daniel Seemaier
 * @date:   29.04.2022
 * @brief:  Utility functions to handle distributed graphs in unit tests.
 ******************************************************************************/
#pragma once

#include <vector>

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/graphutils/communication.h"

#include "kaminpar-shm/datastructures/graph.h"

#include "kaminpar-common/parallel/atomic.h"

namespace kaminpar::dist::testing {
inline std::vector<NodeID> local_neighbors(const shm::Graph &graph, const NodeID u) {
  std::vector<NodeID> neighbors;
  for (const auto &[e, v] : graph.neighbors(u)) {
    neighbors.push_back(v);
  }
  return neighbors;
}

inline std::vector<NodeID> local_neighbors(const DistributedGraph &graph, const NodeID u) {
  std::vector<NodeID> neighbors;
  for (const auto &[e, v] : graph.neighbors(u)) {
    neighbors.push_back(v);
  }
  return neighbors;
}

inline std::vector<GlobalNodeID> global_neighbors(const DistributedGraph &graph, const NodeID u) {
  std::vector<GlobalNodeID> neighbors;
  for (const auto &[e, v] : graph.neighbors(u)) {
    neighbors.push_back(graph.local_to_global_node(v));
  }
  return neighbors;
}

inline DistributedPartitionedGraph make_partitioned_graph(
    const DistributedGraph &graph, const BlockID k, const std::vector<BlockID> &local_partition
) {
  StaticArray<BlockID> partition(graph.total_n());
  StaticArray<BlockWeight> local_block_weights(k);

  std::copy(local_partition.begin(), local_partition.end(), partition.begin());
  for (const NodeID u : graph.nodes()) {
    local_block_weights[partition[u]] += graph.node_weight(u);
  }

  StaticArray<BlockWeight> block_weights(k);
  mpi::allreduce(local_block_weights.data(), block_weights.data(), k, MPI_SUM, MPI_COMM_WORLD);

  struct NodeBlock {
    GlobalNodeID global_node;
    BlockID block_weights;
  };

  mpi::graph::sparse_alltoall_interface_to_pe<NodeBlock>(
      graph,
      [&](const NodeID u) {
        return NodeBlock{graph.local_to_global_node(u), local_partition[u]};
      },
      [&](const auto &buffer) {
        for (const auto &[global_node, block] : buffer) {
          partition[graph.global_to_local_node(global_node)] = block;
        }
      }
  );

  return {&graph, k, std::move(partition), std::move(block_weights)};
}

inline DistributedPartitionedGraph make_partitioned_graph_by_rank(const DistributedGraph &graph) {
  const PEID rank = mpi::get_comm_rank(graph.communicator());
  const PEID size = mpi::get_comm_size(graph.communicator());
  std::vector<BlockID> local_partition(graph.n(), rank);
  return make_partitioned_graph(graph, size, local_partition);
}

//! Return the id of the edge connecting two adjacent nodes \c u and \c v in \c
//! graph, found by linear search.
inline std::pair<EdgeID, EdgeID>
get_edge_by_endpoints(const DistributedGraph &graph, const NodeID u, const NodeID v) {
  EdgeID forward_edge = kInvalidEdgeID;
  EdgeID backward_edge = kInvalidEdgeID;

  if (graph.is_owned_node(u)) {
    for (const auto [cur_e, cur_v] : graph.neighbors(u)) {
      if (cur_v == v) {
        forward_edge = cur_e;
        break;
      }
    }
  }

  if (graph.is_owned_node(v)) {
    for (const auto [cur_e, cur_u] : graph.neighbors(v)) {
      if (cur_u == u) {
        backward_edge = cur_e;
        break;
      }
    }
  }

  // one of those edges might now exist due to ghost nodes
  return {forward_edge, backward_edge};
}

//! Return the id of the edge connecting two adjacent nodes \c u and \c v given
//! by their global id in \c graph, found by linear search
inline std::pair<EdgeID, EdgeID> get_edge_by_endpoints_global(
    const DistributedGraph &graph, const GlobalNodeID u, const GlobalNodeID v
) {
  return get_edge_by_endpoints(graph, graph.global_to_local_node(u), graph.global_to_local_node(v));
}

//! Based on some graph, build a new graph with modified edge weights.
inline DistributedGraph change_edge_weights(
    DistributedGraph graph, const std::vector<std::pair<EdgeID, EdgeWeight>> &changes
) {
  auto edge_weights = graph.take_edge_weights();
  if (edge_weights.empty()) {
    edge_weights.resize(graph.m(), 1);
  }

  for (const auto &[e, weight] : changes) {
    if (e != kInvalidEdgeID) {
      edge_weights[e] = weight;
    }
  }

  return {
      graph.take_node_distribution(),
      graph.take_edge_distribution(),
      graph.take_nodes(),
      graph.take_edges(),
      graph.take_node_weights(),
      std::move(edge_weights),
      graph.take_ghost_owner(),
      graph.take_ghost_to_global(),
      graph.take_global_to_ghost(),
      false,
      graph.communicator()};
}

inline DistributedGraph change_edge_weights_by_endpoints(
    DistributedGraph graph, const std::vector<std::tuple<NodeID, NodeID, EdgeWeight>> &changes
) {
  std::vector<std::pair<EdgeID, EdgeWeight>> edge_id_changes;
  for (const auto &[u, v, weight] : changes) {
    const auto [forward_edge, backward_edge] = get_edge_by_endpoints(graph, u, v);
    edge_id_changes.emplace_back(forward_edge, weight);
    edge_id_changes.emplace_back(backward_edge, weight);
  }

  return change_edge_weights(std::move(graph), edge_id_changes);
}

inline DistributedGraph change_edge_weights_by_global_endpoints(
    DistributedGraph graph,
    const std::vector<std::tuple<GlobalNodeID, GlobalNodeID, EdgeWeight>> &changes
) {
  std::vector<std::pair<EdgeID, EdgeWeight>> edge_id_changes;
  for (const auto &[u, v, weight] : changes) {
    const auto real_u = u % graph.global_n();
    const auto real_v = v % graph.global_n();
    const auto [forward_edge, backward_edge] = get_edge_by_endpoints_global(graph, real_u, real_v);
    edge_id_changes.emplace_back(forward_edge, weight);
    edge_id_changes.emplace_back(backward_edge, weight);
  }

  return change_edge_weights(std::move(graph), edge_id_changes);
}

//! Based on some graph, build a new graph with modified node weights.
inline DistributedGraph change_node_weights(
    DistributedGraph graph, const std::vector<std::pair<NodeID, NodeWeight>> &changes
) {
  auto node_weights = graph.take_node_weights();
  if (node_weights.empty()) {
    node_weights.resize(graph.total_n(), 1);
  }

  for (const auto &[u, weight] : changes) {
    node_weights[u] = weight;
  }

  return {
      graph.take_node_distribution(),
      graph.take_edge_distribution(),
      graph.take_nodes(),
      graph.take_edges(),
      std::move(node_weights),
      graph.take_edge_weights(),
      graph.take_ghost_owner(),
      graph.take_ghost_to_global(),
      graph.take_global_to_ghost(),
      false,
      graph.communicator()};
}
} // namespace kaminpar::dist::testing
