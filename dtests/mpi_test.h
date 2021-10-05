#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/distributed_definitions.h"
#include "dkaminpar/mpi_utils.h"

#include <gmock/gmock.h>
#include <utility>
#include <vector>

namespace dkaminpar::test {
class DistributedGraphFixture : public ::testing::Test {
protected:
  void SetUp() override { std::tie(size, rank) = mpi::get_comm_info(MPI_COMM_WORLD); }

  GlobalNodeID next(const GlobalNodeID u, const GlobalNodeID step = 2, const GlobalNodeID n = 9) {
    return (u + step) % n;
  }

  GlobalNodeID prev(const GlobalNodeID u, const GlobalNodeID step = 2, const GlobalNodeID n = 9) {
    return (u < step) ? n + u - step : u - step;
  }

  PEID size;
  PEID rank;
};

/*
 * Utility function for graphs
 */
namespace graph {
//! Return the id of the edge connecting two adjacent nodes \c u and \c v in \c graph, found by linear search.
std::pair<EdgeID, EdgeID> get_edge_by_endpoints(const DistributedGraph &graph, const NodeID u, const NodeID v) {
  EdgeID forward_edge = kInvalidEdgeID;
  EdgeID backward_edge = kInvalidEdgeID;

  for (const NodeID cur_u : graph.nodes()) {
    for (const auto [cur_e, cur_v] : graph.neighbors(cur_u)) {
      if (cur_v == v) {
        forward_edge = cur_e;
        goto found_forward_edge;
      }
    }
  }
found_forward_edge:

  for (const NodeID cur_v : graph.nodes()) {
    for (const auto [cur_e, cur_u] : graph.neighbors(cur_v)) {
      if (cur_u == u) {
        backward_edge = cur_e;
        goto found_backward_edge;
      }
    }
  }
found_backward_edge:
  ALWAYS_ASSERT(forward_edge != kInvalidEdgeID) << "there is no edge " << u << " --> " << v;
  ALWAYS_ASSERT(backward_edge != kInvalidEdgeID) << "there is no edge " << v << " --> " << u;
  return {forward_edge, backward_edge};
}

//! Return the id of the edge connecting two adjacent nodes \c u and \c v given by their global id in \c graph,
//! found by linear search
std::pair<EdgeID, EdgeID> get_edge_by_endpoints_global(const DistributedGraph &graph, const GlobalNodeID u,
                                                       const GlobalNodeID v) {
  return get_edge_by_endpoints(graph, graph.global_to_local_node(u), graph.global_to_local_node(v));
}

//! Based on some graph, build a new graph with modified edge weights.
DistributedGraph change_edge_weights(DistributedGraph graph,
                                     const std::vector<std::pair<EdgeID, EdgeWeight>> &changes) {
  auto edge_weights = graph.take_edge_weights();
  for (const auto &[e, weight] : changes) {
    ALWAYS_ASSERT(e < edge_weights.size());
    edge_weights[e] = weight;
  }

  return {graph.global_n(),
          graph.global_m(),
          graph.ghost_n(),
          graph.offset_n(),
          graph.offset_m(),
          graph.take_node_distribution(),
          graph.take_edge_distribution(),
          graph.take_nodes(),
          graph.take_edges(),
          graph.take_node_weights(),
          std::move(edge_weights),
          graph.take_ghost_owner(),
          graph.take_ghost_to_global(),
          graph.take_global_to_ghost(),
          graph.communicator()};
}

DistributedGraph change_edge_weights_by_endpoints(DistributedGraph graph,
                                                  const std::vector<std::tuple<NodeID, NodeID, EdgeWeight>> &changes) {
  std::vector<std::pair<EdgeID, EdgeWeight>> edge_id_changes;
  for (const auto &[u, v, weight] : changes) {
    const auto [forward_edge, backward_edge] = get_edge_by_endpoints(graph, u, v);
    edge_id_changes.emplace_back(forward_edge, weight);
    edge_id_changes.emplace_back(backward_edge, weight);
  }

  return change_edge_weights(std::move(graph), edge_id_changes);
}

DistributedGraph change_edge_weights_by_global_endpoints(
    DistributedGraph graph, const std::vector<std::tuple<GlobalNodeID, GlobalNodeID, EdgeWeight>> &changes) {
  std::vector<std::pair<EdgeID, EdgeWeight>> edge_id_changes;
  for (const auto &[u, v, weight] : changes) {
    const auto [forward_edge, backward_edge] = get_edge_by_endpoints_global(graph, u, v);
    edge_id_changes.emplace_back(forward_edge, weight);
    edge_id_changes.emplace_back(backward_edge, weight);
  }

  return change_edge_weights(std::move(graph), edge_id_changes);
}

//! Based on some graph, build a new graph with modified node weights.
DistributedGraph change_node_weights(DistributedGraph graph,
                                     const std::vector<std::pair<NodeID, NodeWeight>> &changes) {
  auto node_weights = graph.take_node_weights();
  for (const auto &[u, weight] : changes) {
    ALWAYS_ASSERT(u < node_weights.size());
    node_weights[u] = weight;
  }

  return {graph.global_n(),
          graph.global_m(),
          graph.ghost_n(),
          graph.offset_n(),
          graph.offset_m(),
          graph.take_node_distribution(),
          graph.take_edge_distribution(),
          graph.take_nodes(),
          graph.take_edges(),
          std::move(node_weights),
          graph.take_edge_weights(),
          graph.take_ghost_owner(),
          graph.take_ghost_to_global(),
          graph.take_global_to_ghost(),
          graph.communicator()};
}
} // namespace graph
} // namespace dkaminpar::test